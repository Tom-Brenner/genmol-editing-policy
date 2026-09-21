#!/usr/bin/env python3
"""Scan ChEMBL parents and summarize untruncated pendant-action counts.

Run on the machine that already holds chembl_37_chemreps.txt.gz. This does not
download the file. ``--offline-no-safe`` is for development only; the decision
record must use real SAFE alignment.

The scan stops after train+validation+test eligible parents (40,000 for V2).
It writes those parents as soon as it stops. A Ctrl-C of a run that was started
before this stop rule does not save anything: the old process holds the list
only in memory.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from pathlib import Path

from rdkit import Chem, rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_manifest import assign_core_disjoint_splits, file_sha
from rollout_policy.chembl import iter_chembl_smiles, standardize_parent
from rollout_policy.eligibility import enumerate_aligned_actions
from rollout_policy.io import (
    artifact_path,
    load_experiment_config,
    resolve_path,
    write_json,
    write_jsonl_atomic,
)
from rollout_policy.scan_limit import should_stop_scan, split_eligible_limit


def _quantile(sorted_values: list[int], q: float) -> int | None:
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return int(sorted_values[index])


def _summarize_cap(rows: list[dict], cap: int | None, cfg: dict) -> dict:
    retained_counts: list[int] = []
    discarded = 0
    eligible_rows: list[dict] = []
    min_actions = int(cfg["parent_filters"]["min_eligible_actions"])
    for row in rows:
        if cap is None:
            kept = int(row["n_verified"]) if int(row["n_verified"]) >= min_actions else 0
        else:
            kept_n = min(int(row["n_verified"]), cap)
            kept = kept_n if kept_n >= min_actions else 0
        if kept == 0:
            continue
        discarded += max(0, int(row["n_verified"]) - kept)
        retained_counts.append(kept)
        eligible_rows.append(row)
    ordered = sorted(retained_counts)
    total_verified = sum(int(row["n_verified"]) for row in eligible_rows)
    split = cfg["split"]
    packing: dict = {"ok": False}
    try:
        splits = assign_core_disjoint_splits(
            [{"parent_id": row["parent_id"], "core_id": row["core_id"]} for row in eligible_rows],
            n_train=int(split["n_train"]),
            n_validation=int(split["n_validation"]),
            n_test=int(split["n_test"]),
            seed=int(split["manifest_seed"]),
        )
        packing = {
            "ok": True,
            "counts": {name: len(group) for name, group in splits.items()},
        }
    except RuntimeError as exc:
        packing = {"ok": False, "error": str(exc)}
    labels_per_action = int(cfg["rollouts"]["labels_per_action"])
    train_val_parents = int(split["n_train"]) + int(split["n_validation"])
    return {
        "cap": "unlimited" if cap is None else cap,
        "n_eligible_parents": len(eligible_rows),
        "n_cores": len({row["core_id"] for row in eligible_rows}),
        "mean_retained": (sum(ordered) / len(ordered)) if ordered else 0.0,
        "median_retained": _quantile(ordered, 0.5),
        "p90_retained": _quantile(ordered, 0.90),
        "p95_retained": _quantile(ordered, 0.95),
        "max_retained": ordered[-1] if ordered else 0,
        "actions_discarded_by_cap": discarded,
        "fraction_actions_discarded": discarded / total_verified if total_verified else 0.0,
        "expected_train_val_label_completions_if_mean_holds": int(
            round(labels_per_action * (sum(ordered) / max(len(ordered), 1)) * train_val_parents)
        ),
        "core_disjoint_packing": packing,
    }


def _write_audit(
    cfg: dict,
    raw_path: Path,
    *,
    rows: list[dict],
    scanned: int,
    stop_reason: str,
    eligible_limit: int,
    max_scan: int | None,
    offline_no_safe: bool,
    counts: Counter[str],
    reason_counts: Counter[str],
    verified_hist: Counter[int],
    examples: dict[str, list[str]],
) -> None:
    parents_path = artifact_path(cfg, "eligible_parents.jsonl")
    write_jsonl_atomic(parents_path, rows)
    n = max(len(rows), 1)
    report = {
        "eligible_limit": eligible_limit,
        "stop_reason": stop_reason,
        "parents_path": str(parents_path),
        "source_sha256": file_sha(raw_path),
        "offline_no_safe": offline_no_safe,
        "scanned": scanned,
        "scan_capped": max_scan is not None or stop_reason == "eligible_limit",
        "n_verified_ge_2": len(rows),
        "verified_histogram": {str(k): verified_hist[k] for k in sorted(verified_hist)},
        "fraction_lt_4": sum(1 for row in rows if row["n_verified"] < 4) / n,
        "fraction_eq_4": sum(1 for row in rows if row["n_verified"] == 4) / n,
        "fraction_gt_4": sum(1 for row in rows if row["n_verified"] > 4) / n,
        "reason_counts": dict(reason_counts),
        "example_ids": examples,
        "stage_counts": dict(counts),
        "caps": [_summarize_cap(rows, cap, cfg) for cap in (4, 6, 8, None)],
        "versions": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
        },
    }
    out = artifact_path(cfg, "action_count_audit.json")
    write_json(out, report)
    print(
        json.dumps(
            {
                "wrote": str(out),
                "parents": str(parents_path),
                "scanned": scanned,
                "n_verified_ge_2": len(rows),
                "stop_reason": stop_reason,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_v2.yaml")
    parser.add_argument("--max-scan", type=int, default=None)
    parser.add_argument(
        "--eligible-limit",
        type=int,
        default=None,
        help="Stop after this many parents with at least min_eligible_actions. "
        "Default is n_train + n_validation + n_test (40000 for V2).",
    )
    parser.add_argument(
        "--offline-no-safe",
        action="store_true",
        help="Development only. Do not use the result to freeze the action cap.",
    )
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    raw_path = resolve_path(cfg["paths"]["chembl_raw"])
    if not raw_path.exists():
        raise SystemExit(f"Missing {raw_path}. Use the chemreps file already on this machine.")

    filters = cfg["parent_filters"]
    min_actions = int(filters["min_eligible_actions"])
    eligible_limit = (
        int(args.eligible_limit)
        if args.eligible_limit is not None
        else split_eligible_limit(cfg)
    )
    counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    verified_hist: Counter[int] = Counter()
    examples: dict[str, list[str]] = {}
    rows: list[dict] = []
    scanned = 0
    stop_reason = "end_of_file"
    try:
        for chembl_id, source_smiles in iter_chembl_smiles(raw_path):
            scanned += 1
            if args.max_scan is not None and scanned > args.max_scan:
                stop_reason = "max_scan"
                break
            parent = standardize_parent(
                chembl_id,
                source_smiles,
                min_heavy_atoms=filters["min_heavy_atoms"],
                max_heavy_atoms=filters["max_heavy_atoms"],
                min_qed=filters["min_qed"],
                max_qed=filters["max_qed"],
            )
            if parent is None:
                counts["standardization_rejected"] += 1
                continue
            mol = Chem.MolFromSmiles(parent.smiles)
            if mol is None:
                counts["standardization_rejected"] += 1
                continue
            diag = enumerate_aligned_actions(
                mol, require_safe_alignment=not args.offline_no_safe
            )
            counts[f"n_rdkit:{diag.n_rdkit}"] += 1
            counts[f"n_aligned:{diag.n_safe_aligned}"] += 1
            counts[f"n_verified:{diag.n_verified}"] += 1
            verified_hist[diag.n_verified] += 1
            for reason in diag.rejection_reasons:
                reason_counts[reason] += 1
                bucket = examples.setdefault(reason, [])
                if len(bucket) < 5:
                    bucket.append(chembl_id)
            if diag.n_verified >= min_actions:
                rows.append(
                    {
                        "parent_id": chembl_id,
                        "core_id": parent.core_smiles,
                        "n_verified": diag.n_verified,
                        "heavy_atoms": parent.heavy_atoms,
                        "qed": parent.qed,
                    }
                )
                if should_stop_scan(len(rows), eligible_limit):
                    stop_reason = "eligible_limit"
                    break
            if scanned % 5000 == 0:
                print(
                    f"scanned={scanned} verified_ge_{min_actions}={len(rows)} "
                    f"stop_at={eligible_limit}",
                    flush=True,
                )
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        print(
            f"Interrupted after scanned={scanned} eligible={len(rows)}. Writing what was kept.",
            flush=True,
        )

    _write_audit(
        cfg,
        raw_path,
        rows=rows,
        scanned=scanned,
        stop_reason=stop_reason,
        eligible_limit=eligible_limit,
        max_scan=args.max_scan,
        offline_no_safe=bool(args.offline_no_safe),
        counts=counts,
        reason_counts=reason_counts,
        verified_hist=verified_hist,
        examples=examples,
    )


if __name__ == "__main__":
    main()
