#!/usr/bin/env python3
"""Scan ChEMBL parents and summarize untruncated pendant-action counts.

Run on the machine that already holds chembl_37_chemreps.txt.gz. This does not
download the file. ``--offline-no-safe`` is for development only; the decision
record must use real SAFE alignment.

The 40,000-parent V2 split is assumed available. This script checks whether a
proposed cap still leaves enough core-disjoint parents to pack 30,000/5,000/5,000.
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
)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_v2.yaml")
    parser.add_argument("--max-scan", type=int, default=None)
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
    counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    verified_hist: Counter[int] = Counter()
    examples: dict[str, list[str]] = {}
    rows: list[dict] = []
    scanned = 0
    for chembl_id, source_smiles in iter_chembl_smiles(raw_path):
        scanned += 1
        if args.max_scan is not None and scanned > args.max_scan:
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
        if diag.n_verified >= 2:
            rows.append(
                {
                    "parent_id": chembl_id,
                    "core_id": parent.core_smiles,
                    "n_verified": diag.n_verified,
                    "heavy_atoms": parent.heavy_atoms,
                    "qed": parent.qed,
                }
            )
        if scanned % 5000 == 0:
            print(
                f"scanned={scanned} standardized_kept={len(rows)} "
                f"verified_ge_2={len(rows)}",
                flush=True,
            )

    at_least_two = [row for row in rows if row["n_verified"] >= 2]
    n = max(len(at_least_two), 1)
    report = {
        "assumption": "40000 eligible parents are assumed available for the V2 split",
        "source_sha256": file_sha(raw_path),
        "offline_no_safe": bool(args.offline_no_safe),
        "scanned": scanned,
        "scan_capped": args.max_scan is not None,
        "n_verified_ge_2": len(at_least_two),
        "verified_histogram": {str(k): verified_hist[k] for k in sorted(verified_hist)},
        "fraction_lt_4": sum(1 for row in at_least_two if row["n_verified"] < 4) / n,
        "fraction_eq_4": sum(1 for row in at_least_two if row["n_verified"] == 4) / n,
        "fraction_gt_4": sum(1 for row in at_least_two if row["n_verified"] > 4) / n,
        "reason_counts": dict(reason_counts),
        "example_ids": examples,
        "stage_counts": dict(counts),
        "caps": [
            _summarize_cap(at_least_two, cap, cfg) for cap in (4, 6, 8, None)
        ],
        "versions": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
        },
    }
    out = artifact_path(cfg, "action_count_audit.json")
    write_json(out, report)
    print(json.dumps({k: report[k] for k in ("scanned", "n_verified_ge_2", "caps")}, indent=2))


if __name__ == "__main__":
    main()
