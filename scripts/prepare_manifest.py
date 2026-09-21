#!/usr/bin/env python3
"""Build a core-disjoint ChEMBL manifest.

Split sizes come from the experiment config. The locked protocol is
1000/200/200. Experiment V2 requests 30000/5000/5000.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rollout_policy.chembl import iter_chembl_smiles, standardize_parent
from rollout_policy.eligibility import enumerate_eligible_actions
from rollout_policy.io import (
    artifact_path,
    ensure_dir,
    load_experiment_config,
    resolve_path,
    write_json,
    write_jsonl_atomic,
)


def try_build_parent(
    chembl_id: str,
    source_smiles: str,
    cfg: dict,
    *,
    require_safe_alignment: bool,
) -> dict | None:
    filters = cfg["parent_filters"]
    parent = standardize_parent(
        chembl_id,
        source_smiles,
        min_heavy_atoms=filters["min_heavy_atoms"],
        max_heavy_atoms=filters["max_heavy_atoms"],
        min_qed=filters["min_qed"],
        max_qed=filters["max_qed"],
    )
    if parent is None:
        return None
    mol = Chem.MolFromSmiles(parent.smiles)
    if mol is None:
        return None
    try:
        actions = enumerate_eligible_actions(
            mol,
            min_actions=filters["min_eligible_actions"],
            max_actions=filters["max_eligible_actions"],
            require_safe_alignment=require_safe_alignment,
        )
    except Exception:
        return None
    if not actions:
        return None
    return {
        "parent_id": parent.chembl_id,
        "source_smiles": parent.source_smiles,
        "smiles": parent.smiles,
        "atom_mapped_smiles": parent.atom_mapped_smiles,
        "core_id": parent.core_smiles,
        "heavy_atoms": parent.heavy_atoms,
        "qed": parent.qed,
        "n_actions": len(actions),
        "actions": [a.to_dict() for a in actions],
        "action_order_keys": [list(a.canonical_order_key) for a in actions],
    }


def assign_core_disjoint_splits(
    records: list[dict],
    *,
    n_train: int,
    n_validation: int,
    n_test: int,
    seed: int,
) -> dict[str, list[dict]]:
    by_core: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_core[rec["core_id"]].append(rec)

    cores = sorted(by_core.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(cores)

    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    quotas = {"train": n_train, "validation": n_validation, "test": n_test}
    fill_order = ["test", "validation", "train"]
    for core in cores:
        group = by_core[core]
        for split in fill_order:
            if len(splits[split]) + len(group) <= quotas[split]:
                for rec in group:
                    row = dict(rec)
                    row["split"] = split
                    splits[split].append(row)
                break
        if all(len(splits[s]) >= quotas[s] for s in quotas):
            break

    for split, n in quotas.items():
        if len(splits[split]) != n:
            raise RuntimeError(
                f"Failed to fill {split}: got {len(splits[split])} want {n}. "
                f"Eligible parents={len(records)}, cores={len(cores)}"
            )

    core_sets = {s: {r["core_id"] for r in splits[s]} for s in splits}
    assert core_sets["train"].isdisjoint(core_sets["validation"])
    assert core_sets["train"].isdisjoint(core_sets["test"])
    assert core_sets["validation"].isdisjoint(core_sets["test"])
    return splits


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-scan", type=int, default=None, help="Optional cap on chemreps rows scanned")
    parser.add_argument(
        "--offline-no-safe",
        action="store_true",
        help="Dev-only: skip SAFE alignment (do not use for the real experiment)",
    )
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg["split"]["manifest_seed"])
    require_safe = not args.offline_no_safe

    raw_path = resolve_path(cfg["paths"]["chembl_raw"])
    if not raw_path.exists():
        raise SystemExit(f"Missing {raw_path}; run scripts/download_chembl.py first")

    need = cfg["split"]["n_train"] + cfg["split"]["n_validation"] + cfg["split"]["n_test"]
    manifest_cfg = cfg.get("manifest", {})
    target_pool = int(manifest_cfg.get("target_eligible_pool", need * 4))
    if target_pool < need:
        raise ValueError("target_eligible_pool must be >= requested split total")
    eligible: list[dict] = []
    scanned = 0
    for chembl_id, smiles in iter_chembl_smiles(raw_path):
        scanned += 1
        if args.max_scan is not None and scanned > args.max_scan:
            break
        rec = try_build_parent(
            chembl_id, smiles, cfg, require_safe_alignment=require_safe
        )
        if rec is not None:
            eligible.append(rec)
            if len(eligible) >= target_pool:
                break
        if scanned % 5000 == 0:
            print(f"scanned={scanned} eligible={len(eligible)}")

    if len(eligible) < need:
        raise SystemExit(
            f"Only {len(eligible)} eligible parents after scanning {scanned}; need {need}"
        )

    splits = assign_core_disjoint_splits(
        eligible,
        n_train=cfg["split"]["n_train"],
        n_validation=cfg["split"]["n_validation"],
        n_test=cfg["split"]["n_test"],
        seed=seed,
    )

    manifests_dir = ensure_dir(cfg["paths"]["manifests_dir"])
    for split, rows in splits.items():
        write_jsonl_atomic(manifests_dir / f"{split}.jsonl", rows)

    all_rows = splits["train"] + splits["validation"] + splits["test"]
    parents_path = resolve_path(cfg["paths"]["parents_parquet"])
    ensure_dir(parents_path.parent)
    parquet_rows = []
    for rec in all_rows:
        row = dict(rec)
        row["actions"] = json.dumps(row["actions"])
        row["action_order_keys"] = json.dumps(row["action_order_keys"])
        parquet_rows.append(row)
    pd.DataFrame(parquet_rows).to_parquet(parents_path, index=False)

    seeds = {
        "manifest_seed": seed,
        "random_policy_seed": seed + 1,
        "label_base_seed": seed + 2,
        "test_base_seed": seed + 3,
        "confirmation_base_seed": seed + 4,
        "probe_base_seed": seed + 5,
        "bootstrap_seed": seed + 6,
    }
    write_json(manifests_dir / "seeds.json", seeds)

    hashes = {
        name: file_sha(manifests_dir / f"{name}.jsonl")
        for name in ("train", "validation", "test")
    }
    hashes["parents_parquet"] = file_sha(parents_path)
    write_json(manifests_dir / "hashes.json", hashes)

    write_json(
        artifact_path(cfg, "manifest_summary.json"),
        {
            "scanned": scanned,
            "eligible_pool": len(eligible),
            "counts": {k: len(v) for k, v in splits.items()},
            "seed": seed,
            "offline_no_safe": bool(args.offline_no_safe),
        },
    )
    print(json.dumps({"counts": {k: len(v) for k, v in splits.items()}, "seed": seed}, indent=2))
    print(f"Freeze test manifest with: chmod a-w {manifests_dir / 'test.jsonl'}")


if __name__ == "__main__":
    main()
