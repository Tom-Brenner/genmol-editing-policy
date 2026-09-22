#!/usr/bin/env python3
"""Local end-to-end dry-run without ChEMBL download or GenMol checkpoint.

Exercises: fixture parents → eligibility → mock remask rollouts → action labels
→ value MLP train → policy selection → metrics JSONL.

Usage:
  python scripts/dry_run_local.py --bf16 false
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rollout_policy.dry_run import (
    action_from_dict,
    add_dry_run_args,
    build_generator,
    load_fixture_parents,
    parse_bf16_flag,
)
from rollout_policy.eligibility import core_mol_from_parent
from rollout_policy.features import FEATURE_NAMES, action_features
from rollout_policy.generate import generate_action_rollouts
from rollout_policy.io import (
    artifact_path,
    ensure_dir,
    existing_record_keys,
    load_experiment_config,
    load_jsonl,
    resolve_path,
    write_json,
    write_jsonl_atomic,
)
from rollout_policy.labels import aggregate_action_labels
from rollout_policy.model import ValueMLP, weighted_huber_loss
from rollout_policy.policies import select_action
from rollout_policy.scoring import parent_success


def _split_fixtures(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    # Tiny dry-run split: keep core-disjoint when possible by unique core_id order.
    by_core: dict[str, list[dict]] = defaultdict(list)
    for i in idx:
        by_core[rows[i]["core_id"]].append(dict(rows[i]))
    cores = list(by_core.keys())
    rng.shuffle(cores)
    splits = {"train": [], "validation": [], "test": []}
    # Aim for ~60/20/20 of available parents.
    n = len(rows)
    quotas = {
        "train": max(2, int(round(n * 0.6))),
        "validation": max(1, int(round(n * 0.2))),
        "test": max(1, n - max(2, int(round(n * 0.6))) - max(1, int(round(n * 0.2)))),
    }
    for core in cores:
        group = by_core[core]
        for name in ("test", "validation", "train"):
            if len(splits[name]) + len(group) <= quotas[name] or (
                name == "train" and len(splits["train"]) < quotas["train"]
            ):
                if len(splits[name]) >= quotas[name] and name != "train":
                    continue
                for rec in group:
                    rec["split"] = name
                    splits[name].append(rec)
                break
    # Ensure non-empty.
    for name in splits:
        if not splits[name] and rows:
            rec = dict(rows[0])
            rec["split"] = name
            splits[name].append(rec)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    add_dry_run_args(parser)
    parser.set_defaults(dry_run=True)
    parser.add_argument("--n-parents", type=int, default=8)
    parser.add_argument("--label-rollouts", type=int, default=4)
    parser.add_argument("--test-rollouts", type=int, default=4)
    parser.add_argument("--outdir", default="results/dry_run")
    args = parser.parse_args()

    bf16 = parse_bf16_flag(args.bf16)
    if bf16:
        raise SystemExit("dry_run_local.py requires --bf16 false")

    cfg = load_experiment_config(args.config)
    outdir = ensure_dir(args.outdir)
    fixtures = load_fixture_parents(
        n=args.n_parents,
        require_safe_alignment=False,
        min_actions=cfg["parent_filters"]["min_eligible_actions"],
        max_actions=cfg["parent_filters"]["max_eligible_actions"],
    )
    if len(fixtures) < 3:
        raise SystemExit(
            f"Only {len(fixtures)} fixture parents passed eligibility; need ≥3"
        )

    splits = _split_fixtures(fixtures, seed=int(cfg["split"]["manifest_seed"]))
    for name, rows in splits.items():
        write_jsonl_atomic(outdir / f"manifest_{name}.jsonl", rows)

    generator = build_generator(dry_run=True, bf16=False, experiment=cfg)
    base_seed = int(cfg["split"]["manifest_seed"]) + 100

    # Label rollouts for train+val actions.
    label_path = outdir / "labels.jsonl"
    done = existing_record_keys(label_path)
    for split_name in ("train", "validation"):
        for parent in splits[split_name]:
            for action_d in parent["actions"]:
                action = action_from_dict(action_d)
                generate_action_rollouts(
                    generator=generator,
                    parent_id=parent["parent_id"],
                    parent_smiles=parent["smiles"],
                    core_id=parent["core_id"],
                    action=action,
                    n_rollouts=args.label_rollouts,
                    phase="dry_run_labels",
                    split=split_name,
                    output_path=str(label_path),
                    experiment=cfg,
                    base_seed=base_seed,
                    done_keys=done,
                )

    labels = aggregate_action_labels(load_jsonl(label_path), cfg)
    write_jsonl_atomic(outdir / "action_rhat.jsonl", labels)

    # Tiny MLP fit (train only scaler/model).
    manifest_train = {r["parent_id"]: r for r in splits["train"]}
    train_rows = []
    for lab in labels:
        if lab["parent_id"] not in manifest_train:
            continue
        parent = manifest_train[lab["parent_id"]]
        action = action_from_dict(
            next(a for a in parent["actions"] if a["action_id"] == lab["action_id"])
        )
        mol = Chem.MolFromSmiles(parent["smiles"])
        core = core_mol_from_parent(mol)
        feats = action_features(mol, core, action)
        train_rows.append((feats, lab["rhat"], lab["reward_variance"]))

    if not train_rows:
        raise SystemExit("No train labels for dry-run MLP")

    x = np.stack([r[0] for r in train_rows]).astype(np.float32)
    y = np.asarray([r[1] for r in train_rows], dtype=np.float32)
    w = 1.0 / (1.0 + np.asarray([r[2] for r in train_rows], dtype=np.float32))
    scaler = StandardScaler()
    xs = scaler.fit_transform(x).astype(np.float32)

    device = torch.device("cpu")
    model = ValueMLP(n_features=xs.shape[1], hidden_sizes=(64, 32)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(xs), torch.from_numpy(y), torch.from_numpy(w)),
        batch_size=8,
        shuffle=True,
    )
    model.train()
    for _ in range(20):
        for xb, yb, wb in loader:
            opt.zero_grad()
            loss = weighted_huber_loss(model(xb), yb, wb)
            loss.backward()
            opt.step()
    model.eval()

    ckpt = {
        "model_state_dict": model.state_dict(),
        "n_features": xs.shape[1],
        "hidden_sizes": [64, 32],
        "feature_names": list(FEATURE_NAMES),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "bf16": False,
        "dry_run": True,
    }
    torch.save(ckpt, outdir / "value_model.pt")

    # Test-policy generation with 3 policies.
    test_path = outdir / "test_rollouts.jsonl"
    done_test = existing_record_keys(test_path)
    selections = []
    for parent in splits["test"]:
        actions = [action_from_dict(a) for a in parent["actions"]]
        mol = Chem.MolFromSmiles(parent["smiles"])
        core = core_mol_from_parent(mol)
        feats = np.stack(
            [action_features(mol, core, action) for action in actions]
        ).astype(np.float32)
        with torch.no_grad():
            preds = (
                model(torch.from_numpy(scaler.transform(feats).astype(np.float32)))
                .numpy()
                .tolist()
            )
        for policy in ("random", "largest_fragment", "learned"):
            chosen = select_action(
                policy,
                actions,
                parent_id=parent["parent_id"],
                seed=base_seed,
                predicted_values=preds,
            )
            selections.append(
                {
                    "parent_id": parent["parent_id"],
                    "policy": policy,
                    "action_id": chosen.action_id,
                }
            )
            generate_action_rollouts(
                generator=generator,
                parent_id=parent["parent_id"],
                parent_smiles=parent["smiles"],
                core_id=parent["core_id"],
                action=chosen,
                n_rollouts=args.test_rollouts,
                phase=f"dry_run_test:{policy}",
                split="test",
                output_path=str(test_path),
                experiment=cfg,
                base_seed=base_seed + 7,
                done_keys=done_test,
            )

    write_json(outdir / "selections.json", selections)
    rows = load_jsonl(test_path)
    by_key: dict[tuple, list] = defaultdict(list)
    for row in rows:
        policy = row["phase"].split(":")[-1]
        by_key[(row["parent_id"], policy)].append(row)

    success = defaultdict(list)
    for (pid, policy), prow in by_key.items():
        success[policy].append(
            parent_success(
                [
                    {
                        "valid": r.get("valid"),
                        "exact_core_ok": r.get("exact_core"),
                        "tanimoto": r.get("tanimoto"),
                        "heavy_atom_delta": r.get("heavy_atom_delta"),
                        "qed_delta_raw": r.get("qed_delta_raw"),
                        "parent_n_heavy_atoms": r.get("parent_n_heavy_atoms"),
                        "max_heavy_atom_delta_allowed": r.get("max_heavy_atom_delta_allowed"),
                    }
                    for r in prow
                ],
                success_qed_delta=cfg["reward"]["success_qed_delta"],
                min_tanimoto=cfg["constraints"]["min_tanimoto_to_parent"],
                constraints=cfg["constraints"],
            )
        )

    report = {
        "dry_run": True,
        "bf16": False,
        "precision": "fp32",
        "generator": "MockFrozenGenMol",
        "n_fixture_parents": len(fixtures),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "n_label_records": len(load_jsonl(label_path)),
        "n_test_records": len(rows),
        "success_rate": {k: float(np.mean(v)) if v else 0.0 for k, v in success.items()},
        "outdir": str(outdir),
        "note": "Mock generator only. Does not validate GenMol/SAFE/checkpoint.",
    }
    write_json(outdir / "report.json", report)
    write_json(artifact_path(cfg, "dry_run_report.json"), report)
    print(json.dumps(report, indent=2))
    print("DRY-RUN PASSED (local fp32 mock path)")


if __name__ == "__main__":
    main()
