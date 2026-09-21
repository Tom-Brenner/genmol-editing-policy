#!/usr/bin/env python3
"""Phase 3: locked held-out evaluation for three policies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rollout_policy.dry_run import action_from_dict
from rollout_policy.eligibility import EligibleAction, core_mol_from_parent
from rollout_policy.features import action_features
from rollout_policy.genmol_adapter import FrozenGenMol
from rollout_policy.generate import generate_action_rollouts
from rollout_policy.io import (
    existing_record_keys,
    load_experiment_config,
    load_jsonl,
    read_json,
    resolve_path,
    write_json,
)
from rollout_policy.model import ValueMLP
from rollout_policy.policies import select_action
from rollout_policy.scoring import parent_success


def load_value_model(ckpt_path: Path, device: torch.device):
    blob = torch.load(ckpt_path, map_location=device)
    model = ValueMLP(blob["n_features"], hidden_sizes=tuple(blob["hidden_sizes"]))
    model.load_state_dict(blob["model_state_dict"])
    model.to(device)
    model.eval()
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(blob["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(blob["scaler_scale"], dtype=np.float64)
    scaler.n_features_in_ = len(blob["scaler_mean"])
    return model, scaler


def predict_values(model, scaler, parent: dict, actions: list[EligibleAction], device) -> list[float]:
    mol = Chem.MolFromSmiles(parent["smiles"])
    core = core_mol_from_parent(mol)
    feats = []
    for action in actions:
        frag = Chem.MolFromSmiles(action.fragment_smiles)
        feats.append(
            action_features(
                mol, core, frag, action.bond_type, action.is_ring, action.is_terminal
            )
        )
    x = scaler.transform(np.stack(feats).astype(np.float32)).astype(np.float32)
    with torch.no_grad():
        pred = model(torch.from_numpy(x).to(device)).cpu().numpy()
    return [float(v) for v in pred]


def paired_bootstrap_ci(
    diffs: np.ndarray, *, seed: int, n_boot: int = 10000, alpha: float = 0.05
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        sample = diffs[rng.integers(0, n, size=n)]
        means.append(float(sample.mean()))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return {
        "mean": float(diffs.mean()),
        "ci95_low": lo,
        "ci95_high": hi,
        "n_boot": n_boot,
        "bootstrap_seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--rollouts", type=int, default=None)
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    n_rollouts = args.rollouts or cfg["rollouts"]["test_per_selected_action"]
    seeds = read_json(resolve_path(cfg["paths"]["manifests_dir"]) / "seeds.json")
    base_seed = int(seeds["test_base_seed"])
    random_policy_seed = int(seeds["random_policy_seed"])

    test_parents = load_jsonl(resolve_path(cfg["paths"]["manifests_dir"]) / "test.jsonl")
    out_dir = resolve_path(cfg["paths"]["results_dir"]) / "test"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler = load_value_model(
        resolve_path(cfg["paths"]["artifacts_dir"]) / "value_model.pt", device
    )
    generator = FrozenGenMol.from_configs(cfg)

    policies = ("random", "largest_fragment", "learned")
    selections = []

    for parent in test_parents:
        actions = [action_from_dict(a) for a in parent["actions"]]
        preds = predict_values(model, scaler, parent, actions, device)
        for policy in policies:
            chosen = select_action(
                policy,
                actions,
                parent_id=parent["parent_id"],
                seed=random_policy_seed,
                predicted_values=preds,
            )
            selections.append(
                {
                    "parent_id": parent["parent_id"],
                    "policy": policy,
                    "action_id": chosen.action_id,
                    "predicted_value": preds[actions.index(chosen)] if policy == "learned" else None,
                }
            )
            out_path = out_dir / f"{policy}.jsonl"
            done = existing_record_keys(out_path)
            generate_action_rollouts(
                generator=generator,
                parent_id=parent["parent_id"],
                parent_smiles=parent["smiles"],
                core_id=parent["core_id"],
                action=chosen,
                n_rollouts=n_rollouts,
                phase="test",
                split="test",
                output_path=str(out_path),
                experiment=cfg,
                base_seed=base_seed,
                done_keys=done,
            )

    write_json(out_dir / "selections.json", selections)

    # Metrics before inspecting examples.
    success_cfg = cfg["reward"]["success_qed_delta"]
    constraints = cfg["constraints"]
    parent_outcomes: dict[str, dict[str, bool]] = {p["parent_id"]: {} for p in test_parents}
    validity: dict[str, list[bool]] = {p: [] for p in policies}

    for policy in policies:
        rows = load_jsonl(out_dir / f"{policy}.jsonl")
        by_parent: dict[str, list[dict]] = {}
        for row in rows:
            by_parent.setdefault(row["parent_id"], []).append(row)
            validity[policy].append(bool(row.get("valid")))
        for parent_id, prow in by_parent.items():
            parent_outcomes[parent_id][policy] = parent_success(
                prow,
                success_qed_delta=success_cfg,
                min_tanimoto=constraints["min_tanimoto_to_parent"],
                max_heavy_atom_delta=constraints["max_absolute_heavy_atom_delta"],
            )

    learned = np.asarray(
        [int(parent_outcomes[pid]["learned"]) for pid in parent_outcomes], dtype=float
    )
    largest = np.asarray(
        [int(parent_outcomes[pid]["largest_fragment"]) for pid in parent_outcomes], dtype=float
    )
    random_s = np.asarray(
        [int(parent_outcomes[pid]["random"]) for pid in parent_outcomes], dtype=float
    )
    primary = paired_bootstrap_ci(
        learned - largest, seed=int(seeds["bootstrap_seed"])
    )
    valid_learned = float(np.mean(validity["learned"])) if validity["learned"] else 0.0
    valid_largest = float(np.mean(validity["largest_fragment"])) if validity["largest_fragment"] else 0.0

    report = {
        "n_parents": len(test_parents),
        "rollouts_per_selected_action": n_rollouts,
        "success_rate": {
            "random": float(random_s.mean()),
            "largest_fragment": float(largest.mean()),
            "learned": float(learned.mean()),
        },
        "primary_learned_minus_largest": primary,
        "candidate_validity": {
            "random": float(np.mean(validity["random"])) if validity["random"] else 0.0,
            "largest_fragment": valid_largest,
            "learned": valid_learned,
            "learned_minus_largest_pp": (valid_learned - valid_largest) * 100.0,
        },
        "success_criterion_met": bool(
            primary["ci95_low"] > 0
            and (valid_largest - valid_learned) * 100.0 <= 3.0
        ),
    }
    write_json(out_dir / "metrics.json", report)
    write_json("artifacts/test_metrics.json", report)
    print(json.dumps(report, indent=2))
    print("Lock results with: chmod -R a-w results/test")


if __name__ == "__main__":
    main()
