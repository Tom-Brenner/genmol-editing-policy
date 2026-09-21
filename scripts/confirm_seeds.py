#!/usr/bin/env python3
"""Phase 4: independent-seed confirmation for learned vs largest-fragment."""

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


def paired_bootstrap_ci(diffs: np.ndarray, *, seed: int, n_boot: int = 10000) -> dict:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = [float(diffs[rng.integers(0, n, size=n)].mean()) for _ in range(n_boot)]
    return {
        "mean": float(diffs.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n_boot": n_boot,
        "bootstrap_seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["learned", "largest_fragment"],
        help="Pre-specified: learned and largest_fragment only",
    )
    parser.add_argument("--rollouts", type=int, default=None)
    parser.add_argument("--seed-schedule", default="confirmation-v1")
    args = parser.parse_args()

    allowed = {"learned", "largest_fragment"}
    if set(args.policies) - allowed:
        raise SystemExit(f"Phase 4 policies must be subset of {allowed}")

    cfg = load_experiment_config(args.config)
    n_rollouts = args.rollouts or cfg["rollouts"]["test_per_selected_action"]
    seeds = read_json(resolve_path(cfg["paths"]["manifests_dir"]) / "seeds.json")
    base_seed = int(seeds["confirmation_base_seed"])
    random_policy_seed = int(seeds["random_policy_seed"])

    # Reuse Phase 3 selections when available to keep the same chosen actions.
    test_dir = resolve_path(cfg["paths"]["results_dir"]) / "test"
    selections_path = test_dir / "selections.json"
    test_parents = load_jsonl(resolve_path(cfg["paths"]["manifests_dir"]) / "test.jsonl")
    parent_by_id = {p["parent_id"]: p for p in test_parents}

    out_dir = resolve_path(cfg["paths"]["results_dir"]) / "confirmation"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler = load_value_model(
        resolve_path(cfg["paths"]["artifacts_dir"]) / "value_model.pt", device
    )
    generator = FrozenGenMol.from_configs(cfg)

    if selections_path.exists():
        selections = [
            s for s in read_json(selections_path) if s["policy"] in args.policies
        ]
    else:
        selections = []
        for parent in test_parents:
            actions = [action_from_dict(a) for a in parent["actions"]]
            preds = predict_values(model, scaler, parent, actions, device)
            for policy in args.policies:
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
                    }
                )

    for sel in selections:
        parent = parent_by_id[sel["parent_id"]]
        action = action_from_dict(
            next(a for a in parent["actions"] if a["action_id"] == sel["action_id"])
        )
        out_path = out_dir / f"{sel['policy']}.jsonl"
        done = existing_record_keys(out_path)
        generate_action_rollouts(
            generator=generator,
            parent_id=parent["parent_id"],
            parent_smiles=parent["smiles"],
            core_id=parent["core_id"],
            action=action,
            n_rollouts=n_rollouts,
            phase=f"confirmation:{args.seed_schedule}",
            split="test",
            output_path=str(out_path),
            experiment=cfg,
            base_seed=base_seed,
            done_keys=done,
        )

    outcomes: dict[str, dict[str, bool]] = {p["parent_id"]: {} for p in test_parents}
    for policy in args.policies:
        rows = load_jsonl(out_dir / f"{policy}.jsonl")
        by_parent: dict[str, list[dict]] = {}
        for row in rows:
            by_parent.setdefault(row["parent_id"], []).append(row)
        for pid, prow in by_parent.items():
            outcomes[pid][policy] = parent_success(
                prow,
                success_qed_delta=cfg["reward"]["success_qed_delta"],
                min_tanimoto=cfg["constraints"]["min_tanimoto_to_parent"],
                constraints=cfg["constraints"],
            )

    learned = np.asarray([int(outcomes[pid].get("learned", False)) for pid in outcomes])
    largest = np.asarray(
        [int(outcomes[pid].get("largest_fragment", False)) for pid in outcomes]
    )
    report = {
        "seed_schedule": args.seed_schedule,
        "success_rate": {
            "learned": float(learned.mean()),
            "largest_fragment": float(largest.mean()),
        },
        "primary_learned_minus_largest": paired_bootstrap_ci(
            learned - largest, seed=int(seeds["bootstrap_seed"]) + 1
        ),
        "note": "Seed-sensitivity confirmation, not an independent test set.",
    }
    write_json(out_dir / "metrics.json", report)
    write_json("artifacts/confirmation_metrics.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
