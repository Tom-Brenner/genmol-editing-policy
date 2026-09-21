#!/usr/bin/env python3
"""Generate per-action rollout labels for train or validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rollout_policy.dry_run import action_from_dict
from rollout_policy.genmol_adapter import FrozenGenMol
from rollout_policy.generate import generate_action_rollouts
from rollout_policy.io import (
    artifact_path,
    existing_record_keys,
    load_experiment_config,
    load_jsonl,
    read_json,
    resolve_path,
    write_json,
)
from rollout_policy.labels import aggregate_action_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--split", choices=["train", "validation"], required=True)
    parser.add_argument("--rollouts", type=int, default=None)
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    n_rollouts = args.rollouts or cfg["rollouts"]["labels_per_action"]
    seeds = read_json(resolve_path(cfg["paths"]["manifests_dir"]) / "seeds.json")
    base_seed = int(seeds["label_base_seed"])

    manifest = load_jsonl(resolve_path(cfg["paths"]["manifests_dir"]) / f"{args.split}.jsonl")
    out = resolve_path(cfg["paths"]["results_dir"]) / "labels" / f"{args.split}.jsonl"
    generator = FrozenGenMol.from_configs(cfg)
    done = existing_record_keys(out)

    for parent in manifest:
        actions = [action_from_dict(a) for a in parent["actions"]]
        for action in actions:
            generate_action_rollouts(
                generator=generator,
                parent_id=parent["parent_id"],
                parent_smiles=parent["smiles"],
                core_id=parent["core_id"],
                action=action,
                n_rollouts=n_rollouts,
                phase="labels",
                split=args.split,
                output_path=str(out),
                experiment=cfg,
                base_seed=base_seed,
                done_keys=done,
            )

    records = load_jsonl(out)
    labels = aggregate_action_labels(records, cfg)
    labels_path = resolve_path(cfg["paths"]["results_dir"]) / "labels" / f"{args.split}_action_rhat.jsonl"
    from rollout_policy.io import write_jsonl_atomic

    write_jsonl_atomic(labels_path, labels)
    write_json(
        artifact_path(cfg, f"labels_{args.split}_summary.json"),
        {
            "split": args.split,
            "n_records": len(records),
            "n_actions": len(labels),
            "mean_rhat": sum(l["rhat"] for l in labels) / max(len(labels), 1),
            "n_successes": sum(l["n_successes"] for l in labels),
            "mean_p_success_observed": sum(l["p_success_observed"] for l in labels)
            / max(len(labels), 1),
            "label_schema_version": 2,
        },
    )
    print(json.dumps({"split": args.split, "n_records": len(records), "n_actions": len(labels)}, indent=2))


if __name__ == "__main__":
    main()
