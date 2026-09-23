#!/usr/bin/env python3
"""Held-out ablation for the shallow MLP and the LightGBM regressor.

Phase 3 and Phase 4 already stored random and largest-fragment completions for
the same test parents. This script does not rewrite those files. It selects
cuts with one frozen tabular checkpoint and generates a completion only when
that (phase, parent, action, candidate, seed) is not already in the Phase 3
or Phase 4 JSONL. Seeds match those runs: test_base_seed with phase "test",
or confirmation_base_seed with phase "confirmation:confirmation-v1".

Training must already be finished. This script opens the test manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rollout_policy.dry_run import action_from_dict  # noqa: E402
from rollout_policy.genmol_adapter import FrozenGenMol  # noqa: E402
from rollout_policy.generate import generate_action_rollouts, seed_for  # noqa: E402
from rollout_policy.io import (  # noqa: E402
    append_jsonl,
    artifact_path,
    existing_record_keys,
    iter_jsonl,
    load_experiment_config,
    load_jsonl,
    read_json,
    record_key,
    resolve_path,
    write_json,
)
from rollout_policy.policies import select_action  # noqa: E402
from rollout_policy.predictors import load_ablation_predictor  # noqa: E402
from rollout_policy.scoring import parent_success  # noqa: E402

SCHEDULES = {
    "test": {
        "phase": "test",
        "seed_name": "test_base_seed",
        "baselines": ("random", "largest_fragment"),
        "archive_dirs": ("test",),
    },
    "confirmation": {
        "phase": "confirmation:confirmation-v1",
        "seed_name": "confirmation_base_seed",
        "baselines": ("largest_fragment",),
        "archive_dirs": ("confirmation",),
    },
}


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


def _archive(results_dir: Path, schedule: str) -> dict[tuple, dict]:
    found: dict[tuple, dict] = {}
    phase = SCHEDULES[schedule]["phase"]
    for dirname in SCHEDULES[schedule]["archive_dirs"]:
        directory = results_dir / dirname
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.jsonl")):
            for record in iter_jsonl(path):
                if record.get("phase") != phase:
                    continue
                found.setdefault(record_key(record), record)
    return found


def _baseline_outcomes(
    path: Path,
    parent_ids: list[str],
    *,
    n_rollouts: int,
    success_qed_delta: float,
    min_tanimoto: float,
    constraints: dict,
) -> dict[str, bool]:
    if not path.is_file():
        raise SystemExit(f"Missing stored baseline {path}. Finish that phase before the ablation.")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in iter_jsonl(path):
        grouped[str(record["parent_id"])].append(record)
    short = [parent_id for parent_id in parent_ids if len(grouped[parent_id]) < n_rollouts]
    if short:
        raise SystemExit(
            f"{path} has fewer than {n_rollouts} completions for {len(short)} parents. "
            "Leave it in place until that phase finishes."
        )
    return {
        parent_id: parent_success(
            grouped[parent_id],
            success_qed_delta=success_qed_delta,
            min_tanimoto=min_tanimoto,
            constraints=constraints,
        )
        for parent_id in parent_ids
    }


def _file_stamp(path: Path) -> tuple[int, int] | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_v2.yaml")
    parser.add_argument("--model", choices=("shallow_mlp", "lightgbm_regressor"), required=True)
    parser.add_argument("--schedule", choices=tuple(SCHEDULES), required=True)
    parser.add_argument("--rollouts", type=int, default=None)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    spec = SCHEDULES[args.schedule]
    n_rollouts = args.rollouts or int(cfg["rollouts"]["test_per_selected_action"])
    seeds = read_json(resolve_path(cfg["paths"]["manifests_dir"]) / "seeds.json")
    base_seed = int(seeds[spec["seed_name"]])
    phase = spec["phase"]

    results_dir = resolve_path(cfg["paths"]["results_dir"])
    test_parents = load_jsonl(resolve_path(cfg["paths"]["manifests_dir"]) / "test.jsonl")
    parent_ids = [parent["parent_id"] for parent in test_parents]
    protected = [
        results_dir / dirname / f"{policy}.jsonl"
        for dirname in spec["archive_dirs"]
        for policy in ("random", "largest_fragment", "learned")
    ]
    stamps = {path: _file_stamp(path) for path in protected}

    out_dir = results_dir / "ablation" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.schedule}.jsonl"
    done = existing_record_keys(out_path)
    archive = _archive(results_dir, args.schedule)
    print(
        f"model={args.model} schedule={args.schedule} phase={phase} "
        f"archived_completions={len(archive)} already_written={len(done)}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = load_ablation_predictor(cfg, args.model, device)
    generator = FrozenGenMol.from_configs(cfg)
    copied = 0
    generated_actions = 0
    selections = []

    for parent_index, parent in enumerate(test_parents, start=1):
        actions = [action_from_dict(action) for action in parent["actions"]]
        scores = [float(value) for value in predictor.predict_actions(parent, actions)]
        chosen = select_action(
            "learned",
            actions,
            parent_id=parent["parent_id"],
            seed=int(seeds["random_policy_seed"]),
            predicted_values=scores,
        )
        selections.append(
            {
                "parent_id": parent["parent_id"],
                "policy": args.model,
                "action_id": chosen.action_id,
                "predicted_value": scores[actions.index(chosen)],
                "model_type": predictor.model_type,
                "feature_schema_version": predictor.feature_schema_version,
                "checkpoint_sha256": predictor.checkpoint_sha256,
                "schedule": args.schedule,
                "phase": phase,
            }
        )
        missing = False
        for candidate_index in range(n_rollouts):
            requested_seed = seed_for(
                phase, parent["parent_id"], chosen.action_id, candidate_index, base_seed
            )
            key = (phase, parent["parent_id"], chosen.action_id, candidate_index, requested_seed)
            if key in done:
                continue
            stored = archive.get(key)
            if stored is None:
                missing = True
                continue
            append_jsonl(out_path, stored)
            done.add(key)
            copied += 1
        if missing:
            generate_action_rollouts(
                generator=generator,
                parent_id=parent["parent_id"],
                parent_smiles=parent["smiles"],
                core_id=parent["core_id"],
                action=chosen,
                n_rollouts=n_rollouts,
                phase=phase,
                split="test",
                output_path=str(out_path),
                experiment=cfg,
                base_seed=base_seed,
                done_keys=done,
            )
            generated_actions += 1
        if parent_index % 250 == 0:
            print(
                f"parents={parent_index}/{len(test_parents)} "
                f"copied_completions={copied} actions_sent_to_genmol={generated_actions}",
                flush=True,
            )

    for path, before in stamps.items():
        if _file_stamp(path) != before:
            raise RuntimeError(f"ablation wrote a protected baseline file: {path}")

    write_json(out_dir / f"{args.schedule}_selections.json", selections)
    success_qed_delta = cfg["reward"]["success_qed_delta"]
    constraints = cfg["constraints"]
    min_tanimoto = constraints["min_tanimoto_to_parent"]
    model_outcomes = _baseline_outcomes(
        out_path,
        parent_ids,
        n_rollouts=n_rollouts,
        success_qed_delta=success_qed_delta,
        min_tanimoto=min_tanimoto,
        constraints=constraints,
    )
    model_flags = np.asarray([int(model_outcomes[parent_id]) for parent_id in parent_ids], dtype=float)
    success_rate = {args.model: float(model_flags.mean())}
    contrasts = {}
    bootstrap_seed = int(seeds["bootstrap_seed"]) + (1 if args.schedule == "confirmation" else 0)
    for baseline in spec["baselines"]:
        directory = "confirmation" if args.schedule == "confirmation" else "test"
        baseline_path = results_dir / directory / f"{baseline}.jsonl"
        baseline_outcomes = _baseline_outcomes(
            baseline_path,
            parent_ids,
            n_rollouts=n_rollouts,
            success_qed_delta=success_qed_delta,
            min_tanimoto=min_tanimoto,
            constraints=constraints,
        )
        baseline_flags = np.asarray(
            [int(baseline_outcomes[parent_id]) for parent_id in parent_ids], dtype=float
        )
        success_rate[baseline] = float(baseline_flags.mean())
        contrasts[f"{args.model}_minus_{baseline}"] = paired_bootstrap_ci(
            model_flags - baseline_flags, seed=bootstrap_seed
        )

    report = {
        "model_type": predictor.model_type,
        "checkpoint_sha256": predictor.checkpoint_sha256,
        "schedule": args.schedule,
        "phase": phase,
        "base_seed": base_seed,
        "n_parents": len(test_parents),
        "rollouts_per_selected_action": n_rollouts,
        "copied_completions": copied,
        "actions_sent_to_genmol": generated_actions,
        "success_rate": success_rate,
        "contrasts": contrasts,
        "baselines_left_in_place": [str(path) for path, stamp in stamps.items() if stamp is not None],
        "output": str(out_path),
        "note": "Ablation against stored Phase 3/4 baselines. Not a replacement for the GATv2 test.",
    }
    write_json(out_dir / f"{args.schedule}_metrics.json", report)
    write_json(artifact_path(cfg, f"ablation_{args.model}_{args.schedule}_metrics.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
