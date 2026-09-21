#!/usr/bin/env python3
"""Phase 0: GenMol compatibility and throughput probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rollout_policy.chembl import iter_chembl_smiles, standardize_parent
from rollout_policy.dry_run import (
    action_from_dict,
    add_dry_run_args,
    build_generator,
    load_fixture_parents,
    parse_bf16_flag,
)
from rollout_policy.eligibility import enumerate_eligible_actions
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
from rollout_policy.policies import select_random


def _load_manifest_ids(manifests_dir: Path) -> set[str]:
    ids: set[str] = set()
    for name in ("train", "validation", "test"):
        path = manifests_dir / f"{name}.jsonl"
        if path.exists():
            for row in load_jsonl(path):
                ids.add(row["parent_id"])
                ids.add(row["smiles"])
    return ids


def collect_probe_parents(cfg: dict, n_parents: int, seed: int, *, dry_run: bool) -> list[dict]:
    if dry_run:
        fixtures = load_fixture_parents(
            n=n_parents,
            require_safe_alignment=False,
            min_actions=cfg["parent_filters"]["min_eligible_actions"],
            max_actions=cfg["parent_filters"]["max_eligible_actions"],
        )
        if len(fixtures) < n_parents:
            raise SystemExit(
                f"Dry-run fixtures only yielded {len(fixtures)} parents; need {n_parents}"
            )
        return [
            {
                "parent_id": r["parent_id"],
                "smiles": r["smiles"],
                "core_id": r["core_id"],
                "actions": [action_from_dict(a) for a in r["actions"]],
            }
            for r in fixtures
        ]

    manifests_dir = resolve_path(cfg["paths"]["manifests_dir"])
    excluded = _load_manifest_ids(manifests_dir)
    raw = resolve_path(cfg["paths"]["chembl_raw"])
    filters = cfg["parent_filters"]
    rng = np.random.default_rng(seed)
    found: list[dict] = []
    for chembl_id, smiles in iter_chembl_smiles(raw):
        if chembl_id in excluded:
            continue
        parent = standardize_parent(
            chembl_id,
            smiles,
            min_heavy_atoms=filters["min_heavy_atoms"],
            max_heavy_atoms=filters["max_heavy_atoms"],
            min_qed=filters["min_qed"],
            max_qed=filters["max_qed"],
        )
        if parent is None or parent.smiles in excluded:
            continue
        mol = Chem.MolFromSmiles(parent.smiles)
        if mol is None:
            continue
        actions = enumerate_eligible_actions(
            mol,
            min_actions=filters["min_eligible_actions"],
            max_actions=filters["max_eligible_actions"],
            require_safe_alignment=True,
        )
        if not actions:
            continue
        found.append(
            {
                "parent_id": parent.chembl_id,
                "smiles": parent.smiles,
                "core_id": parent.core_smiles,
                "actions": actions,
            }
        )
        if len(found) >= n_parents * 5:
            break
    if len(found) < n_parents:
        raise SystemExit(f"Only found {len(found)} probe candidates; need {n_parents}")
    idx = rng.choice(len(found), size=n_parents, replace=False)
    return [found[i] for i in sorted(idx)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--n-parents", type=int, default=None)
    parser.add_argument("--n-completions", type=int, default=None)
    parser.add_argument("--output", default="results/probes.jsonl")
    add_dry_run_args(parser)
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    bf16 = parse_bf16_flag(args.bf16)
    if args.dry_run:
        bf16 = False
        if args.output == "results/probes.jsonl":
            args.output = "results/dry_run/probes.jsonl"
        n_parents = args.n_parents or 4
        n_completions = args.n_completions or 2
    else:
        n_parents = args.n_parents or cfg["rollouts"]["probe_parents"]
        n_completions = args.n_completions or cfg["rollouts"]["probe_completions_per_parent"]

    seeds_path = resolve_path(cfg["paths"]["manifests_dir"]) / "seeds.json"
    if seeds_path.exists():
        base_seed = int(read_json(seeds_path)["probe_base_seed"])
    else:
        base_seed = int(cfg["split"]["manifest_seed"]) + 5

    probes = collect_probe_parents(cfg, n_parents, base_seed, dry_run=args.dry_run)
    generator = build_generator(
        dry_run=args.dry_run, bf16=bf16, experiment=cfg
    )
    out = resolve_path(args.output)
    done = existing_record_keys(out)

    for parent in probes:
        action = select_random(parent["actions"], parent_id=parent["parent_id"], seed=base_seed)
        generate_action_rollouts(
            generator=generator,
            parent_id=parent["parent_id"],
            parent_smiles=parent["smiles"],
            core_id=parent["core_id"],
            action=action,
            n_rollouts=n_completions,
            phase="probe_dry_run" if args.dry_run else "probe",
            split="probe",
            output_path=str(out),
            experiment=cfg,
            base_seed=base_seed,
            done_keys=done,
        )

    rows = load_jsonl(out)
    n = len(rows)
    valid = sum(1 for r in rows if r.get("valid"))
    mean_wall = float(np.mean([r["wall_seconds"] for r in rows])) if rows else float("nan")
    report = {
        "dry_run": bool(args.dry_run),
        "bf16": bf16,
        "n_records": n,
        "n_parents": n_parents,
        "n_completions_per_parent": n_completions,
        "valid_fraction": valid / n if n else 0.0,
        "mean_wall_seconds": mean_wall,
        "gate_max_seconds": cfg["throughput_gate"]["max_seconds_per_completion"],
        "gate_min_valid": cfg["throughput_gate"]["min_valid_fraction"],
        "pass_time": mean_wall <= cfg["throughput_gate"]["max_seconds_per_completion"],
        "pass_valid": (valid / n if n else 0.0) >= cfg["throughput_gate"]["min_valid_fraction"],
    }
    report_name = (
        "throughput_report_dry_run.json" if args.dry_run else "throughput_report.json"
    )
    write_json(artifact_path(cfg, report_name), report)
    print(json.dumps(report, indent=2))
    if args.dry_run:
        print("Probe dry-run finished (mock generator; gate not authoritative)")
        return
    if not (report["pass_time"] and report["pass_valid"]):
        raise SystemExit("Phase 0 throughput gate FAILED")
    print("Phase 0 throughput gate PASSED (still require 50-record manual audit)")


if __name__ == "__main__":
    main()
