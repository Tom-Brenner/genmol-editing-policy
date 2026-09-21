#!/usr/bin/env python3
"""Local single-molecule SAFE alignment + GenMol remask smoke test.

Requires:
  - vendor/genmol installed editable with safe-mol
  - vendor/genmol/checkpoints/model.ckpt
  - a parent TSV (default: committed data/fixtures/CHEMBL521.tsv)

Usage:
  python scripts/test_one_molecule.py --bf16 true
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "genmol" / "src"))

from rollout_policy.chembl import standardize_parent
from rollout_policy.dry_run import parse_bf16_flag
from rollout_policy.eligibility import enumerate_eligible_actions, verify_action_roundtrip
from rollout_policy.genmol_adapter import FrozenGenMol
from rollout_policy.io import load_experiment_config, resolve_path, write_json
from rollout_policy.precision import resolve_precision
from rollout_policy.scoring import score_candidate


def load_parent_tsv(path: Path) -> tuple[str, str]:
    line = path.read_text(encoding="utf-8").strip().splitlines()[0]
    parent_id, smiles = line.split("\t", 1)
    return parent_id.strip(), smiles.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--parent-tsv",
        default="data/fixtures/CHEMBL521.tsv",
        help="parent_id\\tsmiles TSV (committed CHEMBL521 fixture by default)",
    )
    parser.add_argument("--bf16", default="true")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    prefer_bf16 = parse_bf16_flag(args.bf16)
    planned = resolve_precision(prefer_bf16=prefer_bf16)
    cfg = load_experiment_config(args.config)

    parent_path = resolve_path(args.parent_tsv)
    if not parent_path.exists():
        raise SystemExit(
            f"Missing {parent_path}. The ibuprofen fixture is committed as "
            "data/fixtures/CHEMBL521.tsv (git pull). For another ID:\n"
            "  python scripts/download_chembl_molecule.py --chembl-id <ID> [--offline]"
        )

    parent_id, source_smiles = load_parent_tsv(parent_path)
    parent = standardize_parent(parent_id, source_smiles)
    if parent is None:
        raise SystemExit(f"Failed to standardize {parent_id}: {source_smiles}")
    mol = Chem.MolFromSmiles(parent.smiles)
    assert mol is not None

    print(f"Parent {parent.chembl_id}: {parent.smiles}")
    print(f"QED={parent.qed:.3f} heavy={parent.heavy_atoms} core={parent.core_smiles}")

    actions = enumerate_eligible_actions(
        mol,
        min_actions=1,  # allow single-action smoke parents
        max_actions=4,
        require_safe_alignment=True,
    )
    if not actions:
        raise SystemExit("No SAFE-aligned eligible actions for this parent")

    for action in actions:
        verify_action_roundtrip(mol, action)
    print(f"SAFE-aligned actions: {len(actions)}")
    for a in actions:
        print(
            f"  {a.action_id} frag_idx={a.safe_fragment_index} "
            f"heavy={a.fragment_heavy_atoms} span={a.safe_span} smiles={a.fragment_smiles}"
        )

    ckpt = args.checkpoint or cfg["paths"]["genmol_checkpoint"]
    ckpt_path = resolve_path(ckpt)
    if not ckpt_path.exists():
        raise SystemExit(
            f"Missing checkpoint {ckpt_path}. Run:\n"
            "  python scripts/download_genmol_ckpt.py"
        )

    print(f"Loading GenMol from {ckpt_path} (prefer_bf16={prefer_bf16} → plan {planned})")
    t0 = time.perf_counter()
    generator = FrozenGenMol(str(ckpt_path), bf16=prefer_bf16)
    load_s = time.perf_counter() - t0
    print(json.dumps({"load_seconds": load_s, **generator.precision_info}, indent=2))

    action = actions[0]
    t1 = time.perf_counter()
    completion = generator.complete_span(
        parent.smiles,
        action.safe_fragment_index,
        args.seed,
        cut_bonds=action.slicer_cut_bonds,
    )
    gen_s = time.perf_counter() - t1

    from rollout_policy.chembl import assign_atom_maps
    from rollout_policy.core_check import extract_protected_core

    parent_mapped = assign_atom_maps(mol)
    core = extract_protected_core(parent_mapped)
    scored = score_candidate(
        mol,
        parent_mapped,
        completion.output_smiles,
        core=core,
        constraints=cfg["constraints"],
        reward_cfg=cfg["reward"],
        fingerprint=cfg["fingerprint"],
    )

    report = {
        "parent_id": parent.chembl_id,
        "parent_smiles": parent.smiles,
        "action_id": action.action_id,
        "safe_fragment_index": action.safe_fragment_index,
        "safe_span": list(action.safe_span),
        "prefer_bf16": prefer_bf16,
        "precision": generator.precision_info,
        "requested_seed": args.seed,
        "completion_error": completion.error,
        "output_smiles": completion.output_smiles,
        "generation_seconds": gen_s,
        "score": scored,
        "n_actions": len(actions),
        "safe_alignment_ok": True,
    }
    out = resolve_path("artifacts/one_molecule_local_test.json")
    write_json(out, report)
    print(json.dumps(report, indent=2))

    if completion.error:
        raise SystemExit(f"Generation failed: {completion.error}")
    if not scored.get("valid"):
        # Valid GenMol output is required for a green local smoke test; core/reward may be 0.
        raise SystemExit("Output was not RDKit-sanitizable")
    print(f"ONE-MOLECULE LOCAL TEST PASSED → {out}")


if __name__ == "__main__":
    main()
