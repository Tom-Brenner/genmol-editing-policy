"""Shared rollout generation loop used by probe / label / evaluate scripts."""

from __future__ import annotations

import hashlib
import time
from typing import Any

from rdkit import Chem
from rdkit.Chem import QED

from rollout_policy.chembl import assign_atom_maps
from rollout_policy.core_check import extract_protected_core
from rollout_policy.eligibility import EligibleAction
from rollout_policy.genmol_adapter import FrozenGenMol
from rollout_policy.io import append_jsonl, existing_record_keys, make_generation_record
from rollout_policy.scoring import resolve_max_heavy_atom_delta, score_candidate


def seed_for(phase: str, parent_id: str, action_id: str, candidate_index: int, base_seed: int) -> int:
    # Deterministic per-attempt seed schedule; recorded in each JSONL row.
    blob = f"{phase}|{parent_id}|{action_id}|{candidate_index}|{base_seed}".encode()
    digest = int(hashlib.sha256(blob).hexdigest()[:8], 16)
    return int(base_seed + digest % 1_000_000_007)


def generate_action_rollouts(
    *,
    generator: FrozenGenMol,
    parent_id: str,
    parent_smiles: str,
    core_id: str,
    action: EligibleAction,
    n_rollouts: int,
    phase: str,
    split: str,
    output_path: str,
    experiment: dict[str, Any],
    base_seed: int,
    done_keys: set[tuple] | None = None,
) -> list[dict[str, Any]]:
    parent = Chem.MolFromSmiles(parent_smiles)
    if parent is None:
        raise ValueError(f"invalid parent SMILES: {parent_smiles}")
    parent_mapped = assign_atom_maps(parent)
    core = extract_protected_core(parent_mapped)
    parent_qed = float(QED.qed(parent))
    parent_atom_mapped_smiles = Chem.MolToSmiles(parent_mapped, canonical=False)

    if done_keys is None:
        done_keys = existing_record_keys(output_path)

    records: list[dict[str, Any]] = []
    for cand_idx in range(n_rollouts):
        req_seed = seed_for(phase, parent_id, action.action_id, cand_idx, base_seed)
        key = (phase, parent_id, action.action_id, cand_idx, req_seed)
        if key in done_keys:
            continue

        t0 = time.perf_counter()
        completion = generator.complete_span(
            parent_smiles,
            action.safe_fragment_index,
            req_seed,
            cut_bonds=action.slicer_cut_bonds or None,
        )
        wall = time.perf_counter() - t0

        if completion.error or not completion.output_smiles:
            parent_n_heavy = int(parent.GetNumHeavyAtoms())
            record = make_generation_record(
                phase=phase,
                split=split,
                parent_id=parent_id,
                parent_smiles=parent_smiles,
                parent_atom_mapped_smiles=parent_atom_mapped_smiles,
                core_id=core_id,
                action_id=action.action_id,
                safe_span=action.safe_span,
                candidate_index=cand_idx,
                requested_seed=req_seed,
                observed_seed=completion.observed_seed,
                output_safe=completion.output_safe,
                output_smiles=None,
                valid=False,
                exact_core=False,
                feasible=False,
                tanimoto=None,
                heavy_atom_delta=None,
                parent_n_heavy_atoms=parent_n_heavy,
                candidate_n_heavy_atoms=None,
                max_heavy_atom_delta_allowed=resolve_max_heavy_atom_delta(
                    parent_n_heavy, experiment.get("constraints")
                ),
                parent_qed=parent_qed,
                candidate_qed=None,
                qed_delta_raw=None,
                reward=0.0,
                wall_seconds=wall,
                exception=completion.error or "EmptyGeneration",
                extra={"safe_fragment_index": action.safe_fragment_index},
            )
        else:
            scored = score_candidate(
                parent,
                parent_mapped,
                completion.output_smiles,
                core=core,
                constraints=experiment.get("constraints"),
                reward_cfg=experiment.get("reward"),
                fingerprint=experiment.get("fingerprint"),
            )
            record = make_generation_record(
                phase=phase,
                split=split,
                parent_id=parent_id,
                parent_smiles=parent_smiles,
                parent_atom_mapped_smiles=parent_atom_mapped_smiles,
                core_id=core_id,
                action_id=action.action_id,
                safe_span=action.safe_span,
                candidate_index=cand_idx,
                requested_seed=req_seed,
                observed_seed=completion.observed_seed,
                output_safe=completion.output_safe,
                output_smiles=scored.get("output_smiles"),
                valid=bool(scored["valid"]),
                exact_core=bool(scored.get("exact_core_ok")),
                feasible=bool(scored.get("feasible")),
                tanimoto=scored.get("tanimoto"),
                heavy_atom_delta=scored.get("heavy_atom_delta"),
                parent_n_heavy_atoms=scored.get("parent_n_heavy_atoms"),
                candidate_n_heavy_atoms=scored.get("candidate_n_heavy_atoms"),
                max_heavy_atom_delta_allowed=scored.get("max_heavy_atom_delta_allowed"),
                parent_qed=scored["parent_qed"],
                candidate_qed=scored.get("candidate_qed"),
                qed_delta_raw=scored.get("qed_delta_raw"),
                reward=float(scored["reward"]),
                wall_seconds=wall,
                exception=None,
                extra={"safe_fragment_index": action.safe_fragment_index},
            )
        append_jsonl(output_path, record)
        done_keys.add(key)
        records.append(record)
    return records
