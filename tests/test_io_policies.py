"""Tests for I/O helpers and policies."""

from __future__ import annotations

from pathlib import Path

from rollout_policy.eligibility import EligibleAction
from rollout_policy.io import append_jsonl, existing_record_keys, load_jsonl, make_generation_record
from rollout_policy.policies import (
    select_largest_fragment,
    select_learned,
    select_random,
    select_scored_index,
)


def _action(action_id: str, heavy: int, order: tuple) -> EligibleAction:
    return EligibleAction(
        action_id=action_id,
        fragment_atom_indices=(1, 2),
        core_attachment_atom=0,
        fragment_attachment_atom=1,
        bond_type="SINGLE",
        canonical_order_key=order,
        safe_span=(0, 1),
        safe_fragment_index=0,
        fragment_smiles="C*",
        fragment_heavy_atoms=heavy,
        is_ring=False,
        is_terminal=True,
    )


def test_jsonl_atomic_append_and_resume(tmp_path: Path):
    path = tmp_path / "out.jsonl"
    rec = make_generation_record(
        phase="labels",
        split="train",
        parent_id="CHEMBL1",
        parent_smiles="CCO",
        parent_atom_mapped_smiles="[CH3:1][CH2:2][OH:3]",
        core_id="C",
        action_id="a0",
        safe_span=(0, 3),
        candidate_index=0,
        requested_seed=1,
        observed_seed=1,
        output_safe=None,
        output_smiles="CCO",
        valid=True,
        exact_core=True,
        tanimoto=1.0,
        heavy_atom_delta=0,
        parent_qed=0.5,
        candidate_qed=0.5,
        qed_delta_raw=0.0,
        reward=0.0,
        wall_seconds=0.1,
        exception=None,
    )
    append_jsonl(path, rec)
    keys = existing_record_keys(path)
    assert ("labels", "CHEMBL1", "a0", 0, 1) in keys
    assert len(load_jsonl(path)) == 1


def test_policy_selection_deterministic():
    actions = [
        _action("a0", 3, (-3, 0, 1, "x")),
        _action("a1", 5, (-5, 0, 2, "y")),
        _action("a2", 5, (-5, 0, 3, "z")),
    ]
    largest = select_largest_fragment(actions)
    assert largest.action_id == "a1"  # tie on heavy atoms → canonical order
    learned = select_learned(actions, [0.1, 0.2, 0.2])
    assert learned.action_id == "a1"  # tie on value → canonical order
    lexical_vs_canonical = [
        _action("a_z", 1, (0, "first")),
        _action("a_a", 1, (1, "second")),
    ]
    assert select_learned(lexical_vs_canonical, [0.5, 0.5]).action_id == "a_z"
    ordered = sorted(
        lexical_vs_canonical,
        key=lambda action: (action.canonical_order_key, action.action_id),
    )
    ranks = {action.action_id: rank for rank, action in enumerate(ordered)}
    picked = select_scored_index(
        [0.5, 0.5],
        [ranks["a_z"], ranks["a_a"]],
        ["a_z", "a_a"],
    )
    assert lexical_vs_canonical[picked].action_id == "a_z"
    r1 = select_random(actions, parent_id="p", seed=7)
    r2 = select_random(actions, parent_id="p", seed=7)
    assert r1.action_id == r2.action_id
