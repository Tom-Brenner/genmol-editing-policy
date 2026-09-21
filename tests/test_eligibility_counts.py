"""Action-count policy. Caps truncate; they do not create pendants."""

from __future__ import annotations

from rdkit import Chem

from rollout_policy.eligibility import (
    apply_action_count_policy,
    enumerate_aligned_actions,
)

# Three methyl pendants on benzene. Six methyl pendants on benzene.
MESITYLENE = "Cc1cc(C)cc(C)c1"
HEXAMETHYLBENZENE = "Cc1c(C)c(C)c(C)c(C)c1C"


def test_raising_cap_does_not_invent_actions():
    parent = Chem.MolFromSmiles(MESITYLENE)
    diag = enumerate_aligned_actions(parent, require_safe_alignment=False)
    assert diag.n_verified == 3
    kept = apply_action_count_policy(diag.actions, min_actions=2, max_actions=8)
    assert len(kept) == 3


def test_cap_truncates_in_canonical_order():
    parent = Chem.MolFromSmiles(HEXAMETHYLBENZENE)
    diag = enumerate_aligned_actions(parent, require_safe_alignment=False)
    assert diag.n_verified > 4
    kept = apply_action_count_policy(diag.actions, min_actions=2, max_actions=4)
    assert kept == list(diag.actions[:4])


def test_exact_four_rejects_three_action_parent():
    parent = Chem.MolFromSmiles(MESITYLENE)
    diag = enumerate_aligned_actions(parent, require_safe_alignment=False)
    assert apply_action_count_policy(diag.actions, min_actions=4, max_actions=4) == []


def test_ambiguous_element_multiset_is_reported(monkeypatch):
    parent = Chem.MolFromSmiles(MESITYLENE)
    monkeypatch.setattr(
        "rollout_policy.eligibility._encode_safe_fragments",
        lambda smiles, cut_bonds: ("A.A", ["A", "A"]),
    )
    monkeypatch.setattr(
        "rollout_policy.eligibility._safe_fragment_element_multiset",
        lambda frag: (("C", 1),),
    )
    diag = enumerate_aligned_actions(parent, require_safe_alignment=True)
    assert diag.n_verified == 0
    assert diag.rejection_reasons.count("ambiguous_element_multiset") == diag.n_rdkit


def test_roundtrip_failure_is_recorded(monkeypatch):
    parent = Chem.MolFromSmiles(MESITYLENE)
    offline = enumerate_aligned_actions(parent, require_safe_alignment=False)

    monkeypatch.setattr(
        "rollout_policy.eligibility.align_actions_to_safe_spans_diagnostic",
        lambda parent_mol, rdkit_actions: (list(offline.actions), []),
    )

    def _fail(parent_mol, action):
        raise ValueError("bad roundtrip")

    monkeypatch.setattr("rollout_policy.eligibility.verify_action_roundtrip", _fail)
    diag = enumerate_aligned_actions(parent, require_safe_alignment=True)
    assert diag.n_verified == 0
    assert diag.rejection_reasons.count("roundtrip_failed:ValueError") == len(offline.actions)
