"""Unit tests for exact-core verification and reward contracts."""

from __future__ import annotations

import pytest
from rdkit import Chem

from rollout_policy.chembl import assign_atom_maps
from rollout_policy.core_check import (
    check_exact_core,
    exact_core_preserved,
    extract_protected_core,
    propagate_atom_maps_via_match,
)
from rollout_policy.eligibility import enumerate_eligible_actions, enumerate_rdkit_pendant_actions
from rollout_policy.scoring import (
    max_heavy_atom_delta_allowed,
    parent_success,
    reward,
    score_candidate,
)


# Phenylalanine-like: aromatic core with pendant carboxyl and amino methyl-ish side.
# Ibuprofen: clear Murcko core + pendant chains.
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


def _mapped_parent(smiles: str = IBUPROFEN):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    mapped = assign_atom_maps(mol)
    core = extract_protected_core(mapped)
    assert core is not None
    return mol, mapped, core


def test_exact_core_true_retained_parent():
    mol, mapped, core = _mapped_parent()
    assert exact_core_preserved(mapped, core)
    assert check_exact_core(mapped, mapped, core)


def test_exact_core_rejects_invalid_smiles():
    _, mapped, core = _mapped_parent()
    assert check_exact_core(mapped, None, core) is False
    scored = score_candidate(Chem.MolFromSmiles(IBUPROFEN), mapped, "not_a_molecule")
    assert scored["valid"] is False
    assert scored["reward"] == 0.0


def test_exact_core_rejects_core_bond_mutation():
    """Candidate with a broken/changed core bond must be infeasible."""
    mol, mapped, core = _mapped_parent()
    # Build a candidate that is the parent but remove aromaticity on one core bond
    # by converting to a non-matching aliphatic ring system when possible.
    # Simpler adversarial case: mutate one core atom element after mapping.
    cand = Chem.Mol(mapped)
    # Flip formal charge on a core-mapped atom to violate the element/charge policy.
    map_num = core.core_atom_map_nums[0]
    for atom in cand.GetAtoms():
        if atom.GetAtomMapNum() == map_num:
            atom.SetFormalCharge(atom.GetFormalCharge() + 1)
            break
    assert exact_core_preserved(cand, core) is False


def test_exact_core_rejects_unmapped_smarts_copy():
    """A molecule that merely contains the core SMARTS without maps is not enough
    unless unique match propagation succeeds — two copies must fail."""
    mol, mapped, core = _mapped_parent()
    # Two disconnected copies of benzene (core-ish) — ambiguous / wrong topology.
    cand = Chem.MolFromSmiles("c1ccccc1.c1ccccc1")
    assert cand is not None
    assert check_exact_core(mapped, cand, core) is False


def test_exact_core_rejects_mapping_failure_on_unrelated():
    _, mapped, core = _mapped_parent()
    cand = Chem.MolFromSmiles("CCO")
    assert cand is not None
    assert check_exact_core(mapped, cand, core) is False


def test_reward_feasible_and_cap():
    parent = Chem.MolFromSmiles(IBUPROFEN)
    # Same molecule: delta QED = 0, feasible if core ok and constraints pass.
    scored = reward(parent, parent, exact_core_ok=True)
    assert scored["feasible"] is True
    assert scored["reward"] == 0.0
    assert scored["tanimoto"] == pytest.approx(1.0)

    # Infeasible when core fails.
    scored2 = reward(parent, parent, exact_core_ok=False)
    assert scored2["feasible"] is False
    assert scored2["reward"] == 0.0


def test_reward_clips_to_unit_interval(monkeypatch):
    parent = Chem.MolFromSmiles("CCO")
    cand = Chem.MolFromSmiles("CCO")
    # Force a large fake delta via monkeypatching QED.
    from rdkit.Chem import QED as qed_mod

    values = {id(parent): 0.2, id(cand): 0.9}

    def fake_qed(mol):
        return values.get(id(mol), 0.5)

    monkeypatch.setattr(qed_mod, "qed", fake_qed)
    scored = reward(parent, cand, exact_core_ok=True, reward_qed_delta_cap=0.10)
    assert scored["qed_delta_raw"] == pytest.approx(0.7)
    assert scored["reward"] == pytest.approx(1.0)


def test_parent_success_requires_qed_delta():
    rows = [
        {
            "valid": True,
            "exact_core_ok": True,
            "tanimoto": 0.9,
            "heavy_atom_delta": 0,
            "parent_n_heavy_atoms": 15,
            "qed_delta_raw": 0.01,
        },
        {
            "valid": True,
            "exact_core_ok": True,
            "tanimoto": 0.9,
            "heavy_atom_delta": 0,
            "parent_n_heavy_atoms": 15,
            "qed_delta_raw": 0.06,
        },
    ]
    assert parent_success(rows) is True
    assert parent_success(rows[:1]) is False


def test_max_heavy_atom_delta_is_round_one_third():
    assert max_heavy_atom_delta_allowed(15) == 5
    assert max_heavy_atom_delta_allowed(16) == 5
    assert max_heavy_atom_delta_allowed(17) == 6
    assert max_heavy_atom_delta_allowed(45) == 15


def test_parent_success_respects_fraction_budget():
    ok = {
        "valid": True,
        "exact_core_ok": True,
        "tanimoto": 0.9,
        "heavy_atom_delta": 5,
        "parent_n_heavy_atoms": 15,
        "qed_delta_raw": 0.06,
    }
    too_large = {**ok, "heavy_atom_delta": 6}
    assert parent_success([ok]) is True
    assert parent_success([too_large]) is False
    # Alternate success QED threshold can be recomputed from raw fields.
    assert parent_success([ok], success_qed_delta=0.10) is False


def test_score_candidate_records_budget_fields():
    parent = Chem.MolFromSmiles(IBUPROFEN)
    mapped = assign_atom_maps(parent)
    scored = score_candidate(
        parent,
        mapped,
        IBUPROFEN,
        constraints={"max_heavy_atom_delta_fraction": 1.0 / 3.0, "min_tanimoto_to_parent": 0.50},
    )
    n_heavy = parent.GetNumHeavyAtoms()
    assert scored["parent_n_heavy_atoms"] == n_heavy
    assert scored["candidate_n_heavy_atoms"] == n_heavy
    assert scored["max_heavy_atom_delta_allowed"] == max_heavy_atom_delta_allowed(n_heavy)
    assert scored["heavy_atom_delta"] == 0
    assert scored["tanimoto"] == pytest.approx(1.0)
    assert scored["parent_qed"] is not None
    assert scored["candidate_qed"] is not None


def test_enumerate_pendant_actions_ibuprofen():
    mol = Chem.MolFromSmiles(IBUPROFEN)
    actions = enumerate_rdkit_pendant_actions(mol)
    assert len(actions) >= 2
    # Offline eligible path without SAFE.
    eligible = enumerate_eligible_actions(mol, require_safe_alignment=False)
    assert 2 <= len(eligible) <= 4
    # Ordering is deterministic.
    keys = [a.canonical_order_key for a in eligible]
    assert keys == sorted(keys)


def test_propagate_maps_on_identical_unmapped():
    mol, mapped, core = _mapped_parent()
    unmapped = Chem.MolFromSmiles(IBUPROFEN)
    propagated = propagate_atom_maps_via_match(mapped, unmapped, core)
    assert propagated is not None
    assert exact_core_preserved(propagated, core)
