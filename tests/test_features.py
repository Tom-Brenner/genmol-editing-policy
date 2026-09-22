"""Schema-v2 action features and the mixed continuous/binary scaler."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem

from rollout_policy.eligibility import core_mol_from_parent, enumerate_aligned_actions
from rollout_policy.features import (
    ATTACHMENT_FP_BITS,
    FEATURE_KINDS,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    StreamingMixedFeatureScaler,
    action_features,
    induced_fragment,
)

MESITYLENE = "Cc1cc(C)cc(C)c1"


def _parent_actions():
    mol = Chem.MolFromSmiles(MESITYLENE)
    actions = enumerate_aligned_actions(mol, require_safe_alignment=False).actions
    assert len(actions) >= 2
    return mol, actions


def test_feature_names_match_values():
    mol, actions = _parent_actions()
    core = core_mol_from_parent(mol)
    values = action_features(mol, core, actions[0])
    assert values.shape == (len(FEATURE_NAMES),)
    assert len(FEATURE_KINDS) == len(FEATURE_NAMES)
    assert FEATURE_SCHEMA_VERSION == 2
    again = action_features(mol, core, actions[0])
    assert np.array_equal(values, again)


def test_actions_differ_and_fragment_has_no_dummy():
    mol, actions = _parent_actions()
    core = core_mol_from_parent(mol)
    first = action_features(mol, core, actions[0])
    second = action_features(mol, core, actions[1])
    assert not np.array_equal(first, second)
    fragment = induced_fragment(mol, actions[0].fragment_atom_indices)
    assert all(atom.GetAtomicNum() > 0 for atom in fragment.GetAtoms())
    morgan = [name for name in FEATURE_NAMES if name.startswith("core_attachment_morgan_")]
    assert len(morgan) == ATTACHMENT_FP_BITS
    assert len([name for name in FEATURE_NAMES if name.startswith("fragment_attachment_morgan_")]) == ATTACHMENT_FP_BITS


def test_invalid_attachment_is_rejected():
    mol, actions = _parent_actions()
    core = core_mol_from_parent(mol)
    broken = replace(actions[0], fragment_atom_indices=(10_000,))
    with pytest.raises(ValueError):
        action_features(mol, core, broken)


def test_binary_columns_stay_binary_after_scaling():
    generator = np.random.default_rng(0)
    width = len(FEATURE_NAMES)
    batch = generator.normal(size=(40, width)).astype(np.float32)
    binary = np.array([kind == "binary" for kind in FEATURE_KINDS])
    batch[:, binary] = generator.integers(0, 2, size=(40, int(binary.sum())))
    scaler = StreamingMixedFeatureScaler()
    scaler.partial_fit(batch[:20])
    scaler.partial_fit(batch[20:])
    scaler.finalize()
    transformed = scaler.transform(batch)
    assert np.all((transformed[:, binary] == 0) | (transformed[:, binary] == 1))
    restored = StreamingMixedFeatureScaler.from_state(scaler.state_dict())
    assert np.allclose(restored.transform(batch), transformed)


def test_schema_mismatch_rejects_checkpoint_scaler():
    scaler = StreamingMixedFeatureScaler()
    scaler.partial_fit(np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32))
    scaler.finalize()
    payload = scaler.state_dict()
    payload["feature_schema_version"] = 1
    with pytest.raises(ValueError):
        StreamingMixedFeatureScaler.from_state(payload)
