"""Fixed RDKit descriptors for action-value features."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

FEATURE_NAMES: tuple[str, ...] = (
    "parent_molwt",
    "parent_logp",
    "parent_hbd",
    "parent_hba",
    "parent_heavy_atoms",
    "core_heavy_atoms",
    "fragment_heavy_atoms",
    "fragment_heavy_atom_fraction",
    "bond_single",
    "bond_double",
    "bond_aromatic",
    "is_ring_fragment",
    "is_terminal_fragment",
)


def action_features(
    parent: Chem.Mol,
    core: Chem.Mol,
    fragment: Chem.Mol,
    bond_type: str,
    is_ring: bool,
    is_terminal: bool,
) -> np.ndarray:
    """Fixed descriptors only. Fit scaler on train actions only."""
    parent_heavy = parent.GetNumHeavyAtoms()
    if parent_heavy <= 0:
        raise ValueError("parent has no heavy atoms")
    return np.asarray(
        [
            Descriptors.MolWt(parent),
            Descriptors.MolLogP(parent),
            rdMolDescriptors.CalcNumHBD(parent),
            rdMolDescriptors.CalcNumHBA(parent),
            parent.GetNumHeavyAtoms(),
            core.GetNumHeavyAtoms(),
            fragment.GetNumHeavyAtoms(),
            fragment.GetNumHeavyAtoms() / parent_heavy,
            int(bond_type == "SINGLE"),
            int(bond_type == "DOUBLE"),
            int(bond_type == "AROMATIC"),
            int(is_ring),
            int(is_terminal),
        ],
        dtype=np.float32,
    )


def features_from_action_record(
    parent: Chem.Mol,
    core: Chem.Mol,
    fragment: Chem.Mol,
    *,
    bond_type: str,
    is_ring: bool,
    is_terminal: bool,
) -> dict[str, float]:
    values = action_features(parent, core, fragment, bond_type, is_ring, is_terminal)
    return {name: float(val) for name, val in zip(FEATURE_NAMES, values)}
