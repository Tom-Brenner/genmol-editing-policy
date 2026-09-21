"""Action-conditioned molecular graphs for the V2 GATv2 model."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

from rollout_policy.chembl import murcko_core_atom_indices
from rollout_policy.eligibility import EligibleAction

GRAPH_SCHEMA_VERSION = 1

_ELEMENTS = (6, 7, 8, 9, 15, 16, 17, 35, 53)
_HYBRIDIZATIONS = (
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
)
_CHIRAL = (
    Chem.ChiralType.CHI_UNSPECIFIED,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
)
_BOND_TYPES = (
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
)
_BOND_STEREO = (
    Chem.BondStereo.STEREONONE,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOE,
)


NODE_FEATURE_NAMES: tuple[str, ...] = (
    tuple(f"element_{z}" for z in _ELEMENTS)
    + ("element_other",)
    + (
        "degree",
        "formal_charge",
        "is_aromatic",
        "is_in_ring",
        "total_h",
    )
    + tuple(f"hybridization_{h}" for h in _HYBRIDIZATIONS)
    + ("hybridization_other",)
    + tuple(f"chirality_{tag}" for tag in _CHIRAL)
    + ("chirality_other",)
)

EDGE_FEATURE_NAMES: tuple[str, ...] = (
    tuple(f"bond_{bt}" for bt in _BOND_TYPES)
    + ("bond_other", "conjugated", "in_ring")
    + tuple(f"stereo_{st}" for st in _BOND_STEREO)
    + ("stereo_other", "selected_cut")
)


def atom_feature_vector(atom: Chem.Atom) -> list[float]:
    z = atom.GetAtomicNum()
    hybridization = atom.GetHybridization()
    chirality = atom.GetChiralTag()
    values = [float(z == allowed) for allowed in _ELEMENTS]
    values.append(float(z not in _ELEMENTS))
    values += [
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(atom.GetTotalNumHs()),
    ]
    values += [float(hybridization == h) for h in _HYBRIDIZATIONS]
    values.append(float(hybridization not in _HYBRIDIZATIONS))
    values += [float(chirality == tag) for tag in _CHIRAL]
    values.append(float(chirality not in _CHIRAL))
    if len(values) != len(NODE_FEATURE_NAMES):
        raise RuntimeError("node feature width drifted from NODE_FEATURE_NAMES")
    return values


def bond_feature_vector(bond: Chem.Bond) -> list[float]:
    bond_type = bond.GetBondType()
    stereo = bond.GetStereo()
    values = [float(bond_type == bt) for bt in _BOND_TYPES]
    values.append(float(bond_type not in _BOND_TYPES))
    values += [float(bond.GetIsConjugated()), float(bond.IsInRing())]
    values += [float(stereo == st) for st in _BOND_STEREO]
    values.append(float(stereo not in _BOND_STEREO))
    return values


def validate_action_indices(parent: Chem.Mol, action: EligibleAction) -> None:
    n_atoms = parent.GetNumAtoms()
    indices = set(action.fragment_atom_indices)
    if not indices or any(i < 0 or i >= n_atoms for i in indices):
        raise ValueError(f"invalid fragment atom indices for {action.action_id}")
    if action.fragment_attachment_atom not in indices:
        raise ValueError("fragment attachment atom is not in selected fragment")
    if action.core_attachment_atom in indices:
        raise ValueError("core attachment atom is inside selected fragment")
    bond = parent.GetBondBetweenAtoms(
        action.core_attachment_atom, action.fragment_attachment_atom
    )
    if bond is None:
        raise ValueError("persisted attachment bond is missing")


def action_to_graph(
    parent_record: dict[str, Any],
    action: EligibleAction,
    global_features: np.ndarray,
) -> Data:
    """One directed graph for a single parent/action pair."""
    mol = Chem.MolFromSmiles(parent_record["smiles"])
    if mol is None:
        raise ValueError("invalid parent SMILES")
    validate_action_indices(mol, action)
    core_atoms = murcko_core_atom_indices(mol)
    if core_atoms is None:
        raise ValueError("parent has no uniquely usable Murcko core")

    node_rows = [atom_feature_vector(atom) for atom in mol.GetAtoms()]
    fragment_set = set(action.fragment_atom_indices)
    fragment_mask = torch.tensor(
        [i in fragment_set for i in range(mol.GetNumAtoms())], dtype=torch.bool
    )
    core_mask = torch.tensor(
        [i in core_atoms for i in range(mol.GetNumAtoms())], dtype=torch.bool
    )
    core_attachment_mask = torch.zeros(mol.GetNumAtoms(), dtype=torch.bool)
    fragment_attachment_mask = torch.zeros(mol.GetNumAtoms(), dtype=torch.bool)
    core_attachment_mask[action.core_attachment_atom] = True
    fragment_attachment_mask[action.fragment_attachment_atom] = True

    src: list[int] = []
    dst: list[int] = []
    edge_rows: list[list[float]] = []
    selected_pair = {
        (action.core_attachment_atom, action.fragment_attachment_atom),
        (action.fragment_attachment_atom, action.core_attachment_atom),
    }
    selected_hits = 0
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        base = bond_feature_vector(bond)
        for u, v in ((a, b), (b, a)):
            src.append(u)
            dst.append(v)
            selected = float((u, v) in selected_pair)
            selected_hits += int(selected)
            edge_rows.append(base + [selected])
    if selected_hits != 2:
        raise ValueError(
            f"selected attachment edge must appear twice, found {selected_hits}"
        )
    if edge_rows and len(edge_rows[0]) != len(EDGE_FEATURE_NAMES):
        raise RuntimeError("edge feature width drifted from EDGE_FEATURE_NAMES")

    return Data(
        x=torch.tensor(node_rows, dtype=torch.float32),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(edge_rows, dtype=torch.float32),
        fragment_mask=fragment_mask,
        core_mask=core_mask,
        core_attachment_mask=core_attachment_mask,
        fragment_attachment_mask=fragment_attachment_mask,
        global_features=torch.tensor(global_features, dtype=torch.float32).view(1, -1),
        parent_id=parent_record["parent_id"],
        action_id=action.action_id,
    )
