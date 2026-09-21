"""Exact protected-core verification via atom-map tracking."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


@dataclass(frozen=True)
class ProtectedCore:
    """Atom-mapped Bemis–Murcko core of a parent molecule."""

    parent_atom_mapped_smiles: str
    core_atom_map_nums: tuple[int, ...]
    core_internal_bonds: tuple[tuple[int, int, str], ...]
    core_elements: tuple[tuple[int, int, bool, int], ...]  # map, Z, aromatic, charge
    core_smiles: str


def _bond_type_name(bond: Chem.Bond) -> str:
    return str(bond.GetBondType())


def extract_protected_core(parent_atom_mapped: Chem.Mol) -> ProtectedCore | None:
    """Extract the Murcko core using existing atom-map numbers on the parent."""
    if parent_atom_mapped is None:
        return None
    if any(a.GetAtomMapNum() == 0 for a in parent_atom_mapped.GetAtoms()):
        return None

    core = MurckoScaffold.GetScaffoldForMol(parent_atom_mapped)
    if core is None or core.GetNumHeavyAtoms() == 0:
        return None

    matches = parent_atom_mapped.GetSubstructMatches(core)
    if len(matches) != 1:
        # Ambiguous core placement on the parent itself — reject.
        return None
    core_indices = matches[0]
    index_set = set(core_indices)

    map_nums = []
    elements = []
    for idx in core_indices:
        atom = parent_atom_mapped.GetAtomWithIdx(idx)
        map_num = atom.GetAtomMapNum()
        map_nums.append(map_num)
        elements.append(
            (map_num, atom.GetAtomicNum(), atom.GetIsAromatic(), int(atom.GetFormalCharge()))
        )

    bonds: list[tuple[int, int, str]] = []
    for bond in parent_atom_mapped.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in index_set and b in index_set:
            ma = parent_atom_mapped.GetAtomWithIdx(a).GetAtomMapNum()
            mb = parent_atom_mapped.GetAtomWithIdx(b).GetAtomMapNum()
            bonds.append(tuple(sorted((ma, mb))) + (_bond_type_name(bond),))  # type: ignore[operator]

    bonds_t = tuple(sorted(bonds))
    return ProtectedCore(
        parent_atom_mapped_smiles=Chem.MolToSmiles(parent_atom_mapped, canonical=False),
        core_atom_map_nums=tuple(map_nums),
        core_internal_bonds=bonds_t,
        core_elements=tuple(elements),
        core_smiles=Chem.MolToSmiles(core, canonical=True),
    )


def _candidate_map_lookup(candidate: Chem.Mol) -> dict[int, int] | None:
    """Map atom-map-number -> candidate atom index. None if duplicate/missing maps."""
    lookup: dict[int, int] = {}
    for atom in candidate.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num == 0:
            continue
        if map_num in lookup:
            return None
        lookup[map_num] = atom.GetIdx()
    return lookup


def exact_core_preserved(candidate: Chem.Mol, core: ProtectedCore) -> bool:
    """Return True iff the protected core atoms/bonds are exactly recovered.

    Requires the candidate to carry the same atom-map numbers on core atoms
    (propagated through the SAFE round-trip). A newly generated copy of the
    same SMARTS without those maps does not count.
    """
    if candidate is None or core is None:
        return False

    lookup = _candidate_map_lookup(candidate)
    if lookup is None:
        return False

    # Every core map must be present exactly once.
    for map_num, atomic_num, aromatic, charge in core.core_elements:
        if map_num not in lookup:
            return False
        atom = candidate.GetAtomWithIdx(lookup[map_num])
        if atom.GetAtomicNum() != atomic_num:
            return False
        if atom.GetIsAromatic() != aromatic:
            return False
        if int(atom.GetFormalCharge()) != charge:
            return False

    # All original internal core bonds must exist with the same bond type.
    for ma, mb, bond_name in core.core_internal_bonds:
        if ma not in lookup or mb not in lookup:
            return False
        bond = candidate.GetBondBetweenAtoms(lookup[ma], lookup[mb])
        if bond is None:
            return False
        if _bond_type_name(bond) != bond_name:
            return False

    return True


def propagate_atom_maps_via_match(
    parent_mapped: Chem.Mol,
    candidate: Chem.Mol,
    core: ProtectedCore,
) -> Chem.Mol | None:
    """Attempt to copy parent atom maps onto a candidate via unique core match.

    Used when SAFE decoding drops atom maps. Finds the unique substructure match
    of the parent core onto the candidate that is consistent with core topology,
    then copies map numbers for core atoms only. Returns None on ambiguity.
    """
    if candidate is None:
        return None
    parent_core_mol = MurckoScaffold.GetScaffoldForMol(parent_mapped)
    if parent_core_mol is None:
        return None

    matches = candidate.GetSubstructMatches(parent_core_mol)
    if len(matches) != 1:
        return None

    match = matches[0]
    parent_matches = parent_mapped.GetSubstructMatches(parent_core_mol)
    if len(parent_matches) != 1:
        return None
    parent_core_indices = parent_matches[0]

    out = Chem.Mol(candidate)
    for atom in out.GetAtoms():
        atom.SetAtomMapNum(0)
    for parent_idx, cand_idx in zip(parent_core_indices, match):
        map_num = parent_mapped.GetAtomWithIdx(parent_idx).GetAtomMapNum()
        out.GetAtomWithIdx(cand_idx).SetAtomMapNum(map_num)

    if not exact_core_preserved(out, core):
        return None
    return out


def check_exact_core(
    parent_mapped: Chem.Mol,
    candidate: Chem.Mol | None,
    core: ProtectedCore | None = None,
) -> bool:
    """High-level exact-core check used by scoring.

    If the candidate lacks atom maps, try a unique core-match propagation first.
    Ambiguity or failure → False (infeasible).
    """
    if candidate is None:
        return False
    core = core or extract_protected_core(parent_mapped)
    if core is None:
        return False

    if exact_core_preserved(candidate, core):
        return True

    mapped_cand = propagate_atom_maps_via_match(parent_mapped, candidate, core)
    if mapped_cand is None:
        return False
    return exact_core_preserved(mapped_cand, core)
