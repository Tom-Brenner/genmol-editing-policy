"""Eligible pendant-fragment enumeration and SAFE-span alignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from rollout_policy.chembl import assign_atom_maps, murcko_core_atom_indices


@dataclass(frozen=True)
class EligibleAction:
    action_id: str
    fragment_atom_indices: tuple[int, ...]
    core_attachment_atom: int
    fragment_attachment_atom: int
    bond_type: str
    canonical_order_key: tuple
    safe_span: tuple[int, int]
    safe_fragment_index: int
    fragment_smiles: str
    fragment_heavy_atoms: int
    is_ring: bool
    is_terminal: bool
    # Parent-level cut bonds used by the pendant SAFE slicer (needed at generation).
    slicer_cut_bonds: tuple[tuple[int, int], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fragment_atom_indices"] = list(self.fragment_atom_indices)
        payload["safe_span"] = list(self.safe_span)
        payload["canonical_order_key"] = list(self.canonical_order_key)
        payload["slicer_cut_bonds"] = [list(b) for b in self.slicer_cut_bonds]
        return payload


class PendantBondSlicer:
    """SAFE slicer that cuts only the listed parent bonds (core↔pendant attachments)."""

    def __init__(self, cut_bonds: tuple[tuple[int, int], ...] | list[tuple[int, int]]):
        self.cut_bonds = tuple((int(a), int(b)) for a, b in cut_bonds)

    def __call__(self, mol):
        for a, b in self.cut_bonds:
            yield (a, b)


def _bond_type_name(bond: Chem.Bond) -> str:
    return bond.GetBondType().name


def _is_cuttable_bond(bond: Chem.Bond) -> bool:
    return (not bond.IsInRing()) and bond.GetBondType() == Chem.BondType.SINGLE


def _fragment_atom_set_after_cut(
    mol: Chem.Mol, core_atoms: frozenset[int], core_atom: int, frag_atom: int
) -> frozenset[int] | None:
    """BFS from frag_atom without crossing into core via the cut bond."""
    if frag_atom in core_atoms or core_atom not in core_atoms:
        return None
    visited = set()
    stack = [frag_atom]
    while stack:
        idx = stack.pop()
        if idx in visited:
            continue
        visited.add(idx)
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx == core_atom and idx == frag_atom:
                continue
            if nidx in core_atoms:
                # Additional core contact → not a simple pendant fragment.
                return None
            if nidx not in visited:
                stack.append(nidx)
    if not visited:
        return None
    return frozenset(visited)


def _fragment_mol(mol: Chem.Mol, atom_indices: frozenset[int], attach_idx: int) -> Chem.Mol:
    """Build a fragment mol with a dummy attachment at attach_idx."""
    rw = Chem.RWMol(Chem.Mol())
    index_map: dict[int, int] = {}
    for idx in sorted(atom_indices):
        atom = mol.GetAtomWithIdx(idx)
        new_idx = rw.AddAtom(Chem.Atom(atom.GetAtomicNum()))
        index_map[idx] = new_idx
        rw.GetAtomWithIdx(new_idx).SetFormalCharge(atom.GetFormalCharge())
        rw.GetAtomWithIdx(new_idx).SetIsAromatic(atom.GetIsAromatic())
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in atom_indices and b in atom_indices:
            rw.AddBond(index_map[a], index_map[b], bond.GetBondType())
    dummy = rw.AddAtom(Chem.Atom(0))
    rw.AddBond(index_map[attach_idx], dummy, Chem.BondType.SINGLE)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def enumerate_rdkit_pendant_actions(parent: Chem.Mol) -> list[dict[str, Any]]:
    """Enumerate pendant fragments attached to the Murcko core by one cuttable bond."""
    core_atoms = murcko_core_atom_indices(parent)
    if core_atoms is None:
        return []

    candidates: list[dict[str, Any]] = []
    for bond in parent.GetBonds():
        if not _is_cuttable_bond(bond):
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (a in core_atoms) ^ (b in core_atoms):
            core_atom, frag_atom = (a, b) if a in core_atoms else (b, a)
            frag_atoms = _fragment_atom_set_after_cut(parent, core_atoms, core_atom, frag_atom)
            if frag_atoms is None:
                continue
            frag_mol = _fragment_mol(parent, frag_atoms, frag_atom)
            frag_smiles = Chem.MolToSmiles(frag_mol, canonical=True)
            is_ring = any(parent.GetAtomWithIdx(i).IsInRing() for i in frag_atoms)
            # Terminal: fragment has no further cuttable exit beyond the attachment.
            is_terminal = True
            for idx in frag_atoms:
                atom = parent.GetAtomWithIdx(idx)
                for nbr in atom.GetNeighbors():
                    nidx = nbr.GetIdx()
                    if nidx in frag_atoms or nidx == core_atom:
                        continue
                    is_terminal = False
            order_key = (-frag_mol.GetNumHeavyAtoms(), core_atom, frag_atom, frag_smiles)
            candidates.append(
                {
                    "fragment_atom_indices": tuple(sorted(frag_atoms)),
                    "core_attachment_atom": core_atom,
                    "fragment_attachment_atom": frag_atom,
                    "bond_type": _bond_type_name(bond),
                    "canonical_order_key": order_key,
                    "fragment_smiles": frag_smiles,
                    "fragment_heavy_atoms": frag_mol.GetNumHeavyAtoms(),
                    "is_ring": is_ring,
                    "is_terminal": is_terminal,
                    "fragment_mol": frag_mol,
                }
            )

    # Deduplicate identical atom sets (keep first by order key).
    candidates.sort(key=lambda c: c["canonical_order_key"])
    seen: set[tuple[int, ...]] = set()
    unique: list[dict[str, Any]] = []
    for cand in candidates:
        key = cand["fragment_atom_indices"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)
    return unique


def _encode_safe_fragments(
    smiles: str, cut_bonds: tuple[tuple[int, int], ...]
) -> tuple[str, list[str]]:
    """Encode SMILES with the pendant-bond slicer; return full SAFE and fragments."""
    import safe as sf

    encoded = sf.SAFEConverter(
        slicer=PendantBondSlicer(cut_bonds), ignore_stereo=True
    ).encoder(smiles, allow_empty=True)
    fragments = encoded.split(".")
    return encoded, fragments


def _decode_safe_fragment_to_mol(safe_fragment: str) -> Chem.Mol | None:
    """sf.decode returns a SMILES string; convert to Mol."""
    import safe as sf

    try:
        decoded = sf.decode(safe_fragment, canonical=False, ignore_errors=True)
    except Exception:
        decoded = None
    if decoded is None:
        return None
    if isinstance(decoded, Chem.Mol):
        return decoded
    if isinstance(decoded, str):
        return Chem.MolFromSmiles(decoded)
    return None


def _fragment_element_multiset(mol: Chem.Mol) -> tuple[tuple[str, int], ...]:
    from collections import Counter

    counts = Counter(
        atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 0
    )
    return tuple(sorted(counts.items()))


def _safe_fragment_element_multiset(safe_fragment: str) -> tuple[tuple[str, int], ...] | None:
    frag_mol = _decode_safe_fragment_to_mol(safe_fragment)
    if frag_mol is None:
        return None
    return _fragment_element_multiset(frag_mol)


def _rdkit_cut_piece_atom_indices(
    parent: Chem.Mol, cut_bonds: tuple[tuple[int, int], ...]
) -> list[frozenset[int]]:
    """Atom-index sets for each connected component after cutting attachment bonds."""
    bond_idxs = []
    for a, b in cut_bonds:
        bond = parent.GetBondBetweenAtoms(int(a), int(b))
        if bond is None:
            raise ValueError(f"missing bond between {a} and {b}")
        bond_idxs.append(bond.GetIdx())
    fragmented = Chem.FragmentOnBonds(
        parent,
        bond_idxs,
        addDummies=True,
        dummyLabels=[(i + 1, i + 1) for i in range(len(bond_idxs))],
    )
    # Map fragmented atom idx → original atom idx via molAtomMapNumber? FragmentOnBonds
    # keeps original atoms first; dummies are appended. GetMolFrags returns indices into
    # the fragmented mol; originals retain the same indices as in `parent`.
    pieces = Chem.GetMolFrags(fragmented, asMols=False, sanitizeFrags=False)
    n_orig = parent.GetNumAtoms()
    out: list[frozenset[int]] = []
    for piece in pieces:
        orig = frozenset(i for i in piece if i < n_orig)
        if orig:
            out.append(orig)
    return out


def align_actions_to_safe_spans(
    parent: Chem.Mol, rdkit_actions: list[dict[str, Any]]
) -> list[EligibleAction]:
    """Attach verified SAFE fragment indices/spans to RDKit pendant actions.

    Uses a pendant-only slicer (cut bonds = core↔pendant attachments) so each
    eligible pendant is one '.'-delimited SAFE fragment. Alignment matches by
    element multiset (SAFE decode vs RDKit fragment), not fragile substructure
    remapping of attachment-numbered SAFE strings.
    """
    smiles = Chem.MolToSmiles(parent, canonical=True)
    cut_bonds = tuple(
        (a["core_attachment_atom"], a["fragment_attachment_atom"]) for a in rdkit_actions
    )
    encoded, fragments = _encode_safe_fragments(smiles, cut_bonds)
    if not fragments:
        return []

    char_spans: list[tuple[int, int]] = []
    cursor = 0
    for frag in fragments:
        start = cursor
        end = start + len(frag)
        char_spans.append((start, end))
        cursor = end + 1

    frag_multisets = [_safe_fragment_element_multiset(f) for f in fragments]
    aligned: list[EligibleAction] = []
    used_frag_idx: set[int] = set()
    for rank, action in enumerate(rdkit_actions):
        target_ms = _fragment_element_multiset(action["fragment_mol"])
        matches = [
            i
            for i, ms in enumerate(frag_multisets)
            if i not in used_frag_idx and ms == target_ms
        ]
        if len(matches) != 1:
            continue
        frag_idx = matches[0]
        # Confirm the cut-piece atom set equals the RDKit pendant atom set.
        pieces = _rdkit_cut_piece_atom_indices(parent, cut_bonds)
        target_atoms = frozenset(action["fragment_atom_indices"])
        if target_atoms not in pieces:
            continue
        used_frag_idx.add(frag_idx)
        aligned.append(
            EligibleAction(
                action_id=f"a{rank}_{frag_idx}",
                fragment_atom_indices=action["fragment_atom_indices"],
                core_attachment_atom=action["core_attachment_atom"],
                fragment_attachment_atom=action["fragment_attachment_atom"],
                bond_type=action["bond_type"],
                canonical_order_key=action["canonical_order_key"],
                safe_span=char_spans[frag_idx],
                safe_fragment_index=frag_idx,
                fragment_smiles=action["fragment_smiles"],
                fragment_heavy_atoms=action["fragment_heavy_atoms"],
                is_ring=action["is_ring"],
                is_terminal=action["is_terminal"],
                slicer_cut_bonds=cut_bonds,
            )
        )
    return aligned


def verify_action_roundtrip(parent: Chem.Mol, action: EligibleAction) -> None:
    """Raise if SAFE fragment at action index does not match the RDKit pendant."""
    if not action.slicer_cut_bonds:
        raise ValueError("action missing slicer_cut_bonds")
    _, fragments = _encode_safe_fragments(
        Chem.MolToSmiles(parent, canonical=True), action.slicer_cut_bonds
    )
    if action.safe_fragment_index < 0 or action.safe_fragment_index >= len(fragments):
        raise ValueError(f"SAFE fragment index out of range: {action.safe_fragment_index}")
    frag = fragments[action.safe_fragment_index]
    ms = _safe_fragment_element_multiset(frag)
    frag_mol = Chem.MolFromSmiles(action.fragment_smiles)
    if frag_mol is None:
        raise ValueError(f"invalid fragment_smiles: {action.fragment_smiles}")
    if ms != _fragment_element_multiset(frag_mol):
        raise ValueError(
            f"SAFE/RDKit element multiset mismatch for {action.action_id}: "
            f"safe={ms} rdkit={_fragment_element_multiset(frag_mol)}"
        )
    pieces = _rdkit_cut_piece_atom_indices(parent, action.slicer_cut_bonds)
    if frozenset(action.fragment_atom_indices) not in pieces:
        raise ValueError(
            f"RDKit pendant atoms not recovered after cut: {action.fragment_atom_indices}"
        )


def enumerate_eligible_actions(
    parent: Chem.Mol,
    *,
    min_actions: int = 2,
    max_actions: int = 4,
    require_safe_alignment: bool = True,
) -> list[EligibleAction]:
    """Return only pendant fragments attached to the protected core by one cuttable bond.

    If more than `max_actions` qualify, retain the first `max_actions` under the
    deterministic canonical ordering. Parents outside [min_actions, max_actions]
    after alignment yield an empty list (caller should reject).
    """
    rdkit_actions = enumerate_rdkit_pendant_actions(parent)
    if not rdkit_actions:
        return []

    if require_safe_alignment:
        try:
            aligned = align_actions_to_safe_spans(parent, rdkit_actions)
        except Exception:
            return []
        for action in aligned:
            try:
                verify_action_roundtrip(parent, action)
            except Exception:
                return []
    else:
        cut_bonds = tuple(
            (a["core_attachment_atom"], a["fragment_attachment_atom"]) for a in rdkit_actions
        )
        aligned = []
        for rank, action in enumerate(rdkit_actions):
            aligned.append(
                EligibleAction(
                    action_id=f"a{rank}_rdkit",
                    fragment_atom_indices=action["fragment_atom_indices"],
                    core_attachment_atom=action["core_attachment_atom"],
                    fragment_attachment_atom=action["fragment_attachment_atom"],
                    bond_type=action["bond_type"],
                    canonical_order_key=action["canonical_order_key"],
                    safe_span=(rank, rank + 1),
                    safe_fragment_index=rank,
                    fragment_smiles=action["fragment_smiles"],
                    fragment_heavy_atoms=action["fragment_heavy_atoms"],
                    is_ring=action["is_ring"],
                    is_terminal=action["is_terminal"],
                    slicer_cut_bonds=cut_bonds,
                )
            )

    aligned = sorted(aligned, key=lambda a: a.canonical_order_key)
    if len(aligned) > max_actions:
        aligned = aligned[:max_actions]
    if len(aligned) < min_actions or len(aligned) > max_actions:
        return []
    return aligned


def core_mol_from_parent(parent: Chem.Mol) -> Chem.Mol:
    core = MurckoScaffold.GetScaffoldForMol(parent)
    if core is None or core.GetNumAtoms() == 0:
        raise ValueError("empty Murcko core")
    return core


def parent_with_maps(smiles: str) -> tuple[Chem.Mol, Chem.Mol]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    Chem.SanitizeMol(mol)
    return mol, assign_atom_maps(mol)
