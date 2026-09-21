"""ChEMBL download, parsing, and deterministic parent standardization."""

from __future__ import annotations

import gzip
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.request import urlretrieve

from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from rollout_policy.io import ensure_dir, resolve_path


CHEMBL_37_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/"
    "chembl_37/chembl_37_chemreps.txt.gz"
)
CHEMBL_37_SHA256 = "ea6181ce8dc7af41974e35b92e1febb0c9dcbe2c62f7ccc4a5d983ac19f696e7"


@dataclass(frozen=True)
class StandardizedParent:
    chembl_id: str
    source_smiles: str
    smiles: str
    heavy_atoms: int
    qed: float
    core_smiles: str
    atom_mapped_smiles: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def download_chembl_chemreps(
    destination: str | Path = "data/raw/chembl_37_chemreps.txt.gz",
    url: str = CHEMBL_37_URL,
    expected_sha256: str = CHEMBL_37_SHA256,
) -> Path:
    dest = resolve_path(destination)
    ensure_dir(dest.parent)
    if not dest.exists():
        urlretrieve(url, dest)
    actual = sha256_file(dest)
    if actual != expected_sha256:
        raise RuntimeError(f"Checksum mismatch: {actual} != {expected_sha256}")
    return dest


def iter_chembl_smiles(path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield (chembl_id, smiles) from the chemreps TSV.

    Column names are read from the header; do not assume column order.
    """
    resolved = resolve_path(path)
    opener = gzip.open if str(resolved).endswith(".gz") else open
    with opener(resolved, "rt", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        header_lower = [h.strip().lower() for h in header]
        try:
            id_idx = header_lower.index("chembl_id")
        except ValueError as exc:
            raise ValueError(f"Missing chembl_id column in {resolved}: {header}") from exc

        smiles_idx = None
        for candidate in ("canonical_smiles", "smiles", "standard_inchi"):
            if candidate in header_lower:
                # Prefer SMILES columns; skip InChI unless nothing else exists.
                if candidate != "standard_inchi":
                    smiles_idx = header_lower.index(candidate)
                    break
        if smiles_idx is None:
            raise ValueError(f"No SMILES column in {resolved}: {header}")

        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(id_idx, smiles_idx):
                continue
            chembl_id = parts[id_idx].strip()
            smiles = parts[smiles_idx].strip()
            if chembl_id and smiles:
                yield chembl_id, smiles


def _strip_atom_maps(mol: Chem.Mol) -> Chem.Mol:
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def standardize_molecule(smiles: str) -> Chem.Mol | None:
    """Deterministic neutralization / normalization. Returns None if rejected."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    # Reject explicit multi-component inputs early.
    if smiles.count(".") > 0 or len(Chem.GetMolFrags(mol)) > 1:
        # Keep only if standardization collapses to one fragment below.
        pass

    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    if len(Chem.GetMolFrags(mol)) != 1:
        return None

    mol = _strip_atom_maps(mol)
    # Clear stereochemistry policy: keep as-is from standardization; do not invent.
    try:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception:
        pass
    return mol


def assign_atom_maps(mol: Chem.Mol) -> Chem.Mol:
    """Return a copy with 1-based atom map numbers in current atom order."""
    mapped = Chem.Mol(mol)
    for i, atom in enumerate(mapped.GetAtoms(), start=1):
        atom.SetAtomMapNum(i)
    return mapped


def murcko_core_smiles(mol: Chem.Mol) -> str | None:
    core = MurckoScaffold.GetScaffoldForMol(mol)
    if core is None or core.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(core, canonical=True)


def murcko_core_atom_indices(mol: Chem.Mol) -> frozenset[int] | None:
    """Indices of Bemis–Murcko scaffold atoms in `mol` (unmapped)."""
    core = MurckoScaffold.GetScaffoldForMol(mol)
    if core is None or core.GetNumAtoms() == 0:
        return None
    matches = mol.GetSubstructMatches(core)
    if not matches:
        return None
    # Prefer the match that covers the scaffold once; take the first deterministic match.
    return frozenset(matches[0])


def passes_parent_filters(
    mol: Chem.Mol,
    *,
    min_heavy_atoms: int = 15,
    max_heavy_atoms: int = 45,
    min_qed: float = 0.30,
    max_qed: float = 0.85,
) -> bool:
    heavy = mol.GetNumHeavyAtoms()
    if heavy < min_heavy_atoms or heavy > max_heavy_atoms:
        return False
    qed = QED.qed(mol)
    if qed < min_qed or qed > max_qed:
        return False
    if murcko_core_smiles(mol) is None:
        return False
    return True


def standardize_parent(
    chembl_id: str,
    source_smiles: str,
    *,
    min_heavy_atoms: int = 15,
    max_heavy_atoms: int = 45,
    min_qed: float = 0.30,
    max_qed: float = 0.85,
) -> StandardizedParent | None:
    mol = standardize_molecule(source_smiles)
    if mol is None:
        return None
    if not passes_parent_filters(
        mol,
        min_heavy_atoms=min_heavy_atoms,
        max_heavy_atoms=max_heavy_atoms,
        min_qed=min_qed,
        max_qed=max_qed,
    ):
        return None
    core = murcko_core_smiles(mol)
    if core is None:
        return None
    mapped = assign_atom_maps(mol)
    return StandardizedParent(
        chembl_id=chembl_id,
        source_smiles=source_smiles,
        smiles=Chem.MolToSmiles(mol, canonical=True),
        heavy_atoms=mol.GetNumHeavyAtoms(),
        qed=float(QED.qed(mol)),
        core_smiles=core,
        atom_mapped_smiles=Chem.MolToSmiles(mapped, canonical=False),
    )
