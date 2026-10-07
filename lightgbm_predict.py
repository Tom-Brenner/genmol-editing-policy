"""Torch-free LightGBM cut-site scoring for a Hugging Face Space.

Takes a parent SMILES, enumerates RDKit pendant actions (no SAFE alignment),
scores them with the frozen lightgbm_regressor checkpoint, and returns the
learned top-1 cut site plus the full ranked list.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

FEATURE_SCHEMA_VERSION = 2
LABEL_SCHEMA_VERSION = 2
ATTACHMENT_FP_BITS = 128
MIN_ACTIONS = 2
MAX_ACTIONS = 4

_ELEMENT_SYMBOLS = ("C", "N", "O", "F", "P", "S", "Cl", "Br", "I")
_ELEMENT_Z = {
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}
_HYBRIDIZATIONS = (
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
)


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
    slicer_cut_bonds: tuple[tuple[int, int], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fragment_atom_indices"] = list(self.fragment_atom_indices)
        payload["safe_span"] = list(self.safe_span)
        payload["canonical_order_key"] = list(self.canonical_order_key)
        payload["slicer_cut_bonds"] = [list(bond) for bond in self.slicer_cut_bonds]
        return payload


def _schema() -> tuple[tuple[str, ...], tuple[str, ...]]:
    names: list[str] = []
    kinds: list[str] = []

    def add(name: str, kind: str) -> None:
        names.append(name)
        kinds.append(kind)

    for name in (
        "parent_molwt",
        "parent_logp",
        "parent_hbd",
        "parent_hba",
        "parent_heavy_atoms",
        "core_heavy_atoms",
    ):
        add(name, "continuous")
    for name in (
        "fragment_molwt",
        "fragment_logp",
        "fragment_tpsa",
        "fragment_hbd",
        "fragment_hba",
        "fragment_heteroatoms",
        "fragment_formal_charge",
        "fragment_rotatable_bonds",
        "fragment_ring_count",
        "fragment_aromatic_atoms",
        "fragment_heavy_atoms",
        "fragment_heavy_atom_fraction",
        "fragment_molwt_fraction",
        "fragment_tpsa_fraction",
    ):
        add(name, "continuous")
    add("is_ring_fragment", "binary")
    add("is_terminal_fragment", "binary")
    for prefix in ("core_attachment", "fragment_attachment"):
        for symbol in _ELEMENT_SYMBOLS:
            add(f"{prefix}_element_{symbol}", "binary")
        add(f"{prefix}_element_other", "binary")
        for name in ("degree", "total_valence", "formal_charge", "total_h"):
            add(f"{prefix}_{name}", "continuous")
        add(f"{prefix}_is_aromatic", "binary")
        add(f"{prefix}_is_in_ring", "binary")
        for hybrid in _HYBRIDIZATIONS:
            add(f"{prefix}_hybridization_{hybrid.name}", "binary")
        add(f"{prefix}_hybridization_other", "binary")
        for bit in range(ATTACHMENT_FP_BITS):
            add(f"{prefix}_morgan_{bit:03d}", "binary")
    for name in ("bond_single", "bond_double", "bond_aromatic"):
        add(name, "binary")
    return tuple(names), tuple(kinds)


FEATURE_NAMES, FEATURE_KINDS = _schema()

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=ATTACHMENT_FP_BITS,
    includeChirality=False,
)


def murcko_core_atom_indices(mol: Chem.Mol) -> frozenset[int] | None:
    core = MurckoScaffold.GetScaffoldForMol(mol)
    if core is None or core.GetNumAtoms() == 0:
        return None
    matches = mol.GetSubstructMatches(core)
    if not matches:
        return None
    return frozenset(matches[0])


def core_mol_from_parent(parent: Chem.Mol) -> Chem.Mol:
    core = MurckoScaffold.GetScaffoldForMol(parent)
    if core is None or core.GetNumAtoms() == 0:
        raise ValueError("empty Murcko core")
    return core


def _bond_type_name(bond: Chem.Bond) -> str:
    return bond.GetBondType().name


def _is_cuttable_bond(bond: Chem.Bond) -> bool:
    return (not bond.IsInRing()) and bond.GetBondType() == Chem.BondType.SINGLE


def _fragment_atom_set_after_cut(
    mol: Chem.Mol, core_atoms: frozenset[int], core_atom: int, frag_atom: int
) -> frozenset[int] | None:
    if frag_atom in core_atoms or core_atom not in core_atoms:
        return None
    visited: set[int] = set()
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
                return None
            if nidx not in visited:
                stack.append(nidx)
    if not visited:
        return None
    return frozenset(visited)


def _fragment_mol(mol: Chem.Mol, atom_indices: frozenset[int], attach_idx: int) -> Chem.Mol:
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
                }
            )

    candidates.sort(key=lambda cand: cand["canonical_order_key"])
    seen: set[tuple[int, ...]] = set()
    unique: list[dict[str, Any]] = []
    for cand in candidates:
        key = cand["fragment_atom_indices"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)
    return unique


def _offline_actions(rdkit_actions: list[dict[str, Any]]) -> list[EligibleAction]:
    cut_bonds = tuple(
        (a["core_attachment_atom"], a["fragment_attachment_atom"]) for a in rdkit_actions
    )
    aligned: list[EligibleAction] = []
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
    return aligned


def enumerate_eligible_actions(
    parent: Chem.Mol,
    *,
    min_actions: int = MIN_ACTIONS,
    max_actions: int = MAX_ACTIONS,
) -> list[EligibleAction]:
    """RDKit-only pendant enumeration (SAFE alignment intentionally disabled)."""
    rdkit_actions = enumerate_rdkit_pendant_actions(parent)
    if not rdkit_actions:
        return []
    aligned = _offline_actions(rdkit_actions)
    ordered = sorted(aligned, key=lambda action: action.canonical_order_key)
    retained = ordered[:max_actions]
    return retained if len(retained) >= min_actions else []


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


def induced_fragment(parent: Chem.Mol, atom_indices: tuple[int, ...]) -> Chem.Mol:
    smiles = Chem.MolFragmentToSmiles(
        parent,
        atomsToUse=list(atom_indices),
        canonical=True,
        isomericSmiles=False,
    )
    fragment = Chem.MolFromSmiles(smiles)
    if fragment is None:
        raise ValueError("failed to construct selected fragment")
    if any(atom.GetAtomicNum() == 0 for atom in fragment.GetAtoms()):
        raise ValueError("induced fragment contains a dummy atom")
    return fragment


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _add_attachment_atom(names: list[str], values: list[float], atom: Chem.Atom, prefix: str) -> None:
    atomic_number = atom.GetAtomicNum()
    for symbol in _ELEMENT_SYMBOLS:
        names.append(f"{prefix}_element_{symbol}")
        values.append(float(atomic_number == _ELEMENT_Z[symbol]))
    names.append(f"{prefix}_element_other")
    values.append(float(atomic_number not in _ELEMENT_Z.values()))
    names.extend(
        [
            f"{prefix}_degree",
            f"{prefix}_total_valence",
            f"{prefix}_formal_charge",
            f"{prefix}_total_h",
        ]
    )
    values.extend(
        [
            float(atom.GetDegree()),
            float(atom.GetTotalValence()),
            float(atom.GetFormalCharge()),
            float(atom.GetTotalNumHs()),
        ]
    )
    names.append(f"{prefix}_is_aromatic")
    values.append(float(atom.GetIsAromatic()))
    names.append(f"{prefix}_is_in_ring")
    values.append(float(atom.IsInRing()))
    hybridization = atom.GetHybridization()
    for hybrid in _HYBRIDIZATIONS:
        names.append(f"{prefix}_hybridization_{hybrid.name}")
        values.append(float(hybridization == hybrid))
    names.append(f"{prefix}_hybridization_other")
    values.append(float(hybridization not in _HYBRIDIZATIONS))


def centered_morgan_bits(parent: Chem.Mol, atom_idx: int) -> list[float]:
    fingerprint = _MORGAN.GetFingerprint(parent, fromAtoms=[int(atom_idx)])
    bits = [float(fingerprint.GetBit(i)) for i in range(ATTACHMENT_FP_BITS)]
    if len(bits) != ATTACHMENT_FP_BITS:
        raise RuntimeError("centered Morgan width drifted")
    return bits


def action_features(
    parent: Chem.Mol,
    core: Chem.Mol,
    action: EligibleAction,
) -> np.ndarray:
    validate_action_indices(parent, action)
    fragment = induced_fragment(parent, action.fragment_atom_indices)
    parent_heavy = parent.GetNumHeavyAtoms()
    if parent_heavy <= 0:
        raise ValueError("parent has no heavy atoms")
    parent_molwt = float(Descriptors.MolWt(parent))
    parent_tpsa = float(Descriptors.TPSA(parent))
    fragment_molwt = float(Descriptors.MolWt(fragment))
    fragment_tpsa = float(Descriptors.TPSA(fragment))
    bond = parent.GetBondBetweenAtoms(
        action.core_attachment_atom, action.fragment_attachment_atom
    )
    if bond is None:
        raise ValueError("persisted attachment bond is missing")
    bond_name = bond.GetBondType().name

    names: list[str] = []
    values: list[float] = []
    names.extend(
        [
            "parent_molwt",
            "parent_logp",
            "parent_hbd",
            "parent_hba",
            "parent_heavy_atoms",
            "core_heavy_atoms",
            "fragment_molwt",
            "fragment_logp",
            "fragment_tpsa",
            "fragment_hbd",
            "fragment_hba",
            "fragment_heteroatoms",
            "fragment_formal_charge",
            "fragment_rotatable_bonds",
            "fragment_ring_count",
            "fragment_aromatic_atoms",
            "fragment_heavy_atoms",
            "fragment_heavy_atom_fraction",
            "fragment_molwt_fraction",
            "fragment_tpsa_fraction",
            "is_ring_fragment",
            "is_terminal_fragment",
        ]
    )
    values.extend(
        [
            parent_molwt,
            float(Descriptors.MolLogP(parent)),
            float(rdMolDescriptors.CalcNumHBD(parent)),
            float(rdMolDescriptors.CalcNumHBA(parent)),
            float(parent_heavy),
            float(core.GetNumHeavyAtoms()),
            fragment_molwt,
            float(Descriptors.MolLogP(fragment)),
            fragment_tpsa,
            float(rdMolDescriptors.CalcNumHBD(fragment)),
            float(rdMolDescriptors.CalcNumHBA(fragment)),
            float(sum(atom.GetAtomicNum() not in (1, 6) for atom in fragment.GetAtoms())),
            float(sum(atom.GetFormalCharge() for atom in fragment.GetAtoms())),
            float(rdMolDescriptors.CalcNumRotatableBonds(fragment)),
            float(rdMolDescriptors.CalcNumRings(fragment)),
            float(sum(atom.GetIsAromatic() for atom in fragment.GetAtoms())),
            float(fragment.GetNumHeavyAtoms()),
            fragment.GetNumHeavyAtoms() / parent_heavy,
            _safe_div(fragment_molwt, parent_molwt),
            _safe_div(fragment_tpsa, parent_tpsa),
            float(action.is_ring),
            float(action.is_terminal),
        ]
    )
    _add_attachment_atom(
        names, values, parent.GetAtomWithIdx(action.core_attachment_atom), "core_attachment"
    )
    core_bits = centered_morgan_bits(parent, action.core_attachment_atom)
    names.extend(f"core_attachment_morgan_{bit:03d}" for bit in range(ATTACHMENT_FP_BITS))
    values.extend(core_bits)
    _add_attachment_atom(
        names,
        values,
        parent.GetAtomWithIdx(action.fragment_attachment_atom),
        "fragment_attachment",
    )
    fragment_bits = centered_morgan_bits(parent, action.fragment_attachment_atom)
    names.extend(f"fragment_attachment_morgan_{bit:03d}" for bit in range(ATTACHMENT_FP_BITS))
    values.extend(fragment_bits)
    names.extend(["bond_single", "bond_double", "bond_aromatic"])
    values.extend(
        [
            float(bond_name == "SINGLE"),
            float(bond_name == "DOUBLE"),
            float(bond_name == "AROMATIC"),
        ]
    )
    if tuple(names) != FEATURE_NAMES:
        raise RuntimeError("feature name/order drift")
    if len(values) != len(FEATURE_NAMES):
        raise RuntimeError("feature value width drifted")
    return np.asarray(values, dtype=np.float32)


def select_learned_index(
    actions: list[EligibleAction],
    predicted_values: list[float],
) -> int:
    if not actions:
        raise ValueError("no eligible actions")
    if len(actions) != len(predicted_values):
        raise ValueError("actions/values length mismatch")
    return min(
        range(len(actions)),
        key=lambda index: (
            -float(predicted_values[index]),
            actions[index].canonical_order_key,
            actions[index].action_id,
        ),
    )


def default_model_paths() -> tuple[Path, Path]:
    """Prefer Space-local artifacts/, then this repo's artifacts_v2/."""
    here = Path(__file__).resolve().parent
    candidates = (
        here / "artifacts" / "lightgbm_regressor.txt",
        here / "artifacts_v2" / "lightgbm_regressor.txt",
    )
    for model_path in candidates:
        sidecar = model_path.with_suffix(".json")
        if model_path.is_file() and sidecar.is_file():
            return model_path, sidecar
    raise FileNotFoundError(
        "Could not find lightgbm_regressor.txt (+ .json). "
        "Place them under ./artifacts/ or ./artifacts_v2/."
    )


class LightGBMCutSitePredictor:
    def __init__(self, model_path: Path | None = None):
        import lightgbm

        if model_path is None:
            model_path, sidecar = default_model_paths()
        else:
            model_path = Path(model_path)
            sidecar = model_path.with_suffix(".json")
        if not sidecar.is_file():
            raise FileNotFoundError(f"Missing LightGBM sidecar {sidecar}")
        blob = json.loads(sidecar.read_text(encoding="utf-8"))
        if blob.get("model_type") != "lightgbm_regressor":
            raise ValueError(f"{sidecar} is not a lightgbm_regressor sidecar")
        if list(blob["feature_names"]) != list(FEATURE_NAMES):
            raise ValueError("checkpoint feature names do not match schema v2")
        if int(blob["feature_schema_version"]) != FEATURE_SCHEMA_VERSION:
            raise ValueError("checkpoint feature schema version does not match")
        if int(blob.get("label_schema_version", 0)) != LABEL_SCHEMA_VERSION:
            raise ValueError("checkpoint label schema version does not match")
        if blob.get("scaled_features"):
            raise ValueError("lightgbm_regressor checkpoint was fit on scaled features")
        self.model_path = model_path
        self.best_iteration = int(blob.get("best_iteration") or 0)
        self.booster = lightgbm.Booster(model_file=str(model_path))

    def score_actions(
        self, parent_smiles: str, actions: list[EligibleAction]
    ) -> np.ndarray:
        mol = Chem.MolFromSmiles(parent_smiles)
        if mol is None:
            raise ValueError("invalid parent SMILES")
        core = core_mol_from_parent(mol)
        rows = np.stack([action_features(mol, core, action) for action in actions]).astype(
            np.float64
        )
        scores = np.asarray(
            self.booster.predict(rows, num_iteration=self.best_iteration),
            dtype=np.float64,
        ).reshape(-1)
        if scores.shape != (len(actions),):
            raise RuntimeError(f"expected {len(actions)} scores, got {tuple(scores.shape)}")
        return scores


_PREDICTOR: LightGBMCutSitePredictor | None = None


def get_predictor(model_path: str | Path | None = None) -> LightGBMCutSitePredictor:
    global _PREDICTOR
    if model_path is not None:
        return LightGBMCutSitePredictor(Path(model_path))
    if _PREDICTOR is None:
        _PREDICTOR = LightGBMCutSitePredictor()
    return _PREDICTOR


def predict_cut_site(
    smiles: str,
    *,
    model_path: str | Path | None = None,
    min_actions: int = MIN_ACTIONS,
    max_actions: int = MAX_ACTIONS,
) -> dict[str, Any]:
    """Return the learned cut site and ranked eligible actions for one SMILES."""
    smiles = (smiles or "").strip()
    if not smiles:
        return {"ok": False, "error": "empty SMILES"}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"ok": False, "error": "invalid SMILES"}
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # noqa: BLE001 - surface sanitize failures to the UI
        return {"ok": False, "error": f"sanitize failed: {exc}"}

    canonical = Chem.MolToSmiles(mol, canonical=True)
    try:
        core = core_mol_from_parent(mol)
        core_smiles = Chem.MolToSmiles(core, canonical=True)
    except ValueError:
        return {"ok": False, "error": "no Murcko core", "smiles": canonical}

    actions = enumerate_eligible_actions(
        mol, min_actions=min_actions, max_actions=max_actions
    )
    if not actions:
        return {
            "ok": False,
            "error": (
                f"need {min_actions}-{max_actions} eligible pendant cut sites; "
                "none found under the RDKit offline policy"
            ),
            "smiles": canonical,
            "core_smiles": core_smiles,
        }

    predictor = get_predictor(model_path)
    scores = predictor.score_actions(canonical, actions)
    chosen_index = select_learned_index(actions, [float(score) for score in scores])
    chosen = actions[chosen_index]
    order = sorted(
        range(len(actions)),
        key=lambda index: (
            -float(scores[index]),
            actions[index].canonical_order_key,
            actions[index].action_id,
        ),
    )
    ranked = []
    for rank, index in enumerate(order, start=1):
        action = actions[index]
        ranked.append(
            {
                "rank": rank,
                "action_id": action.action_id,
                "score": float(scores[index]),
                "core_attachment_atom": action.core_attachment_atom,
                "fragment_attachment_atom": action.fragment_attachment_atom,
                "bond_type": action.bond_type,
                "fragment_smiles": action.fragment_smiles,
                "fragment_heavy_atoms": action.fragment_heavy_atoms,
                "fragment_atom_indices": list(action.fragment_atom_indices),
                "is_ring": action.is_ring,
                "is_terminal": action.is_terminal,
                "selected": index == chosen_index,
            }
        )
    return {
        "ok": True,
        "smiles": canonical,
        "core_smiles": core_smiles,
        "model_type": "lightgbm_regressor",
        "model_path": str(predictor.model_path),
        "best_iteration": predictor.best_iteration,
        "n_actions": len(actions),
        "selected": {
            "action_id": chosen.action_id,
            "score": float(scores[chosen_index]),
            "core_attachment_atom": chosen.core_attachment_atom,
            "fragment_attachment_atom": chosen.fragment_attachment_atom,
            "bond_type": chosen.bond_type,
            "fragment_smiles": chosen.fragment_smiles,
            "fragment_heavy_atoms": chosen.fragment_heavy_atoms,
            "fragment_atom_indices": list(chosen.fragment_atom_indices),
        },
        "ranked_actions": ranked,
    }


def format_prediction(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        parts = [f"Error: {result.get('error', 'unknown error')}"]
        if result.get("smiles"):
            parts.append(f"SMILES: {result['smiles']}")
        if result.get("core_smiles"):
            parts.append(f"Core: {result['core_smiles']}")
        return "\n".join(parts)

    selected = result["selected"]
    lines = [
        f"SMILES: {result['smiles']}",
        f"Murcko core: {result['core_smiles']}",
        f"Selected action: {selected['action_id']}",
        f"Score: {selected['score']:.6f}",
        (
            "Cut bond: "
            f"core_atom={selected['core_attachment_atom']} — "
            f"fragment_atom={selected['fragment_attachment_atom']} "
            f"({selected['bond_type']})"
        ),
        f"Fragment SMILES: {selected['fragment_smiles']}",
        f"Fragment heavy atoms: {selected['fragment_heavy_atoms']}",
        "",
        "Ranked eligible actions:",
    ]
    for row in result["ranked_actions"]:
        marker = "*" if row["selected"] else " "
        lines.append(
            f"{marker} {row['rank']}. {row['action_id']}  score={row['score']:.6f}  "
            f"cut=({row['core_attachment_atom']},{row['fragment_attachment_atom']})  "
            f"frag={row['fragment_smiles']}"
        )
    return "\n".join(lines)
