"""Schema-v2 action features. Built from the parent and persisted atom indices."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from rollout_policy.eligibility import EligibleAction
from rollout_policy.graph_features import validate_action_indices

FEATURE_SCHEMA_VERSION = 2
ATTACHMENT_FP_BITS = 128

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

# Locked 13-d vector. Kept so an existing ValueMLP checkpoint can still score.
LEGACY_FEATURE_NAMES: tuple[str, ...] = (
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
    if len(names) != len(kinds):
        raise RuntimeError("feature schema drift")
    return tuple(names), tuple(kinds)


FEATURE_NAMES, FEATURE_KINDS = _schema()


def induced_fragment(parent: Chem.Mol, atom_indices: tuple[int, ...]) -> Chem.Mol:
    """Fragment from parent atoms. No dummy atom from the stored cut SMILES."""
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
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        parent,
        radius=2,
        nBits=ATTACHMENT_FP_BITS,
        fromAtoms=[int(atom_idx)],
        useChirality=False,
    )
    bits = [float(fingerprint.GetBit(i)) for i in range(ATTACHMENT_FP_BITS)]
    if len(bits) != ATTACHMENT_FP_BITS:
        raise RuntimeError("centered Morgan width drifted")
    return bits


def action_features(
    parent: Chem.Mol,
    core: Chem.Mol,
    action: EligibleAction,
) -> np.ndarray:
    """Ordered schema-v2 vector. Names and values are checked against FEATURE_NAMES."""
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


def legacy_action_features(
    parent: Chem.Mol,
    core: Chem.Mol,
    action: EligibleAction,
) -> np.ndarray:
    """The 13 descriptors stored in the locked ValueMLP checkpoint."""
    fragment = induced_fragment(parent, action.fragment_atom_indices)
    parent_heavy = parent.GetNumHeavyAtoms()
    if parent_heavy <= 0:
        raise ValueError("parent has no heavy atoms")
    bond = parent.GetBondBetweenAtoms(
        action.core_attachment_atom, action.fragment_attachment_atom
    )
    bond_name = bond.GetBondType().name if bond is not None else action.bond_type
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
            int(bond_name == "SINGLE"),
            int(bond_name == "DOUBLE"),
            int(bond_name == "AROMATIC"),
            int(action.is_ring),
            int(action.is_terminal),
        ],
        dtype=np.float32,
    )


def features_from_action_record(
    parent: Chem.Mol,
    core: Chem.Mol,
    action: EligibleAction,
) -> dict[str, float]:
    values = action_features(parent, core, action)
    return {name: float(value) for name, value in zip(FEATURE_NAMES, values)}


class StreamingMixedFeatureScaler:
    """Standardize continuous columns. Leave binary and zero-variance columns as stored."""

    def __init__(self, kinds: tuple[str, ...] = FEATURE_KINDS):
        if len(kinds) != len(FEATURE_NAMES):
            raise ValueError("feature kinds do not match FEATURE_NAMES")
        self.kinds = kinds
        self.continuous = np.array([kind == "continuous" for kind in kinds])
        self.n = 0
        self._mean = np.zeros(len(kinds), dtype=np.float64)
        self._m2 = np.zeros(len(kinds), dtype=np.float64)
        self.center = np.zeros(len(kinds), dtype=np.float64)
        self.scale = np.ones(len(kinds), dtype=np.float64)
        self.zero_variance: list[str] = []
        self._finalized = False

    def partial_fit(self, batch: np.ndarray) -> None:
        if self._finalized:
            raise RuntimeError("scaler is already finalized")
        rows = np.asarray(batch, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.shape[1] != len(self.kinds):
            raise ValueError("feature width does not match the scaler")
        batch_n = int(rows.shape[0])
        if batch_n == 0:
            return
        batch_mean = rows.mean(axis=0)
        batch_m2 = ((rows - batch_mean) ** 2).sum(axis=0)
        total = self.n + batch_n
        delta = batch_mean - self._mean
        self._m2 = self._m2 + batch_m2 + (delta**2) * (self.n * batch_n / total)
        self._mean = self._mean + delta * (batch_n / total)
        self.n = total

    def finalize(self) -> None:
        variance = self._m2 / self.n if self.n else np.zeros_like(self._mean)
        deviation = np.sqrt(np.maximum(variance, 0.0))
        self.center = np.zeros(len(self.kinds), dtype=np.float64)
        self.scale = np.ones(len(self.kinds), dtype=np.float64)
        self.zero_variance = []
        for index, is_continuous in enumerate(self.continuous):
            if not is_continuous:
                continue
            if deviation[index] < 1e-8:
                self.zero_variance.append(FEATURE_NAMES[index])
                continue
            self.center[index] = self._mean[index]
            self.scale[index] = deviation[index]
        self._finalized = True

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not self._finalized:
            raise RuntimeError("call finalize() before transform()")
        rows = np.asarray(values, dtype=np.float32)
        center = self.center.astype(np.float32)
        scale = self.scale.astype(np.float32)
        return ((rows - center) / scale).astype(np.float32)

    def state_dict(self) -> dict[str, list]:
        if not self._finalized:
            raise RuntimeError("call finalize() before state_dict()")
        return {
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "zero_variance": list(self.zero_variance),
            "feature_names": list(FEATURE_NAMES),
            "feature_kinds": list(FEATURE_KINDS),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "n": self.n,
        }

    @classmethod
    def from_state(cls, payload: dict) -> "StreamingMixedFeatureScaler":
        if list(payload["feature_names"]) != list(FEATURE_NAMES):
            raise ValueError("checkpoint feature names do not match FEATURE_NAMES")
        if list(payload["feature_kinds"]) != list(FEATURE_KINDS):
            raise ValueError("checkpoint feature kinds do not match FEATURE_KINDS")
        if int(payload["feature_schema_version"]) != FEATURE_SCHEMA_VERSION:
            raise ValueError("checkpoint feature schema version does not match")
        scaler = cls()
        scaler.center = np.asarray(payload["center"], dtype=np.float64)
        scaler.scale = np.asarray(payload["scale"], dtype=np.float64)
        scaler.zero_variance = list(payload.get("zero_variance", []))
        scaler.n = int(payload.get("n", 0))
        scaler._finalized = True
        if scaler.center.shape != (len(FEATURE_NAMES),) or scaler.scale.shape != (len(FEATURE_NAMES),):
            raise ValueError("checkpoint scaler width does not match FEATURE_NAMES")
        return scaler
