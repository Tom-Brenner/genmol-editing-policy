"""Reward and success metrics for QED-constrained remasking."""

from __future__ import annotations

from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, QED

from rollout_policy.core_check import ProtectedCore, check_exact_core


def morgan_fp(
    mol: Chem.Mol,
    *,
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = False,
):
    return AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits, useChirality=use_chirality
    )


def sanitize_smiles(smiles: str | None) -> Chem.Mol | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def reward(
    parent: Chem.Mol,
    candidate: Chem.Mol,
    exact_core_ok: bool,
    *,
    min_tanimoto: float = 0.50,
    max_heavy_atom_delta: int = 8,
    reward_qed_delta_cap: float = 0.10,
    fingerprint: dict | None = None,
) -> dict[str, Any]:
    """Compute pre-registered rollout reward and raw components.

    Candidate must already have passed RDKit sanitization.
    """
    fp_cfg = fingerprint or {}
    similarity = DataStructs.TanimotoSimilarity(
        morgan_fp(
            parent,
            radius=int(fp_cfg.get("radius", 2)),
            n_bits=int(fp_cfg.get("n_bits", 2048)),
            use_chirality=bool(fp_cfg.get("use_chirality", False)),
        ),
        morgan_fp(
            candidate,
            radius=int(fp_cfg.get("radius", 2)),
            n_bits=int(fp_cfg.get("n_bits", 2048)),
            use_chirality=bool(fp_cfg.get("use_chirality", False)),
        ),
    )
    heavy_atom_delta = candidate.GetNumHeavyAtoms() - parent.GetNumHeavyAtoms()
    parent_qed = float(QED.qed(parent))
    candidate_qed = float(QED.qed(candidate))
    raw_delta = candidate_qed - parent_qed
    feasible = bool(
        exact_core_ok
        and similarity >= min_tanimoto
        and abs(heavy_atom_delta) <= max_heavy_atom_delta
    )
    value = max(0.0, min(raw_delta / reward_qed_delta_cap, 1.0)) if feasible else 0.0
    return {
        "feasible": feasible,
        "exact_core_ok": bool(exact_core_ok),
        "tanimoto": float(similarity),
        "heavy_atom_delta": int(heavy_atom_delta),
        "parent_qed": parent_qed,
        "candidate_qed": candidate_qed,
        "qed_delta_raw": float(raw_delta),
        "reward": float(value),
    }


def score_candidate(
    parent: Chem.Mol,
    parent_mapped: Chem.Mol,
    candidate_smiles: str | None,
    *,
    core: ProtectedCore | None = None,
    constraints: dict | None = None,
    reward_cfg: dict | None = None,
    fingerprint: dict | None = None,
) -> dict[str, Any]:
    """Sanitize, exact-core check, and reward. Invalid → zero reward."""
    constraints = constraints or {}
    reward_cfg = reward_cfg or {}
    candidate = sanitize_smiles(candidate_smiles)
    if candidate is None:
        parent_qed = float(QED.qed(parent))
        return {
            "valid": False,
            "feasible": False,
            "exact_core_ok": False,
            "tanimoto": None,
            "heavy_atom_delta": None,
            "parent_qed": parent_qed,
            "candidate_qed": None,
            "qed_delta_raw": None,
            "reward": 0.0,
            "output_smiles": None,
        }

    exact_ok = check_exact_core(parent_mapped, candidate, core)
    scored = reward(
        parent,
        candidate,
        exact_ok,
        min_tanimoto=float(constraints.get("min_tanimoto_to_parent", 0.50)),
        max_heavy_atom_delta=int(constraints.get("max_absolute_heavy_atom_delta", 8)),
        reward_qed_delta_cap=float(reward_cfg.get("reward_qed_delta_cap", 0.10)),
        fingerprint=fingerprint,
    )
    return {
        "valid": True,
        "output_smiles": Chem.MolToSmiles(candidate, canonical=True),
        **scored,
    }


def parent_success(
    rewards_or_flags: list[dict[str, Any]],
    *,
    success_qed_delta: float = 0.05,
    min_tanimoto: float = 0.50,
    max_heavy_atom_delta: int = 8,
) -> bool:
    """True if any of 16 samples meets the pre-registered success criterion."""
    for row in rewards_or_flags:
        if not row.get("valid"):
            continue
        if not row.get("exact_core_ok"):
            continue
        if row.get("tanimoto") is None or row["tanimoto"] < min_tanimoto:
            continue
        if row.get("heavy_atom_delta") is None:
            continue
        if abs(row["heavy_atom_delta"]) > max_heavy_atom_delta:
            continue
        if row.get("qed_delta_raw") is None:
            continue
        if row["qed_delta_raw"] < success_qed_delta:
            continue
        return True
    return False
