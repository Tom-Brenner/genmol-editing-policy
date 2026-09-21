"""Reward and success metrics for QED-constrained remasking."""

from __future__ import annotations

from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, QED

from rollout_policy.core_check import ProtectedCore, check_exact_core

# Pre-registered default: abs(Δheavy) ≤ round(fraction × parent heavy atoms).
DEFAULT_HEAVY_ATOM_DELTA_FRACTION = 1.0 / 3.0


def max_heavy_atom_delta_allowed(
    parent_n_heavy: int,
    *,
    fraction: float = DEFAULT_HEAVY_ATOM_DELTA_FRACTION,
) -> int:
    """Per-parent heavy-atom change budget: round(fraction * n_heavy_parent)."""
    if parent_n_heavy < 0:
        raise ValueError("parent_n_heavy must be non-negative")
    return int(round(float(fraction) * int(parent_n_heavy)))


def resolve_max_heavy_atom_delta(parent_n_heavy: int, constraints: dict | None) -> int:
    """Read fraction from config; fall back to legacy absolute key if present alone."""
    constraints = constraints or {}
    if "max_heavy_atom_delta_fraction" in constraints:
        return max_heavy_atom_delta_allowed(
            parent_n_heavy,
            fraction=float(constraints["max_heavy_atom_delta_fraction"]),
        )
    if "max_absolute_heavy_atom_delta" in constraints:
        return int(constraints["max_absolute_heavy_atom_delta"])
    return max_heavy_atom_delta_allowed(parent_n_heavy)


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
    max_heavy_atom_delta: int | None = None,
    reward_qed_delta_cap: float = 0.10,
    fingerprint: dict | None = None,
) -> dict[str, Any]:
    """Compute rollout reward and raw components (always retain raw fields).

    Candidate must already have passed RDKit sanitization.
    Raw tanimoto / QED / heavy-atom delta are stored regardless of feasibility so
    success can be recomputed under alternate thresholds.
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
    parent_n_heavy = int(parent.GetNumHeavyAtoms())
    heavy_atom_delta = int(candidate.GetNumHeavyAtoms() - parent_n_heavy)
    if max_heavy_atom_delta is None:
        max_heavy_atom_delta = max_heavy_atom_delta_allowed(parent_n_heavy)
    parent_qed = float(QED.qed(parent))
    candidate_qed = float(QED.qed(candidate))
    raw_delta = candidate_qed - parent_qed
    feasible = bool(
        exact_core_ok
        and similarity >= min_tanimoto
        and abs(heavy_atom_delta) <= int(max_heavy_atom_delta)
    )
    value = max(0.0, min(raw_delta / reward_qed_delta_cap, 1.0)) if feasible else 0.0
    return {
        "feasible": feasible,
        "exact_core_ok": bool(exact_core_ok),
        "tanimoto": float(similarity),
        "heavy_atom_delta": heavy_atom_delta,
        "parent_n_heavy_atoms": parent_n_heavy,
        "candidate_n_heavy_atoms": int(candidate.GetNumHeavyAtoms()),
        "max_heavy_atom_delta_allowed": int(max_heavy_atom_delta),
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
    """Sanitize, exact-core check, and reward. Invalid → zero reward + raw nulls."""
    constraints = constraints or {}
    reward_cfg = reward_cfg or {}
    parent_n_heavy = int(parent.GetNumHeavyAtoms())
    max_delta = resolve_max_heavy_atom_delta(parent_n_heavy, constraints)
    candidate = sanitize_smiles(candidate_smiles)
    if candidate is None:
        parent_qed = float(QED.qed(parent))
        return {
            "valid": False,
            "feasible": False,
            "exact_core_ok": False,
            "tanimoto": None,
            "heavy_atom_delta": None,
            "parent_n_heavy_atoms": parent_n_heavy,
            "candidate_n_heavy_atoms": None,
            "max_heavy_atom_delta_allowed": max_delta,
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
        max_heavy_atom_delta=max_delta,
        reward_qed_delta_cap=float(reward_cfg.get("reward_qed_delta_cap", 0.10)),
        fingerprint=fingerprint,
    )
    return {
        "valid": True,
        "output_smiles": Chem.MolToSmiles(candidate, canonical=True),
        **scored,
    }


def _exact_core_flag(row: dict[str, Any]) -> bool:
    if "exact_core_ok" in row and row["exact_core_ok"] is not None:
        return bool(row["exact_core_ok"])
    return bool(row.get("exact_core"))


def completion_success(
    row: dict[str, Any],
    *,
    success_qed_delta: float = 0.05,
    min_tanimoto: float = 0.50,
    max_heavy_atom_delta: int | None = None,
    max_heavy_atom_delta_fraction: float | None = DEFAULT_HEAVY_ATOM_DELTA_FRACTION,
    constraints: dict | None = None,
) -> bool:
    """The per-completion event whose OR is ``parent_success``.

    Heavy-atom budget: if ``max_heavy_atom_delta`` is set, use it;
    else per-row ``max_heavy_atom_delta_allowed`` if present; else
    ``round(fraction * parent_n_heavy_atoms)``.
    """
    constraints = constraints or {}
    fraction = (
        float(constraints["max_heavy_atom_delta_fraction"])
        if "max_heavy_atom_delta_fraction" in constraints
        else max_heavy_atom_delta_fraction
    )
    if not row.get("valid") or not _exact_core_flag(row):
        return False
    if row.get("tanimoto") is None or row["tanimoto"] < min_tanimoto:
        return False
    if row.get("heavy_atom_delta") is None or row.get("qed_delta_raw") is None:
        return False
    if max_heavy_atom_delta is not None:
        budget = int(max_heavy_atom_delta)
    elif row.get("max_heavy_atom_delta_allowed") is not None:
        budget = int(row["max_heavy_atom_delta_allowed"])
    else:
        n_heavy = row.get("parent_n_heavy_atoms")
        if n_heavy is None:
            return False
        budget = max_heavy_atom_delta_allowed(
            int(n_heavy),
            fraction=float(fraction or DEFAULT_HEAVY_ATOM_DELTA_FRACTION),
        )
    if abs(row["heavy_atom_delta"]) > budget:
        return False
    if row["qed_delta_raw"] < success_qed_delta:
        return False
    return True


def parent_success(
    rewards_or_flags: list[dict[str, Any]],
    *,
    success_qed_delta: float = 0.05,
    min_tanimoto: float = 0.50,
    max_heavy_atom_delta: int | None = None,
    max_heavy_atom_delta_fraction: float | None = DEFAULT_HEAVY_ATOM_DELTA_FRACTION,
    constraints: dict | None = None,
) -> bool:
    """True if any completion meets ``completion_success``.

    Thresholds are arguments so success can be recomputed from JSONL.
    """
    return any(
        completion_success(
            row,
            success_qed_delta=success_qed_delta,
            min_tanimoto=min_tanimoto,
            max_heavy_atom_delta=max_heavy_atom_delta,
            max_heavy_atom_delta_fraction=max_heavy_atom_delta_fraction,
            constraints=constraints,
        )
        for row in rewards_or_flags
    )
