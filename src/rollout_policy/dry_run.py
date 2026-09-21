"""Local dry-run helpers: mock GenMol + fixture parents (no checkpoint/ChEMBL)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from rdkit import Chem

from rollout_policy.chembl import standardize_parent
from rollout_policy.eligibility import EligibleAction, enumerate_eligible_actions
from rollout_policy.genmol_adapter import Completion, set_global_seeds
from rollout_policy.io import PROJECT_ROOT, resolve_path
from rollout_policy.precision import parse_bool_flag

FIXTURE_PARENTS = PROJECT_ROOT / "data" / "fixtures" / "dry_run_parents.tsv"


def parse_bf16_flag(value: str | bool | None) -> bool:
    """Parse --bf16. True means prefer bf16, with automatic fp16 fallback on GPU."""
    return parse_bool_flag(value, default=False)


def add_dry_run_args(parser) -> None:
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Local pipeline check: fixtures + mock GenMol; no checkpoint/ChEMBL required",
    )
    parser.add_argument(
        "--bf16",
        default="false",
        help="Prefer bfloat16 when available. If CUDA lacks bf16 (e.g. GTX 1080 Ti), "
        "automatically uses fp16. false → float32. Dry-run forces false.",
    )


def load_fixture_parents(
    path: str | Path | None = None,
    *,
    n: int | None = None,
    require_safe_alignment: bool = False,
    min_actions: int = 2,
    max_actions: int = 4,
) -> list[dict[str, Any]]:
    resolved = resolve_path(path or FIXTURE_PARENTS)
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parent_id, smiles = line.split("\t", 1)
            parent = standardize_parent(parent_id, smiles)
            if parent is None:
                continue
            mol = Chem.MolFromSmiles(parent.smiles)
            if mol is None:
                continue
            actions = enumerate_eligible_actions(
                mol,
                min_actions=min_actions,
                max_actions=max_actions,
                require_safe_alignment=require_safe_alignment,
            )
            if not actions:
                continue
            rows.append(
                {
                    "parent_id": parent.chembl_id,
                    "source_smiles": parent.source_smiles,
                    "smiles": parent.smiles,
                    "atom_mapped_smiles": parent.atom_mapped_smiles,
                    "core_id": parent.core_smiles,
                    "heavy_atoms": parent.heavy_atoms,
                    "qed": parent.qed,
                    "n_actions": len(actions),
                    "actions": [a.to_dict() for a in actions],
                    "action_order_keys": [list(a.canonical_order_key) for a in actions],
                    "split": "dry_run",
                }
            )
            if n is not None and len(rows) >= n:
                break
    return rows


class MockFrozenGenMol:
    """Deterministic stand-in for FrozenGenMol during --dry-run.

    Does not load a checkpoint. Returns parent / near-parent / invalid outputs
    so scoring, core-check, and JSONL paths are exercised under float32 CPU.
    """

    def __init__(self, *, bf16: bool = False, fail_rate: float = 0.15):
        if bf16:
            raise ValueError("MockFrozenGenMol dry-run requires --bf16 false")
        self.bf16 = False
        self.fail_rate = fail_rate
        self.device = "cpu"
        self.checkpoint = "mock://dry-run"
        self.sampling_kwargs: dict[str, Any] = {"mock": True}
        self.num_insert_mask = 10

    @classmethod
    def from_configs(cls, experiment=None, sampling_config=None, **kwargs):
        return cls(**kwargs)

    def complete_span(
        self,
        smiles: str,
        safe_fragment_index: int,
        seed: int,
        *,
        input_ids=None,
        cut_bonds=None,
    ) -> Completion:
        return self.complete_spans(
            smiles, safe_fragment_index, [seed], input_ids=input_ids, cut_bonds=cut_bonds
        )[0]

    def complete_spans(
        self,
        smiles: str,
        safe_fragment_index: int,
        seeds,
        *,
        input_ids=None,
        cut_bonds=None,
    ) -> list[Completion]:
        return [
            self._complete_one(
                smiles, safe_fragment_index, int(seed), input_ids=input_ids, cut_bonds=cut_bonds
            )
            for seed in seeds
        ]

    def _complete_one(
        self,
        smiles: str,
        safe_fragment_index: int,
        seed: int,
        *,
        input_ids=None,
        cut_bonds=None,
    ) -> Completion:
        set_global_seeds(seed)
        rng = random.Random(seed)
        if rng.random() < self.fail_rate:
            return Completion(
                output_safe=None,
                output_smiles=None,
                error="MockEmptyGeneration",
                observed_seed=seed,
            )
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return Completion(None, None, "MockInvalidParent", seed)

        mode = rng.randrange(3)
        if mode == 0:
            # Exact parent (feasible, zero QED delta).
            out = Chem.MolToSmiles(mol, canonical=True)
        elif mode == 1:
            # Mild methyl add on a non-ring atom when possible.
            out = _maybe_add_methyl(mol, rng) or Chem.MolToSmiles(mol, canonical=True)
        else:
            # Unrelated tiny molecule → fails core check.
            out = "CCO"

        return Completion(
            output_safe=f"mock_span_{safe_fragment_index}",
            output_smiles=out,
            error=None,
            observed_seed=seed,
        )

    def complete(self, masked_safe: str, seed: int) -> Completion:
        return self.complete_span(masked_safe, 0, seed)


def _maybe_add_methyl(mol: Chem.Mol, rng: random.Random) -> str | None:
    candidates = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6 and (not a.IsInRing()) and a.GetTotalNumHs() > 0
    ]
    if not candidates:
        return Chem.MolToSmiles(mol, canonical=True)
    rw = Chem.RWMol(mol)
    idx = rng.choice(candidates)
    new = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(idx, new, Chem.BondType.SINGLE)
    try:
        Chem.SanitizeMol(rw)
        return Chem.MolToSmiles(rw.GetMol(), canonical=True)
    except Exception:
        return Chem.MolToSmiles(mol, canonical=True)


def build_generator(*, dry_run: bool, bf16: bool, experiment=None, sampling_config=None):
    """Factory used by scripts: mock on dry-run, else real FrozenGenMol."""
    if dry_run:
        if bf16:
            raise SystemExit("Refusing --dry-run with --bf16 true; use --bf16 false")
        return MockFrozenGenMol(bf16=False)
    from rollout_policy.genmol_adapter import FrozenGenMol

    return FrozenGenMol.from_configs(
        experiment=experiment,
        sampling_config=sampling_config,
        bf16=bf16,
    )


def action_from_dict(d: dict) -> EligibleAction:
    bonds = d.get("slicer_cut_bonds") or ()
    return EligibleAction(
        action_id=d["action_id"],
        fragment_atom_indices=tuple(d["fragment_atom_indices"]),
        core_attachment_atom=int(d["core_attachment_atom"]),
        fragment_attachment_atom=int(d["fragment_attachment_atom"]),
        bond_type=d["bond_type"],
        canonical_order_key=tuple(d["canonical_order_key"]),
        safe_span=tuple(d["safe_span"]),
        safe_fragment_index=int(d["safe_fragment_index"]),
        fragment_smiles=d["fragment_smiles"],
        fragment_heavy_atoms=int(d["fragment_heavy_atoms"]),
        is_ring=bool(d["is_ring"]),
        is_terminal=bool(d["is_terminal"]),
        slicer_cut_bonds=tuple(tuple(b) for b in bonds),
    )
