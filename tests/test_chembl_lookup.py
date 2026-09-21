"""Tests for chemreps ID lookup used as a REST fallback."""

from __future__ import annotations

from pathlib import Path

from rollout_policy.chembl import lookup_chembl_smiles


def test_lookup_chembl_smiles_from_tsv(tmp_path: Path):
    path = tmp_path / "chemreps.tsv"
    path.write_text(
        "chembl_id\tcanonical_smiles\n"
        "CHEMBL1\tCCO\n"
        "CHEMBL521\tCC(C)Cc1ccc(C(C)C(=O)O)cc1\n",
        encoding="utf-8",
    )
    assert lookup_chembl_smiles("CHEMBL521", path) == "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
    assert lookup_chembl_smiles("CHEMBL999", path) is None
