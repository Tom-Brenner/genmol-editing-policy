#!/usr/bin/env python3
"""Fetch a single ChEMBL molecule via the public REST API (spot-check only).

Bulk experiment data must still come from the pinned chemreps dump.
This script is for local SAFE/remask smoke tests.

Example:
  python scripts/download_chembl_molecule.py --chembl-id CHEMBL521
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rollout_policy.io import ensure_dir, resolve_path, write_json

API_TMPL = "https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"


def fetch_molecule(chembl_id: str) -> dict:
    url = API_TMPL.format(chembl_id=chembl_id)
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_smiles(payload: dict) -> str:
    structs = payload.get("molecule_structures") or {}
    for key in ("canonical_smiles", "standard_smiles"):
        if structs.get(key):
            return structs[key]
    raise RuntimeError(f"No SMILES in ChEMBL payload for {payload.get('molecule_chembl_id')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    # Ibuprofen — known eligible fixture-like parent with pendant fragments.
    parser.add_argument("--chembl-id", default="CHEMBL521")
    parser.add_argument("--outdir", default="data/fixtures")
    args = parser.parse_args()

    payload = fetch_molecule(args.chembl_id)
    smiles = extract_smiles(payload)
    chembl_id = payload.get("molecule_chembl_id", args.chembl_id)
    outdir = ensure_dir(args.outdir)
    json_path = resolve_path(outdir / f"{chembl_id}.json")
    tsv_path = resolve_path(outdir / f"{chembl_id}.tsv")
    write_json(json_path, payload)
    tsv_path.write_text(f"{chembl_id}\t{smiles}\n", encoding="utf-8")
    print(json.dumps({"chembl_id": chembl_id, "smiles": smiles, "tsv": str(tsv_path)}, indent=2))


if __name__ == "__main__":
    main()
