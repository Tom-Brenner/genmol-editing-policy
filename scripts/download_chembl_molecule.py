#!/usr/bin/env python3
"""Fetch a single ChEMBL molecule for SAFE/remask smoke tests.

Bulk experiment data must still come from the pinned chemreps dump.

Production ChEMBL REST (`www.ebi.ac.uk`) is often 500/timeout. This script
tries production, then the staging API, then a committed fixture TSV, then
the local chemreps dump. `--offline` skips REST entirely.

The one-molecule smoke test does not need this script: `data/fixtures/CHEMBL521.tsv`
is committed.

Example:
  python scripts/download_chembl_molecule.py --chembl-id CHEMBL521
  python scripts/download_chembl_molecule.py --chembl-id CHEMBL521 --offline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rollout_policy.chembl import lookup_chembl_smiles
from rollout_policy.io import ensure_dir, load_experiment_config, resolve_path, write_json

# Production ChEMBL has intermittent 500s; wwwdev is the same REST schema.
API_URLS = (
    "https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json",
    "https://wwwdev.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json",
)
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "genmol-editing-policy/spot-check",
}
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def fetch_molecule_from_api(chembl_id: str, *, attempts: int = 2, timeout: int = 20) -> dict:
    errors: list[str] = []
    for url_tmpl in API_URLS:
        url = url_tmpl.format(chembl_id=chembl_id)
        for attempt in range(attempts):
            req = Request(url, headers=HEADERS)
            try:
                with urlopen(req, timeout=timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                payload["_source"] = url
                return payload
            except HTTPError as exc:
                errors.append(f"{url} HTTP {exc.code}")
                if exc.code in RETRYABLE_STATUS and attempt + 1 < attempts:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                errors.append(f"{url} {exc}")
                if attempt + 1 < attempts:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break
    raise RuntimeError("ChEMBL REST fetch failed: " + "; ".join(errors))


def _payload(chembl_id: str, smiles: str, source: str) -> dict:
    return {
        "molecule_chembl_id": chembl_id,
        "molecule_structures": {"canonical_smiles": smiles},
        "_source": source,
    }


def fetch_molecule_from_chemreps(chembl_id: str, chemreps_path: str | Path) -> dict:
    resolved = resolve_path(chemreps_path)
    smiles = lookup_chembl_smiles(chembl_id, resolved)
    if not smiles:
        raise RuntimeError(f"{chembl_id} not found in {resolved}")
    return _payload(chembl_id, smiles, str(resolved))


def fetch_molecule_from_fixture(chembl_id: str, tsv_path: str | Path) -> dict:
    path = resolve_path(tsv_path)
    if not path.exists():
        raise RuntimeError(f"Missing fixture {path}")
    line = path.read_text(encoding="utf-8").strip().splitlines()[0]
    parent_id, smiles = line.split("\t", 1)
    parent_id, smiles = parent_id.strip(), smiles.strip()
    if parent_id != chembl_id or not smiles:
        raise RuntimeError(f"{path} does not contain {chembl_id}")
    return _payload(chembl_id, smiles, str(path))


def fetch_molecule(
    chembl_id: str,
    chemreps_path: str | Path | None = None,
    fixture_tsv: str | Path | None = None,
    *,
    offline: bool = False,
) -> dict:
    errors: list[str] = []
    if not offline:
        try:
            return fetch_molecule_from_api(chembl_id)
        except RuntimeError as api_error:
            errors.append(str(api_error))

    if fixture_tsv is not None:
        try:
            payload = fetch_molecule_from_fixture(chembl_id, fixture_tsv)
            print(f"Using committed fixture ({payload['_source']})", file=sys.stderr)
            return payload
        except RuntimeError as fixture_error:
            errors.append(str(fixture_error))

    if chemreps_path is not None and resolve_path(chemreps_path).exists():
        try:
            payload = fetch_molecule_from_chemreps(chembl_id, chemreps_path)
            print(f"Using chemreps dump ({payload['_source']})", file=sys.stderr)
            return payload
        except RuntimeError as dump_error:
            errors.append(str(dump_error))

    detail = "; ".join(errors) if errors else "no REST, chemreps, or fixture source available"
    raise RuntimeError(f"ChEMBL molecule fetch failed: {detail}")


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
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--chemreps",
        default=None,
        help="Chemreps dump used when REST is down or --offline is set",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip ChEMBL REST; use chemreps dump or an existing fixture TSV",
    )
    args = parser.parse_args()

    chemreps = args.chemreps
    if chemreps is None:
        cfg = load_experiment_config(args.config)
        candidate = resolve_path(cfg["paths"]["chembl_raw"])
        if candidate.exists():
            chemreps = candidate

    fixture_tsv = resolve_path(Path(args.outdir) / f"{args.chembl_id}.tsv")
    payload = fetch_molecule(
        args.chembl_id,
        chemreps,
        fixture_tsv if fixture_tsv.exists() else None,
        offline=args.offline,
    )
    smiles = extract_smiles(payload)
    chembl_id = payload.get("molecule_chembl_id", args.chembl_id)
    outdir = ensure_dir(args.outdir)
    json_path = resolve_path(outdir / f"{chembl_id}.json")
    tsv_path = resolve_path(outdir / f"{chembl_id}.tsv")
    write_json(json_path, payload)
    tsv_path.write_text(f"{chembl_id}\t{smiles}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "chembl_id": chembl_id,
                "smiles": smiles,
                "tsv": str(tsv_path),
                "source": payload.get("_source"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
