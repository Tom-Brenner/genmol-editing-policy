#!/usr/bin/env python3
"""Download and verify pinned ChEMBL 37 chemical representations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rollout_policy.chembl import CHEMBL_37_SHA256, CHEMBL_37_URL, download_chembl_chemreps
from rollout_policy.io import load_experiment_config, resolve_path, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    dest = resolve_path(cfg["paths"]["chembl_raw"])
    path = download_chembl_chemreps(
        destination=dest,
        url=cfg.get("chembl_chemreps_url", CHEMBL_37_URL),
        expected_sha256=cfg.get("chembl_chemreps_sha256", CHEMBL_37_SHA256),
    )
    meta = {
        "path": str(path),
        "sha256": cfg.get("chembl_chemreps_sha256", CHEMBL_37_SHA256),
        "chembl_release": cfg.get("chembl_release", 37),
        "license": "CC BY-SA 3.0",
        "attribution": "ChEMBL https://www.ebi.ac.uk/chembl/",
    }
    write_json("artifacts/chembl_download.json", meta)
    print(f"Verified {path}")


if __name__ == "__main__":
    main()
