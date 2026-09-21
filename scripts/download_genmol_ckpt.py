#!/usr/bin/env python3
"""Download pinned GenMol V1 checkpoint and record SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rollout_policy.io import ensure_dir, resolve_path, write_json

DEFAULT_URL = (
    "https://huggingface.co/nvidia/NV-GenMol-89M-v1/resolve/"
    "cdaf0c1b45c281315ab10389836bc46093cdbb65/model.ckpt"
)
DEFAULT_DEST = "vendor/genmol/checkpoints/model.ckpt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--dest", default=DEFAULT_DEST)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dest = resolve_path(args.dest)
    ensure_dir(dest.parent)
    if dest.exists() and not args.force:
        digest = sha256_file(dest)
        print(f"Already present: {dest} ({digest})")
    else:
        print(f"Downloading {args.url}")
        print("Ensure you have accepted the NVIDIA Open Model License on Hugging Face.")
        tmp = dest.with_suffix(dest.suffix + ".partial")
        urlretrieve(args.url, tmp)
        tmp.replace(dest)
        digest = sha256_file(dest)
        print(f"Saved {dest} ({digest})")

    write_json(
        "artifacts/genmol-v1.sha256.json",
        {
            "path": str(dest),
            "sha256": digest,
            "url": args.url,
            "revision": "cdaf0c1b45c281315ab10389836bc46093cdbb65",
        },
    )
    (resolve_path("artifacts") / "genmol-v1.sha256").write_text(f"{digest}  {dest}\n")


if __name__ == "__main__":
    main()
