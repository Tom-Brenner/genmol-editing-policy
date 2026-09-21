#!/usr/bin/env python3
"""Sample generation records for manual core/mapping audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rollout_policy.io import load_jsonl, resolve_path, write_jsonl_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="artifacts/audit_sample.jsonl")
    args = parser.parse_args()

    rows = load_jsonl(resolve_path(args.jsonl))
    if not rows:
        raise SystemExit(f"No records in {args.jsonl}")
    rng = np.random.default_rng(args.seed)
    n = min(args.sample, len(rows))
    idx = rng.choice(len(rows), size=n, replace=False)
    sample = [rows[i] for i in sorted(idx)]
    # Keep fields useful for manual inspection.
    slim = []
    for row in sample:
        slim.append(
            {
                "parent_id": row.get("parent_id"),
                "parent_smiles": row.get("parent_smiles"),
                "action_id": row.get("action_id"),
                "safe_span": row.get("safe_span"),
                "safe_fragment_index": row.get("safe_fragment_index"),
                "output_smiles": row.get("output_smiles"),
                "valid": row.get("valid"),
                "exact_core": row.get("exact_core"),
                "tanimoto": row.get("tanimoto"),
                "heavy_atom_delta": row.get("heavy_atom_delta"),
                "qed_delta_raw": row.get("qed_delta_raw"),
                "reward": row.get("reward"),
                "exception": row.get("exception"),
                "audit_notes": "",
            }
        )
    write_jsonl_atomic(args.output, slim)
    print(json.dumps({"sampled": n, "output": args.output}, indent=2))
    print("Fill audit_notes after manual inspection; require zero core/mapping errors.")


if __name__ == "__main__":
    main()
