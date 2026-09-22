#!/usr/bin/env python3
"""Re-check manifest scaffolds and replay exact-core flags on stored completions.

Does not call GenMol, does not rewrite JSONL, and does not read test generations.
Phase A covers every manifest parent. Phase B streams the existing train and
validation completion files. Pass --max-records to time a prefix; that run is
not an acceptance audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import platform
import sys
import time
import zlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rollout_policy.chembl import assign_atom_maps, murcko_core_smiles  # noqa: E402
from rollout_policy.core_check import ProtectedCore, extract_protected_core  # noqa: E402
from rollout_policy.core_check import check_exact_core  # noqa: E402
from rollout_policy.dry_run import action_from_dict  # noqa: E402
from rollout_policy.graph_features import validate_action_indices  # noqa: E402
from rollout_policy.io import (  # noqa: E402
    artifact_path,
    iter_jsonl,
    load_experiment_config,
    resolve_path,
    write_json,
    write_jsonl_atomic,
)
from rollout_policy.scoring import sanitize_smiles  # noqa: E402

EXPECTED_RECORDS = {"train": 584_944, "validation": 98_504}
TOTAL_RECORDS = 683_448
FAILURE_SAMPLE_LIMIT = 500
MISMATCH_SAMPLE_LIMIT = 200

_STATE: dict = {}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class ParentAuditContext:
    split: str
    smiles: str
    core_id: str
    parent_mapped: Chem.Mol
    protected_core: ProtectedCore


def _sample(failures: list[dict], item: dict) -> None:
    if len(failures) < FAILURE_SAMPLE_LIMIT:
        failures.append(item)


def audit_manifest_split(path: Path, split: str) -> dict:
    core_ids: set[str] = set()
    parent_ids: set[str] = set()
    failures: list[dict] = []
    failure_count = 0
    counts: Counter = Counter()
    contexts: dict[str, ParentAuditContext] = {}

    for row in iter_jsonl(path):
        counts["parents"] += 1
        parent_id = str(row["parent_id"])
        if parent_id in parent_ids:
            failure_count += 1
            _sample(failures, {"split": split, "parent_id": parent_id, "reason": "duplicate_parent"})
            continue
        parent_ids.add(parent_id)
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            failure_count += 1
            _sample(failures, {"split": split, "parent_id": parent_id, "reason": "invalid_parent"})
            continue
        recomputed_core_id = murcko_core_smiles(mol)
        if recomputed_core_id != row["core_id"]:
            failure_count += 1
            _sample(
                failures,
                {
                    "split": split,
                    "parent_id": parent_id,
                    "reason": "core_id_mismatch",
                    "stored": row["core_id"],
                    "recomputed": recomputed_core_id,
                },
            )
        if recomputed_core_id is not None:
            core_ids.add(recomputed_core_id)
        parent_mapped = assign_atom_maps(mol)
        protected_core = extract_protected_core(parent_mapped)
        if protected_core is None:
            failure_count += 1
            _sample(
                failures,
                {"split": split, "parent_id": parent_id, "reason": "protected_core_not_unique"},
            )
            continue
        counts["unique_protected_core"] += 1
        contexts[parent_id] = ParentAuditContext(
            split=split,
            smiles=row["smiles"],
            core_id=row["core_id"],
            parent_mapped=parent_mapped,
            protected_core=protected_core,
        )
        for action_row in row["actions"]:
            try:
                validate_action_indices(mol, action_from_dict(action_row))
                counts["valid_actions"] += 1
            except Exception as exc:
                failure_count += 1
                _sample(
                    failures,
                    {
                        "split": split,
                        "parent_id": parent_id,
                        "action_id": action_row.get("action_id"),
                        "reason": f"invalid_action:{type(exc).__name__}",
                    },
                )
    return {
        "split": split,
        "core_ids": core_ids,
        "parent_ids": parent_ids,
        "counts": counts,
        "failures": failures,
        "failure_count": failure_count,
        "contexts": contexts,
    }


def _overlap(left: set[str], right: set[str], limit: int = 100) -> list[str]:
    shared = left & right
    return sorted(shared)[:limit]


def recompute_record_exact_core(record: dict, context: ParentAuditContext) -> bool:
    if not record.get("valid") or not record.get("output_smiles"):
        return False
    candidate = sanitize_smiles(record["output_smiles"])
    if candidate is None:
        return False
    return bool(check_exact_core(context.parent_mapped, candidate, context.protected_core))


def _audit_records(path: Path, split: str, parents: dict[str, ParentAuditContext], all_ids: set[str], owner, max_records: int | None) -> dict:
    counts: Counter = Counter()
    mismatch_samples: list[dict] = []
    seen_keys: set[tuple] = set()
    examined = 0
    for record in iter_jsonl(path):
        if max_records is not None and examined >= max_records:
            break
        parent_id = str(record["parent_id"])
        if owner is not None and owner(parent_id) != 0 and parent_id in all_ids:
            continue
        if owner is not None and parent_id not in all_ids and owner(parent_id) != 0:
            continue
        examined += 1
        counts["records"] += 1
        context = parents.get(parent_id)
        if context is None:
            counts["unknown_parent"] += 1
            continue
        if record.get("split") != split:
            counts["wrong_split"] += 1
        if record.get("parent_smiles") != context.smiles:
            counts["parent_smiles_mismatch"] += 1
        if record.get("core_id") != context.core_id:
            counts["record_core_id_mismatch"] += 1
        key = (
            record.get("phase"),
            parent_id,
            record.get("action_id"),
            int(record["candidate_index"]),
            int(record["requested_seed"]),
        )
        if key in seen_keys:
            counts["duplicate_record_key"] += 1
        else:
            seen_keys.add(key)
        recomputed = recompute_record_exact_core(record, context)
        stored = bool(record.get("exact_core"))
        counts["recomputed_exact_core_true"] += int(recomputed)
        counts["stored_exact_core_true"] += int(stored)
        if recomputed != stored:
            counts["exact_core_mismatch"] += 1
            if len(mismatch_samples) < MISMATCH_SAMPLE_LIMIT:
                mismatch_samples.append(
                    {
                        "split": split,
                        "parent_id": parent_id,
                        "core_id": context.core_id,
                        "action_id": record.get("action_id"),
                        "candidate_index": record.get("candidate_index"),
                        "requested_seed": record.get("requested_seed"),
                        "stored_exact_core": stored,
                        "recomputed_exact_core": recomputed,
                        "output_smiles": record.get("output_smiles"),
                    }
                )
        if counts["records"] % 10_000 == 0:
            print(
                f"{split}: audited={counts['records']} mismatches={counts['exact_core_mismatch']}",
                flush=True,
            )
    return {"counts": counts, "mismatch_samples": mismatch_samples}


def _worker(payload: dict) -> dict:
    path = Path(payload["path"])
    split = payload["split"]
    worker_id = int(payload["worker_id"])
    n_workers = int(payload["n_workers"])
    parents = _STATE["parents"]
    all_ids = _STATE["all_ids"]

    def owner(parent_id: str) -> int:
        return zlib.adler32(parent_id.encode("utf-8")) % n_workers

    counts: Counter = Counter()
    mismatch_samples: list[dict] = []
    seen_keys: set[tuple] = set()
    for record in iter_jsonl(path):
        parent_id = str(record["parent_id"])
        if owner(parent_id) != worker_id:
            continue
        counts["records"] += 1
        context = parents.get(parent_id)
        if context is None:
            counts["unknown_parent"] += 1
            continue
        if record.get("split") != split:
            counts["wrong_split"] += 1
        if record.get("parent_smiles") != context.smiles:
            counts["parent_smiles_mismatch"] += 1
        if record.get("core_id") != context.core_id:
            counts["record_core_id_mismatch"] += 1
        key = (
            record.get("phase"),
            parent_id,
            record.get("action_id"),
            int(record["candidate_index"]),
            int(record["requested_seed"]),
        )
        if key in seen_keys:
            counts["duplicate_record_key"] += 1
        else:
            seen_keys.add(key)
        recomputed = recompute_record_exact_core(record, context)
        stored = bool(record.get("exact_core"))
        counts["recomputed_exact_core_true"] += int(recomputed)
        counts["stored_exact_core_true"] += int(stored)
        if recomputed != stored and len(mismatch_samples) < MISMATCH_SAMPLE_LIMIT:
            mismatch_samples.append(
                {
                    "split": split,
                    "parent_id": parent_id,
                    "core_id": context.core_id,
                    "action_id": record.get("action_id"),
                    "candidate_index": record.get("candidate_index"),
                    "requested_seed": record.get("requested_seed"),
                    "stored_exact_core": stored,
                    "recomputed_exact_core": recomputed,
                    "output_smiles": record.get("output_smiles"),
                }
            )
        if counts["records"] % 10_000 == 0:
            print(
                f"{split} worker={worker_id}: audited={counts['records']} "
                f"mismatches={counts['exact_core_mismatch']}",
                flush=True,
            )
    return {"counts": dict(counts), "mismatch_samples": mismatch_samples}


def audit_record_file(path: Path, *, split: str, parents: dict[str, ParentAuditContext], expected_records: int, max_records: int | None, workers: int) -> dict:
    all_ids = set(parents)
    if max_records is not None or workers <= 1:
        report = _audit_records(path, split, parents, all_ids, owner=None, max_records=max_records)
        counts = report["counts"]
        samples = report["mismatch_samples"]
    else:
        global _STATE
        _STATE = {"parents": parents, "all_ids": all_ids}
        ctx = multiprocessing.get_context("fork")
        payloads = [
            {"path": str(path), "split": split, "worker_id": worker_id, "n_workers": workers}
            for worker_id in range(workers)
        ]
        with ctx.Pool(workers) as pool:
            parts = pool.map(_worker, payloads)
        counts = Counter()
        samples = []
        for part in parts:
            counts.update(part["counts"])
            for sample in part["mismatch_samples"]:
                if len(samples) < MISMATCH_SAMPLE_LIMIT:
                    samples.append(sample)
    count_error = None
    if max_records is None and counts["records"] != expected_records:
        count_error = (
            f"{split}: found {counts['records']} records, expected {expected_records}"
        )
        print(count_error, flush=True)
    return {
        "counts": dict(counts),
        "mismatch_samples": samples,
        "count_error": count_error,
    }


def _public_split(report: dict) -> dict:
    return {
        "split": report["split"],
        "n_parents": len(report["parent_ids"]),
        "n_core_ids": len(report["core_ids"]),
        "counts": dict(report["counts"]),
        "failure_count": report["failure_count"],
        "failure_samples": report["failures"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_v2.yaml")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    cfg = load_experiment_config(args.config)
    manifests = resolve_path(cfg["paths"]["manifests_dir"])
    results = resolve_path(cfg["paths"]["results_dir"])
    expected_parents = {
        "train": int(cfg["split"]["n_train"]),
        "validation": int(cfg["split"]["n_validation"]),
        "test": int(cfg["split"]["n_test"]),
    }
    manifest_paths = {split: manifests / f"{split}.jsonl" for split in expected_parents}
    for split, path in manifest_paths.items():
        if not path.is_file():
            raise SystemExit(f"Missing manifest {path}")

    print("Phase A: manifest scaffolds", flush=True)
    reports = {split: audit_manifest_split(path, split) for split, path in manifest_paths.items()}
    overlaps = {
        "train_validation_cores": _overlap(reports["train"]["core_ids"], reports["validation"]["core_ids"]),
        "train_test_cores": _overlap(reports["train"]["core_ids"], reports["test"]["core_ids"]),
        "validation_test_cores": _overlap(reports["validation"]["core_ids"], reports["test"]["core_ids"]),
        "train_validation_parents": _overlap(reports["train"]["parent_ids"], reports["validation"]["parent_ids"]),
        "train_test_parents": _overlap(reports["train"]["parent_ids"], reports["test"]["parent_ids"]),
        "validation_test_parents": _overlap(reports["validation"]["parent_ids"], reports["test"]["parent_ids"]),
    }
    parent_shortfalls = {
        split: {"expected": expected_parents[split], "observed": len(reports[split]["parent_ids"])}
        for split in expected_parents
        if len(reports[split]["parent_ids"]) != expected_parents[split]
    }
    phase_a_ok = (
        not parent_shortfalls
        and all(item["failure_count"] == 0 for item in reports.values())
        and all(len(ids) == 0 for ids in overlaps.values())
        and all(
            reports[split]["counts"]["unique_protected_core"] == expected_parents[split]
            for split in expected_parents
        )
    )
    manifest_report = {
        "started_at": started_at,
        "command": sys.argv,
        "python": platform.python_version(),
        "rdkit": Chem.rdBase.rdkitVersion,
        "manifest_sha256": {split: file_sha256(path) for split, path in manifest_paths.items()},
        "expected_parents": expected_parents,
        "splits": {split: _public_split(reports[split]) for split in reports},
        "overlaps": overlaps,
        "parent_count_mismatches": parent_shortfalls,
        "passed": phase_a_ok,
    }
    write_json(artifact_path(cfg, "manifest_core_audit.json"), manifest_report)
    reports["test"].pop("contexts", None)
    print(json.dumps({"phase_a_passed": phase_a_ok, "overlaps": {k: len(v) for k, v in overlaps.items()}}, indent=2))

    label_paths = {
        split: results / "labels" / f"{split}.jsonl" for split in ("train", "validation")
    }
    for split, path in label_paths.items():
        if not path.is_file():
            raise SystemExit(f"Missing completion file {path}")

    print("Phase B: replay exact-core flags", flush=True)
    phase_b = {}
    benchmark_seconds = None
    for split, path in label_paths.items():
        tick = time.perf_counter()
        phase_b[split] = audit_record_file(
            path,
            split=split,
            parents=reports[split]["contexts"],
            expected_records=EXPECTED_RECORDS[split],
            max_records=args.max_records,
            workers=1 if args.max_records is not None else args.workers,
        )
        elapsed = time.perf_counter() - tick
        if args.max_records:
            benchmark_seconds = (benchmark_seconds or 0.0) + elapsed
            observed = phase_b[split]["counts"]["records"]
            estimate = elapsed * (EXPECTED_RECORDS[split] / max(observed, 1))
            print(
                f"{split}: prefix {observed} in {elapsed:.1f}s; "
                f"estimated full {EXPECTED_RECORDS[split]} in {estimate:.1f}s",
                flush=True,
            )

    observed_total = sum(item["counts"]["records"] for item in phase_b.values())
    production = args.max_records is None
    count_ok = production and observed_total == TOTAL_RECORDS and all(
        item["count_error"] is None for item in phase_b.values()
    )
    mismatch_ok = all(item["counts"].get("exact_core_mismatch", 0) == 0 for item in phase_b.values())
    duplicate_ok = all(item["counts"].get("duplicate_record_key", 0) == 0 for item in phase_b.values())
    identity_ok = all(
        item["counts"].get("unknown_parent", 0) == 0
        and item["counts"].get("wrong_split", 0) == 0
        and item["counts"].get("parent_smiles_mismatch", 0) == 0
        and item["counts"].get("record_core_id_mismatch", 0) == 0
        for item in phase_b.values()
    )
    phase_b_ok = (not production) or (count_ok and mismatch_ok and duplicate_ok and identity_ok)
    samples = []
    for item in phase_b.values():
        for sample in item["mismatch_samples"]:
            if len(samples) < MISMATCH_SAMPLE_LIMIT:
                samples.append(sample)
    replay_report = {
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "command": sys.argv,
        "python": platform.python_version(),
        "rdkit": Chem.rdBase.rdkitVersion,
        "production": production,
        "max_records": args.max_records,
        "workers": args.workers,
        "expected_records": EXPECTED_RECORDS,
        "expected_total": TOTAL_RECORDS,
        "observed_total": observed_total,
        "label_sha256": {split: file_sha256(path) for split, path in label_paths.items()},
        "manifest_sha256": manifest_report["manifest_sha256"],
        "splits": {
            split: {"counts": item["counts"], "n_mismatch_samples": len(item["mismatch_samples"])}
            for split, item in phase_b.items()
        },
        "benchmark_seconds": benchmark_seconds,
        "passed": bool(phase_a_ok and phase_b_ok and production),
        "phase_a_passed": phase_a_ok,
        "phase_b_passed": bool(phase_b_ok and production),
    }
    write_json(artifact_path(cfg, "exact_core_replay_audit.json"), replay_report)
    write_jsonl_atomic(artifact_path(cfg, "exact_core_mismatch_samples.jsonl"), samples)
    print(json.dumps({"passed": replay_report["passed"], "observed_total": observed_total}, indent=2))
    if not replay_report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
