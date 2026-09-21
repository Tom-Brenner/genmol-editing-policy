"""Atomic JSONL I/O, config loading, and resume helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (root or PROJECT_ROOT) / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_experiment_config(path: str | Path = "configs/experiment.yaml") -> dict[str, Any]:
    return load_yaml(path)


def load_sampling_config(path: str | Path | None = None, experiment: dict | None = None) -> dict[str, Any]:
    if path is None:
        experiment = experiment or load_experiment_config()
        path = experiment["paths"]["genmol_sampling"]
    return load_yaml(path)


def artifact_path(cfg: dict[str, Any], filename: str) -> Path:
    """Experiment report path under the configured artifacts directory."""
    return resolve_path(cfg["paths"]["artifacts_dir"]) / filename


def ensure_dir(path: str | Path) -> Path:
    resolved = resolve_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(path: str | Path, payload: Any) -> None:
    resolved = resolve_path(path)
    ensure_dir(resolved.parent)
    tmp = resolved.with_suffix(resolved.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(resolved)


def read_json(path: str | Path) -> Any:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one complete JSON line and fsync."""
    resolved = resolve_path(path)
    ensure_dir(resolved.parent)
    line = json.dumps(record, sort_keys=True, ensure_ascii=False)
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_atomic(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Write a full JSONL file via temp + rename."""
    resolved = resolve_path(path)
    ensure_dir(resolved.parent)
    tmp = resolved.with_suffix(resolved.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(resolved)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    resolved = resolve_path(path)
    if not resolved.exists():
        return
        yield  # pragma: no cover
    with resolved.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {resolved}:{line_no}") from exc


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def record_key(record: dict[str, Any]) -> tuple:
    """Resume key: (phase, parent_id, action_id, candidate_index, seed)."""
    return (
        record["phase"],
        record["parent_id"],
        record["action_id"],
        int(record["candidate_index"]),
        int(record["requested_seed"]),
    )


def existing_record_keys(path: str | Path) -> set[tuple]:
    return {record_key(r) for r in iter_jsonl(path)}


def make_generation_record(
    *,
    phase: str,
    split: str,
    parent_id: str,
    parent_smiles: str,
    parent_atom_mapped_smiles: str,
    core_id: str,
    action_id: str,
    safe_span: tuple[int, int] | list[int],
    candidate_index: int,
    requested_seed: int,
    observed_seed: int | None,
    output_safe: str | None,
    output_smiles: str | None,
    valid: bool,
    exact_core: bool | None,
    tanimoto: float | None,
    heavy_atom_delta: int | None,
    parent_qed: float,
    candidate_qed: float | None,
    qed_delta_raw: float | None,
    reward: float,
    wall_seconds: float,
    exception: str | None,
    extra: dict[str, Any] | None = None,
    feasible: bool | None = None,
    parent_n_heavy_atoms: int | None = None,
    candidate_n_heavy_atoms: int | None = None,
    max_heavy_atom_delta_allowed: int | None = None,
) -> dict[str, Any]:
    record = {
        "phase": phase,
        "split": split,
        "parent_id": parent_id,
        "parent_smiles": parent_smiles,
        "parent_atom_mapped_smiles": parent_atom_mapped_smiles,
        "core_id": core_id,
        "action_id": action_id,
        "safe_span": list(safe_span),
        "candidate_index": int(candidate_index),
        "requested_seed": int(requested_seed),
        "observed_seed": observed_seed,
        "output_safe": output_safe,
        "output_smiles": output_smiles,
        "valid": bool(valid),
        "exact_core": exact_core,
        "feasible": feasible,
        "tanimoto": tanimoto,
        "heavy_atom_delta": heavy_atom_delta,
        "parent_n_heavy_atoms": parent_n_heavy_atoms,
        "candidate_n_heavy_atoms": candidate_n_heavy_atoms,
        "max_heavy_atom_delta_allowed": max_heavy_atom_delta_allowed,
        "parent_qed": parent_qed,
        "candidate_qed": candidate_qed,
        "qed_delta_raw": qed_delta_raw,
        "reward": float(reward),
        "wall_seconds": float(wall_seconds),
        "exception": exception,
    }
    if extra:
        record.update(extra)
    return record
