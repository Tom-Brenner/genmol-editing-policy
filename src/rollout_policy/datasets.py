"""Sharded action-graph cache. Training loads one shard at a time, not every graph."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data

from rollout_policy.dry_run import action_from_dict
from rollout_policy.eligibility import EligibleAction, core_mol_from_parent
from rollout_policy.features import (
    FEATURE_KINDS,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    StreamingMixedFeatureScaler,
    action_features,
)
from rollout_policy.graph_features import (
    EDGE_FEATURE_NAMES,
    GRAPH_SCHEMA_VERSION,
    NODE_FEATURE_NAMES,
    action_to_graph,
)
from rollout_policy.io import iter_jsonl, resolve_path

SHARD_SIZE = 2000
CACHE_DIRNAME = "graph_cache/schema_v2"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ActionExample:
    parent_id: str
    action_row: dict[str, Any]
    successes: float
    trials: float
    canonical_action_rank: int
    fragment_heavy_atoms: int


def canonical_ranks(action_rows: list[dict[str, Any]]) -> dict[str, int]:
    actions = [action_from_dict(row) for row in action_rows]
    action_ids = [action.action_id for action in actions]
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("duplicate action_id on one parent")
    ordered = sorted(actions, key=lambda action: (action.canonical_order_key, action.action_id))
    return {action.action_id: rank for rank, action in enumerate(ordered)}


def iter_action_examples(manifest_path: Path, label_path: Path) -> list[ActionExample]:
    """Join labels to manifest actions. Does not build graphs and does not read test."""
    if manifest_path.name == "test.jsonl":
        raise ValueError("refusing to open the test manifest")
    if not label_path.is_file():
        raise SystemExit(
            f"Missing {label_path}. Run generate_labels.py for this split before train_gat.py."
        )
    parents = {row["parent_id"]: row for row in iter_jsonl(manifest_path)}
    ranks = {
        parent_id: canonical_ranks(row["actions"]) for parent_id, row in parents.items()
    }
    seen: set[tuple[str, str]] = set()
    examples: list[ActionExample] = []
    for label in iter_jsonl(label_path):
        parent_id = str(label["parent_id"])
        action_id = str(label["action_id"])
        key = (parent_id, action_id)
        if key in seen:
            raise ValueError(f"duplicate label for {parent_id} {action_id}")
        seen.add(key)
        parent = parents.get(parent_id)
        if parent is None:
            raise KeyError(f"label parent {parent_id} is not in {manifest_path.name}")
        action_row = next(
            (row for row in parent["actions"] if row["action_id"] == action_id),
            None,
        )
        if action_row is None:
            raise KeyError(f"label action {parent_id} {action_id} is not in the manifest")
        trials = float(label["n_trials"])
        successes = float(label["n_successes"])
        if trials <= 0 or successes < 0 or successes > trials:
            raise ValueError(f"bad binomial counts for {parent_id} {action_id}")
        examples.append(
            ActionExample(
                parent_id=parent_id,
                action_row=action_row,
                successes=successes,
                trials=trials,
                canonical_action_rank=ranks[parent_id][action_id],
                fragment_heavy_atoms=int(action_row["fragment_heavy_atoms"]),
            )
        )
    if not examples:
        raise SystemExit(f"No action labels in {label_path}")
    return examples


def _parent_mols(manifest_path: Path) -> dict[str, tuple[dict[str, Any], Chem.Mol, Chem.Mol]]:
    cached: dict[str, tuple[dict[str, Any], Chem.Mol, Chem.Mol]] = {}
    for row in iter_jsonl(manifest_path):
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            raise ValueError(f"invalid parent SMILES for {row['parent_id']}")
        cached[row["parent_id"]] = (row, mol, core_mol_from_parent(mol))
    return cached


def _graph_for_example(
    example: ActionExample,
    parent_cache: dict[str, tuple[dict[str, Any], Chem.Mol, Chem.Mol]],
    scaler: StreamingMixedFeatureScaler | None,
) -> Data:
    parent, mol, core = parent_cache[example.parent_id]
    action: EligibleAction = action_from_dict(example.action_row)
    features = action_features(mol, core, action)
    if scaler is not None:
        features = scaler.transform(features)
    graph = action_to_graph(parent, action, features)
    graph.successes = torch.tensor([example.successes], dtype=torch.float32)
    graph.trials = torch.tensor([example.trials], dtype=torch.float32)
    graph.canonical_action_rank = torch.tensor(
        [example.canonical_action_rank], dtype=torch.long
    )
    graph.fragment_heavy_atoms = torch.tensor(
        [example.fragment_heavy_atoms], dtype=torch.float32
    )
    return graph


def cache_dir_for(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg["paths"]["artifacts_dir"]) / CACHE_DIRNAME


def scaler_sha256(scaler_state: dict[str, Any]) -> str:
    blob = json.dumps(
        {"center": scaler_state["center"], "scale": scaler_state["scale"]},
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _fingerprint(
    manifest_path: Path,
    label_path: Path,
    *,
    scaler_token: str | None = None,
) -> dict[str, Any]:
    payload = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "feature_kinds": list(FEATURE_KINDS),
        "node_feature_names": list(NODE_FEATURE_NAMES),
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
        "manifest_sha256": file_sha256(manifest_path),
        "label_sha256": file_sha256(label_path),
        "shard_size": SHARD_SIZE,
        "scaler_version": 1,
    }
    if scaler_token is not None:
        payload["scaler_sha256"] = scaler_token
    return payload


def _metadata_matches(metadata: dict[str, Any], fingerprint: dict[str, Any], split: str) -> bool:
    stored = metadata.get("splits", {}).get(split)
    if not stored:
        return False
    for key, value in fingerprint.items():
        if stored.get(key) != value:
            return False
    cache_dir = Path(metadata["cache_dir"])
    return all((cache_dir / name).is_file() for name in stored["shards"])


def ensure_graph_cache(cfg: dict[str, Any], split: str) -> tuple[Path, dict[str, Any]]:
    """Build or reuse shards for train or validation. Never reads the test split."""
    if split not in {"train", "validation"}:
        raise ValueError(f"graph cache is for train and validation, not {split!r}")
    manifest_path = resolve_path(cfg["paths"]["manifests_dir"]) / f"{split}.jsonl"
    label_path = resolve_path(cfg["paths"]["results_dir"]) / "labels" / f"{split}_action_rhat.jsonl"
    directory = cache_dir_for(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / "metadata.json"
    metadata: dict[str, Any] = {"cache_dir": str(directory), "splits": {}}
    if meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata["cache_dir"] = str(directory)
    scaler_token = None
    if split == "validation":
        if "scaler" not in metadata:
            raise SystemExit("Build the train graph cache before the validation cache.")
        scaler_token = scaler_sha256(metadata["scaler"])
    fingerprint = _fingerprint(manifest_path, label_path, scaler_token=scaler_token)
    if _metadata_matches(metadata, fingerprint, split):
        print(f"{split} graph cache is current ({directory})", flush=True)
        return directory, metadata

    for stale in directory.glob(f"{split}-*.pt"):
        stale.unlink()
    examples = iter_action_examples(manifest_path, label_path)
    parent_cache = _parent_mols(manifest_path)
    if split == "train":
        for stale in directory.glob("validation-*.pt"):
            stale.unlink()
        metadata.get("splits", {}).pop("validation", None)
        scaler = StreamingMixedFeatureScaler(FEATURE_KINDS)
        scale_after_build = True
    else:
        scaler = StreamingMixedFeatureScaler.from_state(metadata["scaler"])
        scale_after_build = False

    shards: list[str] = []
    entries: list[list[int]] = []
    buffer: list[Data] = []
    shard_index = 0

    def flush() -> None:
        nonlocal shard_index, buffer
        if not buffer:
            return
        name = f"{split}-{shard_index:05d}.pt"
        torch.save(buffer, directory / name)
        shards.append(name)
        buffer = []
        shard_index += 1

    feature_batch: list[np.ndarray] = []
    for example in examples:
        raw_graph = _graph_for_example(
            example,
            parent_cache,
            None if scale_after_build else scaler,
        )
        if scale_after_build:
            feature_batch.append(raw_graph.global_features.view(-1).numpy())
            if len(feature_batch) == 512:
                scaler.partial_fit(np.stack(feature_batch))
                feature_batch.clear()
        offset = len(buffer)
        buffer.append(raw_graph)
        entries.append([shard_index, offset])
        if len(buffer) >= SHARD_SIZE:
            flush()
            print(f"{split} cached {len(entries)}/{len(examples)}", flush=True)
    flush()
    del parent_cache
    del examples
    if scale_after_build:
        if feature_batch:
            scaler.partial_fit(np.stack(feature_batch))
        scaler.finalize()
        print(
            f"scaler fit {len(entries)} actions; "
            f"zero-variance continuous columns={len(scaler.zero_variance)}",
            flush=True,
        )
        for name in shards:
            shard_graphs = torch.load(directory / name, weights_only=False)
            for graph in shard_graphs:
                scaled = scaler.transform(graph.global_features.view(-1).numpy())
                graph.global_features = torch.tensor(scaled, dtype=torch.float32).view(1, -1)
            torch.save(shard_graphs, directory / name)
            del shard_graphs
    print(f"{split} cached {len(entries)} graphs in {len(shards)} shards", flush=True)

    metadata.setdefault("splits", {})
    metadata["splits"][split] = {
        **fingerprint,
        "n": len(entries),
        "shards": shards,
        "entries": entries,
    }
    if split == "train":
        metadata["scaler"] = scaler.state_dict()
    temporary = meta_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata), encoding="utf-8")
    temporary.replace(meta_path)
    return directory, metadata


class ShardedActionGraphDataset(Dataset):
    """Loads the shard that holds an index, and keeps that shard until the next one."""

    def __init__(self, cache_dir: Path, split: str):
        metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
        stored = metadata["splits"][split]
        self.entries: list[list[int]] = stored["entries"]
        self.shards: list[str] = stored["shards"]
        self.cache_dir = cache_dir
        self._shard_index: int | None = None
        self._shard: list[Data] | None = None

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Data:
        shard_index, offset = self.entries[index]
        if shard_index != self._shard_index:
            path = self.cache_dir / self.shards[shard_index]
            self._shard = torch.load(path, weights_only=False)
            self._shard_index = shard_index
        if self._shard is None:
            raise RuntimeError("shard failed to load")
        return self._shard[offset]


class ShardGroupedSampler(torch.utils.data.Sampler[int]):
    """Shuffle shards, then examples inside each shard, so a batch stays on one shard."""

    def __init__(self, entries: list[list[int]], seed: int):
        self.entries = entries
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        grouped: dict[int, list[int]] = defaultdict(list)
        for index, (shard_index, _offset) in enumerate(self.entries):
            grouped[shard_index].append(index)
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        shard_order = torch.randperm(len(grouped), generator=generator).tolist()
        shard_ids = list(grouped)
        for position in shard_order:
            indexes = grouped[shard_ids[position]]
            order = torch.randperm(len(indexes), generator=generator).tolist()
            for item in order:
                yield indexes[item]

    def __len__(self) -> int:
        return len(self.entries)
