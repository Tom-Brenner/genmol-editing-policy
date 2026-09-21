"""Join manifest actions to binomial labels and build one GATv2 graph each."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

from rollout_policy.dry_run import action_from_dict
from rollout_policy.eligibility import core_mol_from_parent
from rollout_policy.features import FEATURE_NAMES, action_features
from rollout_policy.graph_features import action_to_graph
from rollout_policy.io import load_jsonl, resolve_path

FEATURE_SCHEMA_VERSION = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_action_graphs(split: str, cfg: dict[str, Any]) -> tuple[list[Data], Path, Path]:
    """Build train or validation graphs. Never reads the test manifest."""
    if split not in {"train", "validation"}:
        raise ValueError(f"GAT training loads train and validation only, not {split!r}")
    manifest_path = resolve_path(cfg["paths"]["manifests_dir"]) / f"{split}.jsonl"
    label_path = (
        resolve_path(cfg["paths"]["results_dir"]) / "labels" / f"{split}_action_rhat.jsonl"
    )
    if manifest_path.name == "test.jsonl":
        raise ValueError("refusing to open the test manifest")
    if not label_path.is_file():
        raise SystemExit(
            f"Missing {label_path}. Run generate_labels.py --split {split} before train_gat.py."
        )
    manifest = {row["parent_id"]: row for row in load_jsonl(manifest_path)}
    labels = load_jsonl(label_path)
    if not labels:
        raise SystemExit(f"No action labels in {label_path}")

    graphs: list[Data] = []
    for index, label in enumerate(labels):
        parent = manifest.get(label["parent_id"])
        if parent is None:
            raise KeyError(f"label parent {label['parent_id']} is not in {manifest_path}")
        action_row = next(
            (row for row in parent["actions"] if row["action_id"] == label["action_id"]),
            None,
        )
        if action_row is None:
            raise KeyError(
                f"label action {label['parent_id']} {label['action_id']} is not in the manifest"
            )
        action = action_from_dict(action_row)
        mol = Chem.MolFromSmiles(parent["smiles"])
        if mol is None:
            raise ValueError(f"invalid parent SMILES for {label['parent_id']}")
        fragment = Chem.MolFromSmiles(action.fragment_smiles)
        if fragment is None:
            raise ValueError(f"invalid fragment SMILES for {action.action_id}")
        global_features = action_features(
            mol,
            core_mol_from_parent(mol),
            fragment,
            action.bond_type,
            action.is_ring,
            action.is_terminal,
        )
        if global_features.shape != (len(FEATURE_NAMES),):
            raise RuntimeError("global feature width drifted from FEATURE_NAMES")
        graph = action_to_graph(parent, action, global_features)
        trials = float(label["n_trials"])
        successes = float(label["n_successes"])
        if trials <= 0 or successes < 0 or successes > trials:
            raise ValueError(
                f"bad binomial counts for {label['parent_id']} {label['action_id']}"
            )
        graph.successes = torch.tensor([successes], dtype=torch.float32)
        graph.trials = torch.tensor([trials], dtype=torch.float32)
        graphs.append(graph)
        if (index + 1) % 2000 == 0:
            print(f"{split} graphs {index + 1}/{len(labels)}", flush=True)
    print(f"{split} graphs {len(graphs)}", flush=True)
    return graphs, manifest_path, label_path


def fit_global_scaler(graphs: list[Data]) -> tuple[torch.Tensor, torch.Tensor]:
    """Train-only mean and scale for the concatenated tabular vector."""
    matrix = torch.cat([graph.global_features.view(1, -1) for graph in graphs], dim=0)
    mean = matrix.mean(dim=0)
    scale = matrix.std(dim=0, unbiased=False)
    scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
    return mean, scale


def apply_global_scaler(graphs: list[Data], mean: torch.Tensor, scale: torch.Tensor) -> None:
    for graph in graphs:
        graph.global_features = (graph.global_features.view(1, -1) - mean) / scale


def validation_top1_success_rate(
    parent_ids: list[str],
    action_ids: list[str],
    logits: list[float],
    successes: list[float],
    trials: list[float],
) -> float:
    """Mean observed success rate of the highest-logit action in each parent."""
    grouped: dict[str, list[tuple[float, str, float]]] = {}
    for parent_id, action_id, logit, k, n in zip(
        parent_ids, action_ids, logits, successes, trials, strict=True
    ):
        if n <= 0:
            raise ValueError(f"non-positive trial count for {parent_id}")
        grouped.setdefault(parent_id, []).append((float(logit), action_id, float(k) / float(n)))
    if not grouped:
        raise ValueError("no validation parents")
    chosen = [
        max(items, key=lambda item: (item[0], item[1]))[2] for items in grouped.values()
    ]
    return float(np.mean(chosen))
