"""Load a frozen action scorer. GAT checkpoints are checked against the live schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Batch

from rollout_policy.eligibility import EligibleAction, core_mol_from_parent
from rollout_policy.features import (
    FEATURE_KINDS,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    LEGACY_FEATURE_NAMES,
    StreamingMixedFeatureScaler,
    action_features,
    legacy_action_features,
)
from rollout_policy.graph_features import (
    EDGE_FEATURE_NAMES,
    GRAPH_SCHEMA_VERSION,
    NODE_FEATURE_NAMES,
    action_to_graph,
)
from rollout_policy.io import resolve_path
from rollout_policy.labels import LABEL_SCHEMA_VERSION
from rollout_policy.model import ValueMLP
from rollout_policy.model_gat import ActionGATv2


def _require_exact(stored, live, label: str) -> None:
    if list(stored) != list(live):
        raise ValueError(f"checkpoint {label} does not match the current code")


class GATActionPredictor:
    """One success probability per action, in the order the caller passed them."""

    def __init__(
        self,
        model: ActionGATv2,
        scaler: StreamingMixedFeatureScaler,
        device: torch.device,
        *,
        checkpoint_sha256: str,
    ):
        self.model = model
        self.scaler = scaler
        self.device = device
        self.model_type = "gatv2"
        self.feature_schema_version = FEATURE_SCHEMA_VERSION
        self.checkpoint_sha256 = checkpoint_sha256
        self.model.eval()

    def predict_actions(self, parent: dict, actions: list[EligibleAction]) -> np.ndarray:
        mol = Chem.MolFromSmiles(parent["smiles"])
        if mol is None:
            raise ValueError(f"invalid parent SMILES for {parent.get('parent_id')}")
        core = core_mol_from_parent(mol)
        graphs = []
        for action in actions:
            features = self.scaler.transform(action_features(mol, core, action))
            graphs.append(action_to_graph(parent, action, features))
        if len(graphs) != len(actions):
            raise RuntimeError("graph count drifted from the action list")
        batch = Batch.from_data_list(graphs).to(self.device)
        with torch.no_grad():
            logits = self.model(batch).view(-1)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        if probabilities.shape != (len(actions),):
            raise RuntimeError(
                f"expected {len(actions)} probabilities, got {tuple(probabilities.shape)}"
            )
        return probabilities.astype(np.float64)


class ValueMLPPredictor:
    """Locked-protocol MLP. Uses the 13-d vector when the checkpoint says so."""

    def __init__(self, model: ValueMLP, mean: np.ndarray, scale: np.ndarray, feature_names: list[str], device: torch.device):
        self.model = model
        self.mean = mean.astype(np.float32)
        self.scale = scale.astype(np.float32)
        self.feature_names = feature_names
        self.device = device
        self.model_type = "value_mlp"
        self.feature_schema_version = 1 if feature_names == list(LEGACY_FEATURE_NAMES) else FEATURE_SCHEMA_VERSION
        self.checkpoint_sha256 = ""
        self.model.eval()

    def predict_actions(self, parent: dict, actions: list[EligibleAction]) -> np.ndarray:
        mol = Chem.MolFromSmiles(parent["smiles"])
        if mol is None:
            raise ValueError(f"invalid parent SMILES for {parent.get('parent_id')}")
        core = core_mol_from_parent(mol)
        if self.feature_names == list(LEGACY_FEATURE_NAMES):
            rows = [legacy_action_features(mol, core, action) for action in actions]
        elif self.feature_names == list(FEATURE_NAMES):
            rows = [action_features(mol, core, action) for action in actions]
        else:
            raise ValueError("value_model.pt feature names do not match a known schema")
        matrix = np.stack(rows).astype(np.float32)
        if matrix.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"value_model.pt expects {self.mean.shape[0]} features, computed {matrix.shape[1]}"
            )
        scaled = (matrix - self.mean) / self.scale
        with torch.no_grad():
            predicted = self.model(torch.from_numpy(scaled).to(self.device)).detach().cpu().numpy()
        return np.asarray(predicted, dtype=np.float64).reshape(-1)


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gat_predictor(path: str | Path, device: torch.device | None = None) -> GATActionPredictor:
    resolved = resolve_path(path)
    device = device or torch.device("cpu")
    blob = torch.load(resolved, map_location=device, weights_only=False)
    if blob.get("model_type") != "gatv2":
        raise ValueError(f"{resolved} is not a gatv2 checkpoint")
    _require_exact(blob["node_feature_names"], NODE_FEATURE_NAMES, "node features")
    _require_exact(blob["edge_feature_names"], EDGE_FEATURE_NAMES, "edge features")
    _require_exact(blob["global_feature_names"], FEATURE_NAMES, "global features")
    if int(blob["graph_schema_version"]) != GRAPH_SCHEMA_VERSION:
        raise ValueError("checkpoint graph schema version does not match")
    if int(blob["feature_schema_version"]) != FEATURE_SCHEMA_VERSION:
        raise ValueError("checkpoint feature schema version does not match")
    if int(blob.get("label_schema_version", 0)) != LABEL_SCHEMA_VERSION:
        raise ValueError("checkpoint label schema version does not match")
    if list(blob.get("feature_kinds", FEATURE_KINDS)) != list(FEATURE_KINDS):
        raise ValueError("checkpoint feature kinds do not match")
    config = blob["model_config"]
    model = ActionGATv2(
        node_dim=int(config["node_dim"]),
        edge_dim=int(config["edge_dim"]),
        global_dim=int(config["global_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        heads=int(config["heads"]),
        n_layers=int(config["n_layers"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(blob["model_state_dict"])
    model.to(device)
    scaler = StreamingMixedFeatureScaler.from_state(
        {
            "center": blob["global_scaler_center"],
            "scale": blob["global_scaler_scale"],
            "zero_variance": blob.get("zero_variance_features", []),
            "feature_names": blob["global_feature_names"],
            "feature_kinds": blob.get("feature_kinds", list(FEATURE_KINDS)),
            "feature_schema_version": blob["feature_schema_version"],
            "n": blob.get("scaler_n", 0),
        }
    )
    return GATActionPredictor(
        model,
        scaler,
        device,
        checkpoint_sha256=_file_sha256(resolved),
    )


def load_value_mlp_predictor(path: str | Path, device: torch.device | None = None) -> ValueMLPPredictor:
    resolved = resolve_path(path)
    device = device or torch.device("cpu")
    blob = torch.load(resolved, map_location=device, weights_only=False)
    model = ValueMLP(blob["n_features"], hidden_sizes=tuple(blob["hidden_sizes"]))
    model.load_state_dict(blob["model_state_dict"])
    model.to(device)
    return ValueMLPPredictor(
        model,
        np.asarray(blob["scaler_mean"], dtype=np.float64),
        np.asarray(blob["scaler_scale"], dtype=np.float64),
        list(blob["feature_names"]),
        device,
    )


def load_action_predictor(cfg: dict, device: torch.device):
    """Prefer the V2 GAT checkpoint. Fall back to the locked value-model checkpoint."""
    artifacts = resolve_path(cfg["paths"]["artifacts_dir"])
    mlp_path = artifacts / "value_model.pt"
    for name in ("best.pt", "gatv2.pt"):
        gat_path = artifacts / name
        if gat_path.is_file():
            return load_gat_predictor(gat_path, device)
    if mlp_path.is_file():
        return load_value_mlp_predictor(mlp_path, device)
    raise SystemExit(f"No best.pt or value_model.pt under {artifacts}")
