#!/usr/bin/env python3
"""Train the action-conditioned GATv2 attention model.

Each example is one parent/action graph. Message passing is GATv2Conv
(Brody et al. 2022): a learned attention weight per neighbor, conditioned on
the selected-cut flag at the attachment edge.

Graphs are built here from the manifest (SMILES and attachment indices) joined
to the binomial counts in results_*/labels/{split}_action_rhat.jsonl.
generate_labels.py stores those counts, not node or edge tensors.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rollout_policy.datasets import (
    FEATURE_SCHEMA_VERSION,
    apply_global_scaler,
    file_sha256,
    fit_global_scaler,
    load_action_graphs,
    validation_top1_success_rate,
)
from rollout_policy.features import FEATURE_NAMES
from rollout_policy.graph_features import (
    EDGE_FEATURE_NAMES,
    GRAPH_SCHEMA_VERSION,
    NODE_FEATURE_NAMES,
)
from rollout_policy.io import artifact_path, load_experiment_config, write_json
from rollout_policy.model import binomial_nll
from rollout_policy.model_gat import ActionGATv2
from torch_geometric.loader import DataLoader


def _id_list(value, n: int) -> list[str]:
    """Batch attributes are lists; a one-graph batch can still be a bare string."""
    if isinstance(value, str):
        if n != 1:
            raise RuntimeError(f"expected {n} ids, got one string")
        return [value]
    values = [str(item) for item in value]
    if len(values) != n:
        raise RuntimeError(f"expected {n} ids, got {len(values)}")
    return values


def _free_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _release_gpu(model: torch.nn.Module | None) -> None:
    if model is not None:
        model.to("cpu")
        for param in model.parameters():
            param.grad = None
    _free_cuda_memory()


def _collect_validation(model: ActionGATv2, loader: DataLoader, device: torch.device) -> dict:
    parent_ids: list[str] = []
    action_ids: list[str] = []
    logits: list[float] = []
    successes: list[float] = []
    trials: list[float] = []
    total_nll = 0.0
    total_trials = 0.0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).view(-1)
            k = batch.successes.view(-1)
            n = batch.trials.view(-1)
            loss = binomial_nll(pred, k, n)
            total_nll += float(loss.item()) * float(n.sum().item())
            total_trials += float(n.sum().item())
            parent_ids.extend(_id_list(batch.parent_id, int(pred.shape[0])))
            action_ids.extend(_id_list(batch.action_id, int(pred.shape[0])))
            logits.extend(float(x) for x in pred.detach().cpu())
            successes.extend(float(x) for x in k.detach().cpu())
            trials.extend(float(x) for x in n.detach().cpu())
    return {
        "parent_ids": parent_ids,
        "action_ids": action_ids,
        "logits": logits,
        "successes": successes,
        "trials": trials,
        "binomial_nll": total_nll / max(total_trials, 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_v2.yaml")
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    gat_cfg = cfg.get("models", {}).get("gatv2", {})
    schedule = cfg.get("value_model", {})

    train_graphs, train_manifest, train_labels = load_action_graphs("train", cfg)
    val_graphs, val_manifest, val_labels = load_action_graphs("validation", cfg)
    mean, scale = fit_global_scaler(train_graphs)
    apply_global_scaler(train_graphs, mean, scale)
    apply_global_scaler(val_graphs, mean, scale)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = train_graphs[0]
    model: ActionGATv2 | None = ActionGATv2(
        node_dim=int(sample.x.size(1)),
        edge_dim=int(sample.edge_attr.size(1)),
        global_dim=int(sample.global_features.size(-1)),
        hidden_dim=int(gat_cfg.get("hidden_dim", 64)),
        heads=int(gat_cfg.get("heads", 4)),
        n_layers=int(gat_cfg.get("n_layers", 3)),
        dropout=float(gat_cfg.get("dropout", 0.15)),
    ).to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"model_type=gatv2 attention_layers={model.n_layers} "
        f"heads={model.heads} parameters={n_parameters} device={device}",
        flush=True,
    )

    batch_size = int(gat_cfg.get("batch_size", schedule.get("batch_size", 64)))
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(gat_cfg.get("learning_rate", schedule.get("learning_rate", 1e-3))),
        weight_decay=float(gat_cfg.get("weight_decay", 1e-4)),
    )
    max_epochs = int(gat_cfg.get("max_epochs", schedule.get("max_epochs", 200)))
    patience = int(gat_cfg.get("patience", schedule.get("patience", 20)))
    log_every = max(1, int(schedule.get("log_every_n_steps", 100)))
    ckpt_path = artifact_path(cfg, "gatv2.pt")
    best = {
        "validation_top1_success_rate": -1.0,
        "epoch": -1,
        "val_binomial_nll": None,
    }
    history: list[dict] = []
    interrupted = False
    stale = 0
    global_step = 0

    try:
        for epoch in range(max_epochs):
            model.train()
            total_loss = 0.0
            n_batches = 0
            window_loss = 0.0
            window_n = 0
            for batch in train_loader:
                batch = batch.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(batch).view(-1)
                loss = binomial_nll(logits, batch.successes.view(-1), batch.trials.view(-1))
                loss.backward()
                opt.step()
                batch_loss = float(loss.item())
                total_loss += batch_loss
                n_batches += 1
                global_step += 1
                window_loss += batch_loss
                window_n += 1
                if global_step % log_every == 0:
                    print(
                        f"step={global_step} epoch={epoch} "
                        f"train_binomial_nll={window_loss / max(window_n, 1):.6f}",
                        flush=True,
                    )
                    window_loss = 0.0
                    window_n = 0
                    _free_cuda_memory()

            held = _collect_validation(model, val_loader, device)
            top1 = validation_top1_success_rate(
                held["parent_ids"],
                held["action_ids"],
                held["logits"],
                held["successes"],
                held["trials"],
            )
            row = {
                "epoch": epoch,
                "train_binomial_nll": total_loss / max(n_batches, 1),
                "val_binomial_nll": held["binomial_nll"],
                "validation_top1_success_rate": top1,
                "step": global_step,
            }
            history.append(row)
            print(
                f"epoch={epoch} train_binomial_nll={row['train_binomial_nll']:.6f} "
                f"val_binomial_nll={held['binomial_nll']:.6f} "
                f"validation_top1_success_rate={top1:.6f}",
                flush=True,
            )
            if top1 > float(best["validation_top1_success_rate"]) + 1e-8:
                best = {
                    "validation_top1_success_rate": top1,
                    "epoch": epoch,
                    "val_binomial_nll": held["binomial_nll"],
                }
                stale = 0
                torch.save(
                    _checkpoint_blob(
                        model=model,
                        gat_cfg=gat_cfg,
                        best=best,
                        n_parameters=n_parameters,
                        mean=mean,
                        scale=scale,
                        train_manifest=train_manifest,
                        val_manifest=val_manifest,
                        train_labels=train_labels,
                        val_labels=val_labels,
                    ),
                    ckpt_path,
                )
            else:
                stale += 1
                if stale >= patience:
                    break
            _free_cuda_memory()
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted (Ctrl+C); releasing GPU and exiting.", flush=True)
    finally:
        report = {
            "model_type": "gatv2",
            "n_parameters": n_parameters,
            "best": best,
            "history": history,
            "n_train_actions": len(train_graphs),
            "n_val_actions": len(val_graphs),
            "checkpoint": str(ckpt_path),
        }
        write_json(artifact_path(cfg, "gatv2_validation_report.json"), report)
        print(json.dumps({"best": best, "n_parameters": n_parameters}, indent=2))
        _release_gpu(model)
        del model
        _free_cuda_memory()

    if interrupted:
        raise SystemExit(130)


def _checkpoint_blob(
    *,
    model: ActionGATv2,
    gat_cfg: dict,
    best: dict,
    n_parameters: int,
    mean: torch.Tensor,
    scale: torch.Tensor,
    train_manifest: Path,
    val_manifest: Path,
    train_labels: Path,
    val_labels: Path,
) -> dict:
    return {
        "model_type": "gatv2",
        "model_state_dict": model.state_dict(),
        "model_config": model.model_config(),
        "hyperparameters": dict(gat_cfg),
        "n_parameters": n_parameters,
        "node_feature_names": list(NODE_FEATURE_NAMES),
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "global_feature_names": list(FEATURE_NAMES),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "global_scaler_mean": mean.tolist(),
        "global_scaler_scale": scale.tolist(),
        "label_schema_version": 2,
        "best_validation": best,
        "training_manifest_sha256": file_sha256(train_manifest),
        "validation_manifest_sha256": file_sha256(val_manifest),
        "training_label_sha256": file_sha256(train_labels),
        "validation_label_sha256": file_sha256(val_labels),
    }


if __name__ == "__main__":
    main()
