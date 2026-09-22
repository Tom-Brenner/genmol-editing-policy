#!/usr/bin/env python3
"""Train the action-conditioned GATv2 attention model.

Message passing is GATv2Conv: a learned attention weight per neighbor, using
the selected-cut flag on the attachment edge. Graphs are read from a sharded
cache under artifacts_*/graph_cache/schema_v2/. The cache is built from the
manifest and the existing binomial labels. This script does not call GenMol
and does not open the test manifest.

Early stopping minimizes validation binomial NLL. Top-1 success is reported
beside that checkpoint and is not the selection rule.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rollout_policy.datasets import (  # noqa: E402
    ShardGroupedSampler,
    ShardedActionGraphDataset,
    ensure_graph_cache,
    file_sha256,
)
from rollout_policy.features import (  # noqa: E402
    FEATURE_KINDS,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
)
from rollout_policy.graph_features import (  # noqa: E402
    EDGE_FEATURE_NAMES,
    GRAPH_SCHEMA_VERSION,
    NODE_FEATURE_NAMES,
)
from rollout_policy.io import artifact_path, load_experiment_config, resolve_path, write_json  # noqa: E402
from rollout_policy.model import binomial_nll  # noqa: E402
from rollout_policy.model_gat import ActionGATv2  # noqa: E402
from rollout_policy.policies import select_scored_index  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402


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


def _id_list(value, n: int) -> list[str]:
    if isinstance(value, str):
        if n != 1:
            raise RuntimeError(f"expected {n} ids, got one string")
        return [value]
    values = [str(item) for item in value]
    if len(values) != n:
        raise RuntimeError(f"expected {n} ids, got {len(values)}")
    return values


def _flat(value) -> list[float]:
    return [float(item) for item in value.detach().cpu().view(-1)]


def _collect_validation(model: ActionGATv2, loader: DataLoader, device: torch.device) -> dict:
    parent_ids: list[str] = []
    action_ids: list[str] = []
    ranks: list[int] = []
    heavies: list[float] = []
    logits: list[float] = []
    successes: list[float] = []
    trials: list[float] = []
    total_nll = 0.0
    total_trials = 0.0
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch).view(-1)
            k = batch.successes.view(-1)
            n = batch.trials.view(-1)
            loss = binomial_nll(pred, k, n)
        trial_count = float(n.sum().item())
        total_nll += float(loss.item()) * trial_count
        total_trials += trial_count
        width = int(pred.shape[0])
        parent_ids.extend(_id_list(batch.parent_id, width))
        action_ids.extend(_id_list(batch.action_id, width))
        ranks.extend(int(item) for item in batch.canonical_action_rank.view(-1).cpu())
        heavies.extend(_flat(batch.fragment_heavy_atoms))
        logits.extend(_flat(pred))
        successes.extend(_flat(k))
        trials.extend(_flat(n))
    return {
        "parent_ids": parent_ids,
        "action_ids": action_ids,
        "ranks": ranks,
        "heavies": heavies,
        "logits": logits,
        "successes": successes,
        "trials": trials,
        "binomial_nll": total_nll / max(total_trials, 1.0),
    }


def _grouped(held: dict) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, parent_id in enumerate(held["parent_ids"]):
        grouped[parent_id].append(index)
    return grouped


def _top1_success(held: dict) -> float:
    chosen: list[float] = []
    for indexes in _grouped(held).values():
        pick = select_scored_index(
            [held["logits"][i] for i in indexes],
            [held["ranks"][i] for i in indexes],
            [held["action_ids"][i] for i in indexes],
        )
        index = indexes[pick]
        chosen.append(held["successes"][index] / held["trials"][index])
    return float(np.mean(chosen))


def _secondary_metrics(held: dict) -> dict:
    logits = np.asarray(held["logits"], dtype=np.float64)
    successes = np.asarray(held["successes"], dtype=np.float64)
    trials = np.asarray(held["trials"], dtype=np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
    observed = successes / trials
    bins = []
    edges = np.linspace(0.0, 1.0, 11)
    for low, high in zip(edges[:-1], edges[1:]):
        if high == 1.0:
            mask = (probabilities >= low) & (probabilities <= high)
        else:
            mask = (probabilities >= low) & (probabilities < high)
        count = int(mask.sum())
        if count == 0:
            continue
        bins.append(
            {
                "lo": float(low),
                "hi": float(high),
                "n": count,
                "mean_predicted": float(probabilities[mask].mean()),
                "mean_observed": float(observed[mask].mean()),
            }
        )
    agreements = []
    for indexes in _grouped(held).values():
        learned = select_scored_index(
            [held["logits"][i] for i in indexes],
            [held["ranks"][i] for i in indexes],
            [held["action_ids"][i] for i in indexes],
        )
        largest = min(
            range(len(indexes)),
            key=lambda local: (
                -held["heavies"][indexes[local]],
                held["ranks"][indexes[local]],
                held["action_ids"][indexes[local]],
            ),
        )
        agreements.append(int(learned == largest))
    return {
        "brier": float(np.mean((probabilities - observed) ** 2)),
        "calibration_bins": bins,
        "largest_fragment_agreement": float(np.mean(agreements)),
    }


def _checkpoint_blob(model, gat_cfg, best, n_parameters, metadata, train_manifest, val_manifest, train_labels, val_labels) -> dict:
    scaler = metadata["scaler"]
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
        "feature_kinds": list(FEATURE_KINDS),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "global_scaler_center": scaler["center"],
        "global_scaler_scale": scaler["scale"],
        "zero_variance_features": scaler["zero_variance"],
        "scaler_n": scaler["n"],
        "label_schema_version": 2,
        "best_validation": best,
        "training_manifest_sha256": file_sha256(train_manifest),
        "validation_manifest_sha256": file_sha256(val_manifest),
        "training_label_sha256": file_sha256(train_labels),
        "validation_label_sha256": file_sha256(val_labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_v2.yaml")
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    gat_cfg = cfg.get("models", {}).get("gatv2", {})
    schedule = cfg.get("value_model", {})

    train_dir, metadata = ensure_graph_cache(cfg, "train")
    val_dir, metadata = ensure_graph_cache(cfg, "validation")
    if train_dir != val_dir:
        raise RuntimeError("train and validation caches diverged")
    train_set = ShardedActionGraphDataset(train_dir, "train")
    val_set = ShardedActionGraphDataset(val_dir, "validation")
    manifests = resolve_path(cfg["paths"]["manifests_dir"])
    labels = resolve_path(cfg["paths"]["results_dir"]) / "labels"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = train_set[0]
    model: ActionGATv2 | None = ActionGATv2(
        node_dim=int(sample.x.size(1)),
        edge_dim=int(sample.edge_attr.size(1)),
        global_dim=int(sample.global_features.size(-1)),
        hidden_dim=int(gat_cfg.get("hidden_dim", 64)),
        heads=int(gat_cfg.get("heads", 4)),
        n_layers=int(gat_cfg.get("n_layers", 3)),
        dropout=float(gat_cfg.get("dropout", 0.15)),
    ).to(device)
    n_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(
        f"model_type=gatv2 attention_layers={model.n_layers} heads={model.heads} "
        f"parameters={n_parameters} device={device} "
        f"train_graphs={len(train_set)} val_graphs={len(val_set)}",
        flush=True,
    )

    batch_size = int(gat_cfg.get("batch_size", schedule.get("batch_size", 64)))
    workers = int(gat_cfg.get("num_workers", 4))
    sampler = ShardGroupedSampler(train_set.entries, seed=int(cfg["split"]["manifest_seed"]))
    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **loader_kwargs)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(gat_cfg.get("learning_rate", schedule.get("learning_rate", 1e-3))),
        weight_decay=float(gat_cfg.get("weight_decay", 1e-4)),
    )
    max_epochs = int(args.max_epochs or gat_cfg.get("max_epochs", 100))
    patience = int(gat_cfg.get("patience", 10))
    min_delta = float(gat_cfg.get("early_stopping_min_delta", 1e-5))
    log_every = max(1, int(schedule.get("log_every_n_steps", 100)))
    ckpt_path = artifact_path(cfg, "gatv2.pt")
    best = {
        "val_binomial_nll": float("inf"),
        "validation_top1_success_rate": None,
        "epoch": -1,
    }
    history: list[dict] = []
    interrupted = False
    stale = 0
    global_step = 0

    try:
        for epoch in range(max_epochs):
            sampler.set_epoch(epoch)
            model.train()
            epoch_nll_sum = 0.0
            epoch_trials = 0.0
            window_nll_sum = 0.0
            window_trials = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(batch).view(-1)
                trials = batch.trials.view(-1)
                loss = binomial_nll(logits, batch.successes.view(-1), trials)
                loss.backward()
                opt.step()
                trial_count = float(trials.sum().item())
                epoch_nll_sum += float(loss.item()) * trial_count
                epoch_trials += trial_count
                window_nll_sum += float(loss.item()) * trial_count
                window_trials += trial_count
                global_step += 1
                if global_step % log_every == 0:
                    print(
                        f"step={global_step} epoch={epoch} "
                        f"train_binomial_nll={window_nll_sum / max(window_trials, 1.0):.6f}",
                        flush=True,
                    )
                    window_nll_sum = 0.0
                    window_trials = 0.0
                    _free_cuda_memory()

            held = _collect_validation(model, val_loader, device)
            top1 = _top1_success(held)
            train_nll = epoch_nll_sum / max(epoch_trials, 1.0)
            row = {
                "epoch": epoch,
                "train_binomial_nll": train_nll,
                "val_binomial_nll": held["binomial_nll"],
                "validation_top1_success_rate": top1,
                "step": global_step,
            }
            history.append(row)
            print(
                f"epoch={epoch} train_binomial_nll={train_nll:.6f} "
                f"val_binomial_nll={held['binomial_nll']:.6f} "
                f"validation_top1_success_rate={top1:.6f}",
                flush=True,
            )
            if held["binomial_nll"] < float(best["val_binomial_nll"]) - min_delta:
                secondary = _secondary_metrics(held)
                best = {
                    "val_binomial_nll": held["binomial_nll"],
                    "validation_top1_success_rate": top1,
                    "brier": secondary["brier"],
                    "calibration_bins": secondary["calibration_bins"],
                    "largest_fragment_agreement": secondary["largest_fragment_agreement"],
                    "epoch": epoch,
                }
                stale = 0
                torch.save(
                    _checkpoint_blob(
                        model,
                        gat_cfg,
                        best,
                        n_parameters,
                        metadata,
                        manifests / "train.jsonl",
                        manifests / "validation.jsonl",
                        labels / "train_action_rhat.jsonl",
                        labels / "validation_action_rhat.jsonl",
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
            "early_stopping_metric": "val_binomial_nll",
            "n_parameters": n_parameters,
            "best": best,
            "history": history,
            "n_train_actions": len(train_set),
            "n_val_actions": len(val_set),
            "checkpoint": str(ckpt_path),
            "opened_test_manifest": False,
        }
        write_json(artifact_path(cfg, "gatv2_validation_report.json"), report)
        print(json.dumps({"best": {k: best[k] for k in best if k != "calibration_bins"}, "n_parameters": n_parameters}, indent=2))
        _release_gpu(model)
        del model
        _free_cuda_memory()

    if interrupted:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
