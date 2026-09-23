#!/usr/bin/env python3
"""Train the schema-v2 shallow MLP and the LightGBM pointwise regressor.

Both models read the same train and validation action table. Neither opens
manifests_*/test.jsonl. The MLP returns a binomial logit and is fit with
binomial NLL. The regressor is fit to n_successes / n_trials with sample
weight n_trials. Early stopping uses validation binomial NLL for the MLP and
LightGBM's regression metric for the trees. Top-1 observed success is reported
beside the checkpoint and is not the stopping rule.

    python scripts/train_tabular_models.py --config configs/experiment_v2.yaml
    python scripts/train_tabular_models.py --config configs/experiment_v2.yaml --model shallow_mlp
    python scripts/train_tabular_models.py --config configs/experiment_v2.yaml --model lightgbm_regressor
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rollout_policy.datasets import file_sha256, iter_action_examples  # noqa: E402
from rollout_policy.dry_run import action_from_dict  # noqa: E402
from rollout_policy.eligibility import core_mol_from_parent  # noqa: E402
from rollout_policy.features import (  # noqa: E402
    FEATURE_KINDS,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    StreamingMixedFeatureScaler,
    action_features,
)
from rollout_policy.io import (  # noqa: E402
    artifact_path,
    iter_jsonl,
    load_experiment_config,
    resolve_path,
    write_json,
)
from rollout_policy.labels import LABEL_SCHEMA_VERSION  # noqa: E402
from rollout_policy.model import TabularMLP, binomial_nll  # noqa: E402
from rollout_policy.policies import select_scored_index  # noqa: E402

_PROB_CLIP = 1e-6


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


def load_feature_table(split: str, cfg: dict) -> list[dict]:
    """Join one train or validation split to schema-v2 action features."""
    if split not in {"train", "validation"}:
        raise ValueError(f"tabular training reads train and validation, not {split!r}")
    manifest_path = resolve_path(cfg["paths"]["manifests_dir"]) / f"{split}.jsonl"
    label_path = (
        resolve_path(cfg["paths"]["results_dir"]) / "labels" / f"{split}_action_rhat.jsonl"
    )
    examples = iter_action_examples(manifest_path, label_path)
    parents = {row["parent_id"]: row for row in iter_jsonl(manifest_path)}
    mols: dict[str, tuple[Chem.Mol, Chem.Mol]] = {}
    rows: list[dict] = []
    for example in examples:
        cached = mols.get(example.parent_id)
        if cached is None:
            parent = parents[example.parent_id]
            mol = Chem.MolFromSmiles(parent["smiles"])
            if mol is None:
                raise ValueError(f"invalid parent SMILES for {example.parent_id}")
            cached = (mol, core_mol_from_parent(mol))
            mols[example.parent_id] = cached
        mol, core = cached
        features = action_features(mol, core, action_from_dict(example.action_row))
        if features.shape != (len(FEATURE_NAMES),):
            raise RuntimeError("feature width drifted from FEATURE_NAMES")
        rows.append(
            {
                "parent_id": example.parent_id,
                "action_id": example.action_row["action_id"],
                "features": features,
                "successes": float(example.successes),
                "trials": float(example.trials),
                "canonical_action_rank": int(example.canonical_action_rank),
                "fragment_heavy_atoms": int(example.fragment_heavy_atoms),
            }
        )
        if len(rows) % 5000 == 0:
            print(f"{split} featurized {len(rows)}/{len(examples)}", flush=True)
    print(f"{split} featurized {len(rows)} actions", flush=True)
    return rows


def _stacks(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.stack([row["features"] for row in rows]).astype(np.float32)
    successes = np.asarray([row["successes"] for row in rows], dtype=np.float64)
    trials = np.asarray([row["trials"] for row in rows], dtype=np.float64)
    return features, successes, trials


def _fit_scaler(features: np.ndarray) -> StreamingMixedFeatureScaler:
    scaler = StreamingMixedFeatureScaler(FEATURE_KINDS)
    for start in range(0, features.shape[0], 4096):
        scaler.partial_fit(features[start : start + 4096])
    scaler.finalize()
    return scaler


def binomial_nll_from_probabilities(
    probabilities: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
) -> float:
    """Trial-weighted binomial NLL, combinatorial term omitted, probabilities clipped."""
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), _PROB_CLIP, 1.0 - _PROB_CLIP)
    k = np.asarray(successes, dtype=np.float64)
    n = np.asarray(trials, dtype=np.float64)
    per_action = -(k * np.log(clipped) + (n - k) * np.log(1.0 - clipped))
    return float(per_action.sum() / np.maximum(n.sum(), 1.0))


def _calibration_bins(probabilities: np.ndarray, observed: np.ndarray) -> list[dict]:
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
    return bins


def selection_metrics(rows: list[dict], scores: np.ndarray, probabilities: np.ndarray) -> dict:
    """Top-1 success on scores. Probabilities are the clipped values used for NLL and Brier."""
    successes = np.asarray([row["successes"] for row in rows], dtype=np.float64)
    trials = np.asarray([row["trials"] for row in rows], dtype=np.float64)
    observed = successes / trials
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["parent_id"]].append(index)

    chosen_rates: list[float] = []
    agreements: list[int] = []
    by_count: dict[int, list[float]] = defaultdict(list)
    for indexes in grouped.values():
        pick = select_scored_index(
            [float(scores[i]) for i in indexes],
            [int(rows[i]["canonical_action_rank"]) for i in indexes],
            [str(rows[i]["action_id"]) for i in indexes],
        )
        chosen = indexes[pick]
        rate = float(observed[chosen])
        chosen_rates.append(rate)
        largest = min(
            range(len(indexes)),
            key=lambda local: (
                -int(rows[indexes[local]]["fragment_heavy_atoms"]),
                int(rows[indexes[local]]["canonical_action_rank"]),
                str(rows[indexes[local]]["action_id"]),
            ),
        )
        agreements.append(int(pick == largest))
        by_count[len(indexes)].append(rate)

    probabilities = np.asarray(probabilities, dtype=np.float64)
    return {
        "validation_top1_success_rate": float(np.mean(chosen_rates)),
        "binomial_nll": binomial_nll_from_probabilities(probabilities, successes, trials),
        "brier": float(np.mean((probabilities - observed) ** 2)),
        "calibration_bins": _calibration_bins(probabilities, observed),
        "largest_fragment_agreement": float(np.mean(agreements)),
        "success_by_n_actions": {
            str(width): {
                "n_parents": len(rates),
                "success_rate": float(np.mean(rates)),
            }
            for width, rates in sorted(by_count.items())
        },
        "n_parents": len(grouped),
        "n_actions": len(rows),
    }


def _manifest_hashes(cfg: dict) -> dict[str, str]:
    manifests = resolve_path(cfg["paths"]["manifests_dir"])
    labels = resolve_path(cfg["paths"]["results_dir"]) / "labels"
    return {
        "training_manifest_sha256": file_sha256(manifests / "train.jsonl"),
        "validation_manifest_sha256": file_sha256(manifests / "validation.jsonl"),
        "training_label_sha256": file_sha256(labels / "train_action_rhat.jsonl"),
        "validation_label_sha256": file_sha256(labels / "validation_action_rhat.jsonl"),
    }


def _public_metrics(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "calibration_bins"}


def train_shallow_mlp(
    cfg: dict,
    train_rows: list[dict],
    val_rows: list[dict],
    *,
    model_seed: int,
    max_epochs: int | None,
) -> None:
    mlp_cfg = cfg.get("models", {}).get("shallow_mlp", {})
    x_train, k_train, n_train = _stacks(train_rows)
    x_val, k_val, n_val = _stacks(val_rows)
    scaler = _fit_scaler(x_train)
    x_train_s = scaler.transform(x_train)
    x_val_s = scaler.transform(x_val)
    print(
        f"scaler fit {scaler.n} actions; "
        f"zero-variance continuous columns={len(scaler.zero_variance)}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    hidden_sizes = tuple(int(width) for width in mlp_cfg.get("hidden_sizes", [64]))
    dropout = float(mlp_cfg.get("dropout", 0.15))
    model: TabularMLP | None = TabularMLP(
        n_features=x_train_s.shape[1],
        hidden_sizes=hidden_sizes,
        dropout=dropout,
    ).to(device)
    n_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(
        f"model_type=shallow_mlp parameters={n_parameters} device={device} "
        f"train_actions={len(train_rows)} val_actions={len(val_rows)}",
        flush=True,
    )
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(mlp_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(mlp_cfg.get("weight_decay", 1e-4)),
    )
    generator = torch.Generator()
    generator.manual_seed(model_seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train_s),
            torch.from_numpy(k_train.astype(np.float32)),
            torch.from_numpy(n_train.astype(np.float32)),
        ),
        batch_size=int(mlp_cfg.get("batch_size", 64)),
        shuffle=True,
        generator=generator,
    )
    x_val_t = torch.from_numpy(x_val_s).to(device)
    k_val_t = torch.from_numpy(k_val.astype(np.float32)).to(device)
    n_val_t = torch.from_numpy(n_val.astype(np.float32)).to(device)
    epochs = int(max_epochs or mlp_cfg.get("max_epochs", 100))
    patience = int(mlp_cfg.get("patience", 10))
    min_delta = float(mlp_cfg.get("early_stopping_min_delta", 1e-5))
    ckpt_path = artifact_path(cfg, "shallow_mlp.pt")
    best: dict = {
        "val_binomial_nll": float("inf"),
        "validation_top1_success_rate": None,
        "epoch": -1,
    }
    history: list[dict] = []
    interrupted = False
    stale = 0

    try:
        for epoch in range(epochs):
            model.train()
            nll_sum = 0.0
            trial_sum = 0.0
            for xb, kb, nb in loader:
                xb, kb, nb = xb.to(device), kb.to(device), nb.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(xb).view(-1)
                loss = binomial_nll(logits, kb, nb)
                loss.backward()
                opt.step()
                trial_count = float(nb.sum().item())
                nll_sum += float(loss.item()) * trial_count
                trial_sum += trial_count
            train_nll = nll_sum / max(trial_sum, 1.0)

            model.eval()
            started = time.perf_counter()
            with torch.no_grad():
                val_logits_t = model(x_val_t).view(-1)
                val_loss = binomial_nll(val_logits_t, k_val_t, n_val_t)
            inference_seconds = time.perf_counter() - started
            val_logits = val_logits_t.detach().cpu().numpy()
            val_nll = float(val_loss.item())
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -60.0, 60.0)))
            metrics = selection_metrics(val_rows, val_logits, probabilities)
            metrics["binomial_nll"] = val_nll
            row = {
                "epoch": epoch,
                "train_binomial_nll": train_nll,
                "val_binomial_nll": val_nll,
                "validation_top1_success_rate": metrics["validation_top1_success_rate"],
                "inference_seconds": inference_seconds,
            }
            history.append(row)
            print(
                f"epoch={epoch} train_binomial_nll={train_nll:.6f} "
                f"val_binomial_nll={val_nll:.6f} "
                f"validation_top1_success_rate={metrics['validation_top1_success_rate']:.6f}",
                flush=True,
            )
            if val_nll < float(best["val_binomial_nll"]) - min_delta:
                best = {
                    "val_binomial_nll": val_nll,
                    "validation_top1_success_rate": metrics["validation_top1_success_rate"],
                    "brier": metrics["brier"],
                    "calibration_bins": metrics["calibration_bins"],
                    "largest_fragment_agreement": metrics["largest_fragment_agreement"],
                    "success_by_n_actions": metrics["success_by_n_actions"],
                    "inference_seconds": inference_seconds,
                    "inference_seconds_per_action": inference_seconds / max(len(val_rows), 1),
                    "epoch": epoch,
                }
                stale = 0
                torch.save(
                    {
                        "model_type": "shallow_mlp",
                        "model_state_dict": model.state_dict(),
                        "model_config": {
                            "n_features": int(x_train_s.shape[1]),
                            "hidden_sizes": list(hidden_sizes),
                            "dropout": dropout,
                        },
                        "hyperparameters": dict(mlp_cfg),
                        "n_parameters": n_parameters,
                        "feature_names": list(FEATURE_NAMES),
                        "feature_kinds": list(FEATURE_KINDS),
                        "feature_schema_version": FEATURE_SCHEMA_VERSION,
                        "scaler": scaler.state_dict(),
                        "label_schema_version": LABEL_SCHEMA_VERSION,
                        "model_seed": model_seed,
                        "best_validation": best,
                        **_manifest_hashes(cfg),
                    },
                    ckpt_path,
                )
                print(
                    f"saved {ckpt_path.name} epoch={epoch} val_binomial_nll={val_nll:.6f}",
                    flush=True,
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
            "model_type": "shallow_mlp",
            "early_stopping_metric": "val_binomial_nll",
            "n_parameters": n_parameters,
            "best": best,
            "history": history,
            "n_train_actions": len(train_rows),
            "n_val_actions": len(val_rows),
            "checkpoint": str(ckpt_path),
            "opened_test_manifest": False,
            "model_seed": model_seed,
        }
        write_json(artifact_path(cfg, "shallow_mlp_validation_report.json"), report)
        print(json.dumps({"best": _public_metrics(best), "n_parameters": n_parameters}, indent=2))
        _release_gpu(model)
        del model
        _free_cuda_memory()
    if interrupted:
        raise SystemExit(130)


def train_lightgbm_regressor(
    cfg: dict,
    train_rows: list[dict],
    val_rows: list[dict],
    *,
    model_seed: int,
) -> None:
    import lightgbm
    from lightgbm import LGBMRegressor

    tree_cfg = cfg.get("models", {}).get("lightgbm_regressor", {})
    x_train, k_train, n_train = _stacks(train_rows)
    x_val, k_val, n_val = _stacks(val_rows)
    y_train = (k_train / n_train).astype(np.float64)
    y_val = (k_val / n_val).astype(np.float64)
    rounds = int(tree_cfg.get("early_stopping_rounds", 50))
    regressor = LGBMRegressor(
        objective=str(tree_cfg.get("objective", "regression")),
        n_estimators=int(tree_cfg.get("n_estimators", 800)),
        learning_rate=float(tree_cfg.get("learning_rate", 0.03)),
        num_leaves=int(tree_cfg.get("num_leaves", 31)),
        max_depth=int(tree_cfg.get("max_depth", -1)),
        min_child_samples=int(tree_cfg.get("min_child_samples", 50)),
        subsample=float(tree_cfg.get("subsample", 0.8)),
        subsample_freq=int(tree_cfg.get("subsample_freq", 1)),
        colsample_bytree=float(tree_cfg.get("colsample_bytree", 0.8)),
        reg_alpha=float(tree_cfg.get("reg_alpha", 1e-3)),
        reg_lambda=float(tree_cfg.get("reg_lambda", 1e-2)),
        random_state=model_seed,
        n_jobs=-1,
        verbosity=-1,
    )
    print(
        f"model_type=lightgbm_regressor train_actions={len(train_rows)} "
        f"val_actions={len(val_rows)} n_features={x_train.shape[1]}",
        flush=True,
    )
    regressor.fit(
        x_train,
        y_train,
        sample_weight=n_train,
        eval_set=[(x_val, y_val)],
        eval_sample_weight=[n_val],
        feature_name=list(FEATURE_NAMES),
        callbacks=[
            lightgbm.early_stopping(rounds),
            lightgbm.log_evaluation(period=50),
        ],
    )
    started = time.perf_counter()
    raw = np.asarray(regressor.predict(x_val), dtype=np.float64)
    inference_seconds = time.perf_counter() - started
    probabilities = np.clip(raw, 0.0, 1.0)
    metrics = selection_metrics(val_rows, raw, probabilities)
    best_iteration = int(getattr(regressor, "best_iteration_", 0) or regressor.n_estimators)
    valid_scores = (getattr(regressor, "best_score_", None) or {}).get("valid_0", {})
    val_l2 = None
    for metric_name in ("l2", "rmse"):
        if metric_name in valid_scores:
            val_l2 = float(valid_scores[metric_name])
            break
    model_path = artifact_path(cfg, "lightgbm_regressor.txt")
    sidecar_path = artifact_path(cfg, "lightgbm_regressor.json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    regressor.booster_.save_model(str(model_path))
    best = {
        "val_l2": val_l2,
        "validation_top1_success_rate": metrics["validation_top1_success_rate"],
        "binomial_nll": metrics["binomial_nll"],
        "brier": metrics["brier"],
        "calibration_bins": metrics["calibration_bins"],
        "largest_fragment_agreement": metrics["largest_fragment_agreement"],
        "success_by_n_actions": metrics["success_by_n_actions"],
        "inference_seconds": inference_seconds,
        "inference_seconds_per_action": inference_seconds / max(len(val_rows), 1),
        "best_iteration": best_iteration,
    }
    write_json(
        sidecar_path,
        {
            "model_type": "lightgbm_regressor",
            "model_path": model_path.name,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "feature_kinds": list(FEATURE_KINDS),
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "scaled_features": False,
            "probability_clip": [0.0, 1.0],
            "selection_uses_unclipped_scores": True,
            "hyperparameters": dict(tree_cfg),
            "n_trees": int(regressor.booster_.num_trees()),
            "model_bytes": model_path.stat().st_size,
            "best_iteration": best_iteration,
            "model_seed": model_seed,
            "best_validation": best,
            **_manifest_hashes(cfg),
        },
    )
    write_json(
        artifact_path(cfg, "lightgbm_regressor_validation_report.json"),
        {
            "model_type": "lightgbm_regressor",
            "early_stopping_metric": "valid_0_l2",
            "best": best,
            "n_train_actions": len(train_rows),
            "n_val_actions": len(val_rows),
            "checkpoint": str(model_path),
            "sidecar": str(sidecar_path),
            "opened_test_manifest": False,
            "model_seed": model_seed,
        },
    )
    print(
        json.dumps(
            {
                "best": _public_metrics(best),
                "best_iteration": best_iteration,
                "checkpoint": str(model_path),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_v2.yaml")
    parser.add_argument(
        "--model",
        choices=("shallow_mlp", "lightgbm_regressor", "both"),
        default="both",
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    model_seed = int(cfg["split"]["manifest_seed"])
    train_rows = load_feature_table("train", cfg)
    val_rows = load_feature_table("validation", cfg)
    if not train_rows or not val_rows:
        raise SystemExit("Missing train or validation action rows")
    if args.model in {"shallow_mlp", "both"}:
        train_shallow_mlp(
            cfg,
            train_rows,
            val_rows,
            model_seed=model_seed,
            max_epochs=args.max_epochs,
        )
    if args.model in {"lightgbm_regressor", "both"}:
        train_lightgbm_regressor(cfg, train_rows, val_rows, model_seed=model_seed)


if __name__ == "__main__":
    main()
