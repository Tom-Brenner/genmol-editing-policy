#!/usr/bin/env python3
"""Train the rollout-value MLP on train actions; select on validation only."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rollout_policy.eligibility import core_mol_from_parent
from rollout_policy.features import FEATURE_NAMES, action_features
from rollout_policy.io import (
    artifact_path,
    load_experiment_config,
    load_jsonl,
    resolve_path,
    write_json,
)
from rollout_policy.model import ValueMLP, weighted_huber_loss


def _load_action_table(split: str, cfg: dict) -> list[dict]:
    manifest = {
        r["parent_id"]: r
        for r in load_jsonl(resolve_path(cfg["paths"]["manifests_dir"]) / f"{split}.jsonl")
    }
    labels = load_jsonl(
        resolve_path(cfg["paths"]["results_dir"]) / "labels" / f"{split}_action_rhat.jsonl"
    )
    rows = []
    for lab in labels:
        parent = manifest[lab["parent_id"]]
        action = next(a for a in parent["actions"] if a["action_id"] == lab["action_id"])
        mol = Chem.MolFromSmiles(parent["smiles"])
        core = core_mol_from_parent(mol)
        frag = Chem.MolFromSmiles(action["fragment_smiles"])
        if mol is None or frag is None:
            continue
        feats = action_features(
            mol,
            core,
            frag,
            action["bond_type"],
            bool(action["is_ring"]),
            bool(action["is_terminal"]),
        )
        rows.append(
            {
                "parent_id": lab["parent_id"],
                "action_id": lab["action_id"],
                "features": feats,
                "rhat": float(lab["rhat"]),
                "reward_variance": float(lab.get("reward_variance", 0.0)),
                "n_actions_on_parent": int(parent["n_actions"]),
            }
        )
    return rows


def _top1_mean_value(rows: list[dict], preds: np.ndarray) -> float:
    by_parent: dict[str, list[tuple[float, float]]] = {}
    for row, pred in zip(rows, preds):
        by_parent.setdefault(row["parent_id"], []).append((pred, row["rhat"]))
    chosen = []
    for items in by_parent.values():
        items.sort(key=lambda t: -t[0])
        chosen.append(items[0][1])
    return float(np.mean(chosen)) if chosen else float("nan")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    vm = cfg["value_model"]

    # Never touch test manifests here.
    train_rows = _load_action_table("train", cfg)
    val_rows = _load_action_table("validation", cfg)
    if not train_rows or not val_rows:
        raise SystemExit("Missing train/validation action labels")

    x_train = np.stack([r["features"] for r in train_rows]).astype(np.float32)
    y_train = np.asarray([r["rhat"] for r in train_rows], dtype=np.float32)
    var_train = np.asarray([r["reward_variance"] for r in train_rows], dtype=np.float32)
    w_train = 1.0 / (float(vm["variance_eps"]) + var_train)

    x_val = np.stack([r["features"] for r in val_rows]).astype(np.float32)
    y_val = np.asarray([r["rhat"] for r in val_rows], dtype=np.float32)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train).astype(np.float32)
    x_val_s = scaler.transform(x_val).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_every = max(1, int(vm.get("log_every_n_steps", 100)))
    model: ValueMLP | None = None
    x_val_t: torch.Tensor | None = None
    best = {"val_mse": float("inf"), "epoch": -1, "val_top1": float("nan")}
    history: list[dict] = []
    interrupted = False

    try:
        model = ValueMLP(
            n_features=x_train_s.shape[1], hidden_sizes=tuple(vm["hidden_sizes"])
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=float(vm["learning_rate"]))
        x_val_t = torch.from_numpy(x_val_s).to(device)

        ds = TensorDataset(
            torch.from_numpy(x_train_s),
            torch.from_numpy(y_train),
            torch.from_numpy(w_train.astype(np.float32)),
        )
        loader = DataLoader(ds, batch_size=int(vm["batch_size"]), shuffle=True)

        patience = int(vm["patience"])
        stale = 0
        global_step = 0
        window_loss = 0.0
        window_n = 0

        for epoch in range(int(vm["max_epochs"])):
            model.train()
            total_loss = 0.0
            n_batches = 0
            for xb, yb, wb in loader:
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = weighted_huber_loss(pred, yb, wb, delta=float(vm["huber_delta"]))
                loss.backward()
                opt.step()
                batch_loss = float(loss.item())
                total_loss += batch_loss
                n_batches += 1
                global_step += 1
                window_loss += batch_loss
                window_n += 1
                del pred, loss, xb, yb, wb

                if global_step % log_every == 0:
                    mean_loss = window_loss / max(window_n, 1)
                    last_val = history[-1]["val_mse"] if history else None
                    val_part = f" val_mse={last_val:.6f}" if last_val is not None else ""
                    print(
                        f"step={global_step} epoch={epoch} "
                        f"train_huber={mean_loss:.6f}{val_part}",
                        flush=True,
                    )
                    window_loss = 0.0
                    window_n = 0
                    _free_cuda_memory()

            model.eval()
            with torch.no_grad():
                val_pred = model(x_val_t).detach().cpu().numpy()
            val_mse = float(np.mean((val_pred - y_val) ** 2))
            val_top1 = _top1_mean_value(val_rows, val_pred)
            del val_pred
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": total_loss / max(n_batches, 1),
                    "val_mse": val_mse,
                    "val_top1_action_value": val_top1,
                    "step": global_step,
                }
            )
            improved = val_mse < best["val_mse"] - 1e-8
            if improved:
                best = {"val_mse": val_mse, "epoch": epoch, "val_top1": val_top1}
                stale = 0
                ckpt_path = resolve_path(cfg["paths"]["artifacts_dir"]) / "value_model.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "n_features": x_train_s.shape[1],
                        "hidden_sizes": list(vm["hidden_sizes"]),
                        "feature_names": list(FEATURE_NAMES),
                        "scaler_mean": scaler.mean_.tolist(),
                        "scaler_scale": scaler.scale_.tolist(),
                        "hyperparameters": vm,
                        "best": best,
                    },
                    ckpt_path,
                )
            else:
                stale += 1
                if stale >= patience:
                    break
            _free_cuda_memory()

        # Random / largest baselines on validation labels for the selection report.
        val_by_parent: dict[str, list[dict]] = {}
        for row in val_rows:
            val_by_parent.setdefault(row["parent_id"], []).append(row)

        def policy_mean(select_fn) -> float:
            vals = []
            for rows in val_by_parent.values():
                chosen = select_fn(rows)
                vals.append(chosen["rhat"])
            return float(np.mean(vals))

        rng = np.random.default_rng(0)

        def random_select(rows):
            return rows[int(rng.integers(0, len(rows)))]

        def largest_select(rows):
            # Approximate via fragment heavy-atom feature index 6.
            return max(rows, key=lambda r: (r["features"][6], r["action_id"]))

        report = {
            "best": best,
            "history": history,
            "n_train_actions": len(train_rows),
            "n_val_actions": len(val_rows),
            "feature_names": list(FEATURE_NAMES),
            "validation_policy_mean_rhat": {
                "random": policy_mean(random_select),
                "largest_fragment": policy_mean(largest_select),
                "learned": best["val_top1"],
            },
        }
        write_json(artifact_path(cfg, "value_model_validation_report.json"), report)
        print(json.dumps({"best": best, "policies": report["validation_policy_mean_rhat"]}, indent=2))
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted (Ctrl+C); releasing GPU and exiting.", flush=True)
    finally:
        if x_val_t is not None:
            del x_val_t
        _release_gpu(model)
        if model is not None:
            del model
        _free_cuda_memory()

    if interrupted:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
