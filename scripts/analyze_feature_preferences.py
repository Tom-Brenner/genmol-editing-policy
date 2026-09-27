#!/usr/bin/env python3
"""Analyze learned site preferences and grouped feature reliance.

The first operation is a strict preflight. No model is loaded and no analysis
output is written until every required input and Python dependency is present.

Outputs are written under results_v2/feature_analysis/ by default:

* preflight.json
* model_summary.json
* grouped_permutation.csv
* feature_preferences.csv
* model_agreement.csv
* lightgbm_feature_importance.csv
* lightgbm_tree_shap.csv

Grouped permutation is evaluated on validation top-1 observed success (k/n).
Action-varying feature groups are permuted among eligible actions of the same
parent. Parent-context features are permuted between parents while remaining
constant across each recipient parent's actions.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

REQUIRED_MODULES = (
    "numpy",
    "yaml",
    "rdkit",
    "torch",
    "torch_geometric",
    "lightgbm",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a mapping")
    return payload


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def required_inputs(config_path: Path) -> dict[str, Path]:
    """Return every input required for the complete analysis."""
    cfg = _load_yaml(config_path)
    manifests = _resolve(ROOT, cfg["paths"]["manifests_dir"])
    results = _resolve(ROOT, cfg["paths"]["results_dir"])
    artifacts = _resolve(ROOT, cfg["paths"]["artifacts_dir"])
    required = {
        "validation_manifest": manifests / "validation.jsonl",
        "test_manifest": manifests / "test.jsonl",
        "seed_manifest": manifests / "seeds.json",
        "validation_action_labels": results / "labels" / "validation_action_rhat.jsonl",
        "gat_checkpoint": artifacts / "best.pt",
        "shallow_mlp_checkpoint": artifacts / "shallow_mlp.pt",
        "lightgbm_model": artifacts / "lightgbm_regressor.txt",
        "lightgbm_sidecar": artifacts / "lightgbm_regressor.json",
        "gat_validation_report": artifacts / "gatv2_validation_report.json",
        "shallow_mlp_validation_report": artifacts / "shallow_mlp_validation_report.json",
        "lightgbm_validation_report": artifacts / "lightgbm_regressor_validation_report.json",
        "phase3_metrics": results / "test" / "metrics.json",
        "phase3_selections": results / "test" / "selections.json",
        "phase4_metrics": results / "confirmation" / "metrics.json",
    }
    for model in ("shallow_mlp", "lightgbm_regressor"):
        for schedule in ("test", "confirmation"):
            prefix = f"{model}_{schedule}"
            directory = results / "ablation" / model
            required[f"{prefix}_metrics"] = directory / f"{schedule}_metrics.json"
            required[f"{prefix}_selections"] = directory / f"{schedule}_selections.json"
    return required


def _validate_json(path: Path) -> None:
    with path.open(encoding="utf-8") as handle:
        json.load(handle)


def _validate_jsonl(path: Path) -> None:
    with path.open(encoding="utf-8") as handle:
        first = next((line for line in handle if line.strip()), None)
    if first is None:
        raise ValueError("contains no records")
    payload = json.loads(first)
    if not isinstance(payload, dict):
        raise ValueError("first record is not an object")


def run_preflight(config_path: Path) -> dict[str, Any]:
    """Fail once with a complete list of missing/bad dependencies and files."""
    problems: list[str] = []
    modules: dict[str, str] = {}
    for name in REQUIRED_MODULES:
        spec = importlib.util.find_spec(name)
        modules[name] = "available" if spec is not None else "missing"
        if spec is None:
            problems.append(f"missing Python module: {name}")

    try:
        inputs = required_inputs(config_path)
    except Exception as exc:
        raise SystemExit(f"Preflight failed while reading {config_path}: {exc}") from exc

    files: dict[str, dict[str, Any]] = {}
    for label, path in inputs.items():
        status: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        files[label] = status
        if not path.is_file():
            problems.append(f"missing file [{label}]: {path}")
            continue
        status["bytes"] = path.stat().st_size
        if path.stat().st_size == 0:
            problems.append(f"empty file [{label}]: {path}")
            continue
        try:
            if path.suffix == ".json":
                _validate_json(path)
            elif path.suffix == ".jsonl":
                _validate_jsonl(path)
            status["parse_check"] = "ok"
        except Exception as exc:
            status["parse_check"] = f"failed: {exc}"
            problems.append(f"unreadable file [{label}]: {path}: {exc}")

    report = {
        "ok": not problems,
        "config": str(config_path),
        "modules": modules,
        "files": files,
        "problems": problems,
    }
    if problems:
        message = "\n".join(f"  - {problem}" for problem in problems)
        raise SystemExit(f"Feature-analysis preflight failed:\n{message}")
    return report


@dataclass
class SplitData:
    name: str
    parents: list[dict[str, Any]]
    features: Any
    parent_slices: list[slice]
    action_ids: list[str]
    canonical_ranks: list[int]
    fragment_heavy_atoms: list[int]
    observed_rates: Any | None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _label_lookup(path: Path) -> dict[tuple[str, str], float]:
    labels: dict[tuple[str, str], float] = {}
    for row in _load_jsonl(path):
        key = (str(row["parent_id"]), str(row["action_id"]))
        if key in labels:
            raise ValueError(f"duplicate action label: {key}")
        trials = float(row["n_trials"])
        successes = float(row["n_successes"])
        if trials <= 0 or successes < 0 or successes > trials:
            raise ValueError(f"invalid binomial label for {key}")
        labels[key] = successes / trials
    return labels


def build_split(
    name: str,
    manifest_path: Path,
    *,
    labels_path: Path | None,
    max_parents: int | None,
) -> SplitData:
    import numpy as np
    from rdkit import Chem

    from rollout_policy.datasets import canonical_ranks
    from rollout_policy.dry_run import action_from_dict
    from rollout_policy.eligibility import core_mol_from_parent
    from rollout_policy.features import FEATURE_NAMES, action_features

    parents = _load_jsonl(manifest_path)
    if max_parents is not None:
        parents = parents[:max_parents]
    labels = _label_lookup(labels_path) if labels_path is not None else None
    feature_rows = []
    parent_slices: list[slice] = []
    action_ids: list[str] = []
    ranks: list[int] = []
    fragment_heavy_atoms: list[int] = []
    observed: list[float] = []

    for parent in parents:
        start = len(feature_rows)
        mol = Chem.MolFromSmiles(parent["smiles"])
        if mol is None:
            raise ValueError(f"invalid parent SMILES: {parent['parent_id']}")
        core = core_mol_from_parent(mol)
        rank_by_id = canonical_ranks(parent["actions"])
        for action_row in parent["actions"]:
            action = action_from_dict(action_row)
            feature_rows.append(action_features(mol, core, action))
            action_ids.append(action.action_id)
            ranks.append(rank_by_id[action.action_id])
            fragment_heavy_atoms.append(int(action_row["fragment_heavy_atoms"]))
            if labels is not None:
                key = (str(parent["parent_id"]), str(action.action_id))
                if key not in labels:
                    raise KeyError(f"missing validation label for {key}")
                observed.append(labels[key])
        parent_slices.append(slice(start, len(feature_rows)))

    matrix = np.stack(feature_rows).astype(np.float32)
    if matrix.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError("computed feature width does not match FEATURE_NAMES")
    return SplitData(
        name=name,
        parents=parents,
        features=matrix,
        parent_slices=parent_slices,
        action_ids=action_ids,
        canonical_ranks=ranks,
        fragment_heavy_atoms=fragment_heavy_atoms,
        observed_rates=np.asarray(observed, dtype=np.float64) if labels is not None else None,
    )


def feature_groups() -> dict[str, list[int]]:
    from rollout_policy.features import FEATURE_NAMES

    groups: dict[str, list[int]] = {
        "parent_context": [],
        "fragment_size": [],
        "fragment_physchem": [],
        "fragment_flags": [],
        "attachment_atoms": [],
        "core_attachment_morgan": [],
        "fragment_attachment_morgan": [],
        "cut_bond": [],
    }
    size_names = {
        "fragment_molwt",
        "fragment_heavy_atoms",
        "fragment_heavy_atom_fraction",
        "fragment_molwt_fraction",
        "fragment_tpsa_fraction",
    }
    for index, name in enumerate(FEATURE_NAMES):
        if name.startswith("parent_") or name == "core_heavy_atoms":
            group = "parent_context"
        elif name in size_names:
            group = "fragment_size"
        elif name in {"is_ring_fragment", "is_terminal_fragment"}:
            group = "fragment_flags"
        elif name.startswith("core_attachment_morgan_"):
            group = "core_attachment_morgan"
        elif name.startswith("fragment_attachment_morgan_"):
            group = "fragment_attachment_morgan"
        elif name.startswith("core_attachment_") or name.startswith("fragment_attachment_"):
            group = "attachment_atoms"
        elif name.startswith("bond_"):
            group = "cut_bond"
        elif name.startswith("fragment_"):
            group = "fragment_physchem"
        else:
            raise RuntimeError(f"feature is not assigned to a group: {name}")
        groups[group].append(index)
    assigned = [index for indexes in groups.values() for index in indexes]
    if sorted(assigned) != list(range(len(FEATURE_NAMES))) or len(assigned) != len(set(assigned)):
        raise RuntimeError("feature groups are not an exact partition")
    return groups


def _chosen_indexes(data: SplitData, scores: Any) -> list[int]:
    from rollout_policy.policies import select_scored_index

    chosen = []
    for parent_slice in data.parent_slices:
        indexes = list(range(parent_slice.start, parent_slice.stop))
        local = select_scored_index(
            [float(scores[index]) for index in indexes],
            [data.canonical_ranks[index] for index in indexes],
            [data.action_ids[index] for index in indexes],
        )
        chosen.append(indexes[local])
    return chosen


def _largest_indexes(data: SplitData) -> list[int]:
    chosen = []
    for parent_slice in data.parent_slices:
        indexes = list(range(parent_slice.start, parent_slice.stop))
        chosen.append(
            min(
                indexes,
                key=lambda index: (
                    -data.fragment_heavy_atoms[index],
                    data.canonical_ranks[index],
                    data.action_ids[index],
                ),
            )
        )
    return chosen


def _validation_metrics(data: SplitData, scores: Any) -> dict[str, float]:
    import numpy as np

    if data.observed_rates is None:
        raise ValueError("validation metrics require observed action rates")
    selected = _chosen_indexes(data, scores)
    largest = _largest_indexes(data)
    chosen_rates = np.asarray([data.observed_rates[index] for index in selected])
    oracle_rates = np.asarray(
        [
            float(np.max(data.observed_rates[parent_slice]))
            for parent_slice in data.parent_slices
        ]
    )
    return {
        "top1_observed_success": float(chosen_rates.mean()),
        "oracle_regret": float((oracle_rates - chosen_rates).mean()),
        "largest_fragment_agreement": float(
            np.mean([left == right for left, right in zip(selected, largest)])
        ),
    }


def _score_shallow(predictor: Any, matrix: Any, batch_size: int = 8192) -> Any:
    import numpy as np
    import torch

    scaled = predictor.scaler.transform(matrix)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(scaled), batch_size):
            batch = torch.from_numpy(scaled[start : start + batch_size]).to(predictor.device)
            logits = predictor.model(batch).view(-1)
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def _score_lightgbm(predictor: Any, matrix: Any) -> Any:
    import numpy as np

    return np.asarray(
        predictor.booster.predict(matrix, num_iteration=predictor.num_iteration),
        dtype=np.float64,
    )


def _score_gat(predictor: Any, data: SplitData) -> Any:
    import numpy as np

    from rollout_policy.dry_run import action_from_dict

    scores = []
    for parent in data.parents:
        actions = [action_from_dict(row) for row in parent["actions"]]
        scores.extend(float(value) for value in predictor.predict_actions(parent, actions))
    return np.asarray(scores, dtype=np.float64)


def _permuted_matrix(
    data: SplitData,
    columns: list[int],
    *,
    parent_context: bool,
    rng: Any,
) -> Any:
    import numpy as np

    matrix = data.features.copy()
    if parent_context:
        donor_order = rng.permutation(len(data.parent_slices))
        donor_values = [
            data.features[parent_slice.start, columns].copy()
            for parent_slice in data.parent_slices
        ]
        for target_index, parent_slice in enumerate(data.parent_slices):
            matrix[parent_slice, columns] = donor_values[int(donor_order[target_index])]
    else:
        for parent_slice in data.parent_slices:
            indexes = list(range(parent_slice.start, parent_slice.stop))
            donor = rng.permutation(indexes)
            matrix[np.ix_(indexes, columns)] = data.features[donor][:, columns]
    return matrix


def grouped_permutation(
    data: SplitData,
    scorers: dict[str, Callable[[Any], Any]],
    baseline_scores: dict[str, Any],
    *,
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    import numpy as np

    groups = feature_groups()
    rows = []
    for model_name, scorer in scorers.items():
        baseline = _validation_metrics(data, baseline_scores[model_name])
        baseline_chosen = _chosen_indexes(data, baseline_scores[model_name])
        for group_index, (group_name, columns) in enumerate(groups.items()):
            losses = []
            changes = []
            regrets = []
            agreements = []
            for repeat in range(repeats):
                rng = np.random.default_rng(seed + group_index * 100003 + repeat)
                matrix = _permuted_matrix(
                    data,
                    columns,
                    parent_context=group_name == "parent_context",
                    rng=rng,
                )
                scores = scorer(matrix)
                metrics = _validation_metrics(data, scores)
                selected = _chosen_indexes(data, scores)
                losses.append(
                    baseline["top1_observed_success"] - metrics["top1_observed_success"]
                )
                changes.append(
                    float(np.mean([a != b for a, b in zip(baseline_chosen, selected)]))
                )
                regrets.append(metrics["oracle_regret"] - baseline["oracle_regret"])
                agreements.append(metrics["largest_fragment_agreement"])
            rows.append(
                {
                    "model": model_name,
                    "feature_group": group_name,
                    "n_features": len(columns),
                    "permutation_scope": (
                        "between_parents" if group_name == "parent_context" else "within_parent"
                    ),
                    "baseline_top1_success": baseline["top1_observed_success"],
                    "mean_top1_success_loss": float(np.mean(losses)),
                    "sd_top1_success_loss": float(np.std(losses, ddof=1)) if repeats > 1 else 0.0,
                    "mean_oracle_regret_increase": float(np.mean(regrets)),
                    "mean_selection_changed": float(np.mean(changes)),
                    "mean_largest_fragment_agreement": float(np.mean(agreements)),
                    "repeats": repeats,
                }
            )
    return rows


def feature_preference_rows(
    data: SplitData,
    scores_by_model: dict[str, Any],
) -> list[dict[str, Any]]:
    import numpy as np

    from rollout_policy.features import FEATURE_KINDS, FEATURE_NAMES

    rows = []
    for model_name, scores in scores_by_model.items():
        selected = _chosen_indexes(data, scores)
        for feature_index, (feature_name, feature_kind) in enumerate(
            zip(FEATURE_NAMES, FEATURE_KINDS)
        ):
            selected_values = []
            alternative_values = []
            deltas = []
            for chosen, parent_slice in zip(selected, data.parent_slices):
                alternatives = [
                    index
                    for index in range(parent_slice.start, parent_slice.stop)
                    if index != chosen
                ]
                chosen_value = float(data.features[chosen, feature_index])
                alternative_mean = float(np.mean(data.features[alternatives, feature_index]))
                selected_values.append(chosen_value)
                alternative_values.append(alternative_mean)
                deltas.append(chosen_value - alternative_mean)
            delta = np.asarray(deltas, dtype=np.float64)
            rows.append(
                {
                    "split": data.name,
                    "model": model_name,
                    "feature": feature_name,
                    "kind": feature_kind,
                    "selected_mean": float(np.mean(selected_values)),
                    "alternatives_mean": float(np.mean(alternative_values)),
                    "mean_selected_minus_alternatives": float(delta.mean()),
                    "median_selected_minus_alternatives": float(np.median(delta)),
                    "standard_error": float(delta.std(ddof=1) / math.sqrt(len(delta))),
                    "fraction_nonzero_parent_contrasts": float(np.mean(delta != 0)),
                    "n_parents": len(delta),
                }
            )
    return rows


def agreement_rows(data: SplitData, scores_by_model: dict[str, Any]) -> list[dict[str, Any]]:
    import numpy as np

    selected = {
        name: _chosen_indexes(data, scores) for name, scores in scores_by_model.items()
    }
    rows = []
    for left, right in combinations(sorted(selected), 2):
        rows.append(
            {
                "split": data.name,
                "model_a": left,
                "model_b": right,
                "same_action_fraction": float(
                    np.mean([a == b for a, b in zip(selected[left], selected[right])])
                ),
                "n_parents": len(data.parents),
            }
        )
    return rows


def lightgbm_rows(predictor: Any, validation: SplitData) -> tuple[list[dict], list[dict]]:
    import numpy as np

    from rollout_policy.features import FEATURE_NAMES

    booster = predictor.booster
    iteration = predictor.num_iteration or None
    gain = np.asarray(
        booster.feature_importance("gain", iteration=iteration), dtype=np.float64
    )
    split = np.asarray(
        booster.feature_importance("split", iteration=iteration), dtype=np.float64
    )
    gain_total = float(gain.sum()) or 1.0
    split_total = float(split.sum()) or 1.0
    importance = [
        {
            "feature": name,
            "gain": float(gain[index]),
            "gain_fraction": float(gain[index] / gain_total),
            "split_count": int(split[index]),
            "split_fraction": float(split[index] / split_total),
        }
        for index, name in enumerate(FEATURE_NAMES)
    ]

    contributions = np.asarray(
        booster.predict(
            validation.features,
            num_iteration=predictor.num_iteration,
            pred_contrib=True,
        ),
        dtype=np.float64,
    )
    if contributions.shape != (len(validation.features), len(FEATURE_NAMES) + 1):
        raise RuntimeError("unexpected LightGBM contribution matrix shape")
    shap_rows = [
        {
            "feature": name,
            "mean_abs_tree_shap": float(np.mean(np.abs(contributions[:, index]))),
            "mean_tree_shap": float(np.mean(contributions[:, index])),
        }
        for index, name in enumerate(FEATURE_NAMES)
    ]
    return importance, shap_rows


def _archived_selection_map(path: Path, *, policy: str | None = None) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("selections", [])
    selected = {}
    for row in rows:
        if policy is not None and row.get("policy") != policy:
            continue
        selected[str(row["parent_id"])] = str(row["action_id"])
    return selected


def verify_test_selections(
    inputs: dict[str, Path],
    test: SplitData,
    scores_by_model: dict[str, Any],
) -> dict[str, Any]:
    paths = {
        "gatv2": (inputs["phase3_selections"], "learned"),
        "shallow_mlp": (inputs["shallow_mlp_test_selections"], None),
        "lightgbm_regressor": (inputs["lightgbm_regressor_test_selections"], None),
    }
    report = {}
    for model_name, (path, policy) in paths.items():
        archived = _archived_selection_map(path, policy=policy)
        chosen = _chosen_indexes(test, scores_by_model[model_name])
        compared = 0
        mismatches = []
        for parent, index in zip(test.parents, chosen):
            parent_id = str(parent["parent_id"])
            if parent_id not in archived:
                continue
            compared += 1
            recomputed = test.action_ids[index]
            if recomputed != archived[parent_id]:
                mismatches.append(
                    {
                        "parent_id": parent_id,
                        "archived": archived[parent_id],
                        "recomputed": recomputed,
                    }
                )
        report[model_name] = {
            "compared": compared,
            "mismatch_count": len(mismatches),
            "first_mismatches": mismatches[:10],
        }
        if compared != len(test.parents) or mismatches:
            raise RuntimeError(
                f"{model_name} selection verification failed: "
                f"compared={compared}/{len(test.parents)}, mismatches={len(mismatches)}"
            )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_v2.yaml")
    parser.add_argument(
        "--output-dir",
        default="results_v2/feature_analysis",
        help="Analysis-only output directory.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--permutation-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument(
        "--max-parents",
        type=int,
        default=None,
        help="Debug only. Selection archive verification is skipped when set.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check dependencies and all required files, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve(ROOT, args.config)

    # This must remain the first substantive operation.
    preflight = run_preflight(config_path)
    print("Feature-analysis preflight passed.", flush=True)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return

    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    import torch

    from rollout_policy.predictors import (
        load_gat_predictor,
        load_lightgbm_regressor_predictor,
        load_tabular_mlp_predictor,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but CUDA is unavailable")
    if args.permutation_repeats < 1:
        raise SystemExit("--permutation-repeats must be positive")

    inputs = required_inputs(config_path)
    output_dir = _resolve(ROOT, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "preflight.json", preflight)

    validation = build_split(
        "validation",
        inputs["validation_manifest"],
        labels_path=inputs["validation_action_labels"],
        max_parents=args.max_parents,
    )
    test = build_split(
        "test",
        inputs["test_manifest"],
        labels_path=None,
        max_parents=args.max_parents,
    )

    device = torch.device(args.device)
    gat = load_gat_predictor(inputs["gat_checkpoint"], device)
    shallow = load_tabular_mlp_predictor(inputs["shallow_mlp_checkpoint"], device)
    lightgbm = load_lightgbm_regressor_predictor(inputs["lightgbm_model"])

    tabular_scorers: dict[str, Callable[[Any], Any]] = {
        "shallow_mlp": lambda matrix: _score_shallow(shallow, matrix),
        "lightgbm_regressor": lambda matrix: _score_lightgbm(lightgbm, matrix),
    }
    validation_scores = {
        "gatv2": _score_gat(gat, validation),
        "shallow_mlp": tabular_scorers["shallow_mlp"](validation.features),
        "lightgbm_regressor": tabular_scorers["lightgbm_regressor"](validation.features),
    }
    test_scores = {
        "gatv2": _score_gat(gat, test),
        "shallow_mlp": tabular_scorers["shallow_mlp"](test.features),
        "lightgbm_regressor": tabular_scorers["lightgbm_regressor"](test.features),
    }

    verification = (
        {"skipped": "debug subset requested"}
        if args.max_parents is not None
        else verify_test_selections(inputs, test, test_scores)
    )
    summary = {
        "validation": {
            name: _validation_metrics(validation, scores)
            for name, scores in validation_scores.items()
        },
        "selection_archive_verification": verification,
        "n_validation_parents": len(validation.parents),
        "n_validation_actions": len(validation.features),
        "n_test_parents": len(test.parents),
        "n_test_actions": len(test.features),
        "device": args.device,
        "permutation_repeats": args.permutation_repeats,
        "seed": args.seed,
        "interpretation": {
            "permutation": (
                "Positive top-1 success loss means the model relied on that group for "
                "validation action ranking. Correlated groups share credit."
            ),
            "preferences": (
                "Selected-minus-alternative contrasts describe association, not a "
                "causal effect of changing a chemical descriptor."
            ),
        },
    }
    _write_json(output_dir / "model_summary.json", summary)

    permutation = grouped_permutation(
        validation,
        tabular_scorers,
        {name: validation_scores[name] for name in tabular_scorers},
        repeats=args.permutation_repeats,
        seed=args.seed,
    )
    _write_csv(
        output_dir / "grouped_permutation.csv",
        permutation,
        list(permutation[0]),
    )

    preferences = feature_preference_rows(validation, validation_scores)
    preferences.extend(feature_preference_rows(test, test_scores))
    _write_csv(
        output_dir / "feature_preferences.csv",
        preferences,
        list(preferences[0]),
    )

    agreements = agreement_rows(validation, validation_scores)
    agreements.extend(agreement_rows(test, test_scores))
    _write_csv(output_dir / "model_agreement.csv", agreements, list(agreements[0]))

    importance, tree_shap = lightgbm_rows(lightgbm, validation)
    _write_csv(
        output_dir / "lightgbm_feature_importance.csv",
        importance,
        list(importance[0]),
    )
    _write_csv(
        output_dir / "lightgbm_tree_shap.csv",
        tree_shap,
        list(tree_shap[0]),
    )
    print(f"Analysis complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
