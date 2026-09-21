"""Endpoint alignment: per-completion success, action counts, binomial likelihood."""

from __future__ import annotations

import torch

from rollout_policy.labels import aggregate_action_labels
from rollout_policy.model import binomial_nll
from rollout_policy.scoring import completion_success, parent_success


def experiment_cfg() -> dict:
    return {
        "reward": {"success_qed_delta": 0.05},
        "constraints": {
            "min_tanimoto_to_parent": 0.50,
            "max_heavy_atom_delta_fraction": 1.0 / 3.0,
        },
    }


def _row(**overrides) -> dict:
    base = {
        "parent_id": "p1",
        "action_id": "a1",
        "core_id": "c1",
        "candidate_index": 0,
        "reward": 0.0,
        "valid": True,
        "exact_core": True,
        "tanimoto": 0.9,
        "heavy_atom_delta": 0,
        "parent_n_heavy_atoms": 15,
        "qed_delta_raw": 0.06,
    }
    base.update(overrides)
    return base


def successful_row() -> dict:
    return _row(reward=0.6, qed_delta_raw=0.06)


def low_qed_row() -> dict:
    return _row(reward=0.0, qed_delta_raw=0.01)


def invalid_row() -> dict:
    return _row(valid=False, reward=0.0, tanimoto=None, heavy_atom_delta=None, qed_delta_raw=None)


def oversized_row() -> dict:
    return _row(reward=0.0, heavy_atom_delta=6, qed_delta_raw=0.2)


def eight_mock_rows() -> list[dict]:
    flags = [successful_row(), low_qed_row(), invalid_row(), oversized_row()]
    flags += [successful_row(), successful_row(), low_qed_row(), successful_row()]
    rows = []
    for index, row in enumerate(flags):
        item = dict(row)
        item["candidate_index"] = index
        rows.append(item)
    return rows


def test_completion_success_matches_single_row_parent_success():
    for row in (successful_row(), low_qed_row(), invalid_row(), oversized_row()):
        assert completion_success(row) == parent_success([row])


def test_action_aggregation_keeps_old_and_new_targets():
    labels = aggregate_action_labels(eight_mock_rows(), experiment_cfg())
    assert len(labels) == 1
    assert labels[0]["n_trials"] == 8
    assert labels[0]["n_successes"] == sum(labels[0]["success_flags"])
    assert labels[0]["p_success_observed"] == labels[0]["n_successes"] / 8
    assert "rhat" in labels[0]
    assert labels[0]["label_schema_version"] == 2
    assert labels[0]["success_flags"] == [1, 0, 0, 0, 1, 1, 0, 1]


def test_binomial_nll_prefers_correct_logit():
    good = binomial_nll(torch.tensor([4.0]), torch.tensor([8.0]), torch.tensor([8.0]))
    bad = binomial_nll(torch.tensor([-4.0]), torch.tensor([8.0]), torch.tensor([8.0]))
    assert good < bad


def test_binomial_nll_rejects_impossible_counts():
    try:
        binomial_nll(torch.tensor([0.0]), torch.tensor([3.0]), torch.tensor([2.0]))
    except ValueError as exc:
        assert "binomial counts" in str(exc)
    else:
        raise AssertionError("impossible counts must fail")
