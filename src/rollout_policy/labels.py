"""Action-level labels. Schema v2 adds binomial success counts beside reward means."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from rollout_policy.scoring import completion_success

LABEL_SCHEMA_VERSION = 2


def aggregate_action_labels(records: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Group completions by parent and action.

    Keeps the original mean-reward fields and adds the per-completion success
    counts used by the V2 binomial objective. Rows in each group are ordered by
    ``candidate_index``.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(row["parent_id"], row["action_id"])].append(row)

    reward_cfg = cfg.get("reward", {})
    constraints = cfg.get("constraints", {})
    success_qed_delta = float(reward_cfg.get("success_qed_delta", 0.05))
    min_tanimoto = float(constraints.get("min_tanimoto_to_parent", 0.50))

    labels: list[dict[str, Any]] = []
    for (parent_id, action_id), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: int(r["candidate_index"]))
        rewards = [float(r["reward"]) for r in rows]
        success_flags = [
            int(
                completion_success(
                    r,
                    success_qed_delta=success_qed_delta,
                    min_tanimoto=min_tanimoto,
                    constraints=constraints,
                )
            )
            for r in rows
        ]
        n_trials = len(success_flags)
        n_successes = int(sum(success_flags))
        mean = sum(rewards) / n_trials
        if n_trials > 1:
            variance = sum((x - mean) ** 2 for x in rewards) / (n_trials - 1)
        else:
            variance = 0.0
        labels.append(
            {
                "parent_id": parent_id,
                "action_id": action_id,
                "core_id": rows[0]["core_id"],
                "n_rollouts": n_trials,
                "rewards": rewards,
                "rhat": float(mean),
                "reward_variance": float(variance),
                "n_valid": sum(bool(r.get("valid")) for r in rows),
                "n_exact_core": sum(bool(r.get("exact_core")) for r in rows),
                "success_flags": success_flags,
                "n_trials": n_trials,
                "n_successes": n_successes,
                "p_success_observed": n_successes / max(n_trials, 1),
                "label_schema_version": LABEL_SCHEMA_VERSION,
            }
        )
    return labels
