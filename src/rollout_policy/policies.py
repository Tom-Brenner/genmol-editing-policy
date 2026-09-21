"""Site-selection policies: random, largest-fragment, learned value."""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from rollout_policy.eligibility import EligibleAction


def _stable_seed(*parts: object) -> int:
    blob = "::".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(blob).hexdigest()[:8], 16)


def select_random(actions: Sequence[EligibleAction], *, parent_id: str, seed: int) -> EligibleAction:
    if not actions:
        raise ValueError("no eligible actions")
    rng = np.random.default_rng(_stable_seed("random", parent_id, seed))
    idx = int(rng.integers(0, len(actions)))
    return actions[idx]


def select_largest_fragment(actions: Sequence[EligibleAction]) -> EligibleAction:
    if not actions:
        raise ValueError("no eligible actions")
    # Max heavy atoms; canonical_order_key already sorts by (-heavy, ...).
    return sorted(
        actions,
        key=lambda a: (-a.fragment_heavy_atoms, a.canonical_order_key, a.action_id),
    )[0]


def select_learned(
    actions: Sequence[EligibleAction],
    predicted_values: Sequence[float],
) -> EligibleAction:
    if not actions:
        raise ValueError("no eligible actions")
    if len(actions) != len(predicted_values):
        raise ValueError("actions/values length mismatch")
    paired = list(zip(actions, predicted_values))
    paired.sort(key=lambda t: (-t[1], t[0].canonical_order_key, t[0].action_id))
    return paired[0][0]


def select_action(
    policy: str,
    actions: Sequence[EligibleAction],
    *,
    parent_id: str,
    seed: int,
    predicted_values: Sequence[float] | None = None,
) -> EligibleAction:
    if policy == "random":
        return select_random(actions, parent_id=parent_id, seed=seed)
    if policy in {"largest_fragment", "largest"}:
        return select_largest_fragment(actions)
    if policy in {"learned", "learned_rollout_value"}:
        if predicted_values is None:
            raise ValueError("learned policy requires predicted_values")
        return select_learned(actions, predicted_values)
    raise ValueError(f"unknown policy: {policy}")
