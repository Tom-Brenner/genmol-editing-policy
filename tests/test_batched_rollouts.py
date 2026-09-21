"""Batched same-parent+action rollouts keep independent recorded seeds."""

from __future__ import annotations

from pathlib import Path

import torch

from rollout_policy.dry_run import MockFrozenGenMol, action_from_dict, load_fixture_parents
from rollout_policy.generate import generate_action_rollouts, seed_for
from rollout_policy.genmol_adapter import generator_for_seed
from rollout_policy.io import load_experiment_config, load_jsonl


def test_generators_are_independent_and_reproducible():
    g0 = generator_for_seed(11, "cpu")
    g1 = generator_for_seed(12, "cpu")
    a = torch.rand(16, generator=g0)
    b = torch.rand(16, generator=g1)
    assert not torch.equal(a, b)
    g0_again = generator_for_seed(11, "cpu")
    assert torch.equal(a, torch.rand(16, generator=g0_again))


def test_seed_for_differs_by_candidate_index():
    s0 = seed_for("probe", "CHEMBL1", "a0", 0, 20260919)
    s1 = seed_for("probe", "CHEMBL1", "a0", 1, 20260919)
    assert s0 != s1


def test_generate_action_rollouts_batches_independent_seeds(tmp_path: Path):
    cfg = load_experiment_config()
    fixtures = load_fixture_parents(n=1, require_safe_alignment=False, min_actions=2, max_actions=4)
    assert fixtures
    parent = fixtures[0]
    action = action_from_dict(parent["actions"][0])

    class Spy(MockFrozenGenMol):
        def __init__(self):
            super().__init__(fail_rate=0.0)
            self.span_calls: list[list[int]] = []

        def complete_spans(self, smiles, safe_fragment_index, seeds, **kwargs):
            self.span_calls.append([int(s) for s in seeds])
            return super().complete_spans(smiles, safe_fragment_index, seeds, **kwargs)

    spy = Spy()
    out = tmp_path / "labels.jsonl"
    rows = generate_action_rollouts(
        generator=spy,
        parent_id=parent["parent_id"],
        parent_smiles=parent["smiles"],
        core_id=parent["core_id"],
        action=action,
        n_rollouts=4,
        phase="test_batch",
        split="train",
        output_path=str(out),
        experiment=cfg,
        base_seed=20260919,
    )
    assert len(rows) == 4
    assert len(spy.span_calls) == 1
    assert len(spy.span_calls[0]) == 4
    assert len(set(spy.span_calls[0])) == 4
    observed = [r["observed_seed"] for r in rows]
    requested = [r["requested_seed"] for r in rows]
    assert observed == requested == spy.span_calls[0]
    assert all(r["rollout_batch_size"] == 4 for r in rows)

    # Resume skips done keys and batches only the remainder.
    spy.span_calls.clear()
    more = generate_action_rollouts(
        generator=spy,
        parent_id=parent["parent_id"],
        parent_smiles=parent["smiles"],
        core_id=parent["core_id"],
        action=action,
        n_rollouts=6,
        phase="test_batch",
        split="train",
        output_path=str(out),
        experiment=cfg,
        base_seed=20260919,
    )
    assert len(more) == 2
    assert spy.span_calls[0] == [r["requested_seed"] for r in more]
    assert len(load_jsonl(out)) == 6
