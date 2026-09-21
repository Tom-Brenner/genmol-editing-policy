"""The action-count scan stops once the split has enough eligible parents."""

from __future__ import annotations

from rollout_policy.scan_limit import should_stop_scan, split_eligible_limit


def test_v2_split_limit_is_40000():
    cfg = {"split": {"n_train": 30000, "n_validation": 5000, "n_test": 5000}}
    assert split_eligible_limit(cfg) == 40000


def test_scan_stops_at_40000_eligible_parents():
    assert should_stop_scan(39999, 40000) is False
    assert should_stop_scan(40000, 40000) is True
    assert should_stop_scan(40001, 40000) is True
