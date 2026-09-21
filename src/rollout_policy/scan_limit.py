"""Stop rules for the ChEMBL parent scan."""

from __future__ import annotations


def split_eligible_limit(cfg: dict) -> int:
    """Number of eligible parents the configured split asks for."""
    split = cfg["split"]
    total = int(split["n_train"]) + int(split["n_validation"]) + int(split["n_test"])
    if total < 1:
        raise ValueError("split total must be positive")
    return total


def should_stop_scan(n_eligible: int, limit: int) -> bool:
    """True once the scan has collected enough eligible parents."""
    if n_eligible < 0:
        raise ValueError("n_eligible must be non-negative")
    if limit < 1:
        raise ValueError("limit must be positive")
    return n_eligible >= limit
