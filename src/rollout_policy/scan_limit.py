"""Stop rules for the ChEMBL parent scan."""

from __future__ import annotations

import os
import signal
import threading

# Backstop so a missed Ctrl+C cannot walk the rest of ChEMBL.
DEFAULT_ROW_CAP = 120_000

_STOP = threading.Event()


def interrupt_requested() -> bool:
    return _STOP.is_set()


def _on_sigint(signum: int, frame) -> None:
    _STOP.set()
    # RDKit replaces this handler during a substructure search. A second
    # Ctrl+C uses the default action and kills the process.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    raise KeyboardInterrupt


def _on_sigterm(signum: int, frame) -> None:
    _STOP.set()
    raise KeyboardInterrupt


def _arm_signals() -> None:
    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.siginterrupt(signal.SIGINT, True)


def _watch_rdkit_stderr() -> None:
    """RDKit handles Ctrl+C inside C++ and only prints that the search was interrupted.

    Copy stderr through, and turn that log line into SIGTERM so the scan exits
    when the current call returns to Python.
    """
    read_fd, write_fd = os.pipe()
    saved_err = os.dup(2)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    notified = threading.Event()

    def _reader() -> None:
        pending = b""
        while True:
            try:
                chunk = os.read(read_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            try:
                os.write(saved_err, chunk)
            except OSError:
                pass
            pending = (pending + chunk)[-512:]
            text = pending.decode("utf-8", errors="replace")
            if rdkit_log_requests_stop(text) and not notified.is_set():
                notified.set()
                _STOP.set()
                os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_reader, name="rdkit-sigint-watch", daemon=True).start()


def install_ctrl_c_kill() -> None:
    """Make one Ctrl+C stop a scan even when RDKit catches SIGINT."""
    _arm_signals()
    _watch_rdkit_stderr()


def row_cap_reached(scanned: int, cap: int = DEFAULT_ROW_CAP) -> bool:
    if scanned < 0:
        raise ValueError("scanned must be non-negative")
    if cap < 1:
        raise ValueError("cap must be positive")
    return scanned >= cap


def split_eligible_limit(cfg: dict) -> int:
    """Number of eligible parents the configured split asks for."""
    split = cfg["split"]
    total = int(split["n_train"]) + int(split["n_validation"]) + int(split["n_test"])
    if total < 1:
        raise ValueError("split total must be positive")
    return total


def rdkit_log_requests_stop(message: str) -> bool:
    """RDKit prints this when Ctrl+C aborts a substructure search and returns."""
    return "interrupted" in message.lower()


def should_stop_scan(n_eligible: int, limit: int) -> bool:
    """True once the scan has collected enough eligible parents."""
    if n_eligible < 0:
        raise ValueError("n_eligible must be non-negative")
    if limit < 1:
        raise ValueError("limit must be positive")
    return n_eligible >= limit
