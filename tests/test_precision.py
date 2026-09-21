"""Precision helper unit tests (no GPU required for logic)."""

from __future__ import annotations

from rollout_policy.precision import parse_bool_flag, resolve_precision


def test_parse_bool_flag():
    assert parse_bool_flag("true") is True
    assert parse_bool_flag("false") is False
    assert parse_bool_flag(None, default=True) is True


def test_resolve_precision_fp32_when_disabled(monkeypatch):
    import rollout_policy.precision as prec

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def get_device_capability(_=0):
                return (8, 0)

            @staticmethod
            def is_bf16_supported():
                return True

    monkeypatch.setattr(prec, "resolve_precision", resolve_precision)
    # Import torch inside resolve_precision — patch sys.modules style via monkeypatch on import.
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_precision(prefer_bf16=False) == "fp32"


def test_resolve_precision_fp16_on_pascal(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _=0: (6, 1))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert resolve_precision(prefer_bf16=True) == "fp16"


def test_resolve_precision_bf16_on_ampere(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _=0: (8, 0))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert resolve_precision(prefer_bf16=True) == "bf16"
