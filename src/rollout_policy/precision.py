"""Precision selection for GenMol inference.

--bf16 true means \"prefer bf16\". On GPUs without bf16 (e.g. GTX 1080 Ti /
Pascal), fall back to fp16. On CPU, use float32.
"""

from __future__ import annotations

from typing import Literal

PrecisionName = Literal["bf16", "fp16", "fp32"]


def parse_bool_flag(value: str | bool | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean flag value: {value!r} (use true/false)")


def resolve_precision(*, prefer_bf16: bool) -> PrecisionName:
    """Choose runtime precision.

    prefer_bf16=True:
      - CUDA + compute capability >= 8.0 (Ampere+) → bf16
      - CUDA without hardware bf16 (e.g. GTX 1080 Ti / Pascal) → fp16
      - no CUDA → fp32
    prefer_bf16=False → fp32 always
    """
    import torch

    if not prefer_bf16:
        return "fp32"
    if not torch.cuda.is_available():
        return "fp32"
    major, _minor = torch.cuda.get_device_capability(0)
    # Hardware bf16 requires Ampere (sm_80+) even if older drivers report support.
    if major >= 8 and torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp16"


def apply_model_precision(model, *, prefer_bf16: bool) -> dict:
    """Cast model weights and return a small provenance record."""
    import torch

    precision = resolve_precision(prefer_bf16=prefer_bf16)
    if precision == "bf16":
        model = model.to(dtype=torch.bfloat16)
    elif precision == "fp16":
        model = model.half()
    else:
        model = model.float()

    info = {
        "prefer_bf16": bool(prefer_bf16),
        "precision": precision,
        "cuda_available": bool(torch.cuda.is_available()),
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "device": str(getattr(model, "device", next(model.parameters()).device)),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = list(torch.cuda.get_device_capability(0))
    return info
