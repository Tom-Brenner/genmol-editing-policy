"""Frozen GenMol adapter: targeted SAFE-span remasking."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from rollout_policy.io import load_sampling_config, resolve_path
from rollout_policy.precision import apply_model_precision, parse_bool_flag


@dataclass(frozen=True)
class Completion:
    output_safe: str | None
    output_smiles: str | None
    error: str | None
    observed_seed: int | None = None


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


class FrozenGenMol:
    """Thin wrapper around GenMol Sampler. Never trains or mutates the checkpoint."""

    def __init__(
        self,
        checkpoint: str,
        sampling_kwargs: dict[str, Any] | None = None,
        *,
        bf16: bool = False,
    ):
        import torch
        from genmol.sampler import Sampler

        self.checkpoint = str(resolve_path(checkpoint))
        self.prefer_bf16 = bool(bf16)
        cfg = sampling_kwargs or {}
        self.sampling_kwargs = {
            "softmax_temp": float(cfg.get("softmax_temp", 1.2)),
            "randomness": float(cfg.get("randomness", 2.0)),
            "gamma": float(cfg.get("gamma", 0.0)),
            "w": float(cfg.get("w", 2.0)),
            "fix": bool(cfg.get("fix", True)),
        }
        self.num_insert_mask = int(cfg.get("num_insert_mask", 10))
        self.sampler = Sampler(self.checkpoint)
        self.precision_info = apply_model_precision(
            self.sampler.model, prefer_bf16=self.prefer_bf16
        )
        # Keep legacy attribute name used by callers/logs.
        self.bf16 = self.prefer_bf16
        self.precision = self.precision_info["precision"]
        self.device = self.sampler.model.device
        self.sampler.model.backbone.eval()
        torch.set_grad_enabled(False)

    @classmethod
    def from_configs(
        cls,
        experiment: dict[str, Any] | None = None,
        sampling_config: dict[str, Any] | None = None,
        *,
        bf16: bool | None = None,
    ) -> "FrozenGenMol":
        from rollout_policy.io import load_experiment_config

        experiment = experiment or load_experiment_config()
        sampling_config = sampling_config or load_sampling_config(experiment=experiment)
        checkpoint = sampling_config.get("checkpoint") or experiment["paths"]["genmol_checkpoint"]
        if bf16 is None:
            bf16 = parse_bool_flag(sampling_config.get("bf16", False))
        return cls(checkpoint, sampling_config.get("sampling", {}), bf16=bf16)

    def encode_safe(
        self,
        smiles: str,
        cut_bonds: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None = None,
    ) -> tuple[str, Any]:
        import safe as sf
        from rollout_policy.eligibility import PendantBondSlicer

        if cut_bonds:
            slicer = PendantBondSlicer(cut_bonds)
        else:
            slicer = self.sampler.slicer
        encoded = sf.SAFEConverter(slicer=slicer, ignore_stereo=True).encoder(
            smiles, allow_empty=True
        )
        input_ids = self.sampler.model.tokenizer(
            [encoded],
            return_tensors="pt",
            truncation=True,
            max_length=self.sampler.model.config.model.max_position_embeddings,
        )["input_ids"]
        return encoded, input_ids

    def complete_span(
        self,
        smiles: str,
        safe_fragment_index: int,
        seed: int,
        *,
        input_ids=None,
        cut_bonds: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None = None,
    ) -> Completion:
        """Remask a specific SAFE fragment index and generate one completion.

        Mirrors `Sampler.remask` but with a caller-chosen fragment index, a
        pinned mask insertion length, and the pendant-bond SAFE slicer.
        """
        import torch

        set_global_seeds(seed)
        try:
            if input_ids is None:
                _, x = self.encode_safe(smiles, cut_bonds=cut_bonds)
            else:
                x = input_ids.clone()

            special_token_idx = (
                [0]
                + (x[0] == self.sampler.dot_index).nonzero(as_tuple=True)[0].tolist()
                + [len(x[0]) - 1]
            )
            n_frags = len(special_token_idx) - 1
            if safe_fragment_index < 0 or safe_fragment_index >= n_frags:
                return Completion(
                    output_safe=None,
                    output_smiles=None,
                    error=f"FragmentIndexError: {safe_fragment_index} not in [0, {n_frags})",
                    observed_seed=seed,
                )

            mask_start_idx = special_token_idx[safe_fragment_index] + 1
            mask_end_idx = special_token_idx[safe_fragment_index + 1]
            num_insert_mask = self.num_insert_mask
            num_insert_mask = min(
                num_insert_mask,
                self.sampler.model.config.model.max_position_embeddings
                - x.shape[-1]
                + mask_end_idx
                - mask_start_idx,
            )
            if num_insert_mask <= 0:
                return Completion(
                    output_safe=None,
                    output_smiles=None,
                    error="MaskLengthError: non-positive insert length",
                    observed_seed=seed,
                )

            x = torch.hstack(
                [
                    x[:, :mask_start_idx],
                    torch.full((1, num_insert_mask), self.sampler.model.mask_index),
                    x[:, mask_end_idx:],
                ]
            )
            samples = self.sampler.generate(x, **self.sampling_kwargs)
            if not samples:
                return Completion(
                    output_safe=None,
                    output_smiles=None,
                    error="EmptyGeneration",
                    observed_seed=seed,
                )
            smiles_out = samples[0]
            return Completion(
                output_safe=None,
                output_smiles=smiles_out,
                error=None,
                observed_seed=seed,
            )
        except Exception as exc:  # noqa: BLE001 — must retain failure text in JSONL
            return Completion(
                output_safe=None,
                output_smiles=None,
                error=f"{type(exc).__name__}: {exc}",
                observed_seed=seed,
            )

    def complete(self, masked_safe: str, seed: int) -> Completion:
        """Compatibility stub matching the implementation.md sketch.

        Prefer `complete_span` for the remasking experiment.
        """
        set_global_seeds(seed)
        try:
            outputs = self.sampler.fragment_completion(
                masked_safe, num_samples=1, **self.sampling_kwargs
            )
            smiles_out = outputs[0] if outputs else None
            return Completion(
                output_safe=masked_safe,
                output_smiles=smiles_out,
                error=None,
                observed_seed=seed,
            )
        except Exception as exc:  # noqa: BLE001
            return Completion(
                output_safe=None,
                output_smiles=None,
                error=f"{type(exc).__name__}: {exc}",
                observed_seed=seed,
            )
