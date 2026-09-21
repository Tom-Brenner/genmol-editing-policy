"""Frozen GenMol adapter: targeted SAFE-span remasking."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from rollout_policy.io import load_sampling_config, resolve_path
from rollout_policy.precision import apply_model_precision, parse_bool_flag


@dataclass(frozen=True)
class Completion:
    output_safe: str | None
    output_smiles: str | None
    error: str | None
    observed_seed: int | None = None


def _torch_seed(seed: int) -> int:
    return int(seed) % (2**63 - 1)


def generator_for_seed(seed: int, device: Any = "cpu"):
    """Return a torch.Generator bound only to this recorded seed.

    Batched rollouts must not share the process-wide RNG: each completion's
    Gumbel / multinomial draws come from this generator.
    """
    import torch

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        gen = torch.Generator(device=torch_device)
    else:
        gen = torch.Generator()
    gen.manual_seed(_torch_seed(seed))
    return gen


def set_global_seeds(seed: int, *, device: Any = None):
    """Seed process-wide RNGs and return a torch.Generator for this seed.

    Call once per recorded completion seed, never once for a whole rollout
    batch (that would couple samples). Prefer the returned generator for
    sampling; libraries that only read the global RNG still see `seed`.
    """
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    try:
        import torch

        torch.manual_seed(_torch_seed(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_torch_seed(seed))
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return generator_for_seed(seed, device)
    except Exception:
        return None


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
        """Remask a specific SAFE fragment index and generate one completion."""
        return self.complete_spans(
            smiles,
            safe_fragment_index,
            [seed],
            input_ids=input_ids,
            cut_bonds=cut_bonds,
        )[0]

    def complete_spans(
        self,
        smiles: str,
        safe_fragment_index: int,
        seeds: Sequence[int],
        *,
        input_ids=None,
        cut_bonds: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None = None,
    ) -> list[Completion]:
        """Batched remask for one parent+action with independent per-seed RNGs.

        The masked prefix is encoded once and stacked. BERT diffusion runs as a
        batch; token sampling uses a dedicated torch.Generator per recorded seed
        so completions stay independent. MCG (`gamma` != 0) indexes row 0 in
        stock GenMol, so that path falls back to sequential generation.
        """
        seed_list = [int(s) for s in seeds]
        if not seed_list:
            return []
        if float(self.sampling_kwargs.get("gamma") or 0) != 0:
            return [
                self._complete_span_sequential(smiles, safe_fragment_index, s, input_ids=input_ids, cut_bonds=cut_bonds)
                for s in seed_list
            ]
        try:
            masked, err = self._masked_prefix(
                smiles, safe_fragment_index, input_ids=input_ids, cut_bonds=cut_bonds
            )
            if err is not None:
                return [
                    Completion(output_safe=None, output_smiles=None, error=err, observed_seed=s)
                    for s in seed_list
                ]
            smiles_out = self._generate_independent(masked, seed_list)
            return [
                Completion(
                    output_safe=None,
                    output_smiles=smi,
                    error=None if smi else "EmptyGeneration",
                    observed_seed=seed,
                )
                for seed, smi in zip(seed_list, smiles_out)
            ]
        except Exception as exc:  # noqa: BLE001 — must retain failure text in JSONL
            return [
                Completion(
                    output_safe=None,
                    output_smiles=None,
                    error=f"{type(exc).__name__}: {exc}",
                    observed_seed=s,
                )
                for s in seed_list
            ]

    def _masked_prefix(
        self,
        smiles: str,
        safe_fragment_index: int,
        *,
        input_ids=None,
        cut_bonds=None,
    ) -> tuple[Any, str | None]:
        import torch

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
            return None, f"FragmentIndexError: {safe_fragment_index} not in [0, {n_frags})"

        mask_start_idx = special_token_idx[safe_fragment_index] + 1
        mask_end_idx = special_token_idx[safe_fragment_index + 1]
        num_insert_mask = min(
            self.num_insert_mask,
            self.sampler.model.config.model.max_position_embeddings
            - x.shape[-1]
            + mask_end_idx
            - mask_start_idx,
        )
        if num_insert_mask <= 0:
            return None, "MaskLengthError: non-positive insert length"
        x = torch.hstack(
            [
                x[:, :mask_start_idx],
                torch.full((1, num_insert_mask), self.sampler.model.mask_index),
                x[:, mask_end_idx:],
            ]
        )
        return x, None

    def _generate_independent(self, x_one: Any, seeds: Sequence[int]) -> list[str | None]:
        """Stack identical masked prefixes; sample each row from its own seed."""
        from genmol.utils.utils_chem import safe_to_smiles

        device = self.device
        x = x_one.to(device).repeat(len(seeds), 1)
        generators = [generator_for_seed(s, device) for s in seeds]
        mdlm = self.sampler.mdlm
        softmax_temp = float(self.sampling_kwargs["softmax_temp"])
        randomness = float(self.sampling_kwargs["randomness"])
        num_steps = max(mdlm.get_num_steps_confidence(x), 2)
        attention_mask = x != self.sampler.pad_index
        for i in range(num_steps):
            logits = self.sampler.model(x, attention_mask)
            x = _step_confidence_independent(
                mdlm,
                logits,
                x,
                i,
                num_steps,
                generators,
                softmax_temp=softmax_temp,
                randomness=randomness,
            )

        raw = self.sampler.model.tokenizer.batch_decode(x, skip_special_tokens=True)
        fix = bool(self.sampling_kwargs["fix"])
        use_bracket = bool(self.sampler.model.config.training.get("use_bracket_safe"))
        if use_bracket:
            from genmol.utils.bracket_safe_converter import bracketsafe2safe
        out: list[str | None] = []
        for text in raw:
            if not text:
                out.append(None)
                continue
            decoded = (
                safe_to_smiles(bracketsafe2safe(text), fix=fix)
                if use_bracket
                else safe_to_smiles(text, fix=fix)
            )
            if not decoded:
                out.append(None)
            else:
                out.append(sorted(decoded.split("."), key=len)[-1])
        return out

    def _complete_span_sequential(
        self,
        smiles: str,
        safe_fragment_index: int,
        seed: int,
        *,
        input_ids=None,
        cut_bonds=None,
    ) -> Completion:
        """MCG fallback: one sample, process-wide seed (stock Sampler.generate)."""
        set_global_seeds(seed, device=self.device)
        try:
            masked, err = self._masked_prefix(
                smiles, safe_fragment_index, input_ids=input_ids, cut_bonds=cut_bonds
            )
            if err is not None:
                return Completion(None, None, err, seed)
            samples = self.sampler.generate(masked, **self.sampling_kwargs)
            if not samples:
                return Completion(None, None, "EmptyGeneration", seed)
            return Completion(None, samples[0], None, seed)
        except Exception as exc:  # noqa: BLE001
            return Completion(None, None, f"{type(exc).__name__}: {exc}", seed)

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


def _step_confidence_independent(
    mdlm: Any,
    logits: Any,
    xt: Any,
    curr_step: int,
    num_steps: int,
    generators: Sequence[Any],
    *,
    softmax_temp: float,
    randomness: float,
    confidence_temperature: float = 1.0,
    num_tokens_unmask: int = 1,
) -> Any:
    """MDLM.step_confidence with a per-row torch.Generator instead of the global RNG."""
    import torch

    xt = xt.clone()
    log_p_x0 = mdlm._subs_parameterization(logits, xt)
    probs = torch.softmax(log_p_x0 / softmax_temp, dim=-1)
    batch, length, vocab = probs.shape
    preds = torch.empty((batch, length), dtype=torch.long, device=probs.device)
    gumbel = torch.empty((batch, length), dtype=logits.dtype, device=logits.device)
    for i, gen in enumerate(generators):
        row = probs[i].reshape(length, vocab).float()
        row = row / row.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        preds[i] = torch.multinomial(row, 1, generator=gen).view(length)
        uniform = torch.rand(
            (length,), device=logits.device, generator=gen, dtype=logits.dtype
        ).clamp(min=1e-20)
        gumbel[i] = -torch.log(-torch.log(uniform))

    confidence = probs.gather(-1, preds.unsqueeze(-1)).squeeze(-1)
    ratio = curr_step / (num_steps - 1)
    gumbel_noise = gumbel * randomness * (1 - ratio)
    confidence = (torch.log(confidence.clamp_min(1e-20)) + gumbel_noise) / confidence_temperature
    mask = xt == mdlm.mask_index
    confidence = confidence.masked_fill(~mask, -torch.inf)
    confidence_threshold, _ = torch.topk(confidence, k=num_tokens_unmask, dim=-1)
    confidence_threshold = confidence_threshold[:, -1].unsqueeze(-1)
    to_replace = (mask.float() * (confidence >= confidence_threshold).float()).bool()
    xt[to_replace] = preds[to_replace]
    return xt
