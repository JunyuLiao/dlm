"""Native text-only adapter for google/diffusiongemma-26B-A4B-it."""

from __future__ import annotations

import copy
import math
import numbers
import time
from collections.abc import Mapping
from typing import Any

import torch

from ..base import GenerationRequest, GenerationResult, ModelAdapter
from .common import dtype, seed_everything


MIN_TRANSFORMERS_VERSION = "5.11.0"
_SAMPLER_KEYS = (
    "max_denoising_steps",
    "t_max",
    "t_min",
    "entropy_bound",
    "confidence_threshold",
    "stability_threshold",
)
_EXTRA_KEYS = frozenset((*_SAMPLER_KEYS, "thinking", "top_p"))


class _ConstantTemperatureLogitsProcessor:
    """Apply an AR-style scalar temperature to DiffusionGemma logits."""

    def __init__(self, temperature: float) -> None:
        self.temperature = temperature

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        del input_ids
        return scores / self.temperature


class _LastDimTopPLogitsProcessor:
    """Top-p filtering for DiffusionGemma's [batch, canvas, vocab] logits."""

    def __init__(self, top_p: float) -> None:
        self.top_p = top_p

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        del input_ids
        # Sorting all 262k vocabulary entries for every token in a 256-token
        # canvas is unnecessarily expensive. Compute the exact global
        # normalization once, then grow an ordered top-k set until it contains
        # at least top_p mass. At that point the nucleus is guaranteed to be
        # entirely inside the set.
        log_normalizer = torch.logsumexp(scores.float(), dim=-1, keepdim=True)
        vocabulary = scores.shape[-1]
        k = min(1024, vocabulary)
        while True:
            values, indices = torch.topk(scores, k=k, dim=-1, sorted=True)
            probabilities = torch.exp(values.float() - log_normalizer)
            cumulative = probabilities.cumsum(dim=-1)
            if k == vocabulary or bool((cumulative[..., -1] >= self.top_p).all()):
                break
            k = min(vocabulary, k * 2)
        keep = (cumulative - probabilities) < self.top_p
        filtered = torch.full_like(scores, -torch.inf)
        return filtered.scatter(-1, indices, values.masked_fill(~keep, -torch.inf))


def _require_diffusion_gemma():
    try:
        import transformers
        from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
        from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
            EntropyBoundSamplerConfig,
        )
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "DiffusionGemma requires the released transformers==5.11.0 stack; "
            "install this project's current dependencies before loading the adapter"
        ) from error
    try:
        from packaging.version import Version

        if Version(transformers.__version__) < Version(MIN_TRANSFORMERS_VERSION):
            raise ImportError(
                f"DiffusionGemma requires transformers>={MIN_TRANSFORMERS_VERSION}; "
                f"found {transformers.__version__}"
            )
    except ImportError:
        raise
    return AutoProcessor, DiffusionGemmaForBlockDiffusion, EntropyBoundSamplerConfig


def _hub_error(error: Exception) -> RuntimeError:
    message = str(error).lower()
    if any(value in message for value in ("401", "403", "gated", "authentication", "unauthorized")):
        return RuntimeError(
            "DiffusionGemma checkpoint access was denied. Accept any model terms, "
            "then authenticate with the standard `hf auth login` command."
        )
    return RuntimeError(f"Unable to load the DiffusionGemma checkpoint: {error}")


def _positive_int(name: str, value: Any, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        comparison = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {comparison} integer")
    return int(value)


def _positive_float(name: str, value: Any, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0 if allow_zero else value <= 0.0:
        comparison = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {comparison}")
    return value


def _normalize_ids(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, numbers.Integral):
        return (int(value),)
    if isinstance(value, (list, tuple, set)):
        return tuple(int(item) for item in value)
    raise ValueError(f"special token IDs must be an integer or list, got {type(value).__name__}")


class DiffusionGemmaAdapter(ModelAdapter):
    name = "diffusion_gemma"
    attention_class_names = ("DiffusionGemmaDecoderTextAttention",)
    attention_integration = "registry"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.processor: Any = None

    def _load_processor(self) -> None:
        AutoProcessor, _, _ = _require_diffusion_gemma()
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                revision=self.revision,
            )
        except Exception as error:
            raise _hub_error(error) from error
        self.tokenizer = self.processor.tokenizer

    def load_tokenizer(self) -> "DiffusionGemmaAdapter":
        self._load_processor()
        return self

    def load(self) -> "DiffusionGemmaAdapter":
        _, model_class, _ = _require_diffusion_gemma()
        if self.device.startswith("cuda") and self.precision != "bfloat16":
            raise ValueError("DiffusionGemma must be loaded in bfloat16 on CUDA")
        self._load_processor()
        target_device = "cuda:0" if self.device == "cuda" else self.device
        try:
            self.model = model_class.from_pretrained(
                self.model_path,
                revision=self.revision,
                dtype=torch.bfloat16 if self.device.startswith("cuda") else dtype(self.precision),
                device_map={"": target_device},
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
        except Exception as error:
            raise _hub_error(error) from error
        self.model.eval()
        return self

    def prompt_configuration(
        self, extra: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        extra = extra or {}
        thinking = extra.get("thinking", False)
        if not isinstance(thinking, bool):
            raise ValueError("thinking must be a boolean")
        return {"thinking": thinking}

    def _prompt_tensor(
        self,
        prompt: str,
        extra: Mapping[str, Any] | None = None,
        *,
        device: Any = None,
    ) -> torch.Tensor:
        if self.processor is None:
            raise RuntimeError("load() or load_tokenizer() must be called first")
        thinking = self.prompt_configuration(extra)["thinking"]
        encoded = self.processor.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=thinking,
            tokenize=True,
            return_tensors="pt",
        )
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        if not isinstance(encoded, torch.Tensor):
            encoded = torch.tensor(encoded, dtype=torch.long)
        if encoded.ndim == 1:
            encoded = encoded.unsqueeze(0)
        if encoded.ndim != 2 or encoded.shape[0] != 1:
            raise ValueError("DiffusionGemma text generation expects one tokenized prompt")
        return encoded.to(device=device, dtype=torch.long) if device is not None else encoded

    def encode_prompt(
        self, prompt: str, extra: Mapping[str, Any] | None = None
    ) -> list[int]:
        return self._prompt_tensor(prompt, extra)[0].tolist()

    def _generation_kwargs(self, request: GenerationRequest) -> dict[str, Any]:
        extra = dict(request.extra)
        unknown = sorted(set(extra) - _EXTRA_KEYS)
        if unknown:
            raise ValueError(
                "unsupported DiffusionGemma generation override(s): " + ", ".join(unknown)
            )
        self.prompt_configuration(extra)
        result: dict[str, Any] = {"max_new_tokens": request.max_new_tokens}
        if request.temperature:
            temperature = _positive_float("temperature", request.temperature)
            top_p = _positive_float("top_p", extra.get("top_p", 1.0))
            if top_p > 1.0:
                raise ValueError("top_p must be at most 1")
            # The checkpoint normally appends its native 0.8 -> 0.4 temperature
            # schedule after external processors. Use a copied config with that
            # schedule disabled so the requested scalar has its conventional
            # meaning and is not compounded with a second temperature.
            generation_config = copy.deepcopy(self.model.generation_config)
            generation_config.t_min = None
            generation_config.t_max = None
            processors = [_ConstantTemperatureLogitsProcessor(temperature)]
            if top_p < 1.0:
                processors.append(_LastDimTopPLogitsProcessor(top_p))
            from transformers.generation import LogitsProcessorList

            result["generation_config"] = generation_config
            result["logits_processor"] = LogitsProcessorList(processors)
        elif "top_p" in extra:
            raise ValueError("top_p requires a positive temperature")
        if request.steps is not None:
            if "max_denoising_steps" in extra and extra["max_denoising_steps"] != request.steps:
                raise ValueError("steps and max_denoising_steps specify conflicting values")
            result["max_denoising_steps"] = _positive_int("steps", request.steps)
        elif "max_denoising_steps" in extra:
            result["max_denoising_steps"] = _positive_int(
                "max_denoising_steps", extra["max_denoising_steps"]
            )
        if "t_min" in extra:
            result["t_min"] = _positive_float("t_min", extra["t_min"], allow_zero=True)
        if "t_max" in extra:
            result["t_max"] = _positive_float("t_max", extra["t_max"], allow_zero=True)
        if "t_min" in result and "t_max" in result and result["t_max"] <= result["t_min"]:
            raise ValueError("t_max must be greater than t_min")
        if "confidence_threshold" in extra:
            result["confidence_threshold"] = _positive_float(
                "confidence_threshold", extra["confidence_threshold"]
            )
        if "stability_threshold" in extra:
            result["stability_threshold"] = _positive_int(
                "stability_threshold", extra["stability_threshold"], allow_zero=True
            )
        if "entropy_bound" in extra:
            _, _, sampler_class = _require_diffusion_gemma()
            result["sampler_config"] = sampler_class(
                entropy_bound=_positive_float("entropy_bound", extra["entropy_bound"])
            )
        return result

    def _effective_denoising_config(self, generation_kwargs: Mapping[str, Any]) -> dict[str, Any]:
        config = copy.deepcopy(
            generation_kwargs.get("generation_config")
            or getattr(self.model, "generation_config", None)
        )
        result: dict[str, Any] = {}
        for name in _SAMPLER_KEYS:
            if name == "entropy_bound":
                sampler = generation_kwargs.get("sampler_config") or getattr(config, "sampler_config", None)
                value = getattr(sampler, "entropy_bound", None)
            else:
                value = generation_kwargs.get(name, getattr(config, name, None))
            if value is not None:
                result[name] = value
        return result

    def _eos_token_ids(self) -> tuple[int, ...]:
        for owner in (
            getattr(self.model, "generation_config", None),
            getattr(self.model, "config", None),
            getattr(getattr(self.model, "config", None), "text_config", None),
            self.tokenizer,
        ):
            value = getattr(owner, "eos_token_id", None)
            if value is not None:
                return _normalize_ids(value)
        return ()

    def _completion(
        self, sequence: torch.Tensor, prompt_length: int, limit: int
    ) -> tuple[list[int], str]:
        tokens = sequence[prompt_length : prompt_length + limit].detach().cpu().tolist()
        eos = set(self._eos_token_ids())
        for index, token in enumerate(tokens):
            if token in eos:
                return tokens[: index + 1], "eos"
        pad = self.pad_token_id
        if pad is not None:
            while tokens and tokens[-1] == pad:
                tokens.pop()
        return tokens, "length"

    @torch.inference_mode()
    def generate(self, request: GenerationRequest) -> GenerationResult:
        if self.model is None:
            raise RuntimeError("load() must be called before generate()")
        generation_kwargs = self._generation_kwargs(request)
        seed_everything(request.seed)
        inputs = self._prompt_tensor(
            request.prompt, request.extra, device=self.model.device
        )
        prompt_length = int(inputs.shape[-1])
        started = time.perf_counter()
        output = self.model.generate(input_ids=inputs, **generation_kwargs)
        elapsed = time.perf_counter() - started
        sequences = output.sequences if hasattr(output, "sequences") else output
        tokens, termination = self._completion(
            sequences[0], prompt_length, request.max_new_tokens
        )
        tokens_per_forward = getattr(output, "tokens_per_forward", None)
        model_evaluations = None
        if tokens_per_forward is not None:
            ratio = float(tokens_per_forward.reshape(-1)[0].item())
            if not math.isfinite(ratio):
                raise RuntimeError("DiffusionGemma returned non-finite tokens_per_forward")
            full_completion = sequences[0, prompt_length:]
            pad = self.pad_token_id
            valid = int((full_completion != pad).sum().item()) if pad is not None else full_completion.numel()
            if ratio > 0:
                model_evaluations = int(round(valid / ratio))
        canvas_length = int(getattr(self.model.config, "canvas_length"))
        special_ids = {
            "mask_token_id": self.mask_token_id,
            "pad_token_id": self.pad_token_id,
            "eos_token_ids": list(self._eos_token_ids()),
        }
        for name in ("image_token_id", "boi_token_id", "eoi_token_id"):
            value = getattr(self.model.config, name, None)
            if value is not None:
                special_ids[name] = int(value)
        metadata = {
            "native_canvas_length": canvas_length,
            "requested_block_size": request.block_size,
            "block_size_applied": False,
            "denoising_configuration": self._effective_denoising_config(generation_kwargs),
            "actual_denoising_step_count": model_evaluations,
            "tokens_per_forward": (
                float(tokens_per_forward.reshape(-1)[0].item())
                if tokens_per_forward is not None
                else None
            ),
            "special_token_ids": special_ids,
            "thinking": self.prompt_configuration(request.extra)["thinking"],
            "sampling": (
                {
                    "temperature": float(request.temperature),
                    "top_p": float(request.extra.get("top_p", 1.0)),
                    "temperature_schedule_disabled": True,
                }
                if request.temperature
                else {
                    "temperature": None,
                    "top_p": None,
                    "native_temperature_schedule": True,
                }
            ),
        }
        return GenerationResult(
            prompt=request.prompt,
            prompt_tokens=inputs[0].detach().cpu().tolist(),
            completion_tokens=tokens,
            text=self.tokenizer.decode(tokens, skip_special_tokens=True),
            elapsed_seconds=elapsed,
            model_evaluations=model_evaluations,
            termination_reason=termination,
            metadata=metadata,
        )

    def is_blasst_attention_module(self, name: str, module: Any) -> bool:
        return type(module).__name__ == "DiffusionGemmaDecoderTextAttention"

    def blasst_query_ids(self, module: Any, args: tuple, kwargs: dict) -> Any:
        return kwargs.get("decoder_input_ids")

    @property
    def blasst_filter_special_query_ids(self) -> bool:
        # A uniformly sampled canvas may legitimately contain pad/mask token IDs.
        return False

    def blasst_dense_kv_prefix(
        self,
        module: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Any,
        kwargs: Mapping[str, Any],
    ) -> int:
        # The read-only encoder KV is part of every decoder query's attention
        # context and must participate in BLASST decisions.  Sliding decoder
        # layers already receive a physically cropped local cache from the
        # native HybridCache; global layers receive the full cache.
        return 0

    def runtime_metadata(self) -> dict[str, Any]:
        metadata = super().runtime_metadata()
        metadata.update(
            diffusion_gemma_min_transformers=MIN_TRANSFORMERS_VERSION,
            native_canvas_length=(
                int(self.model.config.canvas_length) if self.model is not None else None
            ),
        )
        return metadata
