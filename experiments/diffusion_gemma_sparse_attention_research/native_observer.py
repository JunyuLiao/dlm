"""Read-only native DiffusionGemma denoising observer.

The observer wraps the existing ``_denoising_step`` method and sampler calls.
It never changes the tensors passed to the model, the stopping criterion, the
acceptance mask, or RNG state.  It is intentionally diagnostic-only: CPU
transfers and entropy reductions make it unsuitable for clean timing.
"""
from __future__ import annotations

import math
import types
from collections import Counter
from contextlib import contextmanager
from typing import Any, Iterator

import torch


def _percentile(values: torch.Tensor, q: float) -> float:
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        return float("nan")
    return float(torch.quantile(values, q).item())


def _stats(logits: torch.Tensor, confidence_threshold: float) -> dict[str, float]:
    """Compute scalar diagnostics from the already processed native logits."""
    with torch.no_grad():
        values = logits.detach().float()
        entropy = torch.distributions.Categorical(logits=values).entropy()
        top = torch.topk(values, k=2, dim=-1).values
        margin = top[..., 0] - top[..., 1]
        top1 = torch.softmax(values, dim=-1).amax(dim=-1)
        flat_entropy = entropy.reshape(-1)
        flat_margin = margin.reshape(-1)
        flat_top1 = top1.reshape(-1)
        return {
            "entropy_mean": float(flat_entropy.mean().item()),
            "entropy_p50": _percentile(flat_entropy, 0.50),
            "entropy_p90": _percentile(flat_entropy, 0.90),
            "entropy_max": float(flat_entropy.max().item()),
            "margin_p10": _percentile(flat_margin, 0.10),
            "margin_p50": _percentile(flat_margin, 0.50),
            "margin_mean": float(flat_margin.mean().item()),
            "top1_prob_p50": _percentile(flat_top1, 0.50),
            "top1_prob_mean": float(flat_top1.mean().item()),
            "confident_position_fraction": float((flat_entropy < confidence_threshold).float().mean().item()),
            "finite_logits": bool(torch.isfinite(values).all().item()),
        }


class NativeTrajectoryObserver:
    """Collect sanitized per-step records for one or more requests."""

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.005,
        stability_threshold: int = 1,
        max_denoising_steps: int = 48,
    ) -> None:
        if confidence_threshold <= 0 or stability_threshold < 0 or max_denoising_steps <= 0:
            raise ValueError("invalid native stopping configuration")
        self.confidence_threshold = float(confidence_threshold)
        self.stability_threshold = int(stability_threshold)
        self.max_denoising_steps = int(max_denoising_steps)
        self.request_id: str | None = None
        self.canvas_id = -1
        self.call_index = 0
        self._previous_step = None
        self._argmax_history: list[torch.Tensor] = []
        self._event: dict[str, Any] | None = None
        self.records: list[dict[str, Any]] = []

    def begin_request(self, request_id: str) -> None:
        self.request_id = str(request_id)
        self.canvas_id = -1
        self.call_index = 0
        self._previous_step = None
        self._argmax_history = []
        self._event = None
        self.records = []

    def _new_canvas_if_needed(self, cur_step: Any) -> int:
        step = int(cur_step.item()) if isinstance(cur_step, torch.Tensor) else int(cur_step)
        if self._previous_step is None or step > int(self._previous_step):
            self.canvas_id += 1
            self._argmax_history = []
        self._previous_step = step
        return step

    def _record_accept(
        self,
        current_canvas: torch.Tensor,
        denoiser_canvas: torch.Tensor,
        logits: torch.Tensor,
        cur_step: Any,
        accepted_mask: torch.Tensor,
    ) -> None:
        step = int(cur_step.item()) if isinstance(cur_step, torch.Tensor) else int(cur_step)
        metrics = _stats(logits, self.confidence_threshold)
        accepted = accepted_mask.detach().bool()
        current = current_canvas.detach()
        sampled = denoiser_canvas.detach()
        event: dict[str, Any] = {
            "request_id": self.request_id,
            "canvas_id": int(self.canvas_id),
            "call_index": int(self.call_index),
            "denoising_step_reverse": step,
            "denoising_step_forward": self.max_denoising_steps - step + 1,
            **metrics,
            "accepted_count": int(accepted.sum().item()),
            "canvas_tokens": int(accepted.numel()),
            "accepted_fraction": float(accepted.float().mean().item()),
            "renoised_count": int((~accepted).sum().item()),
            "sampled_change_count": int((sampled != current).sum().item()),
            "finished_before": None,
            "finished_after": None,
            "native_stop_stable": None,
            "native_stop_confident": bool(metrics["entropy_mean"] < self.confidence_threshold),
            "native_stop_both": None,
            "stable_position_fraction": None,
            "argmax_flip_count": None,
            "self_conditioning_norm": None,
        }
        self._event = event

    def _record_after_step(
        self,
        previous_argmax: torch.Tensor,
        result: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        finished_before: torch.Tensor,
    ) -> None:
        if self._event is None:
            return
        _, new_argmax, self_conditioning_logits, finished_after = result
        current_argmax = new_argmax.detach()
        previous_argmax = previous_argmax.detach()
        same = current_argmax == previous_argmax
        self._event["stable_position_fraction"] = float(same.float().mean().item())
        self._event["argmax_flip_count"] = int((~same).sum().item())
        self._event["finished_before"] = bool(finished_before.detach().any().item())
        self._event["finished_after"] = bool(finished_after.detach().any().item())
        if self.stability_threshold == 0:
            stable = True
        else:
            stable = len(self._argmax_history) >= self.stability_threshold and all(
                torch.equal(old, current_argmax) for old in self._argmax_history[-self.stability_threshold :]
            )
        confident = bool(self._event["native_stop_confident"])
        self._event["native_stop_stable"] = bool(stable)
        self._event["native_stop_both"] = bool(stable and confident)
        self._event["self_conditioning_norm"] = float(self_conditioning_logits.detach().float().norm().item())
        self._argmax_history.append(current_argmax.cpu())
        if len(self._argmax_history) > max(1, self.stability_threshold + 1):
            self._argmax_history.pop(0)
        self.records.append(self._event)
        self.call_index += 1
        self._event = None

    @contextmanager
    def installed(self, model: Any) -> Iterator[None]:
        """Temporarily wrap one native model instance."""
        original_step = model._denoising_step
        observer = self

        def wrapped_step(
            _model: Any,
            decoder_forward: Any,
            current_canvas: torch.Tensor,
            argmax_canvas: torch.Tensor,
            input_ids: torch.Tensor,
            decoder_position_ids: torch.Tensor,
            self_conditioning_logits: torch.Tensor,
            mask_mapping: dict[str, torch.Tensor],
            past_key_values: Any,
            finished_denoising: torch.Tensor,
            cur_step: Any,
            sampler: Any,
            logits_processor: Any,
            diffusion_stopping_criteria: Any,
            **model_kwargs: Any,
        ):
            step = observer._new_canvas_if_needed(cur_step)
            original_accept = sampler.accept_canvas
            original_renoise = sampler.renoise_canvas

            def accept_wrapper(
                current: torch.Tensor,
                denoiser: torch.Tensor,
                logits: torch.Tensor,
                current_step: Any,
            ) -> torch.Tensor:
                accepted_canvas = original_accept(current, denoiser, logits, current_step)
                mask = getattr(sampler, "accepted_token_mask", None)
                if mask is None:
                    raise RuntimeError("native sampler did not expose accepted_token_mask")
                observer._record_accept(current, denoiser, logits, current_step, mask)
                return accepted_canvas

            def renoise_wrapper(accepted_canvas: torch.Tensor, current_step: Any) -> torch.Tensor:
                return original_renoise(accepted_canvas, current_step)

            sampler.accept_canvas = accept_wrapper
            sampler.renoise_canvas = renoise_wrapper
            try:
                result = original_step(
                    decoder_forward,
                    current_canvas,
                    argmax_canvas,
                    input_ids,
                    decoder_position_ids,
                    self_conditioning_logits,
                    mask_mapping,
                    past_key_values,
                    finished_denoising,
                    cur_step,
                    sampler,
                    logits_processor,
                    diffusion_stopping_criteria,
                    **model_kwargs,
                )
                observer._record_after_step(argmax_canvas, result, finished_denoising)
                return result
            finally:
                sampler.accept_canvas = original_accept
                sampler.renoise_canvas = original_renoise

        model._denoising_step = types.MethodType(wrapped_step, model)
        try:
            yield
        finally:
            model._denoising_step = original_step

    def summary(self) -> dict[str, Any]:
        by_canvas = Counter(int(row["canvas_id"]) for row in self.records)
        stopping = Counter(
            "stability" if row["native_stop_stable"] and not row["native_stop_confident"] else
            "confidence" if row["native_stop_confident"] and not row["native_stop_stable"] else
            "joint" if row["native_stop_both"] else "neither"
            for row in self.records
        )
        return {
            "request_id": self.request_id,
            "denoising_calls": len(self.records),
            "canvas_count_observed": len(by_canvas),
            "calls_per_canvas": {str(k): v for k, v in sorted(by_canvas.items())},
            "stopping_predicate_counts": dict(stopping),
            "terminal_call_was_joint": bool(self.records and self.records[-1]["native_stop_both"]),
        }
