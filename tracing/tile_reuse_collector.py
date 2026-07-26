"""Efficient paired-step tracing for physical attention-result reuse.

The tracer wraps the model's existing FlashAttention call, so normal model
execution remains optimized.  It computes exact statistics only for selected
query tiles and heads, compares them with the previous denoising step, and
writes compact per-tile summaries.  A small configurable subset can retain raw
``(m, l, u)`` tensors for composition and cache-precision validation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from blasst.certified_omission import (
    LogNormalizerInterval,
    advance_log_normalizer_interval,
    attention_output_after_omission,
    certified_tile_mass_upper,
    greedy_mass_budget_omission,
    refresh_log_normalizer_interval,
    temporal_score_perturbation_bound,
)

SCHEMA_VERSION = "blasst-tile-reuse-v3"


@dataclass(frozen=True)
class ReuseTraceStep:
    request_ids: Sequence[str]
    step: int
    remaining_mask_ratio: float
    query_tile: int
    token_states: torch.Tensor


@dataclass
class _PreviousLayer:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    m: torch.Tensor
    l: torch.Tensor
    u: torch.Tensor
    final_output: torch.Tensor


def _row_relative_drift(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(current - previous, dim=-1) / (
        torch.linalg.vector_norm(previous, dim=-1) + 1e-6
    )


def _summaries(values: torch.Tensor, dim: int) -> tuple[torch.Tensor, ...]:
    return (
        values.mean(dim),
        values.amax(dim),
        torch.quantile(values, 0.90, dim=dim),
        torch.quantile(values, 0.99, dim=dim),
    )


class TileReuseTraceCollector:
    """Collect current-step, paired-step, and raw sampled tile statistics."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        heads: Sequence[int] = (0, 7, 15, 31),
        layers: Sequence[int] | None = None,
        raw_layers: Sequence[int] = (0, 15, 31),
        raw_heads: Sequence[int] = (0,),
        q_block_size: int = 128,
        kv_block_size: int = 64,
        stable_drift: float = 0.05,
        certificate_mass_budgets: Sequence[float] = (0.001, 0.005, 0.01, 0.025),
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.heads = tuple(sorted(set(int(value) for value in heads)))
        self.layers = None if layers is None else frozenset(int(value) for value in layers)
        self.raw_layers = frozenset(int(value) for value in raw_layers)
        self.raw_heads = frozenset(int(value) for value in raw_heads)
        self.q_block_size = q_block_size
        self.kv_block_size = kv_block_size
        self.stable_drift = stable_drift
        self.certificate_mass_budgets = tuple(
            sorted(set(float(value) for value in certificate_mass_budgets))
        )
        if any(value <= 0 or value >= 1 for value in self.certificate_mass_budgets):
            raise ValueError("certificate mass budgets must be in (0, 1)")
        self.step: ReuseTraceStep | None = None
        self._previous: dict[int, _PreviousLayer] = {}
        self._certificate: dict[int, dict[float, LogNormalizerInterval]] = {}
        self._chunks: dict[str, list[np.ndarray]] = {}

    @staticmethod
    def _budget_tag(value: float) -> str:
        return f"{value:g}".replace(".", "p").replace("-", "m")

    def set_step(self, step: ReuseTraceStep) -> None:
        if step.token_states.ndim != 2:
            raise ValueError("token_states must have shape [batch, sequence]")
        if len(step.request_ids) != step.token_states.shape[0]:
            raise ValueError("request_ids and token-state batch size differ")
        self.step = step

    def _append(self, **values: np.ndarray) -> None:
        for name, value in values.items():
            self._chunks.setdefault(name, []).append(value)

    @staticmethod
    def _numpy(value: torch.Tensor, dtype: np.dtype | None = None) -> np.ndarray:
        result = value.detach().cpu().numpy()
        return result.astype(dtype, copy=False) if dtype is not None else result

    def record_layer(
        self,
        layer: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        *,
        softmax_scale: float | None = None,
    ) -> None:
        if self.step is None:
            raise RuntimeError("set_step must be called before the model forward")
        if self.layers is not None and layer not in self.layers:
            return
        batch, sequence, query_heads, head_dim = q.shape
        if sequence % self.kv_block_size:
            raise ValueError("reuse tracing currently requires complete KV tiles")
        selected_heads = tuple(head for head in self.heads if head < query_heads)
        if not selected_heads:
            raise ValueError("none of the configured query heads exist")
        if query_heads % k.shape[2]:
            raise ValueError("query heads must be divisible by KV heads")
        repeats = query_heads // k.shape[2]
        kv_heads = tuple(head // repeats for head in selected_heads)
        q_start = self.step.query_tile * self.q_block_size
        q_end = min(q_start + self.q_block_size, sequence)
        if q_start >= sequence or q_end - q_start != self.q_block_size:
            raise ValueError("reuse tracing requires a complete selected query tile")

        qh = q[:, q_start:q_end, selected_heads].permute(0, 2, 1, 3).float()
        kh = k[:, :, kv_heads].permute(0, 2, 1, 3).float()
        vh = v[:, :, kv_heads].permute(0, 2, 1, 3).float()
        scale = float(softmax_scale) if softmax_scale is not None else head_dim**-0.5
        scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
        tile_count = sequence // self.kv_block_size
        scores = scores.reshape(
            batch, len(selected_heads), self.q_block_size, tile_count, self.kv_block_size
        ).permute(0, 1, 3, 2, 4)
        m = scores.amax(-1)
        probabilities = torch.exp(scores - m[..., None])
        l = probabilities.sum(-1)
        v_tiles = vh.reshape(
            batch, len(selected_heads), tile_count, self.kv_block_size, v.shape[-1]
        )
        k_tiles = kh.reshape(
            batch, len(selected_heads), tile_count, self.kv_block_size, k.shape[-1]
        )
        u = torch.einsum("bhtrk,bhtkd->bhtrd", probabilities, v_tiles)
        current_log_normalizer = m + torch.log(l.clamp_min(1e-30))
        final_output = output[:, q_start:q_end, selected_heads].permute(0, 2, 1, 3).float()

        previous = self._previous.get(layer)
        if previous is not None:
            q_drift_rows = _row_relative_drift(qh, previous.q)
            k_drift_rows = _row_relative_drift(kh, previous.k).reshape(
                batch, len(selected_heads), tile_count, self.kv_block_size
            )
            v_drift_rows = _row_relative_drift(vh, previous.v).reshape(
                batch, len(selected_heads), tile_count, self.kv_block_size
            )
            q_mean, q_max, q_q90, q_q99 = _summaries(q_drift_rows, -1)
            k_mean, k_max, k_q90, k_q99 = _summaries(k_drift_rows, -1)
            v_mean, v_max, v_q90, v_q99 = _summaries(v_drift_rows, -1)
            q_fro = torch.linalg.vector_norm(qh - previous.q, dim=(-2, -1)) / (
                torch.linalg.vector_norm(previous.q, dim=(-2, -1)) + 1e-6
            )
            k_fro = torch.linalg.vector_norm(kh - previous.k, dim=-1).reshape(
                batch, len(selected_heads), tile_count, self.kv_block_size
            )
            k_fro = torch.linalg.vector_norm(k_fro, dim=-1) / (
                torch.linalg.vector_norm(
                    torch.linalg.vector_norm(previous.k, dim=-1).reshape(
                        batch, len(selected_heads), tile_count, self.kv_block_size
                    ),
                    dim=-1,
                )
                + 1e-6
            )
            v_fro = torch.linalg.vector_norm(vh - previous.v, dim=-1).reshape(
                batch, len(selected_heads), tile_count, self.kv_block_size
            )
            v_fro = torch.linalg.vector_norm(v_fro, dim=-1) / (
                torch.linalg.vector_norm(
                    torch.linalg.vector_norm(previous.v, dim=-1).reshape(
                        batch, len(selected_heads), tile_count, self.kv_block_size
                    ),
                    dim=-1,
                )
                + 1e-6
            )

            current_tile_output = u / l.clamp_min(1e-30)[..., None]
            previous_tile_output = previous.u / previous.l.clamp_min(1e-30)[..., None]
            output_error_rows = torch.linalg.vector_norm(
                current_tile_output - previous_tile_output, dim=-1
            ) / (torch.linalg.vector_norm(current_tile_output, dim=-1) + 1e-6)
            error_mean, error_max, error_q90, error_q99 = _summaries(output_error_rows, -1)
            tile_cosine = F.cosine_similarity(
                current_tile_output.flatten(-2), previous_tile_output.flatten(-2), dim=-1
            )
            m_change = (m - previous.m).abs()
            l_change = (l - previous.l).abs() / (previous.l.abs() + 1e-6)
            m_mean, m_max, _, _ = _summaries(m_change, -1)
            l_mean, l_max, _, _ = _summaries(l_change, -1)
            final_error_rows = _row_relative_drift(final_output, previous.final_output)
            final_error_mean, final_error_max, _, _ = _summaries(final_error_rows, -1)
            final_cosine = F.cosine_similarity(
                final_output.flatten(-2), previous.final_output.flatten(-2), dim=-1
            )

            # Exact counterfactual effect of replacing one current component
            # with its previous-step (m,l,u), while all other tiles stay fresh.
            # A non-tight global maximum remains a valid softmax reference after
            # subtracting the current tile, avoiding an expensive 64-way rescan.
            global_m = m.amax(2)
            current_scale = torch.exp(m - global_m[:, :, None])
            global_l = (current_scale * l).sum(2)
            global_u = (current_scale[..., None] * u).sum(2)
            baseline_output = global_u / global_l.clamp_min(1e-30)[..., None]
            without_l = global_l[:, :, None] - current_scale * l
            without_u = global_u[:, :, None] - current_scale[..., None] * u
            replacement_m = torch.maximum(global_m[:, :, None], previous.m)
            remaining_scale = torch.exp(global_m[:, :, None] - replacement_m)
            previous_scale = torch.exp(previous.m - replacement_m)
            mixed_l = remaining_scale * without_l + previous_scale * previous.l
            mixed_u = (
                remaining_scale[..., None] * without_u
                + previous_scale[..., None] * previous.u
            )
            mixed_output = mixed_u / mixed_l.clamp_min(1e-30)[..., None]
            one_tile_error_rows = torch.linalg.vector_norm(
                mixed_output - baseline_output[:, :, None], dim=-1
            ) / (torch.linalg.vector_norm(baseline_output[:, :, None], dim=-1) + 1e-6)
            one_tile_error_mean, one_tile_error_max, _, one_tile_error_q99 = _summaries(
                one_tile_error_rows, -1
            )
            tile_mass_rows = current_scale * l / global_l[:, :, None].clamp_min(1e-30)
            tile_mass_mean, tile_mass_max, _, tile_mass_q99 = _summaries(tile_mass_rows, -1)

            previous_k_tiles = previous.k.reshape(
                batch,
                len(selected_heads),
                tile_count,
                self.kv_block_size,
                k.shape[-1],
            )
            score_perturbation = temporal_score_perturbation_bound(
                qh,
                previous.q,
                k_tiles,
                previous_k_tiles,
                scale=scale,
            )
            certificate_fields: dict[str, np.ndarray] = {}
            certificate_states = self._certificate[layer]
            for budget in self.certificate_mass_budgets:
                tag = self._budget_tag(budget)
                propagated = advance_log_normalizer_interval(
                    certificate_states[budget], score_perturbation
                )
                mass_upper = certified_tile_mass_upper(propagated)
                omitted, accumulated_upper = greedy_mass_budget_omission(
                    mass_upper, budget
                )
                omitted_output = attention_output_after_omission(m, l, u, omitted)
                omitted_error_rows = torch.linalg.vector_norm(
                    omitted_output - baseline_output, dim=-1
                ) / (torch.linalg.vector_norm(baseline_output, dim=-1) + 1e-6)
                omitted_error_max = omitted_error_rows.amax(-1)
                actual_omitted_mass = (
                    tile_mass_rows * omitted[..., None]
                ).sum(2)
                interval_violation = torch.maximum(
                    propagated.lower - current_log_normalizer,
                    current_log_normalizer - propagated.upper,
                ).clamp_min(0.0)
                mass_violation = (tile_mass_rows - mass_upper).clamp_min(0.0)

                # Diagnostic decomposition (not causal): replace only the
                # denominator bound with the exact current denominator.  If
                # this remains loose, computing fresh anchor tiles cannot
                # rescue the temporal numerator bound.
                exact_log_denominator = torch.logsumexp(
                    current_log_normalizer, dim=2, keepdim=True
                )
                exact_den_mass_upper = torch.exp(
                    (propagated.upper - exact_log_denominator).clamp(max=0.0)
                )
                exact_den_omitted, exact_den_accumulated = (
                    greedy_mass_budget_omission(exact_den_mass_upper, budget)
                )
                exact_den_output = attention_output_after_omission(
                    m, l, u, exact_den_omitted
                )
                exact_den_error = torch.linalg.vector_norm(
                    exact_den_output - baseline_output, dim=-1
                ) / (torch.linalg.vector_norm(baseline_output, dim=-1) + 1e-6)

                # Noncausal low-mass oracle: establishes how much opportunity
                # exists if both numerator and denominator were known exactly.
                oracle_omitted, oracle_accumulated = greedy_mass_budget_omission(
                    tile_mass_rows, budget
                )
                oracle_output = attention_output_after_omission(m, l, u, oracle_omitted)
                oracle_error = torch.linalg.vector_norm(
                    oracle_output - baseline_output, dim=-1
                ) / (torch.linalg.vector_norm(baseline_output, dim=-1) + 1e-6)
                certificate_states[budget] = refresh_log_normalizer_interval(
                    propagated, current_log_normalizer, omitted
                )
                certificate_fields.update(
                    {
                        f"cert_omit_b{tag}": self._numpy(
                            omitted.reshape(-1), np.bool_
                        ),
                        f"cert_mass_upper_b{tag}": self._numpy(
                            mass_upper.amax(-1).reshape(-1), np.float32
                        ),
                        f"cert_group_aggregate_upper_max_b{tag}": self._numpy(
                            accumulated_upper.amax(-1)[..., None]
                            .expand(batch, len(selected_heads), tile_count)
                            .reshape(-1),
                            np.float32,
                        ),
                        f"cert_group_actual_omitted_mass_max_b{tag}": self._numpy(
                            actual_omitted_mass.amax(-1)[..., None]
                            .expand(batch, len(selected_heads), tile_count)
                            .reshape(-1),
                            np.float32,
                        ),
                        f"cert_group_output_error_max_b{tag}": self._numpy(
                            omitted_error_max[..., None]
                            .expand(batch, len(selected_heads), tile_count)
                            .reshape(-1),
                            np.float32,
                        ),
                        f"cert_group_interval_width_max_b{tag}": self._numpy(
                            (propagated.upper - propagated.lower)
                            .amax(dim=(-2, -1))[..., None]
                            .expand(batch, len(selected_heads), tile_count)
                            .reshape(-1),
                            np.float32,
                        ),
                        f"cert_group_interval_violation_max_b{tag}": self._numpy(
                            interval_violation.amax(dim=(-2, -1))[..., None]
                            .expand(batch, len(selected_heads), tile_count)
                            .reshape(-1),
                            np.float32,
                        ),
                        f"cert_group_mass_violation_max_b{tag}": self._numpy(
                            mass_violation.amax(dim=(-2, -1))[..., None]
                            .expand(batch, len(selected_heads), tile_count)
                            .reshape(-1),
                            np.float32,
                        ),
                        f"diag_exact_den_omit_b{tag}": self._numpy(
                            exact_den_omitted.reshape(-1), np.bool_
                        ),
                        f"diag_exact_den_aggregate_upper_max_b{tag}": self._numpy(
                            exact_den_accumulated.amax(-1)[..., None]
                            .expand(batch, len(selected_heads), tile_count)
                            .reshape(-1),
                            np.float32,
                        ),
                        f"diag_exact_den_output_error_max_b{tag}": self._numpy(
                            exact_den_error.amax(-1)[..., None]
                            .expand(batch, len(selected_heads), tile_count)
                            .reshape(-1),
                            np.float32,
                        ),
                        f"oracle_mass_omit_b{tag}": self._numpy(
                            oracle_omitted.reshape(-1), np.bool_
                        ),
                        f"oracle_mass_aggregate_max_b{tag}": self._numpy(
                            oracle_accumulated.amax(-1)[..., None]
                            .expand(batch, len(selected_heads), tile_count)
                            .reshape(-1),
                            np.float32,
                        ),
                        f"oracle_mass_output_error_max_b{tag}": self._numpy(
                            oracle_error.amax(-1)[..., None]
                            .expand(batch, len(selected_heads), tile_count)
                            .reshape(-1),
                            np.float32,
                        ),
                    }
                )

            token_states = self.step.token_states.to(q.device)
            q_states = token_states[:, q_start:q_end, None].expand(
                batch, self.q_block_size, len(selected_heads)
            ).permute(0, 2, 1).clone()
            q_states[(q_states == 2) & (q_drift_rows <= self.stable_drift)] = 3
            kv_states = token_states.reshape(
                batch, tile_count, self.kv_block_size
            )[:, None].expand(batch, len(selected_heads), tile_count, self.kv_block_size).clone()
            kv_states[(kv_states == 2) & (k_drift_rows <= self.stable_drift)] = 3

            shape = (batch, len(selected_heads), tile_count)
            batch_indices = torch.arange(batch, device=q.device)[:, None, None].expand(shape)
            head_values = torch.tensor(selected_heads, device=q.device)[None, :, None].expand(shape)
            kv_indices = torch.arange(tile_count, device=q.device)[None, None, :].expand(shape)
            request_values = np.asarray(self.step.request_ids, dtype="U64")[
                self._numpy(batch_indices.reshape(-1), np.int64)
            ]
            q_state_fractions = [
                (q_states == state).float().mean(-1)[..., None].expand(shape) for state in range(5)
            ]
            kv_state_fractions = [
                (kv_states == state).float().mean(-1) for state in range(5)
            ]

            def flat(value: torch.Tensor, repeat_tiles: bool = False) -> np.ndarray:
                if repeat_tiles:
                    value = value[..., None].expand(shape)
                return self._numpy(value.reshape(-1), np.float32)

            self._append(
                request_id=request_values,
                layer=np.full(np.prod(shape), layer, dtype=np.int16),
                head=self._numpy(head_values.reshape(-1), np.int16),
                step=np.full(np.prod(shape), self.step.step, dtype=np.int16),
                remaining_mask_ratio=np.full(
                    np.prod(shape), self.step.remaining_mask_ratio, dtype=np.float32
                ),
                query_tile=np.full(np.prod(shape), self.step.query_tile, dtype=np.int16),
                kv_tile=self._numpy(kv_indices.reshape(-1), np.int16),
                diagonal_distance=self._numpy(
                    (kv_indices - self.step.query_tile).abs().reshape(-1), np.int16
                ),
                q_fro=flat(q_fro, True),
                q_mean=flat(q_mean, True),
                q_max=flat(q_max, True),
                q_q90=flat(q_q90, True),
                q_q99=flat(q_q99, True),
                k_fro=flat(k_fro),
                k_mean=flat(k_mean),
                k_max=flat(k_max),
                k_q90=flat(k_q90),
                k_q99=flat(k_q99),
                v_fro=flat(v_fro),
                v_mean=flat(v_mean),
                v_max=flat(v_max),
                v_q90=flat(v_q90),
                v_q99=flat(v_q99),
                tile_output_error_mean=flat(error_mean),
                tile_output_error_max=flat(error_max),
                tile_output_error_q90=flat(error_q90),
                tile_output_error_q99=flat(error_q99),
                tile_output_cosine=flat(tile_cosine),
                m_change_mean=flat(m_mean),
                m_change_max=flat(m_max),
                l_relative_change_mean=flat(l_mean),
                l_relative_change_max=flat(l_max),
                final_output_error_mean=flat(final_error_mean, True),
                final_output_error_max=flat(final_error_max, True),
                final_output_cosine=flat(final_cosine, True),
                one_tile_full_output_error_mean=flat(one_tile_error_mean),
                one_tile_full_output_error_max=flat(one_tile_error_max),
                one_tile_full_output_error_q99=flat(one_tile_error_q99),
                current_tile_mass_mean=flat(tile_mass_mean),
                current_tile_mass_max=flat(tile_mass_max),
                current_tile_mass_q99=flat(tile_mass_q99),
                **{
                    f"query_state_{state}_fraction": flat(q_state_fractions[state])
                    for state in range(5)
                },
                **{
                    f"kv_state_{state}_fraction": flat(kv_state_fractions[state])
                    for state in range(5)
                },
                **certificate_fields,
            )

        if layer not in self._certificate:
            exact = current_log_normalizer.detach()
            self._certificate[layer] = {
                budget: LogNormalizerInterval.exact(exact.clone())
                for budget in self.certificate_mass_budgets
            }

        if layer in self.raw_layers:
            raw_positions = [
                index for index, head in enumerate(selected_heads) if head in self.raw_heads
            ]
            if raw_positions:
                raw_path = self.output_dir / (
                    f"raw-{self.step.request_ids[0]}-step{self.step.step:02d}"
                    f"-layer{layer:02d}-qtile{self.step.query_tile:02d}.pt"
                )
                torch.save(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "request_ids": tuple(self.step.request_ids),
                        "step": self.step.step,
                        "remaining_mask_ratio": self.step.remaining_mask_ratio,
                        "layer": layer,
                        "heads": tuple(selected_heads[index] for index in raw_positions),
                        "query_tile": self.step.query_tile,
                        "m": m[:, raw_positions].to(torch.float16).cpu(),
                        "l": l[:, raw_positions].to(torch.float16).cpu(),
                        "u": u[:, raw_positions].to(torch.bfloat16).cpu(),
                        "tile_normalized_output": (
                            u[:, raw_positions] / l[:, raw_positions].clamp_min(1e-30)[..., None]
                        ).to(torch.bfloat16).cpu(),
                        "final_attention_output": final_output[:, raw_positions].to(torch.bfloat16).cpu(),
                    },
                    raw_path,
                )

        self._previous[layer] = _PreviousLayer(
            q=qh.detach(),
            k=kh.detach(),
            v=vh.detach(),
            m=m.detach(),
            l=l.detach(),
            u=u.detach(),
            final_output=final_output.detach(),
        )

    def flush(self) -> Path | None:
        if not self._chunks:
            return None
        path = self.output_dir / "tile-reuse-summary.npz"
        arrays = {name: np.concatenate(values) for name, values in self._chunks.items()}
        np.savez_compressed(path, schema_version=np.array(SCHEMA_VERSION), **arrays)
        return path


@contextmanager
def install_tile_reuse_tracer(
    model: torch.nn.Module,
    collector: TileReuseTraceCollector,
) -> Iterator[None]:
    """Wrap existing attention calls without replacing their implementation."""

    replaced: list[tuple[torch.nn.Module, object]] = []
    modules = [module for module in model.modules() if hasattr(module, "flash_attn_func")]
    for layer, module in enumerate(modules):
        original = module.flash_attn_func

        def call(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            *,
            _layer: int = layer,
            _original: Callable[..., torch.Tensor] = original,
            **kwargs: object,
        ) -> torch.Tensor:
            output = _original(q, k, v, **kwargs)
            collector.record_layer(
                _layer,
                q,
                k,
                v,
                output,
                softmax_scale=kwargs.get("softmax_scale"),  # type: ignore[arg-type]
            )
            return output

        replaced.append((module, original))
        module.flash_attn_func = call
    try:
        yield
    finally:
        for module, original in replaced:
            module.flash_attn_func = original
