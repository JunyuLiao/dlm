"""Dense/fresh/stale routing emulation gated by temporal observations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from .proxy import _repeat_kv, attention_scores_and_validity, dense_probabilities


@dataclass(frozen=True)
class ReplayConfig:
    mode: str = "fresh"
    proxy: str = "mean"
    beta: float = 1.28
    threshold_mode: str = "gaussian"
    target_density: float = 0.5
    q_block_size: int = 64
    kv_block_size: int = 64
    physical_q_tile_size: int = 128
    layers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in ("dense_eager", "fresh", "previous"):
            raise ValueError("mode must be dense_eager, fresh, or previous")
        if self.proxy not in ("mean", "max"):
            raise ValueError("proxy must be mean or max")
        if self.threshold_mode not in ("gaussian", "empirical_quantile"):
            raise ValueError("threshold_mode must be gaussian or empirical_quantile")
        if not math.isfinite(self.beta):
            raise ValueError("beta must be finite")
        if not 0.0 < self.target_density <= 1.0:
            raise ValueError("target_density must be in (0, 1]")
        if self.q_block_size <= 0 or self.kv_block_size <= 0 or self.physical_q_tile_size <= 0:
            raise ValueError("block and tile sizes must be positive")


@dataclass
class ReplayStats:
    calls: int = 0
    reused_calls: int = 0
    eligible_tiles: dict[str, int] = field(default_factory=lambda: {"local": 0, "global": 0})
    fresh_kept_tiles: dict[str, int] = field(default_factory=lambda: {"local": 0, "global": 0})
    applied_kept_tiles: dict[str, int] = field(default_factory=lambda: {"local": 0, "global": 0})
    physical_tiles: dict[str, int] = field(default_factory=lambda: {"local": 0, "global": 0})
    physical_kept_tiles: dict[str, int] = field(default_factory=lambda: {"local": 0, "global": 0})
    mask_agreement_tiles: dict[str, int] = field(default_factory=lambda: {"local": 0, "global": 0})
    compared_tiles: dict[str, int] = field(default_factory=lambda: {"local": 0, "global": 0})
    fallback_query_rows: int = 0
    output_relative_error_sum: float = 0.0
    output_relative_error_max: float = 0.0
    output_max_absolute_error: float = 0.0
    replay_vs_fresh_relative_error_sum: float = 0.0
    replay_vs_fresh_relative_error_max: float = 0.0
    replay_vs_fresh_max_absolute_error: float = 0.0
    replay_comparison_calls: int = 0

    def summary(self) -> dict[str, Any]:
        regimes = {}
        for attention_type in ("local", "global"):
            eligible = self.eligible_tiles[attention_type]
            physical = self.physical_tiles[attention_type]
            compared = self.compared_tiles[attention_type]
            regimes[attention_type] = {
                "eligible_routing_tiles": eligible,
                "fresh_routing_density": self.fresh_kept_tiles[attention_type] / eligible if eligible else None,
                "applied_routing_density": self.applied_kept_tiles[attention_type] / eligible if eligible else None,
                "physical_tile_sparsity": 1.0 - self.physical_kept_tiles[attention_type] / physical if physical else None,
                "fresh_applied_mask_agreement": self.mask_agreement_tiles[attention_type] / compared if compared else None,
            }
        return {
            "attention_calls": self.calls,
            "routing_work_avoided": self.reused_calls / self.calls if self.calls else 0.0,
            "attention_regimes": regimes,
            "fallback_query_rows": self.fallback_query_rows,
            "mean_attention_output_relative_error": self.output_relative_error_sum / self.calls if self.calls else 0.0,
            "max_attention_output_relative_error": self.output_relative_error_max,
            "max_attention_output_absolute_error": self.output_max_absolute_error,
            "mean_replay_vs_fresh_relative_error": (
                self.replay_vs_fresh_relative_error_sum / self.replay_comparison_calls
                if self.replay_comparison_calls else 0.0
            ),
            "max_replay_vs_fresh_relative_error": self.replay_vs_fresh_relative_error_max,
            "max_replay_vs_fresh_absolute_error": self.replay_vs_fresh_max_absolute_error,
        }


class ReplayAttention:
    def __init__(self, config: ReplayConfig) -> None:
        self.config = config
        self.stats = ReplayStats()
        self.current_step = -1
        self.request_id = "unknown"
        self._previous_masks: dict[int, torch.Tensor] = {}

    def begin_request(self, request_id: str) -> None:
        self.request_id = str(request_id)
        self.current_step = -1
        self._previous_masks.clear()

    def begin_forward(self, *_: Any) -> None:
        self.current_step += 1

    def _selected(self, layer: int) -> bool:
        return not self.config.layers or layer in self.config.layers

    @staticmethod
    def _attention_type(module: torch.nn.Module, sliding_window: int | None) -> str:
        return "local" if (
            bool(getattr(module, "is_sliding", False))
            or getattr(module, "layer_type", "") == "sliding_attention"
            or sliding_window is not None
        ) else "global"

    def attention_forward(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        scaling: float | None = None,
        is_causal: bool | None = None,
        sliding_window: int | None = None,
        dropout: float = 0.0,
        **_: Any,
    ) -> tuple[torch.Tensor, None]:
        if dropout:
            raise ValueError("replay emulation requires evaluation mode with zero dropout")
        layer = int(getattr(module, "layer_idx", -1))
        if not self._selected(layer):
            raise RuntimeError("unselected replay layers must be dispatched to native attention")
        repeated_key, scores, valid = attention_scores_and_validity(
            module,
            query,
            key,
            attention_mask,
            scaling=scaling,
            is_causal=is_causal,
            sliding_window=sliding_window,
        )
        repeats = repeated_key.shape[1] // value.shape[1]
        repeated_value = _repeat_kv(value, repeats) if repeats != 1 else value
        dense_probabilities = globals()["dense_probabilities"](scores, valid)
        dense_output = torch.matmul(dense_probabilities.to(value.dtype), repeated_value)
        attention_type = self._attention_type(module, sliding_window)
        if self.config.mode == "dense_eager":
            sparse_output = dense_output
        else:
            resolved_scale = float(scaling if scaling is not None else query.shape[-1] ** -0.5)
            fresh, eligible, tiles = routing_mask(
                query,
                repeated_key,
                scores,
                valid,
                proxy=self.config.proxy,
                beta=self.config.beta,
                threshold_mode=self.config.threshold_mode,
                target_density=self.config.target_density,
                q_block_size=self.config.q_block_size,
                kv_block_size=self.config.kv_block_size,
                scaling=resolved_scale,
            )
            previous = self._previous_masks.get(layer)
            use_previous = (
                self.config.mode == "previous"
                and previous is not None
                and previous.shape == fresh.shape
            )
            applied = previous if use_previous else fresh
            self.stats.reused_calls += int(use_previous)
            self._previous_masks[layer] = fresh.detach()
            def routed_output(mask: torch.Tensor, *, count_fallback: bool) -> torch.Tensor:
                allowed = expand_routing_mask(mask, tiles, valid, self.config.q_block_size)
                has_kept_key = allowed.any(dim=-1, keepdim=True)
                if bool((valid & ~has_kept_key).any()):
                    # Preserve finite softmax rows. Only rows with no routed
                    # valid key fall back to their single dense argmax key.
                    dense_argmax = scores.argmax(dim=-1, keepdim=True)
                    forced = torch.zeros_like(valid).scatter(-1, dense_argmax, True) & valid
                    allowed |= forced & ~has_kept_key
                    if count_fallback:
                        self.stats.fallback_query_rows += int((~has_kept_key.squeeze(-1) & valid.any(-1)).sum().item())
                sparse_scores = scores.masked_fill(valid & ~allowed, -torch.inf)
                probabilities = globals()["dense_probabilities"](sparse_scores, allowed)
                return torch.matmul(probabilities.to(value.dtype), repeated_value)

            sparse_output = routed_output(applied, count_fallback=True)
            if self.config.mode == "previous":
                fresh_output = routed_output(fresh, count_fallback=False)
                replay_difference = sparse_output.float() - fresh_output.float()
                replay_relative = float(
                    torch.linalg.vector_norm(replay_difference).item()
                    / max(torch.linalg.vector_norm(fresh_output.float()).item(), 1.0e-12)
                )
                self.stats.replay_comparison_calls += 1
                self.stats.replay_vs_fresh_relative_error_sum += replay_relative
                self.stats.replay_vs_fresh_relative_error_max = max(
                    self.stats.replay_vs_fresh_relative_error_max, replay_relative
                )
                self.stats.replay_vs_fresh_max_absolute_error = max(
                    self.stats.replay_vs_fresh_max_absolute_error,
                    float(replay_difference.abs().max().item()),
                )
            self._record_masks(attention_type, fresh, applied, eligible)
        difference = (sparse_output.float() - dense_output.float())
        relative = float(torch.linalg.vector_norm(difference).item() / max(torch.linalg.vector_norm(dense_output.float()).item(), 1.0e-12))
        self.stats.calls += 1
        self.stats.output_relative_error_sum += relative
        self.stats.output_relative_error_max = max(self.stats.output_relative_error_max, relative)
        self.stats.output_max_absolute_error = max(
            self.stats.output_max_absolute_error,
            float(difference.abs().max().item()),
        )
        return sparse_output.transpose(1, 2).contiguous(), None

    def _record_masks(
        self,
        attention_type: str,
        fresh: torch.Tensor,
        applied: torch.Tensor,
        eligible: torch.Tensor,
    ) -> None:
        self.stats.eligible_tiles[attention_type] += int(eligible.sum().item())
        self.stats.fresh_kept_tiles[attention_type] += int((fresh & eligible).sum().item())
        self.stats.applied_kept_tiles[attention_type] += int((applied & eligible).sum().item())
        self.stats.mask_agreement_tiles[attention_type] += int(((fresh == applied) & eligible).sum().item())
        self.stats.compared_tiles[attention_type] += int(eligible.sum().item())
        q_per_physical = math.ceil(self.config.physical_q_tile_size / self.config.q_block_size)
        batch, heads, q_blocks, kv_blocks = eligible.shape
        padded = math.ceil(q_blocks / q_per_physical) * q_per_physical
        if padded != q_blocks:
            pad_shape = (batch, heads, padded - q_blocks, kv_blocks)
            eligible = torch.cat((eligible, torch.zeros(pad_shape, dtype=torch.bool, device=eligible.device)), dim=2)
            applied = torch.cat((applied, torch.zeros(pad_shape, dtype=torch.bool, device=applied.device)), dim=2)
        physical_eligible = eligible.view(batch, heads, -1, q_per_physical, kv_blocks).any(dim=3)
        physical_keep = (applied & eligible).view(batch, heads, -1, q_per_physical, kv_blocks).any(dim=3)
        self.stats.physical_tiles[attention_type] += int(physical_eligible.sum().item())
        self.stats.physical_kept_tiles[attention_type] += int((physical_keep & physical_eligible).sum().item())


def routing_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    scores: torch.Tensor,
    valid: torch.Tensor,
    *,
    proxy: str,
    beta: float,
    threshold_mode: str = "gaussian",
    target_density: float = 0.5,
    q_block_size: int,
    kv_block_size: int,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """Return [B,H,Q-block,KV-block] fresh keep/eligibility masks."""
    batch, heads, query_length, sequence_length = scores.shape
    prefix_length = max(0, sequence_length - query_length)
    segments = []
    if prefix_length:
        segments.append((0, prefix_length))
    segments.append((prefix_length, sequence_length))
    tiles = [
        (start, min(start + kv_block_size, end))
        for segment_start, end in segments
        for start in range(segment_start, end, kv_block_size)
    ]
    q_blocks = math.ceil(query_length / q_block_size)
    padded_query_length = q_blocks * q_block_size
    query_padding = padded_query_length - query_length
    padded_query = torch.nn.functional.pad(query.float(), (0, 0, 0, query_padding))
    query_blocks = padded_query.view(batch, heads, q_blocks, q_block_size, query.shape[-1])
    proxy_segments = []
    eligible_segments = []
    for segment_start, segment_end in segments:
        segment_length = segment_end - segment_start
        kv_blocks = math.ceil(segment_length / kv_block_size)
        padded_kv_length = kv_blocks * kv_block_size
        kv_padding = padded_kv_length - segment_length
        segment_valid = valid[..., :, segment_start:segment_end]
        segment_valid = torch.nn.functional.pad(
            segment_valid,
            (0, kv_padding, 0, query_padding),
            value=False,
        )
        blocked_valid = segment_valid.view(
            batch, heads, q_blocks, q_block_size, kv_blocks, kv_block_size
        ).permute(0, 1, 2, 4, 3, 5)
        eligible_segment = blocked_valid.any(dim=(-2, -1))
        if proxy == "max":
            segment_scores = scores[..., :, segment_start:segment_end]
            segment_scores = torch.nn.functional.pad(
                segment_scores,
                (0, kv_padding, 0, query_padding),
                value=-torch.inf,
            )
            blocked_scores = segment_scores.view(
                batch, heads, q_blocks, q_block_size, kv_blocks, kv_block_size
            ).permute(0, 1, 2, 4, 3, 5)
            proxy_segment = blocked_scores.masked_fill(~blocked_valid, -torch.inf).amax(dim=(-2, -1)).float()
        else:
            q_participates = blocked_valid.any(dim=-1)
            k_participates = blocked_valid.any(dim=-2)
            segment_key = key[..., segment_start:segment_end, :].float()
            segment_key = torch.nn.functional.pad(segment_key, (0, 0, 0, kv_padding))
            key_blocks = segment_key.view(batch, heads, kv_blocks, kv_block_size, key.shape[-1])
            q_sum = torch.einsum("bhqxd,bhqkx->bhqkd", query_blocks, q_participates.float())
            q_count = q_participates.sum(dim=-1, keepdim=True).clamp_min(1)
            k_sum = torch.einsum("bhkyd,bhqky->bhqkd", key_blocks, k_participates.float())
            k_count = k_participates.sum(dim=-1, keepdim=True).clamp_min(1)
            proxy_segment = ((q_sum / q_count) * (k_sum / k_count)).sum(dim=-1) * scaling
        proxy_segments.append(proxy_segment)
        eligible_segments.append(eligible_segment)
    proxy_values = torch.cat(proxy_segments, dim=-1)
    eligible = torch.cat(eligible_segments, dim=-1)
    counts = eligible.sum(dim=-1, keepdim=True).clamp_min(1)
    if threshold_mode == "gaussian":
        safe = proxy_values.masked_fill(~eligible, 0.0)
        mean = safe.sum(dim=-1, keepdim=True) / counts
        variance = ((proxy_values - mean).masked_fill(~eligible, 0.0).square().sum(dim=-1, keepdim=True) / counts)
        threshold = mean + beta * torch.sqrt(variance)
        keep = eligible & (proxy_values > threshold)
    elif threshold_mode == "empirical_quantile":
        # Exact per-row top-density calibration. Ranks avoid a host sync and
        # support a different number of eligible tiles in every query row.
        ranked = proxy_values.masked_fill(~eligible, -torch.inf).argsort(dim=-1, descending=True)
        ranks = torch.empty_like(ranked)
        order = torch.arange(ranked.shape[-1], device=ranked.device, dtype=ranked.dtype)
        ranks.scatter_(-1, ranked, order.expand_as(ranked))
        keep_count = torch.ceil(counts.float() * target_density).to(ranks.dtype)
        keep = eligible & (ranks < keep_count)
    else:
        raise ValueError(f"unknown threshold_mode: {threshold_mode}")
    return keep, eligible, tiles


def expand_routing_mask(
    keep: torch.Tensor,
    tiles: list[tuple[int, int]],
    valid: torch.Tensor,
    q_block_size: int,
) -> torch.Tensor:
    query_length = valid.shape[-2]
    query_index = torch.arange(query_length, device=keep.device) // q_block_size
    kv_index_cpu = torch.empty(valid.shape[-1], dtype=torch.long)
    for tile_index, (kv_start, kv_end) in enumerate(tiles):
        kv_index_cpu[kv_start:kv_end] = tile_index
    kv_index = kv_index_cpu.to(device=keep.device)
    allowed = keep[:, :, query_index[:, None], kv_index[None, :]]
    return allowed & valid
