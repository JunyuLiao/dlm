"""Runtime control for conservative previous-step pre-QK skipping.

The feature has two explicit switches:

* ``use_pre_qk_kernel`` selects the metadata-aware/deferred-V kernel.
* ``enable_pre_skipping`` enables its previous-step score gate.

This makes the original sparse kernel a clean, zero-overhead control and also
allows the specialized kernel to be benchmarked with its gate forced off.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Iterator

import torch

from proxy.proxy_policy import ProxyContext, ThresholdTable

from .triton_bidirectional import (
    DiffusionLambdaSchedule,
    blasst_bidirectional_flash_attn_func,
)


def noise_bucket(ratio: float) -> str:
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("remaining mask ratio must be in [0, 1]")
    if ratio >= 0.75:
        return "high"
    if ratio >= 0.25:
        return "mid"
    return "low"


@dataclass(frozen=True)
class PreQKKernelConfig:
    """Independent feature and kernel switches plus forced-exact anchors."""

    use_pre_qk_kernel: bool = False
    enable_pre_skipping: bool = False
    warmup_tiles: int = 2
    periodic_refresh: int = 8
    anchor_local: bool = True
    anchor_sink: bool = True
    local_radius: int = 1

    def __post_init__(self) -> None:
        if self.enable_pre_skipping and not self.use_pre_qk_kernel:
            raise ValueError("enable_pre_skipping requires use_pre_qk_kernel")
        if self.warmup_tiles < 1:
            raise ValueError("warmup_tiles must be at least 1")
        if self.periodic_refresh < 0 or self.local_radius < 0:
            raise ValueError("refresh period and local radius must be non-negative")


class PreQKMetadataBuffers:
    """Per-layer double buffer; values are base-2 tile score logarithms."""

    def __init__(
        self,
        layers: int,
        batch: int,
        heads: int,
        sequence_length: int,
        device: torch.device | str,
    ) -> None:
        if layers <= 0 or batch <= 0 or heads <= 0 or sequence_length <= 0:
            raise ValueError("metadata dimensions must be positive")
        shape = (layers, batch, heads, math.ceil(sequence_length / 128), math.ceil(sequence_length / 64))
        self.previous = torch.full(shape, torch.inf, dtype=torch.float16, device=device)
        self.current = torch.full_like(self.previous, torch.inf)

    def commit(self) -> None:
        self.previous, self.current = self.current, self.previous
        self.current.fill_(torch.inf)

    def clear(self) -> None:
        self.previous.fill_(torch.inf)
        self.current.fill_(torch.inf)


class PreQKKernelController:
    """Owns diffusion-step state and materializes per-layer thresholds."""

    def __init__(
        self,
        *,
        config: PreQKKernelConfig,
        thresholds: ThresholdTable,
        schedule: DiffusionLambdaSchedule,
        layers: int,
        batch: int,
        heads: int,
        sequence_length: int,
        device: torch.device,
    ) -> None:
        self.config = config
        self.thresholds = thresholds
        self.schedule = schedule
        self.layers = layers
        self.batch = batch
        self.heads = heads
        self.sequence_length = sequence_length
        self.device = device
        self.buffers = PreQKMetadataBuffers(layers, batch, heads, sequence_length, device)
        self._ratios = [1.0] * batch
        self._blasst_lambda = torch.full(
            (batch,), schedule.high_noise_lambda, dtype=torch.float32, device=device
        )
        self._source_buckets: list[str] | None = None
        self._threshold_cache: dict[tuple[int, tuple[float, ...], tuple[str, ...] | None, bool], torch.Tensor] = {}

    @property
    def blasst_lambda(self) -> torch.Tensor:
        return self._blasst_lambda

    def set_remaining_mask_ratio(self, ratio: float | torch.Tensor) -> None:
        if isinstance(ratio, torch.Tensor):
            values = [float(value) for value in ratio.detach().cpu().tolist()]
            if len(values) != self.batch:
                raise ValueError("ratio tensor must have one value per sequence")
        else:
            values = [float(ratio)] * self.batch
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("remaining mask ratios must be in [0, 1]")
        self._ratios = values
        self._blasst_lambda.copy_(torch.tensor(
            [self.schedule.threshold(value) for value in values],
            dtype=torch.float32,
            device=self.device,
        ))

    def proxy_log_thresholds(self, layer: int) -> torch.Tensor:
        source_key = tuple(self._source_buckets) if self._source_buckets is not None else None
        cache_key = (layer, tuple(self._ratios), source_key, self.config.enable_pre_skipping)
        cached = self._threshold_cache.get(cache_key)
        if cached is not None:
            return cached
        values = torch.full((self.batch, self.heads), -torch.inf, dtype=torch.float32, device=self.device)
        if self.config.enable_pre_skipping and self._source_buckets is not None:
            target_buckets = [noise_bucket(ratio) for ratio in self._ratios]
            num_kv_tiles = math.ceil(self.sequence_length / 64)
            for batch_index, (source_bucket, target_bucket) in enumerate(
                zip(self._source_buckets, target_buckets)
            ):
                for head in range(self.heads):
                    context = ProxyContext(
                        sample_id=str(batch_index), layer=layer, head=head, kv_group=head,
                        denoising_iteration=0, noise_bucket=target_bucket,
                        query_tile=0, kv_tile=0, traversal_index=0,
                        num_kv_tiles=num_kv_tiles,
                    )
                    threshold = self.thresholds.get(context, source_bucket)
                    if math.isfinite(threshold) and threshold > 0.0:
                        values[batch_index, head] = math.log2(threshold)
        self._threshold_cache[cache_key] = values
        return values

    def commit_step(self) -> None:
        """Publish exact scores from the completed step as next-step sources."""
        self.buffers.commit()
        self._source_buckets = [noise_bucket(ratio) for ratio in self._ratios]
        self._threshold_cache.clear()

    def reset(self) -> None:
        self.buffers.clear()
        self._source_buckets = None
        self._threshold_cache.clear()


@contextmanager
def install_pre_qk_blasst_kernel(
    model: torch.nn.Module,
    *,
    batch: int,
    sequence_length: int,
    heads: int,
    device: torch.device,
    config: PreQKKernelConfig | None = None,
    thresholds: ThresholdTable | None = None,
    schedule: DiffusionLambdaSchedule | None = None,
    collect_stats: bool = True,
    num_warps: int = 4,
    pipeline_stages: int = 2,
) -> Iterator[PreQKKernelController]:
    """Install either the untouched sparse baseline or the pre-QK variant."""
    selected_config = config or PreQKKernelConfig()
    modules = [module for module in model.modules() if hasattr(module, "flash_attn_func")]
    if not modules:
        raise ValueError("model has no flash_attn_func attention modules")
    controller = PreQKKernelController(
        config=selected_config,
        thresholds=thresholds or ThresholdTable({}),
        schedule=schedule or DiffusionLambdaSchedule(),
        layers=len(modules), batch=batch, heads=heads, sequence_length=sequence_length,
        device=device,
    )
    originals: list[tuple[torch.nn.Module, object]] = []
    for layer, module in enumerate(modules):
        def call(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            *,
            _layer: int = layer,
            **kwargs: object,
        ) -> torch.Tensor:
            return blasst_bidirectional_flash_attn_func(
                q, k, v, **kwargs,
                blasst_lambda=controller.blasst_lambda,
                collect_stats=collect_stats,
                num_warps=num_warps,
                pipeline_stages=pipeline_stages,
                enable_pre_qk=selected_config.use_pre_qk_kernel,
                previous_proxy_log_scores=(
                    controller.buffers.previous[_layer] if selected_config.use_pre_qk_kernel else None
                ),
                current_proxy_log_scores=(
                    controller.buffers.current[_layer] if selected_config.use_pre_qk_kernel else None
                ),
                proxy_log_thresholds=(
                    controller.proxy_log_thresholds(_layer) if selected_config.use_pre_qk_kernel else None
                ),
                proxy_warmup_tiles=selected_config.warmup_tiles,
                proxy_periodic_refresh=selected_config.periodic_refresh,
                proxy_anchor_local=selected_config.anchor_local,
                proxy_anchor_sink=selected_config.anchor_sink,
                proxy_local_radius=selected_config.local_radius,
            )

        originals.append((module, module.flash_attn_func))  # type: ignore[attr-defined]
        module.flash_attn_func = call  # type: ignore[attr-defined]
    try:
        yield controller
    finally:
        for module, original in originals:
            module.flash_attn_func = original  # type: ignore[attr-defined]
