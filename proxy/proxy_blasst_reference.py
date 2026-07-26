"""Counterfactual online-softmax execution for pre-QK proxy policies."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import math
from typing import Callable, Iterator

import torch

from .proxy_policy import (
    Decision,
    DisabledPolicy,
    MetadataStore,
    ProxyContext,
    ProxyPolicy,
    SourceMetadata,
    SourceResolver,
)


@dataclass
class ProxyReferenceStats:
    total_physical_tiles: int = 0
    pre_skipped_tiles: int = 0
    exact_blasst_skipped_tiles: int = 0
    qk_tiles_computed: int = 0
    false_skips: int = 0
    correct_pre_skips: int = 0
    false_skips_new_max: int = 0
    oracle_skip_tiles: int = 0

    def summary(self) -> dict[str, float | int]:
        predicted = self.pre_skipped_tiles
        ratio = lambda a, b: a / b if b else 0.0
        return {
            "total_physical_tiles": self.total_physical_tiles,
            "pre_skipped_tiles": predicted,
            "qk_tiles_computed": self.qk_tiles_computed,
            "qk_mma_avoidance": ratio(predicted, self.total_physical_tiles),
            "skip_precision": ratio(self.correct_pre_skips, predicted),
            "oracle_skip_recall": ratio(self.correct_pre_skips, self.oracle_skip_tiles),
            "physical_pre_qk_sparsity": ratio(self.correct_pre_skips, self.total_physical_tiles),
            "false_skip_rate": ratio(self.false_skips, predicted),
            "false_skips_new_max": self.false_skips_new_max,
            "oracle_skip_tiles": self.oracle_skip_tiles,
            "oracle_physical_skip_rate": ratio(self.oracle_skip_tiles, self.total_physical_tiles),
        }


@dataclass
class ProxyReferenceResult:
    output: torch.Tensor
    stats: ProxyReferenceStats
    metadata: MetadataStore


@dataclass
class ProxyExecutionController:
    """Mutable per-forward context for a model-installed reference path."""

    sample_ids: list[str]
    denoising_iteration: int = 0
    noise_bucket: str = "high"
    stats: ProxyReferenceStats | None = None

    def set_step(self, iteration: int, noise_bucket: str, sample_ids: list[str] | None = None) -> None:
        self.denoising_iteration = iteration
        self.noise_bucket = noise_bucket
        self.stats = None
        if sample_ids is not None:
            self.sample_ids = sample_ids


@contextmanager
def install_proxy_blasst_reference(
    model: torch.nn.Module,
    *,
    blasst_lambda: float | Callable[[], float],
    policy: ProxyPolicy | None = None,
    source_store: SourceResolver | None = None,
    output_store: MetadataStore | None = None,
    oracle_targets: MetadataStore | None = None,
    sample_ids: list[str] | None = None,
    q_block_size: int = 128,
    kv_block_size: int = 64,
) -> Iterator[ProxyExecutionController]:
    """Temporarily install the counterfactual path at LLaDA call sites."""
    originals: list[tuple[torch.nn.Module, object]] = []
    controller = ProxyExecutionController(sample_ids or ["0"])
    emitted = output_store or MetadataStore()
    for layer_index, module in enumerate(module for module in model.modules() if hasattr(module, "flash_attn_func")):
        def call(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            *,
            _layer_index: int = layer_index,
            **kwargs: object,
        ) -> torch.Tensor:
            del kwargs
            threshold = blasst_lambda() if callable(blasst_lambda) else blasst_lambda
            result = proxy_blasst_reference(
                q,
                k,
                v,
                blasst_lambda=threshold,
                policy=policy,
                source_store=source_store,
                output_store=emitted,
                oracle_targets=oracle_targets,
                sample_ids=controller.sample_ids,
                layer=_layer_index,
                denoising_iteration=controller.denoising_iteration,
                noise_bucket=controller.noise_bucket,
                q_block_size=q_block_size,
                kv_block_size=kv_block_size,
            )
            if controller.stats is None:
                controller.stats = result.stats
            else:
                for field_name in ProxyReferenceStats.__dataclass_fields__:
                    setattr(
                        controller.stats,
                        field_name,
                        getattr(controller.stats, field_name) + getattr(result.stats, field_name),
                    )
            return result.output

        originals.append((module, module.flash_attn_func))  # type: ignore[attr-defined]
        module.flash_attn_func = call  # type: ignore[attr-defined]
    try:
        yield controller
    finally:
        for module, original in originals:
            module.flash_attn_func = original  # type: ignore[attr-defined]


def proxy_blasst_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    blasst_lambda: float,
    policy: ProxyPolicy | None = None,
    source_store: SourceResolver | None = None,
    output_store: MetadataStore | None = None,
    oracle_targets: MetadataStore | None = None,
    sample_ids: list[str] | None = None,
    layer: int = 0,
    denoising_iteration: int = 0,
    noise_bucket: str = "mid",
    sequence_lengths: list[int] | None = None,
    q_block_size: int = 128,
    kv_block_size: int = 64,
) -> ProxyReferenceResult:
    """Execute PRE_SKIP as genuine QK/state/PV omission.

    ``oracle_targets`` is consulted only after a decision and only to score
    predictor errors. It is never passed to ``policy``. For realistic online
    mode, pass the same persistent ``MetadataStore`` as ``source_store`` and
    ``output_store``. For oracle-source mode, use a separate exact source
    store and a fresh output store.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must be [batch, sequence, heads, dim]")
    if q.shape[0] != k.shape[0] or k.shape[:3] != v.shape[:3] or q.shape[-1] != k.shape[-1]:
        raise ValueError("incompatible q, k, v shapes")
    if not 0.0 <= blasst_lambda <= 1.0:
        raise ValueError("blasst_lambda must be in [0, 1]")
    batch, padded_len, heads, dim = q.shape
    kv_heads = k.shape[2]
    if heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    repeats = heads // kv_heads
    if repeats != 1:
        k = k.repeat_interleave(repeats, dim=2)
        v = v.repeat_interleave(repeats, dim=2)
    lengths = sequence_lengths or [padded_len] * batch
    identifiers = sample_ids or [str(index) for index in range(batch)]
    if len(lengths) != batch or len(identifiers) != batch:
        raise ValueError("sequence_lengths and sample_ids must match batch size")
    if any(length < 0 or length > padded_len for length in lengths):
        raise ValueError("invalid sequence length")

    policy = policy or DisabledPolicy()
    emitted = output_store or MetadataStore()
    resolver = source_store or MetadataStore()
    output = torch.zeros((batch, padded_len, heads, v.shape[-1]), dtype=q.dtype, device=q.device)
    stats = ProxyReferenceStats()
    scale = dim**-0.5
    log_threshold = math.log(blasst_lambda) if blasst_lambda else -math.inf

    for sample, length in enumerate(lengths):
        num_kv_tiles = math.ceil(length / kv_block_size)
        kv_per_group = heads // kv_heads
        for q_start in range(0, length, q_block_size):
            q_end = min(q_start + q_block_size, length)
            query_tile = q_start // q_block_size
            rows = q_end - q_start
            for head in range(heads):
                query = q[sample, q_start:q_end, head].float()
                running_max = torch.full((rows,), -torch.inf, device=q.device)
                running_sum = torch.zeros_like(running_max)
                accumulator = torch.zeros((rows, v.shape[-1]), device=q.device)
                for traversal_index, kv_tile in enumerate(range(num_kv_tiles - 1, -1, -1)):
                    context = ProxyContext(
                        sample_id=identifiers[sample],
                        layer=layer,
                        head=head,
                        kv_group=head // kv_per_group,
                        denoising_iteration=denoising_iteration,
                        noise_bucket=noise_bucket,
                        query_tile=query_tile,
                        kv_tile=kv_tile,
                        traversal_index=traversal_index,
                        num_kv_tiles=num_kv_tiles,
                    )
                    stats.total_physical_tiles += 1
                    decision = policy.decide(context, resolver)
                    # Direct lookup is deliberately outside the policy-facing
                    # resolver call. This separation is tested for leakage.
                    oracle = None
                    if oracle_targets is not None:
                        oracle = oracle_targets.records.get(MetadataStore.key(context))
                        stats.oracle_skip_tiles += int(oracle is not None and oracle.skipped)
                    if decision is Decision.PRE_SKIP:
                        stats.pre_skipped_tiles += 1
                        if oracle is not None:
                            if oracle.skipped:
                                stats.correct_pre_skips += 1
                            else:
                                stats.false_skips += 1
                                stats.false_skips_new_max += int(oracle.introduced_new_max)
                        emitted.write(
                            context,
                            SourceMetadata(
                                score=None,
                                skipped=True,
                                introduced_new_max=False,
                                was_pre_skipped=True,
                                noise_bucket=context.noise_bucket,
                            ),
                        )
                        # Crucially: no target QK, max update, softmax, or PV.
                        continue

                    stats.qk_tiles_computed += 1
                    k_start = kv_tile * kv_block_size
                    k_end = min(length, k_start + kv_block_size)
                    keys = k[sample, k_start:k_end, head].float()
                    scores = torch.matmul(query, keys.transpose(0, 1)) * scale
                    local_max = scores.amax(-1)
                    gap = local_max - running_max
                    tile_score = float(gap.max().exp().item())
                    exact_skip = bool((gap < log_threshold).all().item())
                    introduced_new_max = bool((local_max > running_max).any().item())
                    emitted.write(
                        context,
                        SourceMetadata(
                            score=tile_score,
                            skipped=exact_skip,
                            introduced_new_max=introduced_new_max,
                            noise_bucket=context.noise_bucket,
                        ),
                    )
                    if exact_skip:
                        stats.exact_blasst_skipped_tiles += 1
                        continue
                    next_max = torch.maximum(running_max, local_max)
                    old_scale = torch.exp(running_max - next_max)
                    old_scale = torch.where(torch.isfinite(old_scale), old_scale, torch.zeros_like(old_scale))
                    probabilities = torch.exp(scores - next_max[:, None])
                    values = v[sample, k_start:k_end, head].float()
                    running_sum = running_sum * old_scale + probabilities.sum(-1)
                    accumulator = accumulator * old_scale[:, None] + torch.matmul(probabilities, values)
                    running_max = next_max
                output[sample, q_start:q_end, head] = (
                    accumulator / running_sum.clamp_min(1e-20)[:, None]
                ).to(q.dtype)
    return ProxyReferenceResult(output=output, stats=stats, metadata=emitted)
