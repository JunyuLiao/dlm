"""Runtime planning and correctness reference for structural BLASST features.

The helpers in this module never use current-step dense QK as a production
predictor.  They consume metadata recorded by the preceding denoising step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch

SignatureAggregation = Literal["majority", "all", "any"]
GroupingMode = Literal[
    "none",
    "mask_position",
    "previous_signature_local",
    "previous_signature_global_experimental",
    "oracle_hamming_debug",
]
KVOrderMode = Literal[
    "reverse_baseline",
    "local_first_static",
    "visible_first_static",
    "previous_step_priority",
    "previous_step_priority_plus_local",
    "oracle_current_step_debug",
]


@dataclass(frozen=True)
class StructuralOptimizationConfig:
    grouping_mode: GroupingMode = "previous_signature_local"
    signature_aggregation: SignatureAggregation = "majority"
    regroup_interval: int = 1
    priority_mode: KVOrderMode = "previous_step_priority"
    priority_prefix_k: int = 4
    skip_group_rows: int = 32
    dense_fallback_threshold: int | None = None
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.05
    delta: float = 0.05

    def __post_init__(self) -> None:
        if self.regroup_interval <= 0:
            raise ValueError("regroup_interval must be positive")
        if self.priority_prefix_k not in (0, 2, 4, 8, 16):
            raise ValueError("priority_prefix_k must be one of 0, 2, 4, 8, 16")
        if self.skip_group_rows not in (16, 32, 64, 128):
            raise ValueError("skip_group_rows must be one of 16, 32, 64, 128")
        if self.dense_fallback_threshold is not None and self.dense_fallback_threshold < 0:
            raise ValueError("dense_fallback_threshold must be nonnegative")


@dataclass
class PreviousStepMetadata:
    """Head-aggregated metadata for one request and transformer layer."""

    row_signature: torch.Tensor  # [valid rows, ceil(num KV tiles / 64)], int64 bit words
    row_sparsity: torch.Tensor  # [valid rows]
    top1_kv_tile: torch.Tensor  # [valid rows]
    top2_kv_tile: torch.Tensor  # [valid rows]
    keep_fraction: torch.Tensor  # [Q supertiles, KV tiles]
    top_score_coverage: torch.Tensor  # [Q supertiles, KV tiles]


class DenoisingMetadataStore:
    """Request-isolated double buffer; ``commit_step`` swaps current/previous."""

    def __init__(self) -> None:
        self._previous: dict[object, dict[int, PreviousStepMetadata]] = {}
        self._current: dict[object, dict[int, PreviousStepMetadata]] = {}

    def previous(self, request_id: object, layer: int) -> PreviousStepMetadata | None:
        return self._previous.get(request_id, {}).get(layer)

    def write_current(self, request_id: object, layer: int, metadata: PreviousStepMetadata) -> None:
        self._current.setdefault(request_id, {})[layer] = metadata

    def commit_step(self) -> None:
        self._previous, self._current = self._current, {}

    def remove_request(self, request_id: object) -> None:
        self._previous.pop(request_id, None)
        self._current.pop(request_id, None)

    def clear(self) -> None:
        self._previous.clear()
        self._current.clear()


def _pack_skip_bits(skip: torch.Tensor) -> torch.Tensor:
    """Pack ``[rows, tiles]`` booleans into signed int64 storage words."""
    rows, tiles = skip.shape
    words = (tiles + 63) // 64
    result = torch.zeros((rows, words), dtype=torch.int64, device=skip.device)
    # Use 63 payload bits per arithmetic pass plus an explicit sign-bit term;
    # this avoids unsupported uint64 arithmetic on several PyTorch backends.
    for word in range(words):
        for bit in range(min(64, tiles - word * 64)):
            value = skip[:, word * 64 + bit].to(torch.int64)
            if bit == 63:
                result[:, word] += value * -(1 << 63)
            else:
                result[:, word] += value * (1 << bit)
    return result


def aggregate_step_metadata(
    row_skip_by_head: torch.Tensor,
    tile_scores_by_head: torch.Tensor,
    *,
    q_block_size: int = 128,
    aggregation: SignatureAggregation = "majority",
) -> PreviousStepMetadata:
    """Build previous-step predictors from ``[heads, rows, KV tiles]`` data."""
    if row_skip_by_head.shape != tile_scores_by_head.shape or row_skip_by_head.ndim != 3:
        raise ValueError("skip decisions and scores must have shape [heads, rows, KV tiles]")
    decisions = row_skip_by_head.bool()
    if aggregation == "majority":
        aggregated_skip = decisions.sum(0) * 2 >= decisions.shape[0]
    elif aggregation == "all":
        aggregated_skip = decisions.all(0)
    elif aggregation == "any":
        aggregated_skip = decisions.any(0)
    else:
        raise ValueError(f"unknown signature aggregation: {aggregation}")
    rows, kv_tiles = aggregated_skip.shape
    top_k = min(2, kv_tiles)
    top = tile_scores_by_head.mean(0).topk(top_k, dim=-1).indices
    top1 = top[:, 0]
    top2 = top[:, 1] if top_k == 2 else top1.clone()
    q_tiles = math.ceil(rows / q_block_size)
    keep_fraction = torch.zeros((q_tiles, kv_tiles), dtype=torch.float32, device=decisions.device)
    coverage = torch.zeros_like(keep_fraction)
    top_flags = torch.zeros_like(decisions)
    top_per_head = tile_scores_by_head.topk(top_k, dim=-1).indices
    top_flags.scatter_(2, top_per_head, True)
    for q_tile in range(q_tiles):
        start, end = q_tile * q_block_size, min(rows, (q_tile + 1) * q_block_size)
        keep_fraction[q_tile] = (~decisions[:, start:end]).float().mean((0, 1))
        coverage[q_tile] = top_flags[:, start:end].float().mean((0, 1))
    return PreviousStepMetadata(
        row_signature=_pack_skip_bits(aggregated_skip),
        row_sparsity=aggregated_skip.float().mean(-1),
        top1_kv_tile=top1,
        top2_kv_tile=top2,
        keep_fraction=keep_fraction,
        top_score_coverage=coverage,
    )


def build_query_permutation(
    mask_state: torch.Tensor,
    *,
    mode: GroupingMode,
    previous: PreviousStepMetadata | None = None,
    q_block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``perm[permuted]=original`` and ``inverse[original]=permuted``."""
    if mask_state.ndim != 1:
        raise ValueError("mask_state must contain one boolean per valid row")
    rows = mask_state.numel()
    if mode == "none":
        perm = torch.arange(rows, device=mask_state.device)
    else:
        # No prior state is the required deterministic mask/position fallback.
        effective_mode = "mask_position" if previous is None else mode
        if previous is not None and previous.row_signature.shape[0] != rows:
            raise ValueError("previous metadata length does not match the request")

        if effective_mode == "oracle_hamming_debug":
            assert previous is not None
            remaining = set(range(rows))
            ordered = []
            while remaining:
                if not ordered:
                    chosen = min(remaining, key=lambda index: (int(mask_state[index]), index))
                else:
                    prior = previous.row_signature[ordered[-1]].tolist()

                    def hamming(index: int) -> tuple[int, int, int]:
                        current = previous.row_signature[index].tolist()
                        distance = sum(
                            ((int(left) & ((1 << 64) - 1)) ^ (int(right) & ((1 << 64) - 1))).bit_count()
                            for left, right in zip(prior, current)
                        )
                        return (int(mask_state[index] != mask_state[ordered[-1]]), distance, index)

                    chosen = min(remaining, key=hamming)
                ordered.append(chosen)
                remaining.remove(chosen)
        else:
            position = torch.arange(rows, dtype=torch.int64, device=mask_state.device)
            local = mode == "previous_signature_local"
            composite = position // q_block_size if local else torch.zeros_like(position)
            composite = (composite << 1) | mask_state.to(torch.int64)
            if effective_mode == "mask_position":
                # Three locality bits retain 16-row neighborhoods while the
                # stable sort uses original row order as its final tie-break.
                composite = (composite << 3) | ((position % q_block_size) // 16)
            else:
                assert previous is not None
                if previous.row_signature.device != mask_state.device:
                    raise ValueError("previous metadata and mask_state must share a device")
                signature_bucket = torch.zeros(rows, dtype=torch.int64, device=mask_state.device)
                for word in range(previous.row_signature.shape[1]):
                    signature_bucket.bitwise_xor_(previous.row_signature[:, word] & 0xFFF)
                sparsity_bucket = (previous.row_sparsity * 8).to(torch.int64).clamp_(0, 7)
                composite = (composite << 16) | previous.top1_kv_tile.to(torch.int64).clamp_(0, 65535)
                composite = (composite << 14) | (previous.top2_kv_tile.to(torch.int64).div(4, rounding_mode="floor") & 0x3FFF)
                composite = (composite << 3) | sparsity_bucket
                composite = (composite << 12) | signature_bucket
                composite = (composite << 2) | ((position % q_block_size) // 32)
            perm = torch.argsort(composite, stable=True)
        if effective_mode == "oracle_hamming_debug":
            perm = torch.tensor(ordered, dtype=torch.long, device=mask_state.device)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(rows, device=perm.device)
    return perm, inverse


def build_kv_priority_order(
    num_kv_tiles: int,
    *,
    mode: KVOrderMode,
    priority_prefix_k: int,
    q_tile: int = 0,
    previous: PreviousStepMetadata | None = None,
    visible_fraction: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.05,
    delta: float = 0.05,
) -> list[int]:
    """Build a unique priority prefix followed by untouched reverse order."""
    if num_kv_tiles <= 0:
        return []
    reverse = list(range(num_kv_tiles - 1, -1, -1))
    if priority_prefix_k == 0 or mode == "reverse_baseline":
        return reverse
    if priority_prefix_k not in (2, 4, 8, 16):
        raise ValueError("priority_prefix_k must be 0, 2, 4, 8, or 16")
    if visible_fraction is None:
        visible_fraction = torch.zeros(num_kv_tiles)
    if visible_fraction.numel() != num_kv_tiles:
        raise ValueError("visible_fraction must have one value per KV tile")
    if mode == "local_first_static":
        importance = torch.tensor([-abs(tile - q_tile) for tile in range(num_kv_tiles)], dtype=torch.float32)
    elif mode == "visible_first_static":
        importance = visible_fraction.float().cpu()
    else:
        if previous is None:
            return reverse
        row = min(q_tile, previous.keep_fraction.shape[0] - 1)
        locality = torch.tensor(
            [1.0 / (1.0 + abs(tile - q_tile)) for tile in range(num_kv_tiles)], dtype=torch.float32
        )
        use_locality = mode == "previous_step_priority_plus_local"
        importance = (
            alpha * previous.keep_fraction[row].float().cpu()
            + beta * previous.top_score_coverage[row].float().cpu()
            + (gamma * locality if use_locality else 0.0)
            + delta * visible_fraction.float().cpu()
        )
    ranked = sorted(range(num_kv_tiles), key=lambda tile: (-float(importance[tile]), tile))
    prefix = ranked[: min(priority_prefix_k, num_kv_tiles)]
    selected = set(prefix)
    return prefix + [tile for tile in reverse if tile not in selected]


@dataclass
class ReferenceStructuralStats:
    row_skipped: int = 0
    row_total: int = 0
    microgroups_skipped: int = 0
    microgroups_total: int = 0
    parent_tiles_skipped: int = 0
    parent_tiles_total: int = 0
    bmm2_groups_skipped: int = 0
    bmm2_groups_total: int = 0
    v_loads_skipped: int = 0
    v_loads_total: int = 0
    dense_fallbacks: int = 0
    partial_tiles: int = 0

    def rates(self) -> dict[str, float]:
        ratio = lambda a, b: a / b if b else 0.0
        return {
            "row_skip_rate": ratio(self.row_skipped, self.row_total),
            "microgroup_skip_rate": ratio(self.microgroups_skipped, self.microgroups_total),
            "full_parent_tile_skip_rate": ratio(self.parent_tiles_skipped, self.parent_tiles_total),
            "bmm2_flop_skip_rate": ratio(self.bmm2_groups_skipped, self.bmm2_groups_total),
            "v_load_skip_rate": ratio(self.v_loads_skipped, self.v_loads_total),
            "dense_fallback_rate": ratio(self.dense_fallbacks, self.parent_tiles_total),
        }


@dataclass
class ReferenceResult:
    output: torch.Tensor
    stats: ReferenceStructuralStats
    row_skip_by_head: list[torch.Tensor] = field(default_factory=list)
    tile_scores_by_head: list[torch.Tensor] = field(default_factory=list)


def structural_blasst_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    blasst_lambda: float,
    query_permutations: list[torch.Tensor] | None = None,
    kv_orders: list[list[list[int]]] | None = None,
    sequence_lengths: list[int] | None = None,
    attention_bias: torch.Tensor | None = None,
    q_block_size: int = 128,
    kv_block_size: int = 64,
    skip_group_rows: int = 128,
    dense_fallback_threshold: int | None = None,
) -> ReferenceResult:
    """Unfused correctness path with real group-level state omission semantics."""
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("reference requires same-shape [batch, sequence, heads, dim] tensors")
    if not 0.0 <= blasst_lambda <= 1.0:
        raise ValueError("blasst_lambda must be in [0, 1]")
    if skip_group_rows not in (16, 32, 64, 128) or q_block_size % skip_group_rows:
        raise ValueError("skip groups must be 16/32/64/128 and divide the parent Q tile")
    batch, padded_len, heads, dim = q.shape
    lengths = sequence_lengths or [padded_len] * batch
    if len(lengths) != batch or any(length < 0 or length > padded_len for length in lengths):
        raise ValueError("invalid sequence_lengths")
    stats = ReferenceStructuralStats()
    output = torch.zeros_like(q)
    all_decisions: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    log_threshold = math.log(blasst_lambda) if blasst_lambda else -math.inf
    scale = dim**-0.5

    for batch_index, length in enumerate(lengths):
        perm = (
            query_permutations[batch_index]
            if query_permutations is not None
            else torch.arange(length, device=q.device)
        )
        if perm.numel() != length or not torch.equal(perm.sort().values, torch.arange(length, device=q.device)):
            raise ValueError("each query permutation must contain every valid row exactly once")
        inverse = torch.empty_like(perm)
        inverse[perm] = torch.arange(length, device=q.device)
        q_rows = q[batch_index, :length].index_select(0, perm).transpose(0, 1).float()
        result_rows = torch.zeros((heads, length, dim), device=q.device)
        num_kv_tiles = math.ceil(length / kv_block_size)
        decisions = torch.zeros((heads, length, num_kv_tiles), dtype=torch.bool, device=q.device)
        score_record = torch.full((heads, length, num_kv_tiles), -torch.inf, device=q.device)

        for q_tile, q_start in enumerate(range(0, length, q_block_size)):
            q_end = min(length, q_start + q_block_size)
            rows = q_end - q_start
            query = q_rows[:, q_start:q_end]
            running_max = torch.full((heads, rows), -torch.inf, device=q.device)
            running_sum = torch.zeros_like(running_max)
            accumulator = torch.zeros((heads, rows, dim), device=q.device)
            order = (
                kv_orders[batch_index][q_tile]
                if kv_orders is not None
                else list(range(num_kv_tiles - 1, -1, -1))
            )
            if sorted(order) != list(range(num_kv_tiles)):
                raise ValueError("KV order must visit every tile exactly once")
            groups = math.ceil(rows / skip_group_rows)
            fallback_threshold = groups if dense_fallback_threshold is None else dense_fallback_threshold
            for kv_tile in order:
                k_start, k_end = kv_tile * kv_block_size, min(length, (kv_tile + 1) * kv_block_size)
                keys = k[batch_index, k_start:k_end].transpose(0, 1).float()
                scores = torch.matmul(query, keys.transpose(-1, -2)) * scale
                if attention_bias is not None:
                    original_rows = perm[q_start:q_end]
                    scores += attention_bias[batch_index].index_select(1, original_rows)[
                        :, :, k_start:k_end
                    ].float()
                local_max = scores.amax(-1)
                gaps = local_max - running_max
                row_skip = gaps < log_threshold  # strict by design
                decisions[:, q_start:q_end, kv_tile] = row_skip
                score_record[:, q_start:q_end, kv_tile] = gaps.exp()
                stats.row_skipped += int(row_skip.sum())
                stats.row_total += row_skip.numel()
                active = []
                for group in range(groups):
                    start, end = group * skip_group_rows, min(rows, (group + 1) * skip_group_rows)
                    active.append(~row_skip[:, start:end].all(-1))
                active_groups = torch.stack(active, dim=-1)
                active_count = active_groups.sum(-1)
                parent_skip = active_count == 0
                dense_fallback = (active_count >= fallback_threshold) & ~parent_skip
                effective_active = torch.where(dense_fallback[:, None], True, active_groups)
                stats.microgroups_skipped += int((~active_groups).sum())
                stats.microgroups_total += active_groups.numel()
                stats.parent_tiles_skipped += int(parent_skip.sum())
                stats.parent_tiles_total += heads
                stats.bmm2_groups_skipped += int((~effective_active).sum())
                stats.bmm2_groups_total += effective_active.numel()
                stats.v_loads_skipped += int(parent_skip.sum())
                stats.v_loads_total += heads
                stats.dense_fallbacks += int(dense_fallback.sum())
                stats.partial_tiles += int(((active_count > 0) & (active_count < groups)).sum())
                values = v[batch_index, k_start:k_end].transpose(0, 1).float()
                for group in range(groups):
                    start, end = group * skip_group_rows, min(rows, (group + 1) * skip_group_rows)
                    head_active = effective_active[:, group]
                    if not bool(head_active.any()):
                        continue
                    next_max = torch.maximum(running_max[:, start:end], local_max[:, start:end])
                    old_scale = torch.exp(running_max[:, start:end] - next_max)
                    old_scale = torch.where(torch.isfinite(old_scale), old_scale, torch.zeros_like(old_scale))
                    probabilities = torch.exp(scores[:, start:end] - next_max[..., None])
                    mask = head_active[:, None]
                    running_sum[:, start:end] = torch.where(
                        mask,
                        running_sum[:, start:end] * old_scale + probabilities.sum(-1),
                        running_sum[:, start:end],
                    )
                    updated = accumulator[:, start:end] * old_scale[..., None] + torch.matmul(probabilities, values)
                    accumulator[:, start:end] = torch.where(mask[..., None], updated, accumulator[:, start:end])
                    running_max[:, start:end] = torch.where(mask, next_max, running_max[:, start:end])
            result_rows[:, q_start:q_end] = accumulator / running_sum.clamp_min(1e-20)[..., None]
        output[batch_index, :length] = result_rows.index_select(1, inverse).transpose(0, 1).to(q.dtype)
        all_decisions.append(decisions)
        all_scores.append(score_record)
    return ReferenceResult(output, stats, all_decisions, all_scores)
