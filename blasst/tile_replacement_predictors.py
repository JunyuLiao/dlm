"""Deterministic mass and value predictors for tile-replacement experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .oracle_mass_pruning import BlasstTileDecisions, ExactTileStatistics
from .tile_replacement_reference import exact_conditional_value, exact_log_z


@dataclass(frozen=True)
class TileLogits:
    scores: torch.Tensor
    values: torch.Tensor
    valid_tokens: torch.Tensor


@dataclass(frozen=True)
class SummarySlots:
    keys: torch.Tensor
    values: torch.Tensor
    log_weights: torch.Tensor

    def __post_init__(self) -> None:
        if self.keys.ndim != 5 or self.values.ndim != 5:
            raise ValueError("summary keys and values must be rank five")
        if self.keys.shape[:-1] != self.values.shape[:-1]:
            raise ValueError("summary keys and values must share batch/head/tile/slot axes")
        if self.log_weights.shape != self.keys.shape[:-1]:
            raise ValueError("summaries must have [batch, heads, tiles, slots, dim]")


def tiled_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    kv_block_size: int = 64,
) -> TileLogits:
    """Return FP32 tile logits and repeated values for one query tile."""

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must use [batch, sequence, heads, dim]")
    repeats = q.shape[2] // k.shape[2]
    if repeats <= 0 or repeats * k.shape[2] != q.shape[2]:
        raise ValueError("query heads must be divisible by KV heads")
    k = k.repeat_interleave(repeats, dim=2)
    v = v.repeat_interleave(repeats, dim=2)
    pad = (-k.shape[1]) % kv_block_size
    valid = torch.ones(k.shape[1] + pad, dtype=torch.bool, device=k.device)
    if pad:
        valid[-pad:] = False
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad))
        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad))
    batch, sequence, heads, dim = k.shape
    tiles = sequence // kv_block_size
    qh = q.permute(0, 2, 1, 3).float()
    kh = k.permute(0, 2, 1, 3).float()
    scale = dim**-0.5 if softmax_scale is None else float(softmax_scale)
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
    scores = scores.reshape(batch, heads, q.shape[1], tiles, kv_block_size).permute(
        0, 1, 3, 2, 4
    )
    valid_tiles = valid.reshape(tiles, kv_block_size)
    scores = scores.masked_fill(~valid_tiles[None, None, :, None, :], -torch.inf)
    values = v.permute(0, 2, 1, 3).reshape(
        batch, heads, tiles, kv_block_size, v.shape[-1]
    ).float()
    return TileLogits(scores=scores, values=values, valid_tokens=valid_tiles)


def analytic_log_z_predictions(logits: TileLogits) -> dict[str, torch.Tensor]:
    """Requested scalar log-mass baselines from token logits."""

    scores = logits.scores
    valid = logits.valid_tokens[None, None, :, None, :]
    count = valid.sum(-1).clamp_min(1).to(scores.dtype)
    finite_scores = torch.where(valid, scores, 0.0)
    mean = finite_scores.sum(-1) / count
    centered = torch.where(valid, scores - mean[..., None], 0.0)
    variance = centered.square().sum(-1) / count
    maximum = scores.amax(-1)
    return {
        "exact": torch.logsumexp(scores, dim=-1),
        "mean_logit": mean + count.log(),
        "maximum_logit": maximum,
        "maximum_logit_count": maximum + count.log(),
        "mean_variance": mean + 0.5 * variance + count.log(),
    }


def _repeat_kv_heads(
    k: torch.Tensor, v: torch.Tensor, query_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    repeats = query_heads // k.shape[2]
    if repeats <= 0 or repeats * k.shape[2] != query_heads:
        raise ValueError("query heads must be divisible by KV heads")
    return k.repeat_interleave(repeats, 2), v.repeat_interleave(repeats, 2)


def uniform_pool_summaries(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    query_heads: int,
    slots: int,
    kv_block_size: int = 64,
) -> SummarySlots:
    """Contiguous uniform K/V pooling with exact group-count biases."""

    if kv_block_size % slots:
        raise ValueError("slots must divide kv_block_size")
    k, v = _repeat_kv_heads(k, v, query_heads)
    if k.shape[1] % kv_block_size:
        raise ValueError("uniform summaries require complete KV tiles")
    batch, sequence, heads, dim = k.shape
    tiles = sequence // kv_block_size
    width = kv_block_size // slots
    kt = k.permute(0, 2, 1, 3).reshape(batch, heads, tiles, slots, width, dim)
    vt = v.permute(0, 2, 1, 3).reshape(
        batch, heads, tiles, slots, width, v.shape[-1]
    )
    keys = kt.float().mean(-2)
    values = vt.float().mean(-2)
    log_weights = torch.full(
        keys.shape[:-1], math.log(width), dtype=torch.float32, device=k.device
    )
    return SummarySlots(keys, values, log_weights)


def summary_predictions(
    q: torch.Tensor,
    summaries: SummarySlots,
    *,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict per-row logZ and conditional U from R K/V summary pairs."""

    qh = q.permute(0, 2, 1, 3).float()
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    score = torch.einsum("bhrd,bhtsd->bhtrs", qh, summaries.keys.float()) * scale
    score = score + summaries.log_weights[:, :, :, None]
    log_z = torch.logsumexp(score, dim=-1)
    weights = score.softmax(-1)
    value = torch.einsum("bhtrs,bhtsd->bhtrd", weights, summaries.values.float())
    return log_z, value


def proxy_query_summaries(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    slots: int,
    proxy_rows: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    kv_block_size: int = 64,
) -> SummarySlots:
    """Attention-pool deterministic token groups using a mean proxy query."""

    if kv_block_size % slots:
        raise ValueError("slots must divide kv_block_size")
    k, v = _repeat_kv_heads(k, v, q.shape[2])
    if k.shape[1] % kv_block_size:
        raise ValueError("proxy summaries require complete KV tiles")
    qh = q.permute(0, 2, 1, 3).float()
    if proxy_rows is None:
        proxy = qh.mean(-2)[:, :, None]
    else:
        if proxy_rows.shape == qh.shape[:-1]:
            proxy_rows = proxy_rows[:, :, None]
        if proxy_rows.ndim != 4 or proxy_rows.shape[:2] != qh.shape[:2] or proxy_rows.shape[-1] != qh.shape[-2]:
            raise ValueError(
                "proxy_rows must have [batch, heads, rows] or [batch, heads, tiles, rows]"
            )
        weights = proxy_rows.float()
        proxy = (qh[:, :, None] * weights[..., None]).sum(-2) / weights.sum(-1).clamp_min(1)[..., None]
    batch, sequence, heads, dim = k.shape
    tiles = sequence // kv_block_size
    width = kv_block_size // slots
    kt = k.permute(0, 2, 1, 3).reshape(batch, heads, tiles, slots, width, dim).float()
    vt = v.permute(0, 2, 1, 3).reshape(
        batch, heads, tiles, slots, width, v.shape[-1]
    ).float()
    scale = dim**-0.5 if softmax_scale is None else float(softmax_scale)
    if proxy.shape[2] == 1:
        proxy = proxy.expand(batch, heads, tiles, dim)
    elif proxy.shape[2] != tiles:
        raise ValueError("per-tile proxy row mask does not match KV tile count")
    score = torch.einsum("bhtd,bhtswd->bhtsw", proxy, kt) * scale
    attention = score.softmax(-1)
    keys = torch.einsum("bhtsw,bhtswd->bhtsd", attention, kt)
    values = torch.einsum("bhtsw,bhtswd->bhtsd", attention, vt)
    entropy = -(attention * attention.clamp_min(1e-30).log()).sum(-1)
    return SummarySlots(keys, values, entropy)


def _batched_kmeans_assignments(
    features: torch.Tensor, slots: int, iterations: int
) -> torch.Tensor:
    """Small deterministic Lloyd solver over the penultimate token axis."""

    tokens = features.shape[-2]
    initial = torch.linspace(0, tokens - 1, slots, device=features.device).round().long()
    centers = features.index_select(-2, initial).clone()
    assignment = torch.zeros(features.shape[:-1], dtype=torch.long, device=features.device)
    for _ in range(iterations):
        distance = (features[..., :, None, :] - centers[..., None, :, :]).square().sum(-1)
        assignment = distance.argmin(-1)
        next_centers = []
        for slot in range(slots):
            selected = assignment == slot
            count = selected.sum(-1).clamp_min(1)
            center = (features * selected[..., None]).sum(-2) / count[..., None]
            next_centers.append(center)
        centers = torch.stack(next_centers, dim=-2)
    return assignment


def kmeans_pool_summaries(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    query_heads: int,
    slots: int,
    joint: bool = False,
    iterations: int = 5,
    kv_block_size: int = 64,
) -> SummarySlots:
    """K-space or joint K/V centroid summaries with population biases."""

    k, v = _repeat_kv_heads(k, v, query_heads)
    if k.shape[1] % kv_block_size:
        raise ValueError("k-means summaries require complete KV tiles")
    batch, sequence, heads, dim = k.shape
    tiles = sequence // kv_block_size
    kt = k.permute(0, 2, 1, 3).reshape(batch, heads, tiles, kv_block_size, dim).float()
    vt = v.permute(0, 2, 1, 3).reshape(
        batch, heads, tiles, kv_block_size, v.shape[-1]
    ).float()
    if joint:
        kn = kt / kt.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
        vn = vt / vt.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
        features = torch.cat((kn, vn), dim=-1)
    else:
        features = kt
    assignment = _batched_kmeans_assignments(features, slots, iterations)
    keys, values, counts = [], [], []
    for slot in range(slots):
        selected = assignment == slot
        count = selected.sum(-1).clamp_min(1)
        keys.append((kt * selected[..., None]).sum(-2) / count[..., None])
        values.append((vt * selected[..., None]).sum(-2) / count[..., None])
        counts.append(count.float())
    return SummarySlots(
        torch.stack(keys, -2),
        torch.stack(values, -2),
        torch.stack(counts, -1).log(),
    )


def simple_value_predictions(
    logits: TileLogits,
    statistics: ExactTileStatistics,
    *,
    top_counts: tuple[int, ...] = (1, 2, 4, 8),
) -> dict[str, torch.Tensor]:
    """Zero, mean, norm-weighted, top-m, and oracle-SVD value baselines."""

    shape = (*logits.scores.shape[:-1], logits.values.shape[-1])
    result: dict[str, torch.Tensor] = {
        "zero": torch.zeros(shape, dtype=torch.float32, device=logits.scores.device),
    }
    mean = logits.values.mean(-2)
    result["mean_v"] = mean[..., None, :].expand(shape)
    norms = torch.linalg.vector_norm(logits.values, dim=-1)
    weighted = (logits.values * norms[..., None]).sum(-2) / norms.sum(-1).clamp_min(1e-8)[
        ..., None
    ]
    result["norm_weighted_mean_v"] = weighted[..., None, :].expand(shape)
    for count in top_counts:
        indices = logits.scores.topk(count, dim=-1).indices
        expanded = logits.values[..., None, :, :].expand(
            *logits.values.shape[:-2], logits.scores.shape[-2], *logits.values.shape[-2:]
        )
        gathered = torch.gather(
            expanded,
            -2,
            indices[..., None].expand(*indices.shape, logits.values.shape[-1]),
        )
        result[f"top_{count}_mean_v"] = gathered.mean(-2)
    exact_u = exact_conditional_value(statistics)
    centered_values = logits.values - mean[..., None, :]
    _, _, vh = torch.linalg.svd(centered_values, full_matrices=False)
    for rank in (1, 2, 4, 8):
        basis = vh[..., :rank, :]
        centered_u = exact_u - mean[..., None, :]
        coefficient = torch.einsum("bhtrd,bhtsd->bhtrs", centered_u, basis)
        projection = torch.einsum("bhtrs,bhtsd->bhtrd", coefficient, basis)
        result[f"oracle_svd_{rank}"] = mean[..., None, :] + projection
    return result


def oracle_row_centroids(exact_u: torch.Tensor, slots: int, iterations: int = 5) -> torch.Tensor:
    """Oracle compression upper bound: nearest centroid of exact row U."""

    features = exact_u.permute(0, 1, 2, 3, 4)
    assignment = _batched_kmeans_assignments(features, slots, iterations)
    centers = []
    for slot in range(slots):
        selected = assignment == slot
        count = selected.sum(-1).clamp_min(1)
        centers.append((features * selected[..., None]).sum(-2) / count[..., None])
    centers_tensor = torch.stack(centers, -2)
    return torch.gather(
        centers_tensor,
        -2,
        assignment[..., None].expand(*assignment.shape, exact_u.shape[-1]),
    )


def normalized_mass_from_log_z(log_z: torch.Tensor) -> torch.Tensor:
    return log_z.softmax(dim=2)


def online_proxy_score(
    statistics: ExactTileStatistics,
    decisions: BlasstTileDecisions | None = None,
    traversal: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Existing online mass relative to the retained prefix plus current tile."""

    tiles = statistics.m.shape[2]
    order = traversal or tuple(range(tiles - 1, -1, -1))
    running_m = torch.full_like(statistics.m[:, :, 0], -torch.inf)
    running_l = torch.zeros_like(running_m)
    result = torch.zeros_like(statistics.m)
    for tile in order:
        valid = statistics.l[:, :, tile] > 0
        local_m = statistics.m[:, :, tile]
        local_l = statistics.l[:, :, tile]
        combined_m = torch.maximum(running_m, local_m)
        previous_scaled = torch.where(
            running_l > 0, torch.exp(running_m - combined_m) * running_l, 0.0
        )
        current_scaled = torch.where(
            valid, torch.exp(local_m - combined_m) * local_l, 0.0
        )
        result[:, :, tile] = current_scaled / (previous_scaled + current_scaled).clamp_min(1e-30)
        physical_keep = (
            valid.any(-1)
            if decisions is None
            else decisions.keep[:, :, tile]
        )
        running_m = torch.where(physical_keep[..., None], combined_m, running_m)
        running_l = torch.where(
            physical_keep[..., None], previous_scaled + current_scaled, running_l
        )
    return result


def per_instance_entropy_identity(logits: TileLogits) -> tuple[torch.Tensor, torch.Tensor]:
    """Oracle one-slot capacity bound using each row's own attention weights."""

    probability = logits.scores.softmax(-1)
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum(-1)
    expected_score = (probability * logits.scores).sum(-1)
    log_z = expected_score + entropy
    value = torch.einsum("bhtrw,bhtwd->bhtrd", probability, logits.values)
    return log_z, value
