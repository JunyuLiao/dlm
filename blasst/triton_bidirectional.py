"""Fused BLASST forward kernels for LLaDA's bidirectional self-attention.

The pruning unit is a physical 2D (query, KV) tile.  BMM1 (QK^T) always
runs.  A tile skips softmax and BMM2 (PV) only when every valid query row
votes that its local maximum is below the running maximum by log2(lambda).

The optional pre-QK path is a separate kernel specialization.  It consults
double-buffered previous-step tile scores before loading K/V or computing QK,
while the production sparse baseline retains zero proxy-metadata overhead.
"""

from __future__ import annotations

import math
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import asdict
from typing import Callable, Iterator, Mapping

import torch
import triton
import triton.language as tl


@triton.jit
def _split_rows_2(x, GROUP_ROWS: tl.constexpr, WIDTH: tl.constexpr):
    grouped = tl.reshape(x, (2, GROUP_ROWS, WIDTH)).permute((1, 2, 0))
    return tl.split(grouped)


@triton.jit
def _split_rows_4(x, GROUP_ROWS: tl.constexpr, WIDTH: tl.constexpr):
    grouped = tl.reshape(x, (2, 2, GROUP_ROWS, WIDTH)).permute((2, 3, 1, 0))
    pair01, pair23 = tl.split(grouped)
    group0, group1 = tl.split(pair01)
    group2, group3 = tl.split(pair23)
    return group0, group1, group2, group3


@triton.jit
def _split_rows_8(x, GROUP_ROWS: tl.constexpr, WIDTH: tl.constexpr):
    grouped = tl.reshape(x, (2, 2, 2, GROUP_ROWS, WIDTH)).permute((3, 4, 2, 1, 0))
    half0, half1 = tl.split(grouped)
    pair01, pair23 = tl.split(half0)
    pair45, pair67 = tl.split(half1)
    group0, group1 = tl.split(pair01)
    group2, group3 = tl.split(pair23)
    group4, group5 = tl.split(pair45)
    group6, group7 = tl.split(pair67)
    return group0, group1, group2, group3, group4, group5, group6, group7


@triton.jit
def _join_rows_2(group0, group1, GROUP_ROWS: tl.constexpr, WIDTH: tl.constexpr):
    grouped = tl.join(group0, group1).permute((2, 0, 1))
    return tl.reshape(grouped, (2 * GROUP_ROWS, WIDTH))


@triton.jit
def _join_rows_4(group0, group1, group2, group3, GROUP_ROWS: tl.constexpr, WIDTH: tl.constexpr):
    pair01 = tl.join(group0, group1)
    pair23 = tl.join(group2, group3)
    grouped = tl.join(pair01, pair23).permute((3, 2, 0, 1))
    return tl.reshape(grouped, (4 * GROUP_ROWS, WIDTH))


@triton.jit
def _join_rows_8(
    group0,
    group1,
    group2,
    group3,
    group4,
    group5,
    group6,
    group7,
    GROUP_ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    pair01 = tl.join(group0, group1)
    pair23 = tl.join(group2, group3)
    pair45 = tl.join(group4, group5)
    pair67 = tl.join(group6, group7)
    half0 = tl.join(pair01, pair23)
    half1 = tl.join(pair45, pair67)
    grouped = tl.join(half0, half1).permute((4, 3, 2, 0, 1))
    return tl.reshape(grouped, (8 * GROUP_ROWS, WIDTH))


@triton.jit
def _split_vector_4(vector):
    grouped = tl.reshape(vector, (2, 2)).permute((1, 0))
    pair01, pair23 = tl.split(grouped)
    group0, group1 = tl.split(pair01)
    group2, group3 = tl.split(pair23)
    return group0, group1, group2, group3


@triton.jit
def _split_vector_8(vector):
    grouped = tl.reshape(vector, (2, 2, 2)).permute((2, 1, 0))
    half0, half1 = tl.split(grouped)
    pair01, pair23 = tl.split(half0)
    pair45, pair67 = tl.split(half1)
    group0, group1 = tl.split(pair01)
    group2, group3 = tl.split(pair23)
    group4, group5 = tl.split(pair45)
    group6, group7 = tl.split(pair67)
    return group0, group1, group2, group3, group4, group5, group6, group7


@triton.jit
def _blasst_bidirectional_fwd(
    Q,
    K,
    V,
    O,
    STATS,
    softmax_scale,
    LOG_THRESHOLDS,
    ROW_LOG_THRESHOLDS,
    QUERY_PERM,
    KV_ORDER,
    ROW_KEEP_MASK,
    stride_qb: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_om: tl.constexpr,
    stride_oh: tl.constexpr,
    seqlen: tl.constexpr,
    nheads: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    COLLECT_STATS: tl.constexpr,
    USE_QUERY_PERM: tl.constexpr,
    USE_KV_ORDER: tl.constexpr,
    DUMP_ROW_MASK: tl.constexpr,
    SKIP_GROUP_ROWS: tl.constexpr,
    DENSE_FALLBACK_THRESHOLD: tl.constexpr,
    ROW_MASK_MODE: tl.constexpr,
    MAX_ACTIVE_ROWS: tl.constexpr,
):
    q_tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // nheads
    head = batch_head % nheads

    offs_m = q_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    valid_m = offs_m < seqlen
    # The permutation maps a physical query row to its original token. RoPE
    # has already been applied at the model call site, and O is scattered to
    # this same original position before the output projection.
    original_m = offs_m
    if USE_QUERY_PERM:
        original_m = tl.load(QUERY_PERM + batch * seqlen + offs_m, mask=valid_m, other=0)
    log_threshold = tl.load(LOG_THRESHOLDS + batch)
    if ROW_MASK_MODE != 0:
        row_log_threshold = tl.load(ROW_LOG_THRESHOLDS + batch)

    q_ptrs = Q + batch * stride_qb + original_m[:, None] * stride_qm + head * stride_qh + offs_d[None, :]
    q = tl.load(q_ptrs, mask=valid_m[:, None], other=0.0)

    running_max = tl.full([BLOCK_M], -float("inf"), tl.float32)
    running_sum = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    # FlashAttention 2 traverses noncausal KV tiles from right to left.
    for reverse_block in tl.range(0, tl.cdiv(seqlen, BLOCK_N), num_stages=PIPELINE_STAGES):
        kv_tile = tl.cdiv(seqlen, BLOCK_N) - 1 - reverse_block
        if USE_KV_ORDER:
            kv_tile = tl.load(
                KV_ORDER
                + (batch * tl.cdiv(seqlen, BLOCK_M) + q_tile) * tl.cdiv(seqlen, BLOCK_N)
                + reverse_block
            )
        start_n = kv_tile * BLOCK_N
        key_pos = start_n + offs_n
        valid_n = key_pos < seqlen
        k_ptrs = K + batch * stride_kb + key_pos[:, None] * stride_kn + head * stride_kh + offs_d[None, :]
        k = tl.load(k_ptrs, mask=valid_n[:, None], other=0.0)

        v_ptrs = V + batch * stride_vb + key_pos[:, None] * stride_vn + head * stride_vh + offs_d[None, :]
        # The calibrated baseline retains its original V prefetch. Structural
        # variants defer V until at least one query microgroup is active.
        if SKIP_GROUP_ROWS == BLOCK_M:
            v = tl.load(v_ptrs, mask=valid_n[:, None], other=0.0)

        # Keep the online-softmax state in base-2 units. Triton's exp2 maps
        # directly to the GPU approximation used by optimized FlashAttention;
        # multiplying both scores and the threshold by log2(e) is
        # mathematically equivalent to the natural-exponential formulation.
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * (softmax_scale * 1.4426950408889634)
        scores = tl.where(valid_m[:, None] & valid_n[None, :], scores, -float("inf"))
        local_max = tl.max(scores, axis=1)

        row_gap = local_max - running_max
        if DUMP_ROW_MASK:
            # Store by original query position even when physical rows are
            # regrouped.  A value of one means that the row votes to keep the
            # KV tile under the exact threshold used by this invocation.
            kv_tiles = tl.cdiv(seqlen, BLOCK_N)
            row_mask_offsets = (
                ((batch * nheads + head) * seqlen + original_m) * kv_tiles
                + kv_tile
            )
            tl.store(
                ROW_KEEP_MASK + row_mask_offsets,
                (row_gap >= log_threshold).to(tl.uint8),
                mask=valid_m,
            )
        if SKIP_GROUP_ROWS == BLOCK_M:
            max_gap = tl.max(tl.where(valid_m, row_gap, -float("inf")), axis=0)
            tile_skip = max_gap < log_threshold
            active_rows = valid_m
            candidate_tile = tile_skip
            active_count = tl.sum(valid_m.to(tl.int32), axis=0)
            valid_count = active_count
            if ROW_MASK_MODE != 0:
                row_votes = valid_m & (row_gap >= row_log_threshold)
                active_count = tl.sum(row_votes.to(tl.int32), axis=0)
                candidate_tile = ~tile_skip & (active_count > 0) & (active_count <= MAX_ACTIVE_ROWS)
                active_rows = valid_m & (row_votes | ~candidate_tile)
                if DUMP_ROW_MASK:
                    # Active-voter diagnostics need the deferred exception
                    # rows, not the baseline vote mask stored above.
                    tl.store(
                        ROW_KEEP_MASK + row_mask_offsets,
                        (row_votes & candidate_tile).to(tl.uint8),
                        mask=valid_m,
                    )
            if COLLECT_STATS:
                tl.atomic_add(STATS + 1, 1)
                tl.atomic_add(STATS + 3, 1)
                tl.atomic_add(STATS + 5, 1)
                tl.atomic_add(STATS + 7, 1)
                if tile_skip:
                    tl.atomic_add(STATS, 1)
                    tl.atomic_add(STATS + 2, 1)
                    tl.atomic_add(STATS + 4, 1)
                if ROW_MASK_MODE != 0:
                    if not tile_skip:
                        tl.atomic_add(STATS + 16, valid_count)
                    if candidate_tile:
                        tl.atomic_add(STATS + 12, 1)
                        tl.atomic_add(STATS + 13, valid_count)
                        tl.atomic_add(STATS + 14, active_count)
                        tl.atomic_add(STATS + 15, valid_count - active_count)
            if not tile_skip:
                softmax_rows = valid_m
                if ROW_MASK_MODE == 1:
                    softmax_rows = active_rows
                next_max = tl.where(softmax_rows, tl.maximum(running_max, local_max), running_max)
                old_scale = tl.exp2(running_max - next_max)
                old_scale = tl.where((running_max == -float("inf")) & softmax_rows, 0.0, old_scale)
                old_scale = tl.where(softmax_rows, old_scale, 1.0)
                probabilities = tl.exp2(scores - next_max[:, None])
                probabilities = tl.where(softmax_rows[:, None], probabilities, 0.0)
                running_sum = running_sum * old_scale + tl.sum(probabilities, axis=1)
                acc = acc * old_scale[:, None]
                output_probabilities = probabilities
                if ROW_MASK_MODE == 2:
                    output_probabilities = tl.where(active_rows[:, None], probabilities, 0.0)
                acc += tl.dot(output_probabilities.to(tl.bfloat16), v, out_dtype=tl.float32)
                running_max = next_max
        else:
            NUM_GROUPS: tl.constexpr = BLOCK_M // SKIP_GROUP_ROWS
            grouped_gap = tl.reshape(tl.where(valid_m, row_gap, -float("inf")), (NUM_GROUPS, SKIP_GROUP_ROWS))
            grouped_valid = tl.reshape(valid_m, (NUM_GROUPS, SKIP_GROUP_ROWS))
            valid_groups = tl.sum(grouped_valid.to(tl.int32), axis=1) > 0
            active_groups = valid_groups & (tl.max(grouped_gap, axis=1) >= log_threshold)
            valid_group_count = tl.sum(valid_groups.to(tl.int32), axis=0)
            active_count = tl.sum(active_groups.to(tl.int32), axis=0)
            if NUM_GROUPS == 2:
                group0, group1 = tl.split(active_groups)
                valid0, valid1 = tl.split(valid_groups)
            if NUM_GROUPS == 4:
                group0, group1, group2, group3 = _split_vector_4(active_groups)
                valid0, valid1, valid2, valid3 = _split_vector_4(valid_groups)
            if NUM_GROUPS == 8:
                group0, group1, group2, group3, group4, group5, group6, group7 = _split_vector_8(
                    active_groups
                )
                valid0, valid1, valid2, valid3, valid4, valid5, valid6, valid7 = _split_vector_8(
                    valid_groups
                )

            parent_skip = active_count == 0
            dense_fallback = (active_count >= DENSE_FALLBACK_THRESHOLD) & ~parent_skip
            group0 = (group0 | dense_fallback) & valid0
            group1 = (group1 | dense_fallback) & valid1
            if NUM_GROUPS >= 4:
                group2 = (group2 | dense_fallback) & valid2
                group3 = (group3 | dense_fallback) & valid3
            if NUM_GROUPS == 8:
                group4 = (group4 | dense_fallback) & valid4
                group5 = (group5 | dense_fallback) & valid5
                group6 = (group6 | dense_fallback) & valid6
                group7 = (group7 | dense_fallback) & valid7
            effective_count = active_count
            if dense_fallback:
                effective_count = valid_group_count

            if COLLECT_STATS:
                tl.atomic_add(STATS + 1, 1)
                tl.atomic_add(STATS + 3, valid_group_count)
                tl.atomic_add(STATS + 2, valid_group_count - active_count)
                tl.atomic_add(STATS + 5, valid_group_count)
                tl.atomic_add(STATS + 4, valid_group_count - effective_count)
                tl.atomic_add(STATS + 7, 1)
                if parent_skip:
                    tl.atomic_add(STATS, 1)
                    tl.atomic_add(STATS + 6, 1)
                if dense_fallback:
                    tl.atomic_add(STATS + 8, 1)
                if (active_count > 0) & (active_count < valid_group_count):
                    tl.atomic_add(STATS + 9, 1)

            if not parent_skip:
                active_rows = (offs_m % BLOCK_M < SKIP_GROUP_ROWS) & group0
                active_rows |= (
                    (offs_m % BLOCK_M >= SKIP_GROUP_ROWS)
                    & (offs_m % BLOCK_M < 2 * SKIP_GROUP_ROWS)
                    & group1
                )
                if NUM_GROUPS >= 4:
                    active_rows |= (
                        (offs_m % BLOCK_M >= 2 * SKIP_GROUP_ROWS)
                        & (offs_m % BLOCK_M < 3 * SKIP_GROUP_ROWS)
                        & group2
                    )
                    active_rows |= (
                        (offs_m % BLOCK_M >= 3 * SKIP_GROUP_ROWS)
                        & (offs_m % BLOCK_M < 4 * SKIP_GROUP_ROWS)
                        & group3
                    )
                if NUM_GROUPS == 8:
                    active_rows |= (
                        (offs_m % BLOCK_M >= 4 * SKIP_GROUP_ROWS)
                        & (offs_m % BLOCK_M < 5 * SKIP_GROUP_ROWS)
                        & group4
                    )
                    active_rows |= (
                        (offs_m % BLOCK_M >= 5 * SKIP_GROUP_ROWS)
                        & (offs_m % BLOCK_M < 6 * SKIP_GROUP_ROWS)
                        & group5
                    )
                    active_rows |= (
                        (offs_m % BLOCK_M >= 6 * SKIP_GROUP_ROWS)
                        & (offs_m % BLOCK_M < 7 * SKIP_GROUP_ROWS)
                        & group6
                    )
                    active_rows |= (offs_m % BLOCK_M >= 7 * SKIP_GROUP_ROWS) & group7
                active_rows &= valid_m
                next_max = tl.where(active_rows, tl.maximum(running_max, local_max), running_max)
                old_scale = tl.exp2(running_max - next_max)
                old_scale = tl.where((running_max == -float("inf")) & active_rows, 0.0, old_scale)
                probabilities = tl.exp2(scores - next_max[:, None])
                probabilities = tl.where(active_rows[:, None], probabilities, 0.0)
                running_sum = running_sum * old_scale + tl.sum(probabilities, axis=1)
                acc = acc * old_scale[:, None]
                v = tl.load(v_ptrs, mask=valid_n[:, None], other=0.0)

                if dense_fallback:
                    acc += tl.dot(probabilities.to(tl.bfloat16), v, out_dtype=tl.float32)
                else:
                    if NUM_GROUPS == 2:
                        probability0, probability1 = _split_rows_2(probabilities, SKIP_GROUP_ROWS, BLOCK_N)
                        pv0 = tl.zeros((SKIP_GROUP_ROWS, HEAD_DIM), tl.float32)
                        pv1 = tl.zeros((SKIP_GROUP_ROWS, HEAD_DIM), tl.float32)
                        if group0:
                            pv0 = tl.dot(probability0.to(tl.bfloat16), v, out_dtype=tl.float32)
                        if group1:
                            pv1 = tl.dot(probability1.to(tl.bfloat16), v, out_dtype=tl.float32)
                        acc += _join_rows_2(pv0, pv1, SKIP_GROUP_ROWS, HEAD_DIM)
                    if NUM_GROUPS == 4:
                        probability0, probability1, probability2, probability3 = _split_rows_4(
                            probabilities, SKIP_GROUP_ROWS, BLOCK_N
                        )
                        pv0 = tl.zeros((SKIP_GROUP_ROWS, HEAD_DIM), tl.float32)
                        pv1 = tl.zeros((SKIP_GROUP_ROWS, HEAD_DIM), tl.float32)
                        pv2 = tl.zeros((SKIP_GROUP_ROWS, HEAD_DIM), tl.float32)
                        pv3 = tl.zeros((SKIP_GROUP_ROWS, HEAD_DIM), tl.float32)
                        if group0:
                            pv0 = tl.dot(probability0.to(tl.bfloat16), v, out_dtype=tl.float32)
                        if group1:
                            pv1 = tl.dot(probability1.to(tl.bfloat16), v, out_dtype=tl.float32)
                        if group2:
                            pv2 = tl.dot(probability2.to(tl.bfloat16), v, out_dtype=tl.float32)
                        if group3:
                            pv3 = tl.dot(probability3.to(tl.bfloat16), v, out_dtype=tl.float32)
                        acc += _join_rows_4(pv0, pv1, pv2, pv3, SKIP_GROUP_ROWS, HEAD_DIM)
                    if NUM_GROUPS == 8:
                        p0, p1, p2, p3, p4, p5, p6, p7 = _split_rows_8(
                            probabilities, SKIP_GROUP_ROWS, BLOCK_N
                        )
                        pv0 = tl.zeros((SKIP_GROUP_ROWS, HEAD_DIM), tl.float32)
                        pv1 = tl.zeros((SKIP_GROUP_ROWS, HEAD_DIM), tl.float32)
                        pv2 = tl.zeros((SKIP_GROUP_ROWS, HEAD_DIM), tl.float32)
                        pv3 = tl.zeros((SKIP_GROUP_ROWS, HEAD_DIM), tl.float32)
                        pv4 = tl.zeros((SKIP_GROUP_ROWS, HEAD_DIM), tl.float32)
                        pv5 = tl.zeros((SKIP_GROUP_ROWS, HEAD_DIM), tl.float32)
                        pv6 = tl.zeros((SKIP_GROUP_ROWS, HEAD_DIM), tl.float32)
                        pv7 = tl.zeros((SKIP_GROUP_ROWS, HEAD_DIM), tl.float32)
                        if group0:
                            pv0 = tl.dot(p0.to(tl.bfloat16), v, out_dtype=tl.float32)
                        if group1:
                            pv1 = tl.dot(p1.to(tl.bfloat16), v, out_dtype=tl.float32)
                        if group2:
                            pv2 = tl.dot(p2.to(tl.bfloat16), v, out_dtype=tl.float32)
                        if group3:
                            pv3 = tl.dot(p3.to(tl.bfloat16), v, out_dtype=tl.float32)
                        if group4:
                            pv4 = tl.dot(p4.to(tl.bfloat16), v, out_dtype=tl.float32)
                        if group5:
                            pv5 = tl.dot(p5.to(tl.bfloat16), v, out_dtype=tl.float32)
                        if group6:
                            pv6 = tl.dot(p6.to(tl.bfloat16), v, out_dtype=tl.float32)
                        if group7:
                            pv7 = tl.dot(p7.to(tl.bfloat16), v, out_dtype=tl.float32)
                        acc += _join_rows_8(
                            pv0, pv1, pv2, pv3, pv4, pv5, pv6, pv7, SKIP_GROUP_ROWS, HEAD_DIM
                        )
                running_max = next_max

    output = acc / running_sum[:, None]
    o_ptrs = O + batch * stride_ob + original_m[:, None] * stride_om + head * stride_oh + offs_d[None, :]
    tl.store(o_ptrs, output, mask=valid_m[:, None])


@triton.jit
def _blasst_bidirectional_pre_qk_fwd(
    Q,
    K,
    V,
    O,
    STATS,
    softmax_scale,
    LOG_THRESHOLDS,
    PREVIOUS_LOG_SCORES,
    CURRENT_LOG_SCORES,
    PROXY_LOG_THRESHOLDS,
    stride_qb: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_om: tl.constexpr,
    stride_oh: tl.constexpr,
    seqlen: tl.constexpr,
    nheads: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    COLLECT_STATS: tl.constexpr,
    WARMUP_TILES: tl.constexpr,
    PERIODIC_REFRESH: tl.constexpr,
    ANCHOR_LOCAL: tl.constexpr,
    ANCHOR_SINK: tl.constexpr,
    LOCAL_RADIUS: tl.constexpr,
):
    """Accepted 128x64 parent-tile path with an early metadata gate.

    This is intentionally separate from ``_blasst_bidirectional_fwd`` so the
    sparse control kernel pays no metadata loads, predicates, or stores.
    """
    q_tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // nheads
    head = batch_head % nheads

    offs_m = q_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    valid_m = offs_m < seqlen
    q_ptrs = Q + batch * stride_qb + offs_m[:, None] * stride_qm + head * stride_qh + offs_d[None, :]
    q = tl.load(q_ptrs, mask=valid_m[:, None], other=0.0)

    exact_log_threshold = tl.load(LOG_THRESHOLDS + batch)
    proxy_log_threshold = tl.load(PROXY_LOG_THRESHOLDS + batch_head)
    q_tiles = tl.cdiv(seqlen, BLOCK_M)
    kv_tiles = tl.cdiv(seqlen, BLOCK_N)
    q_to_kv_ratio = BLOCK_M // BLOCK_N

    running_max = tl.full([BLOCK_M], -float("inf"), tl.float32)
    running_sum = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    metadata_trusted = q_tile == q_tile

    for reverse_block in tl.range(0, kv_tiles, num_stages=PIPELINE_STAGES):
        kv_tile = kv_tiles - 1 - reverse_block
        metadata_offset = ((batch_head * q_tiles + q_tile) * kv_tiles) + kv_tile
        previous_log_score = tl.load(PREVIOUS_LOG_SCORES + metadata_offset)

        protected = reverse_block < WARMUP_TILES
        if PERIODIC_REFRESH > 0:
            protected |= reverse_block % PERIODIC_REFRESH == 0
        if ANCHOR_LOCAL:
            local_start = q_tile * q_to_kv_ratio - LOCAL_RADIUS
            local_end = (q_tile + 1) * q_to_kv_ratio + LOCAL_RADIUS
            protected |= (kv_tile >= local_start) & (kv_tile < local_end)
        if ANCHOR_SINK:
            protected |= kv_tile == 0
        pre_skip = (previous_log_score < proxy_log_threshold) & ~protected

        if COLLECT_STATS:
            tl.atomic_add(STATS + 1, 1)
            tl.atomic_add(STATS + 3, 1)
            tl.atomic_add(STATS + 5, 1)
            tl.atomic_add(STATS + 7, 1)
            tl.atomic_add(STATS + 11, 1)

        if pre_skip:
            # The skipped tile and the rest of this traversal are intentionally
            # non-reusable: a false skip could have changed the online maximum
            # used by every later score.
            tl.store(CURRENT_LOG_SCORES + metadata_offset, float("inf"))
            metadata_trusted = False
            if COLLECT_STATS:
                tl.atomic_add(STATS, 1)
                tl.atomic_add(STATS + 2, 1)
                tl.atomic_add(STATS + 4, 1)
                tl.atomic_add(STATS + 6, 1)
                tl.atomic_add(STATS + 10, 1)
        else:
            start_n = kv_tile * BLOCK_N
            key_pos = start_n + offs_n
            valid_n = key_pos < seqlen
            k_ptrs = K + batch * stride_kb + key_pos[:, None] * stride_kn + head * stride_kh + offs_d[None, :]
            k = tl.load(k_ptrs, mask=valid_n[:, None], other=0.0)
            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * (
                softmax_scale * 1.4426950408889634
            )
            scores = tl.where(valid_m[:, None] & valid_n[None, :], scores, -float("inf"))
            local_max = tl.max(scores, axis=1)
            row_gap = local_max - running_max
            max_gap = tl.max(tl.where(valid_m, row_gap, -float("inf")), axis=0)
            tl.store(
                CURRENT_LOG_SCORES + metadata_offset,
                tl.where(metadata_trusted, max_gap, float("inf")),
            )
            tile_skip = max_gap < exact_log_threshold

            if COLLECT_STATS and tile_skip:
                tl.atomic_add(STATS, 1)
                tl.atomic_add(STATS + 2, 1)
                tl.atomic_add(STATS + 4, 1)
                tl.atomic_add(STATS + 6, 1)

            if not tile_skip:
                # Unlike the accepted sparse baseline, this specialized path
                # defers V until both the proxy and exact gates retain a tile.
                v_ptrs = V + batch * stride_vb + key_pos[:, None] * stride_vn + head * stride_vh + offs_d[None, :]
                v = tl.load(v_ptrs, mask=valid_n[:, None], other=0.0)
                next_max = tl.maximum(running_max, local_max)
                old_scale = tl.exp2(running_max - next_max)
                old_scale = tl.where(running_max == -float("inf"), 0.0, old_scale)
                probabilities = tl.exp2(scores - next_max[:, None])
                probabilities = tl.where(valid_m[:, None], probabilities, 0.0)
                running_sum = running_sum * old_scale + tl.sum(probabilities, axis=1)
                acc = acc * old_scale[:, None]
                acc += tl.dot(probabilities.to(tl.bfloat16), v, out_dtype=tl.float32)
                running_max = next_max

    output = acc / running_sum[:, None]
    o_ptrs = O + batch * stride_ob + offs_m[:, None] * stride_om + head * stride_oh + offs_d[None, :]
    tl.store(o_ptrs, output, mask=valid_m[:, None])


@dataclass
class KernelStats:
    skipped_tiles: int
    total_tiles: int
    skipped_microgroups: int = 0
    total_microgroups: int = 0
    skipped_bmm2_groups: int = 0
    total_bmm2_groups: int = 0
    skipped_v_loads: int = 0
    total_v_loads: int = 0
    dense_fallbacks: int = 0
    partial_tiles: int = 0
    pre_qk_skipped_tiles: int = 0
    proxy_metadata_tiles: int = 0
    active_voter_candidate_tiles: int = 0
    active_voter_candidate_row_work: int = 0
    active_voter_selected_rows: int = 0
    active_voter_removed_rows: int = 0
    active_voter_retained_row_work: int = 0

    @property
    def sparsity_ratio(self) -> float:
        return self.skipped_tiles / self.total_tiles if self.total_tiles else 0.0

    @property
    def microgroup_sparsity_ratio(self) -> float:
        return self.skipped_microgroups / self.total_microgroups if self.total_microgroups else 0.0

    @property
    def bmm2_flop_skip_ratio(self) -> float:
        return self.skipped_bmm2_groups / self.total_bmm2_groups if self.total_bmm2_groups else 0.0

    @property
    def v_load_skip_ratio(self) -> float:
        return self.skipped_v_loads / self.total_v_loads if self.total_v_loads else 0.0

    @property
    def pre_qk_skip_ratio(self) -> float:
        return self.pre_qk_skipped_tiles / self.proxy_metadata_tiles if self.proxy_metadata_tiles else 0.0

    @property
    def active_voter_candidate_work_fraction(self) -> float:
        if not self.active_voter_retained_row_work:
            return 0.0
        return self.active_voter_candidate_row_work / self.active_voter_retained_row_work

    @property
    def active_voter_selected_fraction(self) -> float:
        if not self.active_voter_candidate_row_work:
            return 0.0
        return self.active_voter_selected_rows / self.active_voter_candidate_row_work


@dataclass(frozen=True)
class DiffusionLambdaSchedule:
    """Noise-aware thresholds calibrated for LLaDA's denoising trajectory."""

    high_noise_lambda: float = 0.04858582466840744
    mid_noise_lambda: float = 0.4334796965122223
    low_noise_lambda: float = 1.0
    high_noise_boundary: float = 0.75
    low_noise_boundary: float = 0.25

    def __post_init__(self) -> None:
        for name in ("high_noise_lambda", "mid_noise_lambda", "low_noise_lambda"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 <= self.low_noise_boundary <= self.high_noise_boundary <= 1.0:
            raise ValueError("noise boundaries must satisfy 0 <= low <= high <= 1")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> "DiffusionLambdaSchedule":
        return cls(**values)

    def threshold(self, remaining_mask_ratio: float) -> float:
        if not 0.0 <= remaining_mask_ratio <= 1.0:
            raise ValueError("remaining_mask_ratio must be in [0, 1]")
        if remaining_mask_ratio >= self.high_noise_boundary:
            return self.high_noise_lambda
        if remaining_mask_ratio >= self.low_noise_boundary:
            return self.mid_noise_lambda
        return self.low_noise_lambda


@dataclass
class DiffusionKernelController:
    schedule: DiffusionLambdaSchedule
    current_threshold: float | torch.Tensor = 0.04858582466840744

    def __post_init__(self) -> None:
        self.current_threshold = self.schedule.high_noise_lambda

    @property
    def blasst_lambda(self) -> float | torch.Tensor:
        return self.current_threshold

    def set_remaining_mask_ratio(self, ratio: float | torch.Tensor) -> None:
        # Called once per denoising step before the LLaDA forward.
        if isinstance(ratio, torch.Tensor):
            if ratio.ndim != 1 or bool(((ratio < 0) | (ratio > 1)).any()):
                raise ValueError("per-sequence mask ratios must be a vector in [0, 1]")
            self.current_threshold = torch.where(
                ratio >= self.schedule.high_noise_boundary,
                self.schedule.high_noise_lambda,
                torch.where(
                    ratio >= self.schedule.low_noise_boundary,
                    self.schedule.mid_noise_lambda,
                    self.schedule.low_noise_lambda,
                ),
            ).float()
        else:
            self.current_threshold = self.schedule.threshold(ratio)


_DEVICE_STATS: torch.Tensor | None = None
_SCALAR_LOG_THRESHOLD_CACHE: dict[tuple[int, int, float], torch.Tensor] = {}
_TENSOR_LOG_THRESHOLD_CACHE: dict[int, tuple[weakref.ReferenceType[torch.Tensor], int, torch.Tensor]] = {}
_ROW_SCALAR_LOG_THRESHOLD_CACHE: dict[tuple[int, int, float], torch.Tensor] = {}


def reset_kernel_stats(device: torch.device | str = "cuda") -> None:
    global _DEVICE_STATS
    _DEVICE_STATS = torch.zeros(17, dtype=torch.int64, device=device)


def get_kernel_stats(*, reset: bool = False) -> KernelStats:
    global _DEVICE_STATS
    if _DEVICE_STATS is None:
        return KernelStats(0, 0)
    values = _DEVICE_STATS.cpu().tolist()
    result = KernelStats(*(int(value) for value in values))
    if reset:
        _DEVICE_STATS.zero_()
    return result


def blasst_bidirectional_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    *,
    blasst_lambda: float | torch.Tensor = 0.04858582466840744,
    collect_stats: bool = True,
    num_warps: int = 4,
    pipeline_stages: int = 2,
    query_permutation: torch.Tensor | None = None,
    kv_tile_order: torch.Tensor | None = None,
    query_tile_rows: int = 128,
    skip_group_rows: int = 128,
    dense_fallback_threshold: int | None = None,
    row_keep_mask_out: torch.Tensor | None = None,
    enable_pre_qk: bool = False,
    previous_proxy_log_scores: torch.Tensor | None = None,
    current_proxy_log_scores: torch.Tensor | None = None,
    proxy_log_thresholds: torch.Tensor | None = None,
    proxy_warmup_tiles: int = 2,
    proxy_periodic_refresh: int = 8,
    proxy_anchor_local: bool = True,
    proxy_anchor_sink: bool = True,
    proxy_local_radius: int = 1,
    row_mask_variant: str = "none",
    row_lambda: float | torch.Tensor | None = None,
    max_active_rows: int = 32,
) -> torch.Tensor:
    """Drop-in inference replacement using the accepted 128x64 sparse path.

    Structural experiments remain represented in the kernel source for audit
    and reproducibility, but H100 acceptance testing showed that every such
    configuration regresses latency.  The public production entry point
    therefore rejects them instead of silently selecting a slower kernel.
    """
    del deterministic
    if not q.is_cuda or q.dtype != torch.bfloat16:
        raise ValueError("LLaDA BLASST kernel requires CUDA BF16 tensors")
    if q.requires_grad or k.requires_grad or v.requires_grad:
        raise ValueError("LLaDA BLASST kernel is forward-only inference")
    if dropout_p != 0.0 or causal:
        raise ValueError("kernel supports dropout-free bidirectional attention only")
    if window_size != (-1, -1) or softcap != 0.0 or alibi_slopes is not None or return_attn_probs:
        raise NotImplementedError("windowing, softcap, ALiBi, and returned probabilities are unsupported")
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("LLaDA kernel currently requires same-shape MHA tensors [B, S, H, D]")
    if q.shape[-1] != 128:
        raise ValueError("LLaDA kernel is specialized for head dimension 128")
    if row_mask_variant not in ("none", "full", "output"):
        raise ValueError("row_mask_variant must be none, full, or output")
    if not 1 <= max_active_rows <= 128:
        raise ValueError("max_active_rows must be in [1, 128]")
    if enable_pre_qk and row_mask_variant != "none":
        raise ValueError("active-voter diagnostics cannot be combined with pre-QK skipping")
    structural_requested = (
        query_permutation is not None
        or kv_tile_order is not None
        or query_tile_rows != 128
        or skip_group_rows != 128
        or dense_fallback_threshold not in (None, 1)
    )
    if structural_requested:
        raise NotImplementedError(
            "query grouping, KV reordering, and microgroups are disabled: "
            "H100 acceptance benchmarks found no speedup over the calibrated 128x64 sparse kernel"
        )
    if query_tile_rows not in (64, 128):
        raise ValueError("query_tile_rows must be 64 or 128")
    if skip_group_rows not in (16, 32, 64, 128):
        raise ValueError("skip_group_rows must be one of 16, 32, 64, 128")
    if skip_group_rows > query_tile_rows or query_tile_rows % skip_group_rows:
        raise ValueError("skip_group_rows must evenly divide query_tile_rows")
    num_groups = query_tile_rows // skip_group_rows
    if dense_fallback_threshold is None:
        dense_fallback_threshold = num_groups
    if not 1 <= dense_fallback_threshold <= num_groups:
        raise ValueError("dense_fallback_threshold must be in [1, number_of_groups]")
    if enable_pre_qk and (query_tile_rows != 128 or skip_group_rows != 128):
        raise ValueError("pre-QK kernel is specialized for 128x64 parent tiles")
    if proxy_warmup_tiles < 1:
        raise ValueError("proxy_warmup_tiles must be at least 1 so every query has a retained KV tile")
    if proxy_periodic_refresh < 0 or proxy_local_radius < 0:
        raise ValueError("proxy refresh period and local radius must be non-negative")

    global _DEVICE_STATS
    if _DEVICE_STATS is None or _DEVICE_STATS.device != q.device or _DEVICE_STATS.numel() != 17:
        reset_kernel_stats(q.device)
    assert _DEVICE_STATS is not None

    batch, seqlen, nheads, head_dim = q.shape
    q_tiles, kv_tiles = triton.cdiv(seqlen, query_tile_rows), triton.cdiv(seqlen, 64)
    dump_row_mask = row_keep_mask_out is not None
    if enable_pre_qk and dump_row_mask:
        raise ValueError("row-mask dumping is only supported by the post-QK BLASST kernel")
    if row_keep_mask_out is not None:
        expected_row_mask_shape = (batch, nheads, seqlen, kv_tiles)
        if row_keep_mask_out.shape != expected_row_mask_shape:
            raise ValueError(f"row_keep_mask_out must have shape {expected_row_mask_shape}")
        if row_keep_mask_out.device != q.device or row_keep_mask_out.dtype != torch.uint8:
            raise ValueError("row_keep_mask_out must be a CUDA uint8 tensor on the Q device")
        if not row_keep_mask_out.is_contiguous():
            raise ValueError("row_keep_mask_out must be contiguous")
    else:
        row_keep_mask_out = torch.empty(1, dtype=torch.uint8, device=q.device)
    use_query_permutation = query_permutation is not None
    use_kv_order = kv_tile_order is not None
    if query_permutation is not None:
        if query_permutation.shape != (batch, seqlen) or query_permutation.device != q.device:
            raise ValueError("query_permutation must be a CUDA tensor shaped [batch, seqlen]")
        if query_permutation.dtype not in (torch.int32, torch.int64):
            raise ValueError("query_permutation must have integer dtype")
        query_permutation = query_permutation.contiguous()
    else:
        # A valid pointer is still supplied; constexpr flags ensure it is not read.
        query_permutation = torch.empty(1, dtype=torch.int32, device=q.device)
    if kv_tile_order is not None:
        if kv_tile_order.shape != (batch, q_tiles, kv_tiles) or kv_tile_order.device != q.device:
            raise ValueError("kv_tile_order must be CUDA [batch, Q tiles, KV tiles]")
        if kv_tile_order.dtype not in (torch.int32, torch.int64):
            raise ValueError("kv_tile_order must have integer dtype")
        kv_tile_order = kv_tile_order.contiguous()
    else:
        kv_tile_order = torch.empty(1, dtype=torch.int32, device=q.device)
    if isinstance(blasst_lambda, torch.Tensor):
        if blasst_lambda.shape != (batch,) or blasst_lambda.device != q.device:
            raise ValueError("tensor blasst_lambda must be a CUDA vector with one value per sequence")
        if bool(((blasst_lambda < 0) | (blasst_lambda > 1)).any()):
            raise ValueError("blasst_lambda values must be in [0, 1]")
        cached = _TENSOR_LOG_THRESHOLD_CACHE.get(id(blasst_lambda))
        if cached is not None and cached[0]() is blasst_lambda and cached[1] == blasst_lambda._version:
            log_thresholds = cached[2]
        else:
            log_thresholds = torch.where(
                blasst_lambda > 0,
                blasst_lambda.log2(),
                torch.full_like(blasst_lambda, -torch.inf),
            ).float()
            _TENSOR_LOG_THRESHOLD_CACHE[id(blasst_lambda)] = (
                weakref.ref(blasst_lambda),
                blasst_lambda._version,
                log_thresholds,
            )
    else:
        if not 0.0 <= blasst_lambda <= 1.0:
            raise ValueError("blasst_lambda must be in [0, 1]")
        cache_key = (q.device.index or 0, batch, float(blasst_lambda))
        log_thresholds = _SCALAR_LOG_THRESHOLD_CACHE.get(cache_key)
        if log_thresholds is None:
            value = math.log2(blasst_lambda) if blasst_lambda else -math.inf
            log_thresholds = torch.full((batch,), value, dtype=torch.float32, device=q.device)
            _SCALAR_LOG_THRESHOLD_CACHE[cache_key] = log_thresholds
    if row_lambda is None:
        row_log_thresholds = log_thresholds
    elif isinstance(row_lambda, torch.Tensor):
        if row_lambda.shape != (batch,) or row_lambda.device != q.device:
            raise ValueError("tensor row_lambda must be a CUDA vector with one value per sequence")
        if bool(((row_lambda < 0) | (row_lambda > 1)).any()):
            raise ValueError("row_lambda values must be in [0, 1]")
        row_log_thresholds = torch.where(
            row_lambda > 0,
            row_lambda.log2(),
            torch.full_like(row_lambda, -torch.inf),
        ).float()
    else:
        if not 0.0 <= row_lambda <= 1.0:
            raise ValueError("row_lambda must be in [0, 1]")
        row_cache_key = (q.device.index or 0, batch, float(row_lambda))
        row_log_thresholds = _ROW_SCALAR_LOG_THRESHOLD_CACHE.get(row_cache_key)
        if row_log_thresholds is None:
            row_value = math.log2(row_lambda) if row_lambda else -math.inf
            row_log_thresholds = torch.full(
                (batch,), row_value, dtype=torch.float32, device=q.device
            )
            _ROW_SCALAR_LOG_THRESHOLD_CACHE[row_cache_key] = row_log_thresholds
    output = torch.empty_like(q)
    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5
    grid = (triton.cdiv(seqlen, query_tile_rows), batch * nheads)
    if enable_pre_qk:
        expected_metadata_shape = (batch, nheads, q_tiles, kv_tiles)
        for name, tensor in (
            ("previous_proxy_log_scores", previous_proxy_log_scores),
            ("current_proxy_log_scores", current_proxy_log_scores),
        ):
            if tensor is None or tensor.shape != expected_metadata_shape:
                raise ValueError(f"{name} must have shape {expected_metadata_shape}")
            if tensor.device != q.device or tensor.dtype not in (torch.float16, torch.float32):
                raise ValueError(f"{name} must be a CUDA FP16/FP32 tensor on the Q device")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if proxy_log_thresholds is None or proxy_log_thresholds.shape != (batch, nheads):
            raise ValueError(f"proxy_log_thresholds must have shape {(batch, nheads)}")
        if proxy_log_thresholds.device != q.device or proxy_log_thresholds.dtype != torch.float32:
            raise ValueError("proxy_log_thresholds must be CUDA FP32 on the Q device")
        if not proxy_log_thresholds.is_contiguous():
            raise ValueError("proxy_log_thresholds must be contiguous")
        _blasst_bidirectional_pre_qk_fwd[grid](
            q, k, v, output, _DEVICE_STATS, scale, log_thresholds,
            previous_proxy_log_scores, current_proxy_log_scores, proxy_log_thresholds,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            seqlen, nheads,
            BLOCK_M=128,
            BLOCK_N=64,
            HEAD_DIM=128,
            PIPELINE_STAGES=pipeline_stages,
            COLLECT_STATS=collect_stats,
            WARMUP_TILES=proxy_warmup_tiles,
            PERIODIC_REFRESH=proxy_periodic_refresh,
            ANCHOR_LOCAL=proxy_anchor_local,
            ANCHOR_SINK=proxy_anchor_sink,
            LOCAL_RADIUS=proxy_local_radius,
            num_warps=num_warps,
            num_stages=pipeline_stages,
        )
    else:
        _blasst_bidirectional_fwd[grid](
            q,
            k,
            v,
            output,
            _DEVICE_STATS,
            scale,
            log_thresholds,
            row_log_thresholds,
            query_permutation,
            kv_tile_order,
            row_keep_mask_out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            seqlen,
            nheads,
            BLOCK_M=query_tile_rows,
            BLOCK_N=64,
            HEAD_DIM=128,
            PIPELINE_STAGES=pipeline_stages,
            COLLECT_STATS=collect_stats,
            USE_QUERY_PERM=use_query_permutation,
            USE_KV_ORDER=use_kv_order,
            DUMP_ROW_MASK=dump_row_mask,
            SKIP_GROUP_ROWS=skip_group_rows,
            DENSE_FALLBACK_THRESHOLD=dense_fallback_threshold,
            ROW_MASK_MODE={"none": 0, "full": 1, "output": 2}[row_mask_variant],
            MAX_ACTIVE_ROWS=max_active_rows,
            num_warps=num_warps,
            num_stages=pipeline_stages,
        )
    return output


@contextmanager
def install_bidirectional_blasst_kernel(
    model: torch.nn.Module,
    *,
    blasst_lambda: float | torch.Tensor | Callable[[], float | torch.Tensor] = 0.04858582466840744,
    collect_stats: bool = True,
    num_warps: int = 4,
    pipeline_stages: int = 2,
    query_permutation: torch.Tensor | Mapping[int, torch.Tensor] | Callable[[], torch.Tensor | None] | None = None,
    kv_tile_order: torch.Tensor | Mapping[int, torch.Tensor] | Callable[[], torch.Tensor | None] | None = None,
    query_tile_rows: int = 128,
    skip_group_rows: int = 128,
    dense_fallback_threshold: int | None = None,
    row_mask_callback: Callable[[int, torch.Tensor], None] | None = None,
    row_mask_variant: str = "none",
    row_lambda: float | torch.Tensor | Callable[[], float | torch.Tensor] | None = None,
    max_active_rows: int = 32,
    active_voter_layers: set[int] | None = None,
) -> Iterator[None]:
    """Install the accepted fused kernel at LLaDA FlashAttention call sites."""
    if (
        query_permutation is not None
        or kv_tile_order is not None
        or query_tile_rows != 128
        or skip_group_rows != 128
        or dense_fallback_threshold not in (None, 1)
    ):
        raise NotImplementedError(
            "query grouping, KV reordering, and microgroups are disabled: "
            "use the calibrated 128x64 sparse kernel"
        )
    originals: list[tuple[torch.nn.Module, object]] = []

    def resolve(
        provider: torch.Tensor | Mapping[int, torch.Tensor] | Callable[[], torch.Tensor | None] | None,
        layer: int,
    ) -> torch.Tensor | None:
        if callable(provider):
            return provider()
        if isinstance(provider, Mapping):
            return provider.get(layer)
        return provider

    for layer_index, module in enumerate(module for module in model.modules() if hasattr(module, "flash_attn_func")):
        def call(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            *,
            _layer_index: int = layer_index,
            **kwargs: object,
        ) -> torch.Tensor:
            threshold = blasst_lambda() if callable(blasst_lambda) else blasst_lambda
            current_row_lambda = row_lambda() if callable(row_lambda) else row_lambda
            current_variant = (
                row_mask_variant
                if active_voter_layers is None or _layer_index in active_voter_layers
                else "none"
            )
            row_keep_mask = None
            if row_mask_callback is not None:
                kv_tiles = triton.cdiv(q.shape[1], 64)
                row_keep_mask = torch.empty(
                    (q.shape[0], q.shape[2], q.shape[1], kv_tiles),
                    dtype=torch.uint8,
                    device=q.device,
                )
            output = blasst_bidirectional_flash_attn_func(
                q,
                k,
                v,
                **kwargs,
                blasst_lambda=threshold,
                collect_stats=collect_stats,
                num_warps=num_warps,
                pipeline_stages=pipeline_stages,
                query_permutation=resolve(query_permutation, _layer_index),
                kv_tile_order=resolve(kv_tile_order, _layer_index),
                query_tile_rows=query_tile_rows,
                skip_group_rows=skip_group_rows,
                dense_fallback_threshold=dense_fallback_threshold,
                row_keep_mask_out=row_keep_mask,
                row_mask_variant=current_variant,
                row_lambda=current_row_lambda,
                max_active_rows=max_active_rows,
            )
            if row_keep_mask is not None:
                row_mask_callback(_layer_index, row_keep_mask)
            return output

        if hasattr(module, "flash_attn_func"):
            originals.append((module, module.flash_attn_func))  # type: ignore[attr-defined]
            module.flash_attn_func = call  # type: ignore[attr-defined]
    try:
        yield
    finally:
        for module, original in originals:
            module.flash_attn_func = original  # type: ignore[attr-defined]


@contextmanager
def install_diffusion_blasst_kernel(
    model: torch.nn.Module,
    *,
    schedule: DiffusionLambdaSchedule | None = None,
    collect_stats: bool = True,
    num_warps: int = 4,
    pipeline_stages: int = 2,
    query_permutation: torch.Tensor | Mapping[int, torch.Tensor] | Callable[[], torch.Tensor | None] | None = None,
    kv_tile_order: torch.Tensor | Mapping[int, torch.Tensor] | Callable[[], torch.Tensor | None] | None = None,
    query_tile_rows: int = 128,
    skip_group_rows: int = 128,
    dense_fallback_threshold: int | None = None,
) -> Iterator[DiffusionKernelController]:
    """Install a noise-aware kernel and yield its per-step controller."""
    controller = DiffusionKernelController(schedule or DiffusionLambdaSchedule())
    with install_bidirectional_blasst_kernel(
        model,
        blasst_lambda=lambda: controller.blasst_lambda,
        collect_stats=collect_stats,
        num_warps=num_warps,
        pipeline_stages=pipeline_stages,
        query_permutation=query_permutation,
        kv_tile_order=kv_tile_order,
        query_tile_rows=query_tile_rows,
        skip_group_rows=skip_group_rows,
        dense_fallback_threshold=dense_fallback_threshold,
    ):
        yield controller
