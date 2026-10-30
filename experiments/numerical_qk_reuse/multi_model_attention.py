"""D128 cross-model ports of Junyu's value selectors and Yuhan's map reuse.

All output PV uses current full-dimensional V. Discovery is separately counted.
Native block-causal prefix attention is dense; only denoising calls are routed.
"""
from __future__ import annotations

from collections import Counter
import math

import torch
import triton as tr
import triton.language as tl

from . import v27_fa4
from .v31_value_kernels import statistics_cuda, online_cuda
from .v31_value_summary import full_support_cuda


ARMS = ("dense_native", "allkept_fa4", "mass_topk_reuse_r4",
        "value_v2_gaussian32_reuse_r4", "value_v3a_singleton_reuse_r4")


def budget_map(scores, mandatory, eligible):
    """Exactly round(J/2), including protections, with stable low-ID ties."""
    target = torch.maximum(torch.round(eligible.sum(-1).float() * .5).long(),
                           (mandatory & eligible).sum(-1))
    rank = scores.masked_fill(~eligible, -math.inf).masked_fill(mandatory & eligible, math.inf)
    order = torch.argsort(rank, descending=True, stable=True)
    result = torch.zeros_like(eligible)
    result.scatter_(1, order, torch.arange(scores.shape[1], device=scores.device)[None] < target[:, None])
    return result & eligible


def singleton_scores(stats):
    alpha, output, g = full_support_cuda(stats)
    a, residual = alpha.sum(1), g.sum(1)
    remain = a[:, None] - alpha
    error = (residual[:, None] - g).norm(dim=-1)
    error /= remain.clamp_min(torch.finfo(alpha.dtype).tiny) * stats.nu[:, None].clamp_min(1e-12)
    live = stats.valid
    error = torch.where(live, error, 0.)
    score = error.amax(-1)
    score.masked_fill_((live & (remain <= 0)).any(-1), math.inf)
    return score


@tr.jit
def _dense_block(Q, K, V, OUT, NQ, NK, QOFF, SQH, SQQ, SKH, SKK, SVH, SVK,
                 H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
                 B: tl.constexpr, SCALE: tl.constexpr,
                 BM: tl.constexpr = 32, BN: tl.constexpr = 64):
    mi, h = tl.program_id(0), tl.program_id(1)
    qi = mi * BM + tl.arange(0, BM)
    dd = tl.arange(0, D)
    kh = h // (H // HK)
    q = tl.load(Q + h * SQH + qi[:, None] * SQQ + dd[None], qi[:, None] < NQ, 0.)
    m = tl.full((BM,), -float('inf'), tl.float32)
    den = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    # Block-causal visibility, including the whole current denoising block.
    stop = tl.minimum(NK, ((QOFF + mi * BM + BM - 1) // B + 1) * B)
    for j in range(tl.cdiv(stop, BN)):
        kk = j * BN + tl.arange(0, BN)
        k = tl.load(K + kh * SKH + kk[:, None] * SKK + dd[None], kk[:, None] < NK, 0.)
        s = tl.dot(q, tl.trans(k)) * SCALE
        legal = (kk[None] < NK) & (kk[None] // B <= (qi[:, None] + QOFF) // B)
        s = tl.where(legal, s, -float('inf'))
        new_m = tl.maximum(m, tl.max(s, 1))
        safe = tl.where(new_m > -float('inf'), new_m, 0.)
        old_w = tl.exp(m - safe)
        p = tl.exp(s - safe[:, None])
        v = tl.load(V + kh * SVH + kk[:, None] * SVK + dd[None], kk[:, None] < NK, 0.)
        acc = acc * old_w[:, None] + tl.dot(p.to(tl.bfloat16), v)
        den = den * old_w + tl.sum(p, 1)
        m = new_m
    result = acc / den[:, None]
    dest = (qi[:, None] * H + h) * D + dd[None]
    tl.store(OUT + dest, result, qi[:, None] < NQ)


def dense_block(q, k, v, offset, block_size, scale):
    _, h, nq, d = q.shape
    out = torch.empty((1, nq, h, d), dtype=q.dtype, device=q.device)
    _dense_block[(tr.cdiv(nq, 32), h)](
        q, k, v, out, nq, k.shape[2], offset, q.stride(1), q.stride(2),
        k.stride(1), k.stride(2), v.stride(1), v.stride(2),
        h, k.shape[1], d, block_size, scale, num_warps=4)
    return out


class AttentionRuntime:
    def __init__(self, arm="dense_native", threshold=.02, backend="fa4", statistics_query_blocks=1):
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm, self.threshold, self.backend = arm, threshold, backend
        self.statistics_query_blocks = statistics_query_blocks
        self.maps, self.matrices = {}, {}
        self.map_kept_counts = {}
        self.counts = Counter()
        self.block_start = 0
        self.block_end = 0
        self.call_in_block = 0
        self.phase = "prefill"
        self.invalidations = 0

    def reset(self):
        self.maps.clear()
        self.map_kept_counts.clear()
        self.counts.clear()
        self.call_in_block = 0

    def block(self, start, end):
        self.block_start, self.block_end = start, end
        self.call_in_block = 0
        self.maps.clear()
        self.map_kept_counts.clear()
        self.phase = "denoise"
        self.counts["canvases"] += 1

    def begin_forward(self):
        self.counts["denoising_forwards"] += 1
        self.call_in_block += 1
        callback = getattr(self, "heartbeat", None)
        if callback is not None:
            callback()

    def invalidate(self):
        self.maps.clear()
        self.map_kept_counts.clear()
        self.invalidations += 1

    def project(self, v, layer):
        hk, d = v.shape[1], v.shape[-1]
        key = (layer, hk, d, str(v.device))
        if key not in self.matrices:
            gen = torch.Generator(device="cpu").manual_seed(1729 + layer)
            self.matrices[key] = (torch.randn((hk, d, 32), generator=gen) / math.sqrt(32)).to(v.device)
        # Batched projection shares one matrix per KV head, independent of Q heads.
        z = torch.matmul(v[0].float(), self.matrices[key]).unsqueeze(0).contiguous()
        nu = v[0].float().square().sum(-1).mean(-1).sqrt().clamp_min(1e-12)
        self.counts["projected_value_tokens"] += hk * v.shape[2]
        return z, nu

    def attention(self, q, k, v, layer, scale, *, offset=0, block_size=0):
        nq, nk, heads = q.shape[2], k.shape[2], q.shape[1]
        if self.phase != "denoise":
            self.counts["dense_prefix_attention_calls"] += 1
            if block_size:
                return dense_block(q, k, v, offset, block_size, scale)
            return v27_fa4.dense_plain(q, k, v, scale)
        if block_size and nq > block_size:
            raise ValueError("block-causal routed call must contain only the active block")
        qt, kt = tr.cdiv(nq, 128), tr.cdiv(nk, 64)
        legal_tiles = heads * qt * kt
        self.counts["eligible_tiles"] += legal_tiles
        if self.arm == "dense_native":
            self.counts["kept_tiles"] += legal_tiles
            # Match the publisher's SDPA consumer on the active legal extent.
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False, scale=scale,
                enable_gqa=q.shape[1] != k.shape[1]).transpose(1, 2)
        if self.arm == "allkept_fa4":
            self.counts["kept_tiles"] += legal_tiles
            return v27_fa4.dense(q, k, v, scale)
        key = (layer, nq, nk)
        refresh = key not in self.maps or (self.call_in_block - 1) % 4 == 0
        if refresh:
            self.counts["selection_layer_calls"] += 1
            z, nu = self.project(v, layer) if "value" in self.arm else (
                torch.zeros((1, k.shape[1], nk, 32), device=q.device),
                torch.ones((k.shape[1],), device=q.device))
            kept = torch.empty((1, heads, qt, kt), device=q.device, dtype=torch.bool)
            # Bound statistics memory by one Q128 group, even for 128K full attention.
            chunk_qb = max(1, min(self.statistics_query_blocks, 134217728 // max(1, heads * kt * 128 * 32)))
            for qb in range(0, qt, chunk_qb):
                end_qb = min(qt, qb + chunk_qb)
                blocks = end_qb - qb
                part = q[:, :, qb * 128:end_qb * 128]
                stats = statistics_cuda(part, k, z, nu, scale, kt)
                mandatory = torch.zeros((heads * blocks, kt), device=q.device, dtype=torch.bool)
                mandatory[:, self.block_start // 64:tr.cdiv(self.block_end, 64)] = True
                eligible = stats.valid.any(-1)
                mandatory &= eligible
                if self.arm == "value_v2_gaussian32_reuse_r4":
                    keep, _, _, _ = online_cuda(stats, self.threshold, True, mandatory,
                                                torch.zeros_like(mandatory), 0.)
                else:
                    if self.arm == "mass_topk_reuse_r4":
                        lm = stats.log_mass
                        denom = torch.logsumexp(lm, 1, keepdim=True)
                        score = (lm - torch.where(torch.isfinite(denom), denom, 0.)).amax(-1)
                    else:
                        score = singleton_scores(stats)
                    keep = budget_map(score, mandatory, eligible)
                kept[:, :, qb:end_qb] = keep.reshape(1, heads, blocks, kt)
                self.counts["discovery_qk_pairs"] += heads * part.shape[2] * nk
            self.maps[key] = (kept, v27_fa4.block_sparse_tensors(kept))
            # A held map has an unchanged integer tile count. Avoid a fresh
            # GPU reduction and host synchronization on every held layer call.
            self.map_kept_counts[key] = int(kept.sum().item())
        else:
            self.counts["held_layer_calls"] += 1
        keep, lists = self.maps[key]
        self.counts["kept_tiles"] += self.map_kept_counts[key]
        return v27_fa4.sparse_lists(q, k, v, lists, scale)
