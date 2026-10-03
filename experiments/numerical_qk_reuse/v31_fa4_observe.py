"""v31 FA4 in-kernel observation: the official FA4 SM90 dense forward also writes the per-(query row, 64-key tile)
block log-mass of the prefix tiles (execution variant of the v27 fused observation for compact-mu configs).

Why: the v27 fused observation is a Triton dense pass (4.3 ms per GLOBAL layer at 65K keys vs 1.7 ms for vLLM's FA4
dense call). For selectors whose V term does not need a per-row weighted projected V (M2 compact pooled mu, MAGE),
the observation only needs z[h, row, tile] = logsumexp_k(scale * q_row . k) over each wholly-prefix 64-key tile, which
FA4 can produce inside its own pass.

How: FA4's SM90 kernel calls ``mask_fn(acc_S, n_block)`` on every KV tile right before its online softmax, with the
raw FP32 scores of that tile in registers and (batch, head, m_block) bound. ``ObservingMask`` extends the official
``AttentionMask`` (masking unchanged, then): for prefix tiles (n_block < PT) each row's tile max and sum of exponentials
(relative to the TILE's own max, so no underflow against the running max) are quad-reduced and one thread of warp
group 0 stores z. The dense output and LSE are bit-identical to FA4's (the extra values never feed back). The class
is swapped into ``flash_fwd_sm90`` only while our call compiles; the call passes the z buffer as an FA4 aux tensor,
so its compile-cache key differs from every plain FA4 call and no other FA4 kernel is affected.

hdim 512 on SM90 uses tile_m = 64 and two MMA warp groups that both hold the full 64-column S tile; only warp group 0
writes. Exponentials are ex2.approx (as FA4's softmax and Triton's tl.exp); z matches the Triton fused observation to
FP32 rounding (tests/test_v31_fa4_observe.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from quack import layout_utils

from vllm.vllm_flash_attn.cute import flash_fwd_sm90 as _sm90
from vllm.vllm_flash_attn.cute import utils as _fa_utils
from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd
from vllm.vllm_flash_attn.cute.mask import AttentionMask
from vllm.vllm_flash_attn.cute.utils import AuxData

LOG2E = math.log2(math.e)
LN2 = math.log(2.0)


@dataclass(frozen=True)
class ObservingMask(AttentionMask):
    @cute.jit
    def apply_mask(
        self,
        acc_S: cute.Tensor,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
        mask_mod: cutlass.Constexpr = None,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
    ) -> None:
        AttentionMask.apply_mask(self, acc_S, batch_idx, head_idx, m_block, n_block, thr_mma, mask_seqlen,
                                 mask_causal, mask_local, mask_mod, aux_data, fastdiv_mods)
        mZ = aux_data.tensors[0]                       # [H, QB, PT, 128] FP32
        mScale = aux_data.tensors[1]                   # [1] FP32 softmax scale
        if n_block < mZ.shape[2]:
            acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
            cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
            tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cS))
            scale_log2 = mScale[0] * LOG2E
            tid = cute.arch.thread_idx()[0]
            writer = (tid >= 128) and (tid < 256) and (tid % 4 == 0)   # warp group 0 of the consumers, one per quad
            for r in cutlass.range_constexpr(cute.size(tScS_mn.shape[0])):
                row = acc_S_mn[r, None].load()
                mx = _fa_utils.fmax_reduce(row, init_val=None)
                mx = cute.arch.warp_reduction_max(mx, threads_in_group=4)
                safe = mx if mx != -Float32.inf else Float32(0.0)
                e = cute.math.exp2(row * scale_log2 - safe * scale_log2, fastmath=True)
                s = _fa_utils.fadd_reduce(e, init_val=None)
                s = cute.arch.warp_reduction_sum(s, threads_in_group=4)
                grow = m_block * self.tile_m + tScS_mn[r, 0][0]
                if writer and grow < self.seqlen_q:
                    z = -Float32.inf
                    if s > Float32(0.0):
                        z = (safe * scale_log2 + cute.math.log2(s, fastmath=True)) * LN2
                    mZ[head_idx, grow // 128, n_block, grow % 128] = z


_SCALES = {}
_DYN_FALSE = {}


def _scale_tensor(scale, device):
    key = (float(scale), str(device))
    t = _SCALES.get(key)
    if t is None:
        t = torch.tensor([float(scale)], device=device, dtype=torch.float32)
        _SCALES[key] = t
    return t


def observe_dense(q, k, v, scale, z_out, page_table=None, seqused_k=None, num_splits=0):
    """Dense bidirectional attention of the canvas queries over all keys (vLLM's GLOBAL decode call: causal kernel with
    dynamic causal off) plus the prefix-tile log-mass.
    q [1, n, H, D] bf16; k / v [1, nk, HK, D] (strided views allowed) or paged caches [pages, page, HK, D] with
    page_table [1, pages] and seqused_k [1]; z_out [H, QB, PT, 128] FP32 (QB = ceil(n / 128)), written for every row
    < n and tile < PT. Returns the output [1, n, H, D] bf16."""
    dev = q.device
    dyn = _DYN_FALSE.get(str(dev))
    if dyn is None:
        dyn = _DYN_FALSE[str(dev)] = torch.tensor([False], device=dev)
    if z_out.dtype != torch.float32 or z_out.dim() != 4 or z_out.shape[0] != q.shape[2] or z_out.shape[3] != 128:
        raise ValueError('z_out must be FP32 [H, QB, PT, 128]')
    old = _sm90.AttentionMask
    _sm90.AttentionMask = ObservingMask
    try:
        out = _flash_attn_fwd(q, k, v, softmax_scale=float(scale), causal=True, dynamic_causal=dyn,
                              num_splits=num_splits, pack_gqa=False, page_table=page_table, seqused_k=seqused_k,
                              aux_tensors=[z_out, _scale_tensor(scale, dev)])[0]
    finally:
        _sm90.AttentionMask = old
    return out


def fused_observe_fa4(q, k, v, sketch, scale, prefix_tiles, summary, splits=2, mu=False, mu_precision='bf16',
                      output=True):
    """Drop-in for ``v27_consumer64.fused_observe`` when mu=False (compact pooled mu): dense output [1, Q, H, D] from
    FA4, summary.z / active / bad for the wholly-prefix tiles, and the FP32 score tail [1, H, Q, K - 64 * PT] of the
    remaining tiles. q [1, H, Q, D] (view), k / v [1, HK, K, D]."""
    if mu:
        raise ValueError('FA4 observation writes no per-row projected V; use it with mu_mode=pooled_compact')
    b, h, nq, d = q.shape
    hk, nk = k.shape[1], k.shape[2]
    pt = int(prefix_tiles)
    koff = pt * 64
    if b != 1 or summary is None or summary.z.shape[3] != pt:
        raise ValueError('FA4 observation expects batch 1 and a summary sized for the prefix tiles')
    z = summary.z[0]                                              # [H, QB, PT, 128] view
    out = observe_dense(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale, z)
    summary.bad[0].copy_(torch.isnan(z) | (z == float('inf')))
    summary.active[0].copy_(z > float('-inf'))
    tail_k = k[0, :, koff:].float()                               # [HK, T, D]
    qg = q[0].float().reshape(hk, h // hk, nq, d)
    tf32 = torch.backends.cuda.matmul.allow_tf32                  # FP32 products of BF16 values, as the Triton tail
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        tail = torch.matmul(qg, tail_k.transpose(-1, -2).unsqueeze(1)) * float(scale)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return (out if output else None), tail.reshape(1, h, nq, nk - koff).contiguous()
