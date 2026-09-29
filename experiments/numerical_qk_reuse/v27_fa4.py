"""v27: official FlashAttention-4 for DiffusionGemma's 512-dim GLOBAL layers.

The DiffusionGemma technical report serves the canvas with FlashAttention-4, and vLLM's flash-attention
fork accepts head_dim <= 512 on SM90 (``is_sm90_range = 8 <= head_dim <= 512`` in cute/interface.py).
This module loads that kernel (CuTe DSL, vendored unchanged in a dyh overlay, path in V27_FA4_OVERLAY)
into the project's pinned environment and exposes:
  dense(q, k, v, scale)                      -- FA4, bidirectional, GQA
  sparse(q, k, v, skipped, eligible, scale)  -- the SAME FA4 kernel through its official block-sparse
                                                interface: the Q128 x KV64 keep map of M1/M2/M3 becomes
                                                FA4 "full blocks" (block_size (128, 64))
Both take the model's [1, heads, len, D] views and return the output in model-major [1, Q, H, D].

Compatibility shim (documented): the pinned torch 2.12 lacks some low-precision dtypes (e.g.
``float4_e2m1fn_x2``, ``float8_e8m0fnu``) that quack's dtype tables reference at import time. A unique
sentinel attribute is installed only when missing; no FP8/FP4 path is reachable from bf16 attention.
"""
from __future__ import annotations

import os
import sys

import torch

_FWD = None
_BST = None
_LOW_PRECISION_DTYPES = ('float4_e2m1fn_x2', 'float8_e8m0fnu', 'float8_e4m3fnuz', 'float8_e5m2fnuz')


def load():
    global _FWD, _BST
    if _FWD is not None:
        return _FWD
    overlay = os.environ.get('V27_FA4_OVERLAY')
    if overlay:
        for path in (overlay, os.path.join(overlay, 'nvidia_cutlass_dsl', 'dsl_packages')):
            if path not in sys.path:
                sys.path.append(path)
    for name in _LOW_PRECISION_DTYPES:
        if not hasattr(torch, name):
            # import-time dtype-table keys only (quack's FP8/FP4 maps); a unique sentinel, never a real
            # dtype, so no existing table entry is shadowed and no low-precision path becomes reachable
            setattr(torch, name, type(f'missing_{name}', (), {})())
    from vllm.vllm_flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd
    _FWD, _BST = _flash_attn_fwd, BlockSparseTensorsTorch
    return _FWD


def _layout(q, k, v):
    if q.shape[0] != 1 or q.shape[-1] != k.shape[-1]:
        raise ValueError('FA4 path expects batch 1 and equal Q/K head dims')
    return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)   # [1, len, heads, D] views


def dense(q, k, v, scale):
    fwd = load()
    qs, ks, vs = _layout(q, k, v)
    return fwd(qs, ks, vs, softmax_scale=scale, causal=False)[0]


def block_sparse_tensors(kept):
    """[1, H, QB128, KT64] keep map -> FA4 BlockSparseTensorsTorch (all kept blocks are full blocks)."""
    load()
    b, h, qb, kt = kept.shape
    order = torch.argsort((~kept).to(torch.int8), dim=-1, stable=True).to(torch.int32)
    zeros = torch.zeros((b, h, qb), device=kept.device, dtype=torch.int32)
    return _BST(mask_block_cnt=zeros,
                mask_block_idx=torch.zeros((b, h, qb, 1), device=kept.device, dtype=torch.int32),
                full_block_cnt=kept.sum(-1).to(torch.int32).contiguous(), full_block_idx=order.contiguous(),
                block_size=(128, 64))


def sparse(q, k, v, skipped, eligible, scale):
    fwd = load()
    qs, ks, vs = _layout(q, k, v)
    kept = eligible & ~skipped
    return fwd(qs, ks, vs, softmax_scale=scale, causal=False, block_sparse_tensors=block_sparse_tensors(kept))[0]
