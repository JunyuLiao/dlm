"""v27: official FlashAttention-4 for DiffusionGemma's 512-dim GLOBAL layers.

The DiffusionGemma technical report serves the canvas with FlashAttention-4, and vLLM's flash-attention
fork accepts head_dim <= 512 on SM90 (``is_sm90_range = 8 <= head_dim <= 512`` in cute/interface.py).
This module loads that kernel (CuTe DSL, vendored unchanged in a dyh overlay, path in V27_FA4_OVERLAY)
into the project's pinned environment and exposes:
  dense(q, k, v, scale)                      -- FA4, bidirectional, GQA, in its fastest configuration: the kernel's
                                                block-sparse interface with every tile kept. Bitwise identical to
                                                FA4's plain dense path (also for a partial last tile) and 4-6%
                                                faster at the GLOBAL decode shape (official_baseline/README.md)
  dense_plain(q, k, v, scale)                -- FA4's plain dense path (reference only)
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
_ALLKEPT = {}
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


def dense_plain(q, k, v, scale):
    fwd = load()
    qs, ks, vs = _layout(q, k, v)
    return fwd(qs, ks, vs, softmax_scale=scale, causal=False)[0]


def _allkept(heads, qb, kt, device):
    # the all-kept full-block lists block_sparse_tensors(ones) would build, cached per shape (no per-call build)
    key = (heads, qb, kt, str(device))
    lists = _ALLKEPT.get(key)
    if lists is None:
        load()
        cnt = torch.full((1, heads, qb), kt, device=device, dtype=torch.int32)
        idx = torch.arange(kt, device=device, dtype=torch.int32).expand(1, heads, qb, kt).contiguous()
        lists = _BST(mask_block_cnt=torch.zeros((1, heads, qb), device=device, dtype=torch.int32),
                     mask_block_idx=torch.zeros((1, heads, qb, 1), device=device, dtype=torch.int32),
                     full_block_cnt=cnt, full_block_idx=idx, block_size=(128, 64))
        if len(_ALLKEPT) >= 64:
            _ALLKEPT.clear()
        _ALLKEPT[key] = lists
    return lists


def dense(q, k, v, scale):
    lists = _allkept(q.shape[1], -(-q.shape[2] // 128), -(-k.shape[2] // 64), q.device)
    return sparse_lists(q, k, v, lists, scale)


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
    return sparse_lists(q, k, v, block_sparse_tensors(eligible & ~skipped), scale)


def sparse_lists(q, k, v, lists, scale):
    fwd = load()
    qs, ks, vs = _layout(q, k, v)
    # block sparsity needs the KV64 block to be a multiple of the kernel's tile_n: FA4's SM90 default is 64 at
    # head_dim 512 (GLOBAL) but 80 at 256 (LOCAL layers, G75 S15/S30), where FA4 itself uses 64 for local attention
    tile = {} if q.shape[-1] > 256 else dict(tile_mn=(128, 64))
    return fwd(qs, ks, vs, softmax_scale=scale, causal=False, block_sparse_tensors=lists, **tile)[0]
