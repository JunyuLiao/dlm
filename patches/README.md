# Local patches to third-party kernels

## `vllm_0.30.0_fa4_sm90_paged_block_sparse_tma.patch` (2026-10-02)

**Target.** `vllm/vllm_flash_attn/cute/flash_fwd_sm90.py` in vLLM 0.30.0, the bundled FlashAttention-4 CuTe forward
for SM90. It is applied only inside our dyh env on dllm, and the original is kept as `flash_fwd_sm90.py.orig`.

**Bug.** With a paged KV cache on the TMA path (page size = tile_n = 64), the block-sparse producer passed the
**logical** KV-block index of each sparse block straight to the TMA load. Paged TMA indexes physical pages, so it read
the wrong pages: max abs error about 21–22 against the masked reference when unused pages hold other data. The dense
path translates `n_block` through `mPageTable[batch_idx, n_block]` and was correct.

**Fix.** Before calling `produce_block_sparse_loads`, wrap the K/V TMA load closures so that each sparse block index
goes through the page table. This is the same translation as the dense path; nothing else changes.

**Verification** (fresh compile caches; `scripts/v27_vllm_paged_sparse_probe2.py` and `probe3.py`; GLOBAL geometry
256 × 16/2 heads × hd 512, 65K keys, other data in unused pages):
- paged block-sparse now equals contiguous block-sparse exactly: max abs vs fp32 3.4e-4 / 4.6e-4 / 3.3e-4, including
  a partial last tile;
- dense paged is unchanged (max abs 0.0 vs contiguous).

**Not covered.** Paged KV with page size ≠ 64 (cp.async path, page 32 is vLLM's GLOBAL default) still fails in the
raw entry for dense and sparse alike. Using block-sparse inside vLLM therefore needs page size 64 for the GLOBAL KV
group, or a cp.async block-sparse port.

Upstream report (vllm-project/flash-attention) is pending the user's go-ahead.
