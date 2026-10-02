# vLLM port feasibility probe: FA4 block-sparse over vLLM's paged KV cache (2026-10-02)

Random data, no model; vLLM 0.30.0's bundled FA4 (CuTe, SM90). GLOBAL decode geometry: 256 queries, 16 Q / 2 KV heads,
hd 512, bidirectional, bf16, 64 × 64 tiles. Pages are shuffled through the page table.
Scripts: `scripts/v27_vllm_paged_sparse_probe.py` (`paged_probe.json`) and `scripts/v27_vllm_paged_sparse_probe2.py`.

| check | result |
|---|---|
| page size 16 (any paged call) | **fails**: `TypeError: 'NoneType' object is not callable` (the cp.async paged path for page size ≠ tile_n is unavailable) |
| page size 64, dense: paged vs contiguous | **identical** (max abs 0.0); 2.84 ms at 65K either way |
| page size 64, block-sparse: contiguous vs fp32 masked reference | correct (max abs 3–5e-4) |
| page size 64, block-sparse: paged vs fp32 reference | **wrong**: max abs 0.15 when unused pages hold zeros, about 22 when they hold other data (also with keys a multiple of 64, and with the last tile passed as a mask block) |
| speed at 65K, page 64 | dense 2.84 ms, all-kept lists 2.56–2.59 ms, 9% sparse 0.29 ms, the same paged and contiguous |

**Reading.** In this FA4 build the block-sparse iteration does not go through the page table: it reads the wrong pages.
Dense paged attention is fine. So the method cannot simply hand block-sparse lists to vLLM's paged FA4 call.

**Options for the port, to evaluate next:**
1. Pass a contiguous K/V view of the request's cache. Possible when its pages are physically contiguous (batch 1;
   control the allocation or verify `page_table == arange`), and then exactly our tested contiguous path.
2. Fix the paged block-sparse path in FA4's CuTe kernel: translate the sparse n-block index through the page table
   (as the dense path does), add a regression test and report it upstream to vllm-project/flash-attention.
3. FlashInfer's block-sparse (BSR) attention over paged KV, if its hd 512 SM90 speed is competitive (unverified;
   earlier reports put its FA2-template hd 512 far below FA4).
4. Gather kept tiles into a contiguous buffer per decision (extra copies, likely too slow).

## Follow-up (same day): vLLM's real page sizes and block tables

- **vLLM uses KV page size 32 for the GLOBAL layers and 16 for the LOCAL layers** (hybrid KV groups;
  `block_table_probe.json`, from wrapping vLLM's own `_FA4_DENSE_ATTENTION_KERNEL` in eager mode, two real requests).
- **Block tables are not contiguous**, from the first real request on (prefill and canvas calls alike). A contiguous
  view of the request's cache is therefore not available without changing vLLM's allocator.
- **The raw FA4 entry only runs paged attention at page size 64** (= tile_n; TMA path), via
  `scripts/v27_vllm_paged_sparse_probe3.py`.
  - Page sizes 32 and 128 fail (`'NoneType' object is not callable`).
  - At page 64 dense is exact; block-sparse is wrong (max abs 21).
  - vLLM itself serves page 32 through its pre-compiled dense dispatch (`compile_flash_attn_varlen_func_from_specs`).

**Consequence for the port.** The work is at kernel level:
1. Add `block_sparse_tensors` to vLLM's FA4 dense dispatch.
2. Make the sparse n-block iteration translate through the page table, at page 32 (two pages per 64-key tile, the
   cp.async producer path).
3. Verify against the masked reference with garbage in unused pages (the probe scripts are the regression tests).

Alternatives:
- change vLLM's KV block size for the GLOBAL group to 64 and fix only the TMA paged block-sparse path;
- use a different official kernel with paged block-sparse support (FlashInfer BSR; its hd 512 SM90 speed is
  unverified).

## Fix (same day): paged block-sparse corrected at page size 64

- Patch: `patches/vllm_0.30.0_fa4_sm90_paged_block_sparse_tma.patch`. The block-sparse producer now translates sparse
  KV-block indices through the page table, like the dense path.
- With fresh compile caches, paged block-sparse equals contiguous block-sparse exactly: max abs vs fp32 3.4e-4,
  4.6e-4 and 3.3e-4 at keys 65,536 / 65,000 (partial last tile, also passed as a mask block); unused pages hold other
  data.
- Page sizes 32 and 128 still fail through the raw entry (cp.async path, a separate gap).
