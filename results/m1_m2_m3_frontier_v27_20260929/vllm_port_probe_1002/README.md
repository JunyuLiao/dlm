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
