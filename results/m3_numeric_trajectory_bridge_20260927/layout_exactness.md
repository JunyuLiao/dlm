# Exact output layout checkpoint

The legacy consumer stores logical B,H,Q,D in head-major storage; integration transposes to B,Q,H,D and materializes a contiguous output. The new private `model_major` mode writes the same logical BF16 elements with explicit output strides into physical B,Q,H,D storage. The existing transpose then yields a contiguous view without that output copy. Arithmetic and support decisions are unchanged. The legacy API remains selectable.

On both H100 hosts, the 19-test operator qualification passed. Coverage includes GQA, noncontiguous Q/K/V, partial query/key tiles, D64/128/256/512, all-kept and sparse support, score-consuming PV layout, and invalid retained-score rejection. Actual-model replay at source `f65de1e431c1e1cbdb980692d5bb03df47ad0fca` verified equal complete-forward logits and support for both layout-only parent/child pairs on all **17** reached target states. Separate native-step replay also matched current-canvas, argmax, self-conditioning and stop digests in **34/34** comparisons (17 states × two precision parents).

Observed layer-5 output shape is B=1,H=16,Q=256,D=512: **4,194,304 BF16 bytes** per routed layer's materialized output. Removing one read/write copy of this buffer removes 4 MiB of allocation and 8 MiB of logical copy traffic per such layer, subject to actual allocator behavior. Five routed GLOBAL layers imply 20 MiB of output materialization avoided per full canvas forward when geometry matches; this is a byte accounting statement, not measured bandwidth or latency.

The observed V tensors in all 17 states are already contiguous. Therefore the existing `v.contiguous()` does not create a copy in these captures, and no V-copy saving is claimed or implemented. Prefix ownership/lease tracking and device guards remain intact.

Complete model-forward and denoising-step costs, and separate profiler allocation/copy evidence, are still being measured. Sanitizer qualification is pending. Exact layout equivalence alone is an engineering improvement, not a paper novelty claim or a guarantee of material E2E gain.
