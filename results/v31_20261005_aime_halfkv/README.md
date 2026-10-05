# V31 AIME26 half-context budget reproduction

This is the requested three-seed reproduction of the current V31 progress-aware
method on all 30 AIME26 problems.

## Frozen comparison

- panel seeds: `42,43,44` (90 item × seed cells);
- native DiffusionGemma sampler, thinking enabled, 256-token canvases, adaptive
  stopping, maximum 48 denoising calls per canvas, and an 8192-token output
  budget;
- dense reference: official vLLM FA4 (`dense:default`);
- sparse arm: V31 MAGE port with `qblock_max` selection, selection after the
  first exact call, carry to the first call of the next canvas, settledness
  trigger `0.15`, and sticky score `1.386`;
- GLOBAL budget: `4096` tokens, the nearest 64-token tile multiple to half of
  the approximately 8192-token AIME context;
- requested LOCAL reference budget: `512` tokens (half of the native 1024-token
  sliding window).

The current V31 adapter routes only the five GLOBAL layers through the sparse
consumer. Its 25 sliding LOCAL layers remain native dense, so `512` is recorded
as the requested reference budget but produces **0% achieved LOCAL sparsity** in
this reproduction. The report must give GLOBAL sparsity and the attention-work
weighted overall sparsity separately; it must not claim that the LOCAL budget was
applied.

## Required report

For each seed and pooled over the 90 cells, report:

- achieved GLOBAL and overall attention sparsity (tile/work accounting);
- exact-match AIME accuracy;
- total sampler calls and denoising calls, plus denoising calls per canvas;
- end-to-end wall time (prefill + decode) and ratio/speedup versus dense;
- decode-only time and ratio/speedup versus dense.

Use `scripts/v31_score_aime.py` for exact-match scoring and
`scripts/v31_paired_summary.py` for paired timing ratios. Never score or commit
the private completion text.

The GPU worker writes public JSONL receipts and private completions under its
own authorized `dyh` run directory. Only sanitized summaries and receipts may
be copied back into this repository.
