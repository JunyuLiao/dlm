# Generation records, v27 panels of 2026-09-30 (5382 requests)

Redacted per-request first outputs (`attempt00.json`, the scored output) of every successful execution of the
panels reported in `../progress_20260930.md` and `../weekly_slides_20260930.md`, with the same redaction as
`../generation_records_v27_lb_fa4_v3/`: original prompts, prompt tokens, prompt hashes, gold/reference answers
and absolute server paths are removed; completion tokens, raw completion, extracted prediction, per-canvas
trajectory, router counters and timing are kept. LongBench-v2 is multiple choice and `raw_completion` may
quote or paraphrase the context, so zero leakage of benchmark content is not claimed.

Layout: `<panel>/<dataset>/<host>/<cell_id>/attempt00.json`. `index.csv` joins each file to its run directory,
host, question id, seed, arm and first-output score; `manifest.json` holds sha256 of every file.
Requests that failed (the 96K prompts above about 95K tokens ran out of memory in prefill for every arm) have
no record.

| panel | dataset | requests |
|---|---|---:|
| aime_m2_c1 | aime26 | 180 |
| aime_sparsity_hi | aime26 | 300 |
| aime_sparsity_t1 | aime26 | 630 |
| final_aime_f1 | aime26 | 960 |
| final_lb_f1 | longbench_v2 | 288 |
| final_lb_f1 | longbench_v2_32k | 288 |
| final_lb_f1 | longbench_v2_64k | 288 |
| lb96k_c2 | longbench_v2_96k | 72 |
| m2_c1 | longbench_v2_32k | 144 |
| m2_c1 | longbench_v2_64k | 144 |
| sparsed_c1 | longbench_v2_32k | 252 |
| sparsed_c1 | longbench_v2_64k | 252 |
| threshold_sweep_s1 | longbench_v2_32k | 396 |
| threshold_sweep_s1 | longbench_v2_64k | 396 |
| variants_v5 | longbench_v2_32k | 396 |
| variants_v5 | longbench_v2_64k | 396 |
