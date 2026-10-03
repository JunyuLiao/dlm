# Closed length accounting — mechanical evidence audit

This is a CPU-only recomputation/transcription of already-closed evidence.
It adds no inference, confidence interval, performance experiment, causal claim,
or change to any queue. Ratios are candidate/reference; below1 means less of
that quantity. S/N is amortized decode time including commits and host work,
not an isolated forward or attention-kernel price. Different substrates and
inventories are listed separately; their cells are never pooled across studies.

## Source pins and coverage

| Evidence | Generation source | Available input | Coverage |
|---|---|---|---|
| HF E14 | `0fc74fcbfe7b9e58b6cf6061f7581f49585f6403` | Published paired CSV, rounded4decimals |24/24/11 questions,144/144/66 cells per arm at32/64/96K |
| V18b | `8704cd072582dbb766762f955152bc5533c61986` | Private immutable numeric records and closed terminals, read only |8closed workers;568 timed rows;main/dense236each, native/all-kept48each |
| V28 preview | `dcfdb8730efff131f92d0a8bbbab514ef1eedf18` | Closed repaired public summary/receipts only |24closed workers;288timed rows declared;6variants×48;2questions/16cells per bin |
| V29 diagnostic002 | `39e08c521d15b458a1e4c19e98de42a312ff0c25` | Published numeric request CSV |One32K profiled request per arm after one warm |
| V29 diagnostic003 | `54163f4e7c8766aaa4187807420b3cf3167dba81` | Published numeric request CSV |One32K profiled request per arm after one warm; allPIECEWISE |

V18 raw join keys are(dataset,index,repeat). Each pair's host,GPU,engine seed,
deploy and protocol fields match. Duplicate keys fail; control comparisons use
only their intersection(4questions/16cells per bin), while main/dense require
identical complete inventories(24/24/11questions,96/96/44cells). Every worker's
terminal count matches rows. Actual N equals scheduler N plus unused N; W=P+S;
timed capture counts are zero. Recomputed pairs with archived summary entries
agree to relative1e-12. This checks recorded generation pins, not a fresh remote
source/environment rehash. Neither completion ledger nor generated text is read.
V18 has **two engine seeds(7001/7002), two sequential repeats per engine**,
four timed requests per question. Its repeat labels are globally unique across
the two blocks, so the three-field join key does not conflate blocks. Four
repetitions are not four independent seeds. V28 has four engine seeds and two
sequential repeats per engine, eight timed requests per question.

V28 raw worker rows were not available in the local repair directory. Existing
family pair values are copied from its published paired summary. The additional
main/native point values use per-bin geometric means over the same complete
16-cell inventories: their ratio equals the paired geometric mean algebraically.
They are explicitly summary-derived; no new private-key pairing or CI is claimed.
The repaired scorer pin is `7c8c608ed3adb7fd0c7edba7f8d77dcbcbe7bf10`.

## Main comparisons by length

| Study | Length | Candidate/reference | Cells | W | S | N | S/N |
|---|---|---|---:|---:|---:|---:|---:|
| HF E14 | 32k | main/HF FA4 all-kept (S1) | 144 | 0.941900 | 0.930200 | 1.007500 | 0.923300 |
| HF E14 | 64k | main/HF FA4 all-kept (S1) | 144 | 0.865400 | 0.778200 | 0.956400 | 0.813700 |
| HF E14 | 96k | main/HF FA4 all-kept (S1) | 66 | 0.755300 | 0.649700 | 0.860400 | 0.755200 |
| V18b | 32k | method/dense | 96 | 0.781432 | 0.759990 | 0.737702 | 1.030212 |
| V18b | 64k | method/dense | 96 | 0.743935 | 0.684003 | 0.731500 | 0.935069 |
| V18b | 96k | method/dense | 44 | 0.806980 | 0.760477 | 0.862162 | 0.882059 |
| V18b | 32k | method/native | 16 | 1.137749 | 1.161599 | 1.177841 | 0.986210 |
| V18b | 64k | method/native | 16 | 1.006496 | 1.003404 | 1.098214 | 0.913669 |
| V18b | 96k | method/native | 16 | 1.044560 | 1.089306 | 1.280989 | 0.850363 |
| V28 | 32k | main_legacy/dense | 16 | 0.595671 | 0.561446 | 0.540351 | 1.039040 |
| V28 | 64k | main_legacy/dense | 16 | 1.023572 | 0.997973 | 1.090326 | 0.915298 |
| V28 | 96k | main_legacy/dense | 16 | 0.871872 | 0.821552 | 0.946024 | 0.868426 |
| V28 | 32k | main_legacy/native | 16 | 1.171120 | 1.197553 | 1.218397 | 0.982892 |
| V28 | 64k | main_legacy/native | 16 | 1.084097 | 1.080299 | 1.197263 | 0.902307 |
| V28 | 96k | main_legacy/native | 16 | 0.849983 | 0.826040 | 0.955624 | 0.864398 |

The complete comparison table is[comparisons.csv](comparisons.csv), and the
full numeric accounting, including V18 paired absolute N/C/output-count means,
is[aggregate.json](aggregate.json). These descriptive point values retain both
favorable and unfavorable bins. V29 diagnostics contain no64K/96K requests;
their profiled32K observations cannot answer long-prompt performance.

## Actual sparsity coverage

V18 main records explicitly have `telemetry="minimal"` and
`per_tile_statistics="N/A (minimal telemetry; use a token-matched full audit run)"`.
No actual tile-density/physical-skip percentage can be recovered from those
records. Path/refresh/list-build counters are not a density measurement.
The supplied HF E14 paired CSV and V28 summary artifacts also do not carry
actual tile-density measurements. Unknown sparsity is not reported as zero,
and configured sparsity or another token-matched audit is not substituted.

## Public inputs and reproduction

- [HF E14 paired summary](../../m1_m2_m3_frontier_v27_20260929/lb_q64c_panel_e14/summary.csv)
  and[receipts](../../m1_m2_m3_frontier_v27_20260929/lb_q64c_panel_e14/receipts.md).
- [V18b publication](https://github.com/coconight01/dlm_test/blob/e1d2f89c29b6a713ffd3419cf7149d1a34abbd50/results/m1_m2_m3_frontier_v27_20260929/vllm_panel_v18b/summary.md).
- [V28 preview publication](https://github.com/coconight01/dlm_test/blob/91db65396/results/v28_20261002/seed4_preview001/summary.json)
  and[receipts](https://github.com/coconight01/dlm_test/blob/91db65396/results/v28_20261002/seed4_preview001/receipts.json).
- [V29 diagnostic requests](../../v29_20261002/dense_native_accounting001/requests.csv)
  and[accounting limitations](../../v29_20261002/dense_native_accounting001/README.md).

Run `python -m scripts.v30_closed_length_accounting --help` for explicitly supplied
local private inputs; outputs reject overwrite. CPU checks:
`python -m unittest tests.test_v30_closed_length_accounting -v`(5tests pass,
including cell identity, subset coverage, duplicate joins, strata and invalid
values). No Torch/CUDA/pytest dependency is needed. Private paths, join keys,
prompts, generated tokens/text, answers and private file hashes are absent from
published outputs. No remote access or GPU execution occurred in this audit.

Reproduction command(private inputs are placeholders; no private path is published):

```text
python -m scripts.v30_closed_length_accounting --v18-root <V18_CLOSED_NUMERIC_MIRROR> --v28-summary <V28_CLOSED_SUMMARY_JSON> --v28-receipts <V28_CLOSED_RECEIPTS_JSON> --hf-summary results/m1_m2_m3_frontier_v27_20260929/lb_q64c_panel_e14/summary.csv --v29-requests results/v29_20261002/dense_native_accounting001/requests.csv --output <NEW_PUBLIC_AGGREGATE_JSON>
```
