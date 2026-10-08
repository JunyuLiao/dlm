# Calibration result — LongBench-v2 `0shot_think` dev15, frozen v31 substrate

**15 held-out dev cells, one seed, one repeat. This is calibration, not a target claim.**
The threshold was chosen here, on cells disjoint from the 488-cell target, before any target
generation. Accuracy is the official LongBench-v2 `pred.py` judge with `result.py` aggregation
(`scripts/v31_score_longbench_official.py`); it is never recomputed by the aggregator.

Substrate pins, asserted per arm by `scripts/v32_panel_aggregate.py --require-pins 32,16384`:
LongBench-v2 `BLOCK=32 CHUNK=16384 MAX_MODEL_LEN=136401`, `MEM=0.80` (verified KV pool 174,193
tokens), FA4 local fix on, native mask fix 51994 on, LOCAL native dense, GLOBAL layers
5/11/17/23/29, `MAGE_K=1728` (27 KV64 tiles per unit), settle/0.15, sticky 1.386.
Dev cells: mean prompt 72,424 tokens, max 120,016.

`S/N` is the amortized per-step decode cost, not a per-forward price. GLOBAL sparsity uses the
GLOBAL eligible-tile count as its denominator; LOCAL is native dense and is never routed, so it has
no denominator and is reported as such rather than as 0%.

## Result

| arm | Overall % | GLOBAL sparsity | N/C | S/N (s) | decode s/cell | objective_max | retained mass | forced keep | forced skip |
|---|---|---|---|---|---|---|---|---|---|
| dense_full_fix51994 | 60.0 | - | 14.83 | 0.03110 | 10.24 | - | - | - | - |
| dense_piecewise | 60.0 | - | 14.83 | 0.03196 | 10.52 | - | - | - | - |
| allkept_fa4 | 66.7 | 0.00% | 16.60 | 0.03259 | 10.57 | - | - | - | - |
| current_v31_control | 73.3 | 87.60% | 18.42 | 0.02443 | 7.26 | - | - | - | - |
| value_v1_t0.005 | 66.7 | 6.51% | 15.81 | 0.34447 | 81.32 | - | - | 504,002 | 36,480 |
| value_v1_t0.01 | 66.7 | 6.38% | 15.84 | 0.35740 | 91.73 | - | - | 515,357 | 39,040 |
| value_v1_t0.02 | 66.7 | 5.91% | 15.60 | 0.34558 | 83.40 | - | - | 527,487 | 37,504 |
| value_v1_t0.05 | 66.7 | 9.16% | 19.74 | 0.27034 | 80.04 | - | - | 533,223 | 36,608 |
| value_v2_t0.005 | 66.7 | 5.71% | 15.34 | 0.04438 | 10.12 | 0.0150 | 0.9909 | 519,918 | 36,352 |
| value_v2_t0.01 | 66.7 | 6.73% | 16.36 | 0.04400 | 11.13 | 0.0168 | 0.9863 | 527,676 | 37,504 |
| value_v2_t0.02 | 66.7 | 6.84% | 16.45 | 0.04430 | 11.66 | 0.0246 | 0.9844 | 532,376 | 38,528 |
| value_v2_t0.05 | 60.0 | 6.76% | 16.54 | 0.04414 | 12.46 | 0.0274 | 0.9830 | 535,592 | 39,552 |
| value_v3a | 66.7 | 6.72% | 16.51 | 0.04460 | 11.39 | 0.0060 | 0.9965 | - | - |
| value_v3b | 66.7 | 87.76% | 18.62 | 0.03143 | 9.40 | - | - | - | - |
| value_v3b_drop | 46.7 | 10.01% | 29.88 | 0.04681 | 24.61 | 0.3640 | 0.8819 | - | 40,416 |
| value_v3b_short | 66.7 | 6.72% | 16.48 | 0.04804 | 12.25 | 0.0062 | 0.9961 | - | 36,704 |

## What this says

1. **No value-aware selector beats the inherited mass-only control on any axis.** The control is
   simultaneously the most accurate (73.3 vs 66.7), by far the sparsest (87.60% vs 5.7-10.0%), and
   the cheapest per step (`S/N` 0.0244 vs 0.044-0.357). Everything this study added is a
   regression against the baseline it was meant to improve.
2. **The selectors barely prune.** V1/V2 reach only 5.7-9.2% GLOBAL sparsity against the control's
   87.60%, and the reason is visible in the counters: `forced_keep` is 504k-536k against
   `forced_skip` of 36k-40k, about 93% of all (row, tile) decisions. Raising the V2 threshold from
   0.005 to 0.05 barely moves sparsity (5.71% to 6.76%) while its retained mass falls from 0.9909 to
   0.9830. The value-aware rho test is passing almost every tile, so the decision is not a budget
   decision in practice.
3. **V1 is 11-14x slower than the control** (`S/N` 0.27-0.36 vs 0.0244; 80-92 s per cell against
   7.3 s). V1 is the only mass-only selector, so it carries no sketch and cannot use the fused
   Triton scan, which needs MU to build its rho term; it falls back to the batched reference scan.
   Its receipt records `kernel=batched_reference`, so this cost is attributable, not a measurement
   artefact. Its 66.7 equals the all-kept consumer's 66.7, which is what keeping ~94% of tiles buys.
4. **V3b never ran.** All 2405 of its selection calls exceeded `value_exact_max=256` and fell back
   to the control, so its 87.76% sparsity and 66.7 are the control's behaviour, not a V3b result.
   Exact greedy deletion over up to 1876 prefix tiles per call is not affordable at this context.
5. **`v3b_drop` (drop fraction 0.25) is the clearest negative**: 131.9 billion deletion evaluations,
   `N/C` 29.9 against the control's 18.4, 24.6 s per cell against 7.3 s, retained mass down to 0.8819,
   sketch objective up to 0.3640, and accuracy down to **46.7** from the control's 73.3. Dropping a
   quarter of the candidate set up front destroys quality. This confirms and extends the earlier
   single-cell probe that already flagged `v3b_drop`.
6. **V3a and `v3b_shortlist` are the least bad value arms** (66.7, ~6.7% sparsity, `S/N` 0.045-0.048,
   objective_max 0.006, retained mass 0.996) and they match the all-kept consumer's accuracy exactly,
   which is consistent with them also keeping almost everything.

## Honest limits

* n = 15, one seed, one repeat. One item is 6.7 points, so only the large gaps above are meaningful.
  In particular the 66.7-vs-73.3 control gap is two items and must not be read as established.
* The sparsity gap (6% vs 88%) is a mechanism-level result, not a sampling result, and is the part of
  this table that carries weight.
* The all-kept control scoring 66.7 against dense's 60.0 is the consumer difference AGENTS.md asks to
  be retained rather than hidden; on 15 cells it is within noise.
* No target-suite (488 or 120 cells) or AIME26 panel was run under the frozen pins. The AIME26 dense
  and control arms that completed earlier were withdrawn: they carried `BLOCK=32/CHUNK=16384/
  MAX_MODEL_LEN=13312` instead of the frozen AIME26 `(64, 4096, 9216)`.
* RULER v33 and HumanEval remain unavailable on this host and are not addressed.

## Reproduce

```
bash scripts/v32_lbdev_calibration.sh          # this panel
python scripts/v32_panel_aggregate.py --out RESULTS --require-pins 32,16364 --label ARM=PRIVATE.jsonl ...
```

Artifacts: `calibration_dev15/aggregate.json` (all counters, denominators, paired differences),
`calibration_dev15/tables.md`, `calibration_dev15/score.summary.json`,
`calibration_dev15/score_v3.summary.json`. Raw private records with completions stay in the
ignored user-owned run directory and are not committed.
