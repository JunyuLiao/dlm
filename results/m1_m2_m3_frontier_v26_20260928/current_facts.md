# Current facts (established, with citations)

All numbers below are copied exactly from the cited reports under
`results/m3_numeric_trajectory_bridge_20260927/`. Nothing here is recomputed;
where this v26 study adds its own computation (the clock-opportunity grid),
it is filed separately in `v26_clock_opportunity.csv` / `.json` and
summarized in the handback report, not mixed into this fact list.

## 1. v23 bootstrap6 panel (`v23_bootstrap6/bootstrap6_report.md`)

First-generation outputs only were scored for quality; warm-repeat wall
times are timing measurements, not independent samples; all intervals are
exploratory question-cluster bootstraps on an exposed development panel,
not noninferiority tests.

### aime26 (complete blocks used: 8)

| arm | n | strict_correct | total_calls | total_canvases | calls/canvas | router A/D/H | warm geo ratio vs D_native |
|---|---|---|---|---|---|---|---|
| D_native | 8 | 4/8 | 1961 | 168 | 11.67 | n/a | 1.0 (reference) |
| T_scope | 8 | 6/8 | 2020 | 162 | 12.47 | n/a | 1.011 |
| M3_R3_A8_incumbent | 8 | 6/8 | 1855 | 173 | 10.72 | 1530/2090/5655 | 1.024 |
| M1_boot | 8 | 6/8 | 1847 | 156 | 11.84 | 600/7075/0 | 0.975 |
| M3_boot | 8 | 6/8 | 1967 | 164 | 11.99 | 655/2040/5500 | 1.042 |
| B_boot | 8 | 5/8 | 1931 | 169 | 11.43 | 605/0/7360 | 1.059 |

### longbench_v2 (complete blocks used: 12)

| arm | n | strict_correct | total_calls | total_canvases | calls/canvas | router A/D/H | warm geo ratio vs D_native |
|---|---|---|---|---|---|---|---|
| D_native | 12 | 6/12 | 2360 | 140 | 16.86 | n/a | 1.0 (reference) |
| T_scope | 12 | 6/12 | 2223 | 128 | 17.37 | n/a | 0.904 |
| M3_R3_A8_incumbent | 12 | 5/12 | 2604 | 145 | 17.96 | 1940/3070/8010 | 0.990 |
| M1_boot | 12 | 5/12 | 2623 | 147 | 17.84 | 1140/10505/0 | 1.058 |
| M3_boot | 12 | 6/12 | 2415 | 143 | 16.89 | 995/2685/6965 | 0.990 |
| **B_boot** | 12 | **4/12** | 2524 | 144 | 17.53 | 1060/0/10120 | 0.993 |

**v23 B was 4/12** on longbench_v2 (`bootstrap6_report.md` line 52), the fact
the task text points at explicitly. v23 matched B is not equal-quality to M3
(6/12) on this panel; v25b's memory/gate audit corrects a wrong reading of
this ("T and B faster than M3 at similar quality in v23": wrong for B — see
`v25b_memory_and_gate_audit.md` correction 3).

## 2. v24 direct forward ratios and per-phase costs (`v23_bootstrap6/v24_direct_cost.md`)

Direct complete-forward / denoising-step timing, same captured native
incoming states for all six arms, 1 warmup + 3 blocks, arm order rotated,
`model_forward` = full `model.forward(...).logits`.

- **Per-call cost vs native at the same call index (model_forward):**
  - B0 is priced as native (0.996-1.016).
  - **BO is the most expensive call: +19-31% of a native forward on LB, +6-8% on AIME/RULER.**
  - The later A costs +5-16% on LB.
  - H saves 8-12% per call on LB and nothing on short contexts; D is roughly neutral.
- Over whole N16 sequences on LB: bootstrap M3 is 0.988 / 0.941 of native
  (two states), matched B 0.965 / 0.919, incumbent 0.976 / 0.930, fresh T
  0.991 / 0.977. On AIME and RULER every method is 1.5-3.7% slower than
  native.
- **Bootstrap M3 is not cheaper per forward than matched B or the
  incumbent.** Its N16 advantage over native on LB comes from H calls, net
  of BO and A.
- Where observation time goes (median ms, isolated CUDA-event pieces, not
  additive to a forward price): the **route + summary store is the dominant
  observation cost on LB** (mpk LB 13,703 keys (odd): 6.58-6.67 ms; dllm LB
  17,540 keys (÷4): 3.89-3.91 ms). On odd key counts the route kernel is
  ~2.6× slower per key than on ÷4 counts (0.49 vs 0.22 µs/key) (superseded
  to ~2.2× by v25's precise measurement, see §4 below).
- RULER coverage: 13 task-balanced RULER-4K inputs × seed 101 × six arms,
  90/90 executions, 0 failures. Every arm scored 12/13 (official 13-task
  macro 0.923), 46-55 decoder calls over 13 canvases. RULER requests are
  about 4 calls, so they carry no timing signal.
- Decision (`v24_direct_cost.md` §5): no exact execution improvement was
  found; matched B is cheaper per forward than bootstrap M3, and bootstrap
  M3's request ratio is geoN 1.04 × geoW/N 0.95, so the optional LB
  expansion was **not** run. Fresh T remains the fastest arm at the request
  point estimate.

## 3. v25 aligned16 qualification (`v23_bootstrap6/v25_route_storage_report.md`)

Candidate `route_storage=aligned16`: grouped-Q producer unchanged at real
K; FP32 bits copied into a 16-key-pitch score buffer with a -inf tail;
projected-V sketch padded per route.

- **Level 2 (behaviour):** 18/18 replayed sequences (Parent vs aligned16
  bootstrapped M1/M3/B, AIME/LB/RULER × host) have every call's logits
  bitwise equal and every per-layer decision identical.
- **Level 1 (intermediates):** stored scores bit-identical by construction.
  Stored prefix summaries bitwise equal wherever K was already ÷4/÷8; on
  odd-K states (K = 365, 13,703) 8-35% of summary elements differ (z max
  abs 9.5e-7, mu max abs 5.7e-6, no non-finite change).
- **Pilot** (`v25_aligned6_pilot_56d3b98fe1261756`, 90/90 executions, 0
  failures): aligned16 M3 vs logical M3 tokens identical on all 8
  first-output cells, identical calls and canvases. Paired warm request
  time LB 1.001 (0.998-1.003), AIME 0.992, RULER 0.998. **The component
  gain does not reach the request level: BO + A are ~14% of calls, and only
  odd-K canvases benefit.**
- Decision: `aligned16` is qualified (same logits/decisions/tokens on every
  tested state) but **not claimed as a speedup** in v25 — its measured
  request effect is ~0. (v25b below corrects the scope: the v25 LB pilot
  inputs never actually exercised the odd-K path.)

## 4. v25b odd-K bridge and KDIV8 control (`v23_bootstrap6/v25b_memory_and_gate_audit.md`)

CP0 found neither v25 pilot LB input exercised the KDIV = 1 (odd-K) route
path (`…aacb1f149` KDIV 4, `…067c4480` KDIV 8), so v25's LB request ratio
of 1.001 measured only copy cost on already-fast classes.

CP2 bridge (`v25b_aligned_bridge_2dafaf7cd531bff3`, 16/16, 0 failures),
logical vs aligned16 bootstrapped M3, host-counterbalanced, same GPU per
cell, tokens and calls identical between arms in every row:

| Input | Host / seed | Warm request aligned / logical |
|---|---|---|
| `…067c5456` (odd K, K mod 16 = 7) | mpk / 101 | **0.9742** |
| `…067c5456` (odd K, K mod 16 = 7) | dllm / 202 | **0.9739** |
| `…067c4480` (KDIV 8 control) | dllm / 101 | **1.0011** |
| `…067c4480` (KDIV 8 control) | mpk / 202 | **1.0017** |

- On the odd-K input the request saving is **≈2.6% on both hosts and both
  seeds**, consistent with the direct same-state sequence ratio 0.977.
- On the KDIV-8 control the copy cost is **+0.1-0.2%**.
- **The 8-9% applies to BO/A calls only.** Direct profile figures: the
  odd-K state's model-forward sums were native 2268.8 ms, logical M3
  2243.7 ms, aligned M3 2191.4 ms — a ≈2.3% saving of the *whole forward
  sequence*, not 8-9%. "The 8-9% refers only to BO/A calls, and must not be
  multiplied again by the BO/A share." (`v25b_memory_and_gate_audit.md`,
  correction 2, verbatim.)
- Caveats stated in the same file: one warm repeat per cell, one odd-K and
  one control question — "a mechanism check, not a population benchmark";
  the KDIV-2 class was not measured.
- Status: `aligned16` stays a selectable, qualified implementation; about
  2.6% faster on the odd-K LB request, +0.1-0.2% on the aligned-class
  control; "an execution change, not a method contribution."
