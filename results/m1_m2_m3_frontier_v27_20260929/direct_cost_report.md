# v27 Tier 1: direct per-forward cost, time breakdown and sparsity sweep

> **Erratum (2026-09-29, see [tier3_report.md](tier3_report.md) §4):** every per-forward ratio below is relative to the *native* dense path, whose GLOBAL SDPA dispatch (`enable_gqa`) is about 2× slower than a repeated-KV SDPA call. Against that strongest dense path (`D_fast`), the best sparse variants are within ±0.5% on LB and 1.5–2.5% slower on AIME; the ~8–10% gains are not sparsity gains.

**Setup.**
- H100; DiffusionGemma-26B-A4B, BF16.
- Only the 5 GLOBAL layers are routed; the 25 LOCAL layers stay native.
- Replays are complete `model.forward(...).logits` calls (and the native denoising step), taken from canvas start on captured native states.
- Each arm replays the same states on the same GPU with native brackets, rotated order, 3 repetitions and no JIT misses.
- Physical work comes from an untimed counter twin.
- Raw profiles: private. Reduced tables:
  - [direct_cost_breakdown.csv](direct_cost_breakdown.csv) (13 arms × 6 states);
  - [threshold_sweep.csv](threshold_sweep.csv);
  - [policy_screen.csv](policy_screen.csv).

These are **fixed-state timing diagnostics, not scored methods**. Request-level effects need Tier 2/3.

## 1. Where one forward's time goes (LB, K ≈ 14–17.5K)

**Per GLOBAL layer** (component probe, layer 5, LB K = 17,540, P0, 51% of tiles kept):

| piece | ms |
|---|---:|
| native SDPA attention | 5.23 |
| same Triton consumer, all tiles kept (D_matched) | ≈ 4.7 (inferred from full-forward 0.982; see §2) |
| sparse consumer with a given bitmap (pure sparse execution) | 2.10 |
| route with prefix summary LOAD (D decision, M1/M3) | 1.59 |
| compact M2 D: route LOAD 1.04 + tile pool 0.18 | 1.22 |
| route with summary STORE (A/BO), exact / compact | 3.92 / 2.56 |
| current-QK observation (grouped-Q producer) | 1.18 |
| projection lease update (current canvas) | 0.09 |

Phase cost = sum of the pieces it runs:
- **H** = consumer
- **D** = route LOAD + consumer
- **A** = observation + route STORE + consumer; this is more than native attention

**Per complete forward** (LB states, ratio to native on the same state):

| state | native ms | H | D | A |
|---|---:|---:|---:|---:|
| LB KDIV8 c0 | 155.9 | 0.908 | 0.96–0.99 | – |
| LB odd-K c0 | 172.6 | 0.918 | 0.96–0.98 | 1.05–1.06 |
| LB odd-K c6 | 163.0 | 0.908 | 0.95–0.99 | 1.06–1.15 |

- **The saving ceiling.** The prepared-support floor (bitmap given for free) is 0.91–0.92, the same as H. The whole GLOBAL-attention share is about 10% of a forward at this length, so no sparsity setting can save more than about 10% per forward.
- **The two overheads.**
  - The selection step (D − H) costs 8–14 ms per forward.
  - A true-QK re-observation (A − H) costs 25–40 ms per forward.

## 2. Kernel vs sparsity attribution (strong dense baseline)

The same-consumer all-kept dense `D_matched` measured **0.982** (LB c0) and 0.991 (AIME c12) of native per forward. About 2% of any gain over native therefore comes from our consumer being faster than native SDPA at head_dim 256, not from skipping.

Against D_matched on LB c0 (8 calls, two of them native bootstrap), the gain attributable to sparsity plus reuse is:
- B A64: 4.3%
- M3 R6/A64: 3.2%
- M3 R3/A64: 2.2%

## 3. Sparsity sweep (M3 R3/A8, same states)

GLOBAL legal-pair skip fraction vs held-call cost (LB KDIV8 c0):

| threshold | QK/PV skipped | H / native |
|---|---:|---:|
| −ln2 | 53% | 0.921 |
| P0 | 69% | 0.909 |
| +ln2 | 81% | 0.901 |
| +2ln2 | 88% | 0.899 |
| +3ln2 | 93% | 0.899 |
| +4ln2 | 96% | 0.899 |

- Per layer, the consumer time is roughly linear in the kept fraction (kept 71% → 2.72 ms, 35% → 1.52 ms, 8% → 0.43 ms).
- Per forward, the saving saturates because the non-attention work is fixed.
- Going beyond P0 buys at most about 1% per forward, while its quality risk is unmeasured.
- **AIME (short early prefix) and RULER-4K:** every threshold stays at 1.01–1.04 × native.

## 4. Per-call mean over whole canvases

**Measured** (LB odd-K c6, 21 calls):

| arm | per call / native |
|---|---:|
| B A64 | 0.921 |
| B A16 | 0.931 |
| M3 R6/A16 | 0.942 |
| M3 R3/A64 | 0.945 |
| M3 R6/A8 | 0.951 |
| M3 R3/A16 | 0.954 |
| M3 R3/A8 | 0.958 |
| compact M2 | 0.969 |
| M2 reference | 0.979 |
| M1 | 0.989 |
| fresh T | 0.991 |

**Composed** ([policy_screen.csv](policy_screen.csv)). All 45 frozen points (A {8,16,64} × R {1,3,6} × three thresholds, B, compact M2) are priced by combining:
- the measured per-phase ratios;
- the clock applied to recorded native LB canvas lengths.

This is a *conditional estimate* and does not model the candidate's own trajectory. At P0 on LB:

| point | est. per call / native |
|---|---:|
| B A64 | 0.922 |
| M3 R6/A64 | 0.930 |
| M3 R6/A16 | 0.933 |
| M3 R3/A64 | 0.941 |
| M3 R3/A8 | 0.951 |
| compact M2 A64 | 0.970 |
| M1 A64 | 0.983 |

- In every R, A64 is about 1% cheaper than A8/A16.
- R6 is about 1% cheaper than R3.
- No point beats native on AIME or RULER.

## 5. Decisions taken from this tier (before any Tier 2 output)

- **Threshold: P0.** Higher thresholds buy ≤ 1% at unmeasured quality risk.
- **M3 candidates:** R3/A64 (primary) and R6/A64 (secondary).
- **Matched baseline:** B A64 (hold-only).
- **Reference:** M3 R3/A8 kept.
- **Also in Tier 2:** M1, compact M2, native, D_matched and fresh T.
- **Not done: the parallel prefix-summary builder** (from the separate A64 task document). With A64 the only summary STORE per canvas is at BO. Removing its full cost would save ≤ 0.5% of an average LB call, which is below that document's own stop rule.
- **Length gate** (`min_route_keys = 8192`, native below it):
  - It is a named variant, measured only on AIME c12, where it cuts the overhead from 1.1% to 0.7%. The remainder is the router's Python prelude.
  - It is not in the main comparison.
