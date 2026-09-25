# v11 CP1 kernel qualification

Build `e3c283b8cbb9b79c`, commit `ce32230`. Tolerances were fixed before any result.
- **Unit tests** (`tests/test_v11_support_consumer.py`, 25 passed on H100):
  - FP32 reference within the envelope (max-abs <= 1.5x Triton consumer error + 1e-3, relative Frobenius
    <= 1e-2) on LOCAL/GLOBAL model geometries (prefix 63/64/65/365/1023/1645 local, 129/1645/5004 global,
    all-kept, sparse);
  - constructed supports: all-dropped gives zero work; also one kept, last only, alternating plus an
    ineligible head, long runs, and a Q tail of 77;
  - NaN is flagged only when retained;
  - read-only maps, exact counters, a non-default stream, model-layout output;
  - **bit-identical to fresh T on fresh T's own support**;
  - NaN poisoning of dropped tiles leaves the output bit-identical.
- **compute-sanitizer:** memcheck 0 errors, racecheck 0 hazards, synccheck 0 errors. The first build failed
  synccheck: an aligned `bar.sync` was reached from two roles. It was fixed with non-aligned `barrier.sync`.
- **Real model states** (aime26/2, production M1 S-selector supports with nonuniform T; canvas 1 and canvas 8;
  660 ordinary-step calls; the model consumed the Triton output):
  - relative Frobenius error vs FP32 <= 0.0025 everywhere;
  - counters exact; no invalid rows;
  - **max-abs error was outside the pre-fixed envelope on 60/660 calls**: new/Triton max-abs ratio median
    1.32, max 2.08. The cause is fresh T's P convention (per-tile normalized, alpha-scaled BF16 P) vs the
    Triton consumer's unnormalized running-max P.
- **Not loosened.** Consequence: H1 is numerically different from O (it has fresh T's arithmetic), so H1 vs O
  token differences are expected and are not an execution-only claim.
