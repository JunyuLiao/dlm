# v10 CP1: bounded specialization (length-generic kernels)

Commit `901e360`. The static exact-length kernels are kept as the reference (`variant='static'`, default).
The generic twins (`experiments/numerical_qk_reuse/generic_kernels.py`) make the key count `K`, the tile count
`KT`, and the prefix-summary extent `PREFIX_TILES` runtime scalars (`do_not_specialize`). Loops were already
runtime `range` loops, so scan order, masks, arithmetic and the physical skip branches are unchanged. A dropped
tile still issues no K/V load and no dot.

## First mismatch and the one fallback
Without a hint, K = 8156 and 8556 (K % 4 == 0, K % 16 != 0) differed from static in the last ulp: both were
within 4.8e-7 of an FP32 reference and gave the same decisions. The cause: a constexpr K lets the compiler prove
fp32 row alignment, which selects a different load layout and so a different reduction order. Triton 3.2's
integer specialization only tracks %16, and `tl.multiple_of` on a scalar argument had no measurable effect.
The fallback passes `K // KDIV` and rebuilds `K = K * KDIV` in the kernel, where `KDIV` is a constexpr power of
two (at most 16). This restores bit identity for every tested K.

## Executable qualification (`tests/test_v10_generic_kernels.py`, 35 passed on H100)
Every production entry point was compared bit for bit, static vs generic:
- entry points: anchor route+PV with summary STORE; route_only legacy and with summary LOAD under changed live T;
  the preqk consumer (trace off and on); the held-bitmap `_held_eligible` + `_pv` path;
- geometry: LOCAL h16/hk8/d256/window 1024 and GLOBAL h16/hk2/d512;
- prefixes: 63/64/65, 127/128/129, 365, saturated/cropped local, global up to K = 8556, and all 5 alignment classes;
- adversarial cases: NaN, all-masked rows, all-kept, sparse;
- compared: bitmaps, eligibility, malformed flags, BF16 outputs, LSE, projected state, risk, invalid flags,
  summary buffers, and physical counters.
Variant growth is bounded, and zero compiles occur for unseen lengths after initialization.

## Variant counts (fresh process, empty private cache; `specialization_contract_and_tests.json`)
| | static | generic |
|---|---:|---:|
| 12-canvas request length sequence (local+global) | 115 compiles, 165 s | 13 compiles, 19 s |
| 8 further unseen lengths incl. new alignment classes | 91 more | 52 more (new classes only) |
| repeat | 0 | 0 |
| total distinct | 206, grows with every length | 65 = the whole bounded set |

Within one natural request, K's low 4 bits are constant: each canvas adds 256 and the local crop is a
multiple of 64. So a real request touches only 1-2 alignment classes (7 misses on /2 and on /8).

## Request level (`cold_disk_warm_process_comparison.csv`)
Arm: S (M1, LOCAL prefix summary). Each variant ran in its own new process with its own EMPTY private cache.
| | static | generic |
|---|---:|---:|
| /2 cold (empty cache, new process) | 94.67 s (56 compiles, 71.2 s) | **32.54 s** (7, 9.6 s) |
| /8 first-seen lengths (same process after /2) | 85.18 s (61 new, 49.5 s) | **38.64 s** (7 new, 5.0 s) |
| /2 warm repeat | 19.65 s | 19.45 s |
| /8 warm repeat | 34.43 s | 33.38 s |
| fresh process, populated disk cache (synthetic sequence) | 206 misses, 0.61 s load | 65 misses, 0.36 s load |
Tokens, per-canvas calls and termination are identical: static == generic == v9, on /2 and /8.
Optional startup warmup of the whole generic set: 56.6 s from an empty cache, 0.5 s from a populated one
(outside request latency).

## What this is not
It removes accidental compilation growth: engineering, not a sparse-attention speedup. The warm walls are
unchanged (19.45 vs 19.65 s, 33.38 vs 34.43 s: single repeats, within noise). Warm L/S remain slower per call
than native (v9: ~20%).
