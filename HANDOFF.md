# v9: qualified measurement path + clean paired E2E (DONE: no E2E gain)

## Identity / authority
- Spec: user-supplied v9. Resumed from reviewed `3b67ea16261a5241c4bee3175ee9859dc9038da0`
  (local == remote, no live jobs, GPU idle). Branch `research/numerical-qk-reuse-native-20260924`.
- Host: this machine (hostname `mpk`, STATE's exouser@149.165.151.254), 1x H100 80GB.
- ACTUAL imports (verified, not assumed): torch 2.6.0+cu124 and **triton 3.2.0 from
  `~/.local/lib/python3.10/site-packages`** (user site shadows the conda env's triton 3.7.1);
  transformers 5.11.0 from conda env `ljy_dlm`; `dllm` from this checkout via PYTHONPATH=src:.
- Runs execute from immutable snapshots `../deploy/<sha>/` (git archive, read-only).
- v8 prefix-summary selector and all older results preserved; v8 reports got appended
  `CORRECTED (v9)` notes only.
- Output root: `results/numerical_qk_request_timing_20260924/` (compact, in Git);
  private receipts `../results/v9/` (raw answers, never in Git); bulky traces
  `/media/volume/dllm-1/dyh/numerical_qk_request_timing_20260924/`.

## Storage preflight (section 0)
Root /dev/sda1 ext4 had 58 GB free, 98% inodes free (>30 GB target) -> NO migration,
nothing moved or deleted. Large local disk /dev/sdb ext4 at /media/volume/dllm-1 (969 GB
free) used only for new dyh-owned trace dir. Receipt: `storage_preflight.json`.

## CP1 (done): measurement path
- 3A: v8 "native_dense" step row = `_install_dense` -> dispatcher -> `dense_eager` (spy:
  30/30 calls). True native = unbound, registry entry `sdpa_attention_forward` (spy:
  30/30 native, 0 eager, 0 selector). Test pins dispatcher selection.
- 3B/3C/3D: `scripts/replay_harness.py` restores RNG, native sampler, stopping criterion,
  logits processors and T controller BEFORE EACH rep (untimed); strips capture-time
  `query_adaptive.Sampler` wrappers; digests inputs+state and outputs per rep (all
  identical); execution flags asserted equal to those read inside `adapter.generate`
  (inference_mode on, grad off, no autocast, sdpa, tf32 off, default stream).
- 3E: M3 held = anchor0 -> held1 on real ScoreCache; telemetry cleared per rep; the
  simulated same-input ordinary phase of the per-call microbench is relabeled.
  Sparse step 'ordinary' uses age-1 history from an actual replay of captured step 0.
- 3F: "206-183=23 ms, no room" withdrawn (appended notes).
- 4: live/peak score, summary and total bytes; separate enforced summary budget
  (exact fallback to legacy recompute); summary dtype/shape/prefix-range guard.
- 2: `v8_smoke_receipt_verification.json`: /2, /8 tokens + per-canvas calls identical
  v7(L) vs v8(S), 2/2 correct on this subset; v8 walls diagnostic-ON (not timing).

Corrected step replay (aime26/2, ms, median of 10, event span):
| row | canvas1 (prefix 365) | canvas6 (local 1023/global 1645) |
|---|---|---|
| native decoder forward | 113.8 | 115.7 |
| native step, step-1 inputs | 122.5 | 132.2 |
| dense_eager same mask | 129.0 | 131.5 |
| L ordinary (age 1) | 153.3 | 165.8 |
| S ordinary (age 1) | 153.4 | 167.4 |
L/S anchor 160-167. First observations of new shapes: up to 1426 ms (Triton JIT).

## CP2 (done): clean paired request timing (`clean_request_timing.{json,csv}`)
One process, deployed `28f0f76`, 18 executions (2 IDs x D/L/S x attempt0 + 2 warm repeats,
order alternated), diagnostic off, seed 42, native adaptive, thinking ON, 8192, EOS.
| | D native | L legacy | S summary | L/S |
|---|---:|---:|---:|---:|
| /2 warm wall s | 22.91 | 19.53 | 19.59 | 0.997 |
| /8 warm wall s | 20.36 | 33.72 | 33.98 | 0.992 |
| warm ms/decoder call | 154-157 | 186-192 | 188-192 | |
| attempt0 cold s (/2, /8) | 24.16, 20.37 | 63.85, 38.32 | 34.66, 82.27 | order-dependent |
Quality attempt 0: 6/6 correct. Every warm repeat is token/call/termination-identical to its attempt 0;
L==S tokens; D == v7 native tokens; L/S == v7/v8 M1 tokens. Calls: D 146/132, L/S 102/181.
**S is not faster than L. L/S are ~20% slower per call than native.** Request totals differ by trajectory
(support/mask), which is not execution speed.

## CP3 (done): trace accounting (`request_trace_accounting.{json,md}`, deployed `3e88d0b`)
- `K/KT/PREFIX_TILES` are tl.constexpr, and global key length grows every canvas (10 distinct K by
  canvas 8), so there are new specializations every canvas. Real compiles cost +44 s (L /2) and
  +48 s (S /8) in attempt 0. With a warm disk cache the misses cost only 0.35 s. This explains
  v7's "430 ms/forward".
- Warm step is launch-bound: ~10.7k launches/step (MoE experts ~5k, shared with native). The GPU
  is active 68-80 ms of ~170 ms. The selector adds 14-27 ms kernel time (mostly `_route`) and ~1.4k launches.
- Pass A wall partition: prefill 0.60, steps 14.37, 8 commits 1.13, unattributed 0.28 s.
- Unassigned: native was not traced; replay-vs-request residual ~10-18 ms/step at canvas 1.
- No remedy implemented (request budget spent; a kernel-signature change needs paired verification).

## Artifacts
Git: `results/numerical_qk_request_timing_20260924/` {storage_preflight, v8_smoke_receipt_verification,
measurement_contract_and_dispatch.{json,md}, replay_state_tests, clean_request_timing.{json,csv},
request_trace_accounting.{json,md}, decision.md}.
Private (not in Git): `../results/v9/{replay,request_timing/{ledger.jsonl,private/},trace}`.
Traces: `/media/volume/dllm-1/dyh/numerical_qk_request_timing_20260924/traces_3e88d0b7c25f/` (2x50 MB).
Job scripts: `../run_v9_{replay,request_timing,trace}.sh`; snapshots `../deploy/<sha12>/`.

## Next (single decision; see decision.md)
Make `_route/_pv/_preqk_pv` shape-generic (runtime K/KT/PREFIX_TILES or pow2-bucketed KT),
prove bit-identity (existing CUDA suites) and L==S==v9 tokens, then rerun the same bounded
D/L/S protocol on /2,/8 with a FRESH Triton cache dir for every arm (TRITON_CACHE_DIR under
/media/volume/dllm-1/dyh). If warm per-call stays ~20% above native, the next lever is launch count
(fusing per-layer selector glue), before any accuracy/multimodel expansion.
Not done: 240-request matrix, seeds, M2, mask unification, native-mask/multimodel/bitmap
ablations, Haowei JAX kernel integration, I-DLM.
