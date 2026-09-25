# v9: qualified measurement path + clean paired E2E (IN PROGRESS)

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

## Next
CP2: `scripts/v9_clean_request_timing.py` D/L/S on /2,/8 (<=18 executions).
