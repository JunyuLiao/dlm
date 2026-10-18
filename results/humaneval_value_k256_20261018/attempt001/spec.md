# HumanEval164 value methods, K256, seed42 — 2026-10-18

User authorization: evaluate all 164 HumanEval tasks with thinking on and off,
seed42, V2, V3a, full-value swaps, direction-guarded residual, V31 and dense;
calibrate V2 to first-canvas work and report native quality, trajectories,
clean request latency and effective physical sparsity. Older LongBench jobs
remain paused. This protocol authorizes this separate HumanEval campaign only.

## Frozen comparison

- Official pinned HumanEval source (SHA256 b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef), all task identities exactly once per arm/setting/pass.
- Same complete-function instruction as the historical thinking-enabled v27
  panel; model chat template with enable_thinking true/false. Gold tests never
  enter generation. Native canvas256, maximum48 denoising calls/canvas,
  confidence0.005, stability1, entropy0.1, temperature0.8→0.4; output8192.
- Pinned current vLLM/SM90 FA4, FULL mask fix51994 and LOCAL window fix;
  max_model_len9216, block64, prefill chunk4096, batch1, memory0.80.
  All settings/source/runtime/model fingerprints are frozen before generation.
- V31 is Yuhan's inherited mass qblock_max selection/reuse: select step1,
  carry first, settledness trigger0.15, sticky1.386; K256 tokens = four KV64
  complete prefix blocks per query-head/Q128 tile. Boundary/canvas protected.
- V2 is the existing Gaussian32 online preserve-mass selector on all five
  GLOBAL layers, with a uniform threshold per thinking setting. Its running
  softmax correction is selector state only; native retained FA4 output remains
  fresh. V2 has variable tile counts and is explicitly exempt from fixed K.
- V3a is the existing Gaussian32 full-row singleton-deletion ranking, fixedK4,
  on all five GLOBAL layers. No greedy search or new selector is introduced.
- Full swaps and direction-guarded residual retain the qualified device_exact_v1
  implementations: V31 K4 on all GLOBAL layers, with the existing value change
  only at GLOBAL layer29. Swaps retain their up-to-eight disjoint swap cap,
  naturally bounded by available removals/additions; residual changes held
  output explicitly. This layer-scope difference is reported.
- LOCAL, prefill and commits remain native dense. Include dense FULL fix51994,
  matched PIECEWISE dense and all-kept FA4 references. No sampler intervention,
  hidden forward, output shortening, forced canvas, or runtime package edit.

## Stages and gates

1. CPU tests for request dispatch/reset, fixedK accounting, threshold selection,
   input/coverage binding and thinking-aware extraction. Existing independent
   selector and refinement tests; H100 short-context/K4 formula parity. Native
   same-state oracle in qualification for swapped maps/correction caches.
2. Scorer preflight and canonical solutions through the exact extraction and
   official full-prompt check_program. Require164/164 with both channel modes.
   Non-thinking raw final text must not be discarded for lacking a channel tag.
3. Work-only calibration on24 deterministic prompt-length strata, with cohort
   weights. This overlaps the evaluation set and is explicitly transductive
   work calibration, with no accuracy used for threshold selection. In one
   native engine compare V31 with V2 thresholds0,0.01,0.03,0.1,0.3,1; then test
   up to two geometric interior points in an observed work bracket (neighboring
   midpoints if no crossing exists). All native calls run unchanged.
   Select smallest absolute first-canvas GLOBAL kept-fraction error, breaking
   ties by kept-rectangle error, then smaller threshold. Nonmonotonic results
   are retained. Freeze each selected threshold and calibration receipt.
4. Two-problem native qualification (longest and median prompt), audit and clean,
   both thinking modes and all controls. Require intended live arms, fixedK4
   checks, oracle parity, dense LOCAL, unchanged native geometry, nonempty
   trajectories and zero measured CUDA captures. Qualify warmups explicitly.
5. Full audit in two resumable index-strided shards per setting. Shared engines
   run V31 bookends around rotated V2/V3a/swaps/residual requests; controls use
   separate matched engines. Validate first-canvas work on all164. Target a
   GLOBAL sparsity difference <=0.25 percentage points and kept-fraction ratio
   within1%. If missed, permit up to two additional native V2-only full audits,
   using observed work to interpolate/refine the threshold; preserve every
   attempt, select nearest work only, disclose any remaining mismatch. No
   accuracy-derived threshold change is allowed.
6. Freeze the final work-based threshold, then full clean runs using identical
   cells, seeds and source. Request/prefill boundaries synchronize; no per-step
   synchronization, routing audits, diagnostic kernels or CUDA capture inside
   measured requests. Identical warmups per engine family precede measurement.
   Report clean/audit answer and trajectory drift, and V31 bookend drift.
7. Official tests, independent source/input/coverage/fairness verification,
   paired question-bootstrap confidence intervals and sanitized final report.
   Completion marker requires every planned arm/setting/pass and scorer record.

## Recorded measurements

Official pass@1 and pass+finished; extraction failures; generation caps; N,
median/P90/P95/maxN; C; pooled N/C; T; S; W; amortized S/N; paired W speedup;
canvas cap counts where available; GLOBAL eligible/kept/skipped rectangles,
LOCAL window rectangles/skips, overall denominator, first-canvas and later
work, observation/selection/correction events, selector QK/PV rectangles and
audit module times. Physical consumer work and selector discovery are separate.
Accuracy and latency conclusions use clean requests; audit scores are also
reported. A single generation seed is exploratory, not seed-robust confirmation.

One worker per H100; check idleness before launch. Startup, warmups, failed
attempts and native work receive GPU reservation accounting. Progress is emitted
about every30minutes; structural failures stop the queue and preserve attempts.
Resume uses a new attempt leaf and skips hash-verified completed shards. A
campaign pause marker is checked before each launch. No old campaign is resumed.
Expected scope is roughly6,000 measured requests plus bounded calibration and
qualification, with several GPU-hours depending on native trajectory tails.
