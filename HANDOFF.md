## Independent GLOBAL value selectors — 2026-10-08

Verified source `01f37fe33d4ad8a4f11ef4a4d5066360c96c49ab`, branch
`value-aware_cross-step-reuse_v2`, study `results/v31_value_selectors_20261008/`.
V1/V2, singleton deletion, exact cumulative greedy and separately named batch8
were derived independently from the requested formulas. Yuhan's inherited
reuse lifecycle and later FA4 consumer are preserved. Junyu's prior Gaussian32
and C_gate families remain distinct from these new integrations.

Done: 63 focused H100 tests (42 CPU passes with21 CUDA skips), including
online mass semantics, direct masked-attention oracles, row support, GQA,
bounded RMS above128K, scratch nonaliasing, inherited clocks, and bitwise paged
consumer parity with GPU counters enabled/disabled. All nine arms passed the
one-cell original-S1 model gate at source `d63d8b1e`, official scoring binding,
zero timed CUDA captures, and receipt validation. These are qualification
receipts, not target accuracy or clean latency results. Exact greedy is costly.
The failed V2 whole-sequence-RMS warm-up and preliminary attempts are retained.

The updated source separates clean timing from complete target audits, records
actual denoising graph modes, and counts native dense/LOCAL geometry without
changing their consumers. All three final instrumentation model gates passed at `01f37fe3`: every native
decode call used FULL, every control call used PIECEWISE, and clean all-kept
removed GPU counters. All three match their earlier outputs and trajectories
exactly. Next/running: audited original64-cell control reference, development-only
uniform V1/V2 calibration, then separate503-cell clean and audit passes for every arm.
Four native-order shards use identical first-cell warm-ups and a frozen rotated
arm order; completed shards survive restart, failed workers receive new attempts.
The execution freeze is `protocol_execution_attempt002_20261008.json`.

The exact original S1 schedule is32 development items at seeds1,2; its historical
SHA matches. The15-item preliminary control (76.45235% GLOBAL sparsity) is
excluded from calibration because11 items belong to the original holdout.
`protocol_original_s1_20261008.json` and the supersession receipt preserve that
mistake. Target evaluation retains all503 official seed-1 items, with the
original160/343 split and332-item sensitivity subset. This historically examined
pool does not provide a fresh-confirmation claim. Thresholds are still unfrozen.
Exact RULER v33 and HumanEval manifests/cell schedules remain unavailable; no
replacement prompts or seeds are generated. No complete target panel, accuracy
improvement, end-to-end gain, or study completion marker exists.

## V31 GLOBAL + LOCAL AIME26 run — 2026-10-06

The complete 30-problem × seeds 42/43/44 panel is under
`results/v31_20261006_aime_global_local/summary.md`. Dense and MAGE + LOCAL
records are complete (90 cells per arm), scored 51/90 for both arms. The
1728-token GLOBAL budget realizes 45.90% physical GLOBAL sparsity. The
512-token LOCAL budget realizes 25.33% over all LOCAL calls (32.81% on
sparse-only LOCAL calls), and 33.41% combined attention-work sparsity.
End-to-end mean is 5.9326 s dense vs 7.1250 s sparse (0.833x speedup);
decode-only is 5.9127 s vs 7.1055 s (0.832x).

The clean 50-sample H100 qualification found dense local FA4 at 0.0531 ms
and best tested sparse local FA4 at 0.0701 ms at prefix 8192 (0.757x), so
LOCAL sparse has no actual kernel speedup at the requested budget. Do not
headline a local speed gain. The installed FlashInfer alternative could not
JIT-build because its nvcc command uses unsupported `--compress-mode=size`.
The full result is a verified negative for this local kernel path.

Source and runner changes are on the current branch; CPU qualification is
21 passed. The raw attempt directory is private and ignored by Git.

## LOCAL compact consumer qualification — 2026-10-06

The named `local_compact_triton_q64` consumer is implemented in
`experiments/numerical_qk_reuse/v31_local_kernel.py` and selected by
`LOCAL_KERNEL=compact_triton` (the default when LOCAL routing is enabled).
It packs the existing Q128 selector map into Q64 tile lists and runs the
short `(1023, 1023)` window directly over native paged K/V. `LOCAL_KERNEL=fa4`
retains the previous direct FA4 path.

The final-source qualification is under
`results/v31_20261006_local_kernel_opt/attempt002/`. With a 512-token budget,
the compact consumer is 1.30x, 1.10x, and 1.25x the dense FA4 call at prefixes
187, 3500, and 8192, respectively. Its maximum absolute difference from the
old FA4 sparse consumer is 0.001953125. These are clean held-map kernel
measurements, not end-to-end or accuracy results. The AIME26 panel above is
unchanged; a fresh panel is needed to establish total attention and request
time after this integration. Verified commit: `121535d9`.

## V31 status (2026-10-03 23:20 UTC−5) — read this first

- **Machines:** dllm (149.165.159.64) is retired and can be shut down by the user; it is idle.
  - Its unique data is backed up to dlm2:`/home/exouser/dyh/backup_dllm_20261004/` (four `.tar.zst` archives,
    every entry and byte count verified; the README has the sha256 values and the list of skipped rebuildable items:
    models, envs, caches, HF text).
  - Private RULER gold, the pool and the v31 records are also at local `E:/dlm/v31_private/backup_dllm/`.
  - RULER scoring now runs on mpk (`/media/volume/dllm-1/dyh/ruler_scoring`, ljy_dlm env, read only). It reproduces
    dllm's panel t scores exactly.
- **Accounting fix (78644966f):** the method's legacy `kept_prefix_fraction` counted its bootstrap dense call as
  kept. Use `sparse_kept_prefix_fraction` and `global_prefix_work_fraction` (exact from 2026-10-04 on; older
  records are reconstructed by `scripts/v31_ruler_compare.py`).
- **RULER at matched compute** (v31 pool, 174 cells; doc section "Correction: realized-sparsity accounting"):
  - m2c k12 mass reaches 138/174 at 0.43 of dense GLOBAL work. Dense scores 140; MAGE k=6144 at the same work
    scores 130 (8 / 0 discordant, p = 0.008).
  - MAGE needs 0.58–0.84 of dense work for 136–141.
  - This is exploratory (adaptive reuse of cells). It is being confirmed on a held-out pool.
- **Panel w (pre-registered, held-out RULER v32, 390 cells): primary NOT confirmed.**
  - m2c k12 mass 310 vs MAGE k=6144 308 at matched work 0.44 (p = 0.83). The exploratory 8 / 0 was a winner's curse.
  - Dense FULL = dense PIECEWISE on every cell, so RULER correctness has no trajectory noise.
  - All sparse arms lose to dense only through cwe (aggregation); without cwe they are within 6 / 360 of dense.
- **Exploratory queue (mpk + dlm2, code 44dd964df after a second audit):**
  - x: selection-granularity ladder (`MAGE_GRAN`, `MAGE_FRAC`) plus accuracy–compute curve points;
  - y: pooled residual (`RESIDUAL=centroid`) on both arms;
  - z: dropped-mass guard (`DROP_GUARD`) on both arms, plus the B (adapter overhead) and E (query drift)
    diagnostics on LongBench 64K;
  - l128: LongBench-v2 128K scaling (new pool `lb_long_v31_128k`, 24 natural-length items, seeds 1–3).
- Winners need a fresh pre-registered confirmation (RULER v33 plus LongBench seeds).

## V31 update (2026-10-03 17:05 UTC−5; branch `research/vllm-paired-20261003`) — read this first

- **Dense baseline:** vLLM 0.30.0 default (FULL) + the exact upstream fix of
  [PR #51994](https://github.com/vllm-project/vllm/pull/51994) (`FIX_51994=1`), seed-paired. The unfixed FULL default
  froze DiffusionGemma's causal / bidirectional mask (19–28% more forwards). Details:
  `docs/V31_BASELINE_ROOT_CAUSE_20261003.md`.
- **Efficiency switches** (opt-in, formulas unchanged, all unit-tested on GPU):
  - `LOGIT_STATS=fused`: C-gate hook, −4–5% per forward;
  - `DP_BUILD=chunked`: dense-prefix build 4–7× faster;
  - `OBSERVE=fa4`: FA4 in-kernel tile log-mass, observation at dense cost for compact mu;
  - `KV_COPY=triton` / `MERGE=triton`;
  - `MAGE_SELECT=fa4`: fair MAGE selection.
  - Overlay7 is the deployed set. Kernel bench: `results/v31_20261003/kernels/`.
- **Scored frontier** (`results/v31_20261003/panels/frontier_all_arms_vs_fixed_default.md`, 48 cells per length):
  - Accuracy is kept everywhere except MAGE k=1024 at 32K.
  - Best per forward: MAGE k=4096 0.84, m2c 0.88, main 0.91 at 64K. No W ratio is significant.
  - The C gate keeps N/C at 0.99–1.03; MAGE is at 1.09–1.30.
  - 96K: m2c S/N 0.83 but N/C 1.19; exact-mu arms OOM at MEM 0.92.
- **Honest status:** with the same efficient execution, the fair MAGE port is at least as good as the method per forward
  and end to end. Panel k (running, ~23:00 UTC) tests selection quality at a fixed budget:
  - keep 12% / 5%;
  - risk vs mass ranking;
  - with and without the C gate.
- **Prior art to cite:** SparseD (ICLR 2026, fixed steps), MAGE, LoSA, PulseCol, SeerAttention (in-kernel block stats).
- **LLaDA2.1-mini:**
  - Subagent branch `research/llada21-sparse-port-20261003`. CPU side is done; 18 tests pass.
  - The S1 GPU step failed because SGLang's breakable CUDA graph rejects `LogitsProcessorOutput`. The subagent is
    moving S1 to eager, gated on panel k.
  - Stray `~/.cache/sglang` on dlm2 (jit log and empty dirs from 10:45 UTC) is reported to the user for removal.

## Paused by user after short-suite completion — 2026-10-03 04:40 (UTC-5)

AIME32/32 and HumanEval32/32 completed generation, unscored. LongBench11/32
complete; b2_native interrupted in warm-up (0 timed rows), 20 workers unstarted.
All 75 closed workers succeeded. Three GPUs confirmed idle; all six local
follow-up/scoring coordinators stopped. NO automatic resume. Older queue
armed/active fields are preserved historical snapshots, not current liveness.

Published 6,857 sanitized request rows, per-worker accounting, effective_method
and numeric counters; 0 timed CUDA captures. Private originals and completion
files backed up locally and retained remotely. Evidence/resume instructions:
results/v30_20261003/pause_after_short001. Closed GPU seconds46574.813729;
interrupted-worker age643.62s recorded separately as approximate reservation.
Three exporter privacy/format tests passed. No new score or speed conclusion.

On explicit resume, verify pins and reuse completed workers; rerun interrupted
and unstarted workers in new directories, form a new complete family, then
strictly score. Do not simply restart original supervisors/frozen queues.

## Independent Junyu vLLM adapter checkpoint — 2026-10-03 04:13 (UTC-5)

User-facing concise status: [V30_WORKING_STATUS_20261003.md](docs/V30_WORKING_STATUS_20261003.md).
Added separate no-M3 GLOBAL adapter source: bare peer dense, projected all-kept,
fresh value-only, previous-confidence and previous-temporal variants. Official
FA4 remains the external dense reference; the peer consumer is explicitly named.
Ten CPU tests passed, no skips/CUDA initialization, 0 GPU seconds. Per-case
evidence/source hashes: results/v30_20261003/standalone_cpu002. Real peer CUDA,
paged-KV numerical qualification, finite policy binding and E2E runner remain
pending; this adapter is NOT armed. Do not mistake earlier M3 sensitivity or
original peer kernel queues for its execution. Current peer SHA remains b890ff494;
uniform attained policy JSON absent, README zero is a placeholder. Public old
RULER8K policies may only be separately labeled transfer candidates.

Read-only 04:09 formal snapshot: LB10/32, AIME28/32, HumanEval29/32; 67 closed,
0 failed, 41710.304596 closed GPU seconds. Earlier frozen queues unchanged.

## V30 follow-up queue armed — 2026-10-03 03:38 (UTC-5)

The new continuation is actually armed on94134a91b, phase waiting_prior_queues,
0newGPU seconds. All three own-host campaign003 CPU preparations passed, with
nine full-inventory sensitivity bindings and eight dense diagnostic bindings;
CUDA uninitialized. Source CPU tests33passed; private queue/payload guards6;
remote sensitivity/config/adapter regressions26. Earlier source freezes and
running worker deployments are unchanged. Receipt:
results/v30_20261003/campaign_queue001/status.json.

After original formal strict scores, original component/dense/engineering work,
and peer29-case qualification finish, run32fresh-engine dense diagnostics,
then60sensitivity pipeline qualification workers and nine complete strict CPU
scores. No automatic formal launch or quality/speed claim. Full eight-seed
specs exist for59LongBench,30AIME,164HumanEval questions per mode. Raw first
logits/canvas/RNG traces and hashes stay private; sanitized phase/count/graph
traces and boolean comparisons are retained. The first sampled state only
locates an initial difference; it cannot rule out a later divergence.

Original formal snapshot: longbench 8/32, aime 25/32, humaneval 25/32;
zero closed-worker failures. GPU resource seconds per suite remain in the
receipt; active worker spans are not yet included. Diagnosis is pending.

Operational continuation: private campaign_queue001.py/PID file, inputs in
prepare_*_003, scorer campaign_score001.py. Do not edit these now-pinned helper
files; a correction requires a new queue/run. Inspect status, preserve failures,
then review first-divergence comparisons and all nine strict qualifications
before choosing the matched engineering baseline for full confirmation.

## V30 configuration qualification and final shape guard — 2026-10-03 03:33 (UTC-5)

All three hosts successfully CPU-froze three sensitivity bindings each and dllm
also froze eight diagnostic bindings at26f3f03b6; CUDA stayed uninitialized.
No GPU execution. These unused campaign002 bindings are retained and superseded
by the next source freeze adding a pre-routing confidence geometry check.
A within-canvas Q-size change now fails before sparse selection; a new canvas
may change Q normally. Updated CPU regression26/26passed, no skips,78stages
CUDA-uninitialized,0GPU seconds. Per-case receipts and source pins:
results/v30_20261003/sensitivity_cpu002. New immutable deployment follows.

The prepared private continuation has32dense diagnostic workers and60pipeline
qualification workers (three modes x four arms x five datasets), then nine
complete strict CPU scores. Full eight-seed specs are committed; qualification
is one longest input/dataset, one warm+one timed and is NOT a quality/speed claim.
It waits for original formal strict scores, earlier component/dense/engineering
queues, and the29-case peer kernel qualification. It has six local guard/payload
tests but is NOT armed at this commit. No frozen prior queue is modified.

## V30 source-freeze guard correction — 2026-10-03 03:21 (UTC-5)

First CPU-only preparation on dlm2 stopped before GPU: the original parent
provenance also pins v21.py, so a byte-identical parent rebind correctly rejected
the reviewed wrapper change. Preserve failed campaign001. New config migration
now verifies every old pin, explicitly allows only reviewed v21.py/adapter.py
changes at any nesting, keeps all other dependencies byte-identical, and asserts
all mathematical fields unchanged except the named sensitivity. Two CPU tests
cover nested fingerprints, old-byte drift, unreviewed kernels and basename
spoofing. New immutable preparation is pending; no GPU was started.

## Dense divergence and M3 query-weight ports — 2026-10-03 03:16 (UTC-5)

Root reviewed installed vLLM sampling and adapter code. Native-hook prepare
performs a GPU-to-CPU metadata stack.tolist() each forward; no-hook dense does
not. The official compiled sampler draws from CUDA RNG without a request-local
generator argument. These facts do NOT prove the cause of divergent steps.
The new independent sync-only arm isolates the metadata read. Separate invasive
first-sample protocols retain initial RNG/canvas and first-sample input/output
privately; traces change synchronization and cannot establish timing gains.
Four arms, two engine seeds and two fresh-engine repeats per protocol (32
engines across sync-only/trace protocols); original V29 queues remain pinned.
New diagnostics have CPU tests, but are NOT deployed/armed in this commit.

Two named query-weight variants now have vLLM code: unit_v30 and confidence_v30
(previous completed call's processed p_top, 1+3*sqrt(1-p_top); first call fully
protected). This ports the confidence prior, NOT Junyu's C_gate or fresh
value-aware router. Preserve M3 DP/R6/A64/Q128/carry0 and torch copy/merge.
C_gate remains rejected in vLLM because the actual accepted mask is unavailable.
All three modes have new full-inventory eight-seed specs for LongBench (59),
AIME (30), HumanEval (164). Qualification/freeze/deploy/launch remains pending;
no variant GPU or E2E performance result exists. Short-task accuracy is required.

Remote CPU regression: 25 passed, no skips, CUDA uninitialized, 0 GPU seconds.
Per-case evidence: results/v30_20261003/sensitivity_cpu001. Local diagnostic,
privacy-comparator and spec/receipt tests: 12 passed. Existing adapter guard,
lifecycle and merge tests also passed. Last direct GPU check found one active
original worker on each of dllm/dlm2/mpk; no new GPU launch. Verified parent:
ed711c8b4afd5bf93a457f03bfc5714a5b36dcb9. Do not mislabel the new source/config freeze as a byte-identical rebind.

## Peer kernel GPU qualification armed — 2026-10-03 02:43 (UTC-5)

The29-case peer C ABI acceptance queue is armed on sourcee2ec0ae4b, currently
waiting for all original formal strict scores. Prepared source/kernel bytes
and pinned SM90 were checked remotely; no GPU was started. It then requires two
idle checks and an exclusive own worker lock on dlm2. Launch is single-attempt;
unknown transport outcomes are preserved, never automatically relaunched.
Every frozen case must pass; skips/duplicates/wrong parameter counts fail.
Per-test receipts and outer reservation seconds are retained. No automatic
ATen/model/E2E rollout or public performance claim follows a passing kernel test.
See results/v30_20261003/peer_gpu_queue001/status.json. Original dllm component,
dense-equivalence and engineering queues retain their existing order and pins.
Formal snapshot:LongBench6/32,AIME18/32,HumanEval18/32,zero closed-worker failures;
these counts are operational progress, not partial scientific results.

## Peer build and bounded regroup screen — 2026-10-03 02:37 (UTC-5)

Root implemented the isolated build, ABI guard fix and GLOBAL selector wrapper.
Junyu b890ff494 CUDA source compiled successfully with own nvcc13.0.88 and
NVIDIA TensorRT-LLM headers796584295. Peer default flags all remain false;
fast-SFU/inline variants are not silently substituted. C++/CUDA source unchanged.
Only the Python debug buffer check now accepts both ABI3 and ABI4, preserving
shape/dtype/device/contiguity checks. Complete chosen header trees are privately
hashed; public receipt, source hashes and29.6KB compiler log are in
results/v30_20261003/junyu_build002. Preparation001 failed before compilation
on the archive root directory entry; it is retained, with0GPU seconds.
Compiler warnings include WGMMA serialization and register spills; they do not
establish a measured bottleneck. The built kernel has NOT run on GPU yet.

GLOBAL-only scope is implemented in peer_global_scope.py using a narrowed
adapter module selector, leaving LOCAL untagged and native call eligibility
unchanged. Four CPU tests cover exact module identities, pass-through arguments,
result identity, scope rejection and cleanup. ABI/build guards add three tests.
This is an HF peer-reproduction component, not a vLLM port or model parity proof.
Twenty-nine original peer kernel acceptance cases collected successfully with
CUDA hidden. Their new frozen spec rejects skips and changed parameter counts;
three queue-contract tests pass. GPU execution will follow the original formal
strict scoring barrier on idle dlm2; it is not armed in this source-freeze commit.
A matched ATen bridge and complete model/scale/scope qualification remain pending.

New independent regroup_key6 CPU prototype samples six fixed support bits,
stably groups inside each head/Q128, rebuilds exact unions, and accepts only
blocks with non-worsening total/max alias2 tile work. Five tests cover coverage,
bijection, ties, nonnested equal-count supports and adversarial/random supports.
Executed all144existing held-inventory snapshots:603/4608blocks accepted,
12.1179%rows moved, aggregate tile ratio0.997058, load-proxy sum ratio0.996582.
CPU construction median1.774874ms (range1.045137–3.961188), excluding loading.
These are prefix-only development proxies, NOT GPU/request gains. Improvement
is too small to justify claiming that row movement/build costs are covered.
Preserve raw anonymous per-snapshot values in results/v30_20261003/regroup_key_cpu001.
Haowei document snapshot pins6f7279c16 and is explicitly attributed.

## V30 cooperative campaign active — 2026-10-03 02:17 (UTC-5)

Branch `research/cooperative-sparsity-20261003` starts from verified V29
`035b70c3f`. User authorizes attributed peer-document snapshots and isolated
execution; root owns algorithm/fairness review. Peer branches stay read-only.
See docs/V30_COOPERATIVE_CAMPAIGN_20261003.md.

Correction: profiled singleton32K cannot establish that main is slower at64K/96K.
Closed V28 preview main/native S/N ratios are0.902307 and0.864398; N ratios
1.197263 and0.955624; W ratios1.084097 and0.849983. Each bin has only two
questions, so these are preview point estimates. V18 controls have different
warm/request inventories; its four repeats are two seeds times two repeats.
Full accounting, numeric tables and five CPU tests are committed under
results/v30_20261003/closed_length_accounting001. Physical density was not
recorded in these requests; high HF sparsity is not evidence for their density.

Junyu b890ff494 docs copied with attribution and redaction. Original peer CPU
formula tests actually executed:31passed, no CUDA initialization, GPU seconds0;
per-test receipt in results/v30_20261003/junyu_cpu001. Kernel/E2E remains pending.
ABI4 debug guard defect needs an isolated patch before affected qualification.
Peer reproduction design is explicitly a draft, not launched or frozen.

Formal workers continue on their original sources and eight seeds. The previous
completion monitor failed on a read-only SSH timeout, not a GPU worker failure.
New completion002 retries only read-only transient transport errors. Components003,
dense003 and engineering003 are armed after strict formal scoring in that order;
new directories preserve failed originals. No scientific settings or deployments
were changed. Snapshot and source pins:results/v30_20261003/queue_recovery003/status.json.
Successor CPU checks passed28/26/23/35. Counts at snapshot are operational status,
not partial performance results. Three GPUs remain reserved for formal workers.

Next: isolate peer ABI/toolchain fixes and GLOBAL-only routing tests; compile
without touching shared environments; execute numerical qualification only on
an idle GPU. Await frozen dense12-engine diagnostic before attributing the large
dense trajectory discrepancy. New regroup candidates must include construction,
movement and restoration costs against the same optimized natural-order consumer.


## Additional peer kernel and timing boundaries

The peer kernel advances through KV tiles sequentially with per-tile cluster
votes (value_direction.cu:306–316,404–416,571–578). Its split variant separates
router/PV roles across CTAs (:623–666); this is not independent KV-range
splitting followed by an LSE merge. validate.py:63–83 measures graph replay,
excluding ordinary Python dispatch and the host-side TMA descriptor construction
in value_direction.cu:754–758. Cached-sketch component timing is not dynamic
request latency. All locations refer to the audited b890ff494 peer ref.

A cooperation candidate is register-fed rank32 computation and on-chip running
state (value_direction.cu:359–401): investigate whether our observation/DP
boundary can avoid some intermediate-summary materialization. Our all-dense-prefix
risk dependencies differ from the peer retained-state router. Preserve our
mathematics and independently qualify any experiment; no equivalence, port,
new variant or measured gain is claimed. Running queues remain unchanged.

## Junyu newly published implementation audited (2026-10-03 01:39 (UTC-5))

Read-only peer pin b890ff49488474c5d476df44019597dc045a5969 supersedes the
previous859c0c8fc publication boundary. All15 missing linked implementation/test
targets and the named report test are now present;118 changed Python files
parse, with no peer code import/test/build/GPU execution. New diff has127files
and no results changes. No peer branch merge, source absorption or modification.
See docs/JUNYU_VALUE_AWARE_CODE_REVIEW_20261003.md and the numeric publication
receipt results/v29_20261002/junyu_publication_update001/audit.json.

Confirmed static defect: ABI4 kernel is rejected by the ABI3-only debug_scores
guard used in same-logit qualification/tests. Ordinary no-debug calls are not
implicated by that check. Historical dense cache reuse has incomplete protocol
identity validation; this is a code risk, not proof that prior outcomes are wrong.
Peer kernel physically skips full PV but still computes current QK/rank32 risk;
its SDPA native adapter is not our vLLM/FA4 baseline. New C_gate agrees with our
P17 port mathematically at matched parameters, while complete routing systems
differ. P17 remains configuration-specific; current vLLM adapter still rejects
C_gate pending accepted-mask lifecycle support. Treat integration as a cited
cooperation candidate requiring separate qualification/calibration, not a speed
or accuracy result. Current formal deployments and follow-up queues unchanged.

## Dense diagnostic and performance follow-ups rearmed (2026-10-03 01:26 (UTC-5))

The first component coordinator failed during a local Windows atomic status
replacement (PermissionError), before deployment or GPU launch. Engineering
stopped at its dependency gate; the first dense coordinator failed check-only.
All original scripts, plans, statuses and failed evidence are preserved.
Successor002 coordinators use new local/remote directories and bounded retry
for transient status-file replacement contention, with retry/exhaustion tests.
All three successor queues are armed and waiting; no new GPU evidence exists.
See results/v29_20261002/queue_recovery002/status.json for sanitized receipts.

Order remains original formal generation and strict CPU scores, then two
component jobs, then12 fresh-engine dense diagnostic workers, then only the
engineering fusion qualification and its strict scores. Component/engineering
science stays at1342ec0b1; dense diagnostic source isbdd59ee97. No current formal
deployment or scientific protocol changed. Public diagnostic CPU tests:21;
private successor tests: components26, dense23, engineering35.
Only source/status evidence is committed here, not hypothetical GPU results.

## Dense reproducibility diagnostic frozen; engineering queue armed (2026-10-03 01:16 (UTC-5))

New public worker/specs: scripts/v29_dense_reference_diag.py and three
v29_dense_reference_diag_* specs. Same original32K index0, two engine seeds
28001/28002, two fresh engines per seed/condition,12workers. Conditions are
default dense, PIECEWISE no-hook dense and PIECEWISE native-hook. Preserve the
original sampler; read only default-device CUDA-generator state at existing
request boundaries, retain fingerprints privately, publish only comparisons.
Record actual graph modes per execution phase, warm and timed N/C/P/output
length, and original counts/JIT receipts. No new per-step synchronization or
profiler is added; boundary instrumentation is diagnostic, never formal W.
Twenty-one public CPU tests pass. Read-only actual cost003 spec comparison
confirms exact equality with all three public protocols after make_spec; old
input, method and deployment are unchanged. The private12worker coordinator
passes20CPU toys and honors the frozen block/repeat/variant launch order.
It is not armed yet in this source-freeze commit. GPU validation remains pending.
See docs/V29_DENSE_REFERENCE_DIAG_20261003.md.

Engineering fusion qualification is now armed after30private CPU tests. It
waits three closed predecessors: original formal generation+strict scoring,
two component jobs, then the complete12worker dense diagnostic. It will run
only40warm+40timed qualification requests and strict score all six families;
no full variant panel or automatic public claims. Its source remains1342ec0b1.
Component source also remains1342ec0b1. No current formal deployment changes.

New telemetry note distinguishes sparse-path execution from unknown per-tile
keep fractions, and documents observation storage: exact32K prefix geometry
needs268MiB summary storage per GLOBAL layer (FP32z/mu32+int8flags),1340MiB for
five layers, excluding DP/output/allocator costs. This is capacity arithmetic,
not measured traffic or causal latency. Original003 summary_peak_bytes is
1569751040. QK sharing does not make statistics storage/DP/full output free.
See results/v29_20261002/dense_native_accounting001/TELEMETRY_LIMITATION.md.

## Dense/native accounting correction and component queue (2026-10-03 00:57 (UTC-5))
Source-audit supplement: docs/V29_DENSE_NATIVE_AUDIT_20261003.md documents the
actual installed vLLM sampler and runtime dispatcher. Native forwards the original
attention arguments and sampler result unchanged; six CPU wrapper-contract tests
pass. This does not rule out GPU execution, graph, hook or RNG bugs.002/003 inputs
and core source bytes match, while profiler instrumentation differs. Official
sampling uses global GPU random draws without a supplied request generator; no
saved RNG boundary state identifies the first divergence. Default initialization
FULL_AND_PIECEWISE is not evidence of each forward's actual graph mode. A separate
three-condition graph/RNG diagnostic is being implemented; no new GPU validation.


The public raw002/003 records were re-tabulated by
scripts/v29_compare_dense_diagnostics.py into
results/v29_20261002/dense_native_accounting001/ (8request rows,27event scopes).
These are original profiled singleton durations, explicitly excluded from formal
performance claims.003 includes PIECEWISE dense without adapter hooks: N374,
output4761,profiled S/N30.029ms. Same-mode native-hook has N99,output2107,
S/N32.657ms; main has N115,output2006,S/N33.248ms. Thus003 alone has slower
main amortized S/N, despite its lower GLOBAL span/N. Different trajectory and
instrumentation prevent causal kernel/request claims. The intermediate no-hook
control is missing from the formal panel, not from all previous diagnostics.

The native374-to99 change between002/003 remains unexplained. Do not dismiss it
as seed noise or assume PIECEWISE explains a same-PIECEWISE discrepancy. Warm
request N/output length and sampling RNG state were not published, so the new
table marks them unknown. Equivalent dense math with identical random inputs
and deterministic execution should reproduce; current engine-seed labels alone
do not establish those conditions. Investigate graph dispatch, hooks, sampling
state and first numerical divergence in a separately frozen diagnostic.

Correct a small old narrative transcription error without altering original
artifacts: raw native LOCAL639.6247041076422ms/99=6.460856ms/N, not6.461866.
All nested/side-stream leaf spans remain non-additive; no fake wall breakdown.

The two frozen component jobs are now armed on source1342ec0b193a929295f08964c132aee96c28c5a1,
waiting for all current formal generation and strict serial CPU scores before
any remote deployment/GPU launch.21 private queue tests passed; own-directory,
registered GPU/idleness, source, full numerical-oracle, every raw timed sample,
zero JIT/capture and failure/resource preservation gates are enforced. No new
component GPU result exists yet. The real-model fusion queue is still being
prepared; no source/binding/run of the current formal campaign is changed.

## Performance follow-ups and Junyu source audit (2026-10-03 00:44 (UTC-5))

The new same-state cost benchmark, raw component timing helper and sparse-cycle
Q movement prototype are CPU-qualified, not GPU-qualified.47 focused public CPU
tests pass (15 cycle,12 homogeneous,5 timing,15 engineering specs). Frozen public
protocols and six legacy/fused real-model qualification specs are committed with
the source. No new model/request gain is claimed. Component samples retain every
arm/repetition/order and host/event duration; no Chrome timeline is fabricated.
The official JIT monitor must activate before FA4/consumer imports, numerical
oracles precede warm timing, and all monitored timed JIT/capture deltas must be0.
See docs/V29_PERFORMANCE_NEXT_20261003.md and the new public specs.

Junyu ljy/value_aware was reviewed read-only at859c0c8fc2a4509a148a3270915f79961e565dee.
Its latest three commits only change READMEs; the newer Hopper/C_gate sources and
tests linked there are absent from that ref. Historical current-QK retained-state
routing is distinct from our dense-prefix surrogate and temporal reuse. Its old
quality/budget/backend results do not establish performance against vLLM/FA4.
The documented C_gate is a cooperation candidate, not a verified port or new
contribution of this branch. See docs/JUNYU_VALUE_AWARE_REVIEW_20261003.md.

The user's dense/native equivalence challenge requires a separate audit. Native
hooks currently call original FlashAttentionImpl.forward, but graph mode/hook and
RNG-trajectory equivalence are not proved by that fact. V18b warmed59 questions
for dense/main and only12 for the control subset, without per-request reseeding;
matched labels were not matched random states. Preserve its measurements but do
not causally attribute default/native W or N differences to PIECEWISE, sparse
quality, or seed noise alone. V29 already uses identical full warm inventories
for all arms; more seeds still do not establish graph/hook equivalence.

Current formal panels remain immutable and running. Closed workers at this
snapshot: LB1/32,AIME3/32,HE4/32; no reported failures.
Their outer closed-worker GPU seconds are recorded in STATE.current; active
unclosed time is excluded and inner worker times are not added. One-shot strict
completion/scoring is armed (17 CPU tests passed), waits all32/32/32 workers,
then scores HE/LB/AIME serially and exports detailed sanitized evidence for
review. New component and engineering queues are still being qualified; not yet
armed in this commit and no new GPU work overlaps the formal campaign.

## Detailed evidence index and receipt correction (2026-10-03 00:20, UTC-5)

Public evidence navigation: results/v29_20261002/EVIDENCE_INDEX.md, with executed
source commits, saved artifact links and interpretation limits. Both old diagnostic
campaigns saved measurement aggregates/records, not individual Chrome timelines.
Do not reconstruct event timing from their aggregates. The44GPU tests comprise
40BF16 independent FP32-oracle cases and4original TF32x3 STORE/LOAD comparisons.

AIME now also publishes path_execution_receipts.jsonl with actual adapter,
effective-method, KV-layout and method counters. The separate correction explains
the earlier README error: only dense has null receipts; native/allkept/method do
have them. Previously published per-row availability flags were already correct.
Old files and original generation are preserved; strict validation had checked
the real receipts all along. This is an export/documentation correction, not a
method change or new accuracy/speed evidence.

All three formal panels remain in their first full-question worker,0closed and
0failed at00:19 snapshot; active GPU time is not yet closed. Completion/scoring
helper is being CPU-tested before arming; current source/bindings remain frozen.

## Three formal panels launched (2026-10-03 00:12, UTC-5)

All three complete qualifications passed strict scoring and the actual frozen
formal-launch guard. LongBench12/12 and AIME4/4 detailed measurements, per-request
receipts and worker accounting are published under longbench_qualification001/
and aime_qualification002/. HumanEval4/4 detailed evidence was already published.
Wrong, unparsed and capped outputs remain failures; qualification is only pipeline
validation. In particular all four AIME qualification outputs capped/unparsed.
No accuracy or speed conclusion follows from these singleton qualifications.

The three formal panels now run: LongBench59 questions on dllm, AIME30 on dlm2,
HumanEval164 on mpk; each has8 engine seeds29001-29008 and4 arms,32 workers per
suite,8096 timed plus8096 warm requests total. LB/HE source54163f4e7, AIME1931db5a6.
Main stays frozen Q128/alias2 with legacy torch copy/merge and no2K route gate.
All qualification proofs/source pins, GPU identity/idleness and CPU scorer idle
checks passed. Private coordinator17CPU tests pass. mpk launched last; a shared
CPU-scoring barrier remains active until all three strict32/32 inventories close.
No existing source, binding or run directory was changed. Any failed worker stops
its supervisor and requires a new run directory; retain failed-attempt evidence.

Closed qualification reserved GPU seconds: HE672.373426352, LB2413.676399794
(including original interrupted001 outer244.358443453), AIME780.407498523
(including reused dense001239.994346702 exactly once). Together3866.457324669s.
With prior V29 components/diagnostics2262.065190672s, pre-formal closed total is
6128.522515341s. Formal workers are active; their unclosed time is excluded.
Use outer supervisor spans; inner worker times are not additional consumption.

Current new-model status clarified: LLaDA2.1-mini and I-DLM-8B official dense toy
GPU executions have completed. Sparse main/variant adapters on these models are
NOT implemented; CPU event-prototype tests do not qualify GPU integration.
See docs/NEW_MODEL_PORT_AUDIT_V28_20261002.md. Older pending-smoke notes are history.

Next: preserve all formal records, complete strict full-family scoring after the
three-suite barrier, export anonymous per-request numbers/receipts plus paired
geometric means and question-cluster95% intervals. Completion/scoring automation
is being prepared; no automatic interpretation or noninferiority claim.

## HumanEval qualified; trace exporter frozen (2026-10-02 23:51, UTC-5)

HumanEval's full four-arm qualification passes strict54163f4e7 scoring, original
bwrap sandbox, public correct/wrong toys and the actual formal-launch guard.
All four single-question outputs pass official tests; this is pipeline evidence
only, not accuracy noninferiority or speed. Publish detailed numeric requests,
receipts, workers and aggregate tables under humaneval_qualification001/.
Reserved outer GPU time672.373426352s; inner worker663.996s is not additive.
Dlm2 CPU scoring stopped on unavailable OS sandbox; mpk CPU scoring uses the
unchanged isolation after an authorized pinned-gold copy. An independent exporter
integer-key bug was repaired without changing the strict proof or GPU generation.
Formal launch is held until all three suite qualifications/scoring are ready, to
avoid heavy mpk CPU scoring contention during HumanEval formal timing.

LongBench qualification002 has11/12 workers closed without failure, final96K
method running,1992.535591133 closed-worker GPU seconds. Resource correction:
qualification001 outer reserved time is244.358443453s; earlier241.732s is its
inner worker terminal span. Preserve earlier records and use the complete outer
value in current accounting, never add both. A read-only original-source-ID audit
confirms59 distinct questions (24/24/11), with zero overlap between length groups.
Eight seeds remain repeats within question clusters; these are previously used
questions, not a newly held-out question pool. Audit receipt is published.

AIME qualification001 completed dense,239.994346702 outer GPU seconds, then its
launcher stopped at transient post-exit NVML activity. Dense artifact stays valid;
new002 runs only the remaining3arms at the same1931db5a6/input002 binding and merges
the preserved dense by exact byte pins. Four launcher CPU toys cover bounded
stable-idle wait, occupied GPU rejection, timeout and worker failure. Formal pending.

The independent trace wrapper and9 new CPU tests plus16 unchanged-profiler tests
pass. Both wrapper and original profiler must be source-pinned; spec/CLI flags and
ordinal agree; output is a separate wholly new private directory with no overwrite.
Trace export is off by default and occurs after the original request boundary.
No existing GPU deployment is changed, no GPU trace run has occurred, and no trace
is automatically published. See docs/V29_TRACE_EXPORT_20261002.md.

## Detailed evidence published and qualification progress (2026-10-02 23:34, UTC-5)

Publish the actual per-arm profiler outputs, sanitized request measurements and
terminal receipts from both closed diagnostics, not just their conclusions:
results/v29_20261002/cost32k_002/raw/ and cost32k_events003/raw/.
There are24measurement files,80,316uncompressed bytes and28,647bytes of gzip
copies, plus manifests/README. Root independently verified all24 gzip roundtrips,
public artifact digests and private-field/path scans. Full numeric measurements,
counts, method/allocator/layout receipts remain; private identities and data are
removed. No timeline was originally exported, so none is reconstructed. Resource
accounting uses each supervisor's complete reserved span, not inner terminal
clock values. These profiled requests remain excluded from formal speed/quality.

AIME CPU setup passed actual v21 validation and all42Linux tests on both original
mpk and destination dlm2; public preparation receipts and test inventory retained.
Destination frozen1931db5a6 now has a verified AIME001 input binding. An initial
minimal-input transfer stopped at strict byte verification due to Windows CRLF
versus Linux LF serialization (0GPU). New002 transfers original bytes, validates
both the byte pin and parsed30rows, and passes freeze/read_frozen; original001
is preserved. No gold contents were transferred. AIME GPU qualification pending.

HumanEval qualification generation closed4/4 with no failure; strict CPU scoring
and formal proof remain pending. LongBench qualification002 has closed5/12 with
no failures and is running64K native; frozen54163f4e7 unchanged. Its closed-worker
reserved time currently944.897625s; active-worker time remains unclosed. No formal
panel has launched. Preserve the separate previous241.732s idle-guard attempt.


User also requests timely commit/push and reasonably sized traces/measurement
records, not only conclusions. Full diagnostic trace privacy/size audit is
in progress; publish sanitized compressed traces and detailed numerical receipts
where suitable, retaining private originals unchanged.

AIME002 fallback CPU preparation did run on mpk before the latest switch: strict
rebind, actual v21 validation and all42Linux tests passed without CUDA initialization.
No binding, generation input/gold transfer or GPU launch was performed there.
This preserved CPU check does not qualify a full model request.

## User clarified private host transfers (2026-10-02 23:20, UTC-5)

The user explicitly authorizes transfers needed for research between their own
three machines. Private data/configuration/artifacts still must not enter Git
or public outputs. Writes stay inside the user's own dyh directories; chw/ljy
and all other collaborators' files are read-only and must not be deleted or
overwritten. Prior auto-review rejections remain recorded with zero execution;
new attempts must use normal approval review with this updated authorization.

Resume original AIME001 on dlm2 at frozen1931db5a6, with exact-source provenance
mirrors, original byte checks, actual v21 validation and Linux42 tests before
GPU qualification. AIME002 same-machine fallback is retained but not launched.
Gold can remain on its existing scoring host; generation/scorer artifact mirrors
are now explicitly authorized. HumanEval/LongBench qualification continues at
frozen54163f4e7. Formal launch still requires complete scored qualification proof.

## Completed event diagnosis and current qualifications (2026-10-02 23:17, UTC-5)

Diagnostic003 closed all four PIECEWISE arms at source54163f4e7, captures0 and
adapter ordering errors0; reserved1092.569460863946GPU seconds. Complete scope
counts match actual N/C/prefill. Matched native/all-kept/main denoise GLOBAL
span/N is4.494699/4.894036/4.329062ms and LOCAL6.461866/6.618996/6.611120ms.
These instrumented, different-trajectory observations are not paired speed gains.
No-hook dense phase attribution remains unknown. Main's40 fused observations
sum99.534944ms (19.99% of inclusive GLOBAL span), including normal dense output:
this is not incremental observation overhead. DP build40 and route105 match
receipts; side-stream overlap and nested consumer/selector ranges prohibit
adding these scopes as wall time. Full source/caveats:
results/v29_20261002/cost32k_events003/. No new task-quality result.

HumanEval four-arm qualification remains active on mpk. LongBench qualification
001 passed host CPU tests and one dense worker, then stopped at the next idle
check during transient NVML activity; preserve its241.732GPU seconds. New002
qualification on dllm reruns all12 workers at unchanged54163f4e7/source/binding,
with bounded stable-idle checks; any foreign compute process stops the launcher.
No formal panel has launched. Score full qualification before formal work.

Automatic approval review rejected AIME cross-host input transfer, then its
private config/provenance-only subset; both stopped before SSH/transfer,0GPU.
Do not retry those transfers. Prepare same-machine AIME using existing mpk
inputs/config/pins. New immutablev29_aime_confirmation002 spec changes only
name/protocol_id and all8block hosts from dlm2 to mpk; original001 retained.
Seeds/questions/method/arm order/budget/scoring are unchanged. CPU bind and GPU
qualification remain pending. Queue behind current mpk worker, never overlap.

## Expanded qualification and provenance freeze (2026-10-02 23:14, UTC-5)

HumanEval four-arm qualification is running on mpk from frozen54163f4e7,
following33CPU checks, real correct/wrong scorer and bwrap toys, private-cache
and single-idle-GPU checks. LongBench qualification is queued on dllm behind
independent diagnostic003; no formal panel has launched. The four diagnostic
workers closed successfully; complete anonymous event aggregation is pending.
Existing generation sources and bindings remain unchanged.

Explicit external provenance rebinding is CPU-qualified:42tests run,41passed
and1Windows symlink-permission skip. Default behavior is unchanged. Optional
complete mappings require unchanged original bytes in a new own deployment's
.external_sources; method/runtime fields stay fixed. This prepares AIME CPU
deployment without moving gold or executing inherited historical binaries.
Actual v21 validation and all42Linux checks are required before GPU qualification.
See docs/V29_EXTERNAL_PROVENANCE_REBIND_20261002.md. No new speed/quality result.

## GPU correctness passed; event diagnostic active (2026-10-02 22:51, UTC-5)

Fused observation44GPU tests pass on H100, source54163f4e7:40BF16 FP32-oracle
cases plus4original STORE/LOAD tests. Includes actual scale1.0, full V output,
GQA, splits1/2 and boundary tails. Fixed tolerances unchanged; synthetic numeric
qualification, not task accuracy, bit-exactness or speed. Reserved32.737210GPU
seconds. First missing-pytest attempt stopped on CPU (0GPU); second used isolated
runner pytest directory, leaving the qualified vLLM environment unchanged.
Source `results/v29_20261002/fused_observe_bf16_001/`.

Corrected four-PIECEWISE-arm event diagnostic003 is active on dllm, frozen
54163f4e7,16CPU checks passed, idle gate passed. It includes native dense with
no hooks and GLOBAL/LOCAL plus fused observation/DP leaf event spans. Profiler
overhead remains excluded from formal evidence; no breakdown result yet.

LongBench59x8x4 source/binding frozen on dllm, no GPU qualification launched.
First CPU preparation missed an archived provenance spec and stopped (0GPU);
new002directory includes the committed E14 source, all parent input/source
hashes pass, strict rebind passes and full1888timed/1888warm inventory verifies.
Generation-host CPU checks and scorer qualification remain pending. AIME and
HumanEval CPU deployment preparation proceeds on their assigned hosts; no formal
panel launched. No existing run or frozen binding was edited.

## New diagnostic and expanded-panel freeze (2026-10-02 22:45, UTC-5)

Expanded runner/scorer CPU validation passes33tests, including canonical model
snapshot paths, complete scorer-only artifact mirrors, original byte pins,
implicit Q128 defaults, native task contracts and same-host qualification proof.
Committed specs select engine seeds29001-29008, all59LongBench/all30AIME/all164
HumanEval questions and all4arms (8096timed plus8096warm total); suites assigned
dllm/dlm2/mpk respectively. No formal or qualification GPU worker is launched
yet; full qualification and real scorer toys remain required. Main stays Q128,
legacy canvas, torch copy/merge, no2K gate. New code never silently changes it.

Corrected independent profiler has16CPU checks: actual model.layers naming,
copy API wait caveat, optional same-stream event spans and fused-observe/DP-build/
DP-route leaf scopes. New four-engine PIECEWISE diagnostic (dense-nohook/native/
allkept/main) is prepared, not yet launched. Default FULL result remains unknown
for layer breakdown. Timings are diagnostic only, never formal panel evidence.

BF16 observation correctness protocol now covers actual scale1.0 and the prior
scale512**-.5:40BF16 cases plus4original=44. First CPU preflight found no pytest
and stopped before GPU work (0s). A separate own runner dependency directory
now supplies pytest; qualified vLLM environment unchanged. New attempt002 GPU
correctness awaits this source freeze. No tolerance was changed.

V28 complete preview was repaired/scored/pushed on its own branch91db65396.
See `docs/V29_V28_PREVIEW_INTERPRETATION_20261002.md`: Q64 request ratio.94245
against matchedQ128, but S/N1.00516 and seed/length signs vary; six questions
only. Canvas release W1.01228[.99035,1.03470], no supported gain. Main/allkept
W.99576[.91015,1.08942], no demonstrated increment. Keep matched native whose
preview accuracy is43/48 versus main40/48 and Q6441/48. No noninferiority claim.

## Latest audit and diagnostic status (2026-10-02 22:35, UTC-5)

First independent32K diagnostic closed four arms successfully, reserved1081.689752
GPU seconds. Source39e08c521; result `results/v29_20261002/cost32k_002/`.
GLOBAL/LOCAL name classification and partial GPU associations prevent a valid
GPU cost breakdown. Preserve the original diagnostic; fix/test the profiler and
add separately qualified event scopes before repeating. No profile speed claim.

Source audit confirms main fused observation shares one QK pass between complete
V output and rank32 summaries; observation itself is custom Triton, while the
sparse output consumer and native dense use FA4 through different dispatches.
See `docs/V29_BASELINE_CODE_AUDIT_20261002.md`. Main has no2K length gate; short
prompt panels cannot inherit the HF AIME gated-method conclusion. New BF16-mode
oracle protocol adds20cases to4existing CUDA tests, source core unchanged;
GPU correctness execution pending, see `docs/V29_FUSED_OBSERVATION_AUDIT_20261002.md`.

Expanded panel infrastructure remains under CPU review and is not launched.
Separately, V28 generation closed24workers; its CPU scoring stopped on a
snapshot-symlink comparison bug, fixed on that branch7c8c608ed with18CPU tests.
Original generation/bindings are unchanged; repaired CPU rescoring is pending.

## Running diagnostic (2026-10-02 22:14 (UTC-5))

Four-arm32K independent profile active on dllm, frozen39e08c521.
Expanded formal dataset panels still preparing, none launched. dlm2 own vLLM
environment CPU-qualified and patched; GPU qualification pending.

## Latest V29 stage (2026-10-02 22:05 (UTC-5))

Fused regroup writeback is approximately tied (.99818 vs equally fused natural);
standard merge fusion reduces component span7.4-9.7%, real-model qualification
pending. Copy component exact; both standard optimizations opt-in with torch
default. Expanded dataset preparation continues. No V29 formal panel launched.

# V29 active handoff (2026-10-02 21:48, UTC-5)

Branch `research/vllm-confirmation-20261002`, from V28 `e5e1ffcc8`.
Preparing expanded four-arm LongBench59/AIME30/HumanEval164 panels with8
engine seeds and1 timed repeat; qualification/scoring must pass before launch.
No new panel is running yet. Existing V28 preview remains on its original branch.
32K overhead attribution is pending independent profiling. Opt-in fused paged
KV copy is implemented with42 CPU tests across copy and adapter lifecycle guards;
GPU component001 passed10 exact cases; long tail event-span ratio about.556;
real-model and W/accuracy qualification pending. Reserved3.867908GPU seconds. Regroup output-scatter/LSE fusion
CPU prototype and11 tests pass, against identically fused natural Q64.
Independent32K profiler and10 CPU tests also ready; no GPU diagnostic yet. No new speed claim.
Read `docs/CONFIRMATION_CAMPAIGN_V29_20261002.md` for predeclared component protocol.

## Inherited V28 snapshot (historical)

# V28 active handoff (2026-10-02 21:02 (UTC-5))

- Branch `research/vllm-variants-20261002`, parent base `9b027df8f`; generation deploy `dcfdb8730` frozen.
- Qualification004 closed all five request workers successfully: Q128/Q64 legacy canvas, Q128/Q64 release, allkept release. Timed monitored JIT events and allocator retries zero; no performance/accuracy inference from one item.
- Four engine seeds x two sequential repeats x six questions are frozen for six variants (48 timed/variant). Family spec committed before launch. seed4_preview001 now running on mpk:24engines,288timed+144warm. Repeats are not independent seeds.
- Component evidence: Q64/Q128 0.94575; held-regroup/natural Q64 total 1.02753 (negative). Full144 alias4/2 mean0.98836 with32K slower; retain alias2.
- Parent V18b completed and pushed at `e1d2f89c29b6a713ffd3419cf7149d1a34abbd50`:568 timed requests,59 questions,2 engine seeds. Main/default-dense W ratios .7814/.7439/.8070, but matched native W 1.1377/1.0065/1.0446; no general incremental or noninferiority claim. See `docs/V18B_INTERPRETATION_V28_20261002.md`. Parent result files stay on their own branch.
- LLaDA official dense environment smoke passed. I-DLM diagnostic003 naturally stops in all3toy calls at868/549/1200 tokens; general task quality remains unqualified. Neither model has our sparse variants implemented. See new-model adaptation audit.
- New-model attempts total916.2097GPU seconds; dlm2 idle after diagnostic003. mpk qualification004 total867.9727GPU seconds (includes alias sweep and startup/teardown).
- Next: close six-arm seed4 preview and run strict scoring, then only supported refinements. Completion pipeline CPU-qualified; arm against this documentation commit after push, then verify private waiting status. It only finishes the existing campaign and publishes validated aggregates; no new GPU job. Question coverage must expand before request-level or noninferiority claims.
- Update HANDOFF/docs/STATE.current with every result or status change and push. Private prompts, generated text, gold, token arrays and private paths stay outside Git.

## Inherited parent handoff (historical; current V28 overlay above takes precedence)

# HANDOFF — current frontier (2026-10-02, local UTC−5)

Read `AGENTS.md` first. Stable context: `docs/RESEARCH_CONTEXT.md`. History and negatives: `docs/DECISIONS.md`.
Verified numbers: `docs/RESULTS_LEDGER.md`. The previous handoff (v27c, 2026-09-29) is archived at
`docs/handoff_archive/HANDOFF_v27c_20260929.md`.

- **Branch:** The current working branch is `research/humaneval-v27-20261001`. A GPT session created it from the v27
  checkpoint `research/m3-output-numerics-20260927` (`add23afa9`) on 2026-10-01 23:59; all work since then
  (HumanEval, step statistics, regrouping, docs) is on it. The old branch stays at `add23afa9` as the v27
  checkpoint; fast-forwarding it is the user's call.
- **Method code:** `9d8ae5e` (named keep k5) on top of `efea33024` (k12), `c36b1f933` (V-term ablation variants)
  and `56de98fe2` (`carry_first`). Docs and decks are newer commits; trust `git log` over this line.
- **Docs:** committed and pushed. Keep them current with every change; this is a user instruction.
- **Intake review (2026-10-01, 22:45 UTC−5):** audited base `6b13fe178`, confirmed as the live remote head.
  See `docs/INTAKE_AUDIT_20261001.md`. Follow-up panels may select the stronger relevant arms instead of repeating
  plain M1/M2/M3 each time (user update). Same-host paired cells and matched optimized references remain mandatory.
  Classmates' A/B branches are pending; their exact refs are unknown, so no merge has been attempted.
- **Group-context review (2026-10-01, 23:10 UTC−5):** user-authorized read-only search of the specified four-person
  research group completed on base `073d87ece`. See `docs/GROUP_CONTEXT_20261001.md`; peer claims remain separate
  from frozen v27 evidence. No HumanEval result or exact A/B refs was found in that search. GPU seconds = 0.
- **Two local checkouts.**
  - `E:/dlm/m3_output_numerics_20260927` is the working checkout used for all v27 work.
  - `E:/dlm/dlm_state_adaptive_router_20260828` is the same branch at an older commit (`3d48ebcdd`); pull before
    using it.

## V18 live status

2026-10-02 17:54 (UTC-5): All four longest-96K V18b qualification arms pass execution/config/count/graph guards and unchanged NeMo scoring. All four outputs parse and all four are incorrect on this one item; this is scorer qualification only, not accuracy evidence. Qualification 005 reserved 640.802 s; total closed V18 work 2154.955 GPU seconds. Formal V18b panel_001 launched from the identical 8704cd072 deploy/binding: 24/24/11 items at 32K/64K/96K, four repeats, dense/main plus matched controls, 568 timed requests. Expected 2-4 hours; mpk CPU scorer ready, both other GPUs idle.

## Variant/fairness audit (2026-10-02 18:19 (UTC-5))

See `docs/VARIANT_FAIRNESS_AUDIT_20261002.md` for the superseding interpretation of historical claims.
No new blocker was found for frozen V18b. Added constructor rejection for unsupported C/density gates
for future deployments only. Historical HF accuracy is descriptive, not proven noninferiority;
cell-level repeated-seed p values are exploratory. HF W excluded method setup/cleanup, and its
`new_graphs` tracked Dynamo compilation rather than CUDA capture. Mass-only removes V from prefix
ranking only. R17 92K main discordance is +0/-0 (91/91 unchanged). CHW/PPT/Slack reviewed read-only.

## New peer PDF evidence (2026-10-02 18:33 (UTC-5))

User-supplied 10-page source fills the missing C-gate formulas. HF state updates match
Eq. (1)-(7); three existing CPU tests pass. P17 is still not a full replication: R6 holds
old maps and carry_first does not reapply first-call s=4 protection. The paper reports
stronger V-direction evidence at severe RULER sparsity, but substantial dense-relative
quality loss and no end-to-end speed measurement. Keep our V negative scoped to our
settings. Table 1 versus Table 11 seed-42 dense differs (17/30 versus 14/30), requiring
protocol provenance before pooling. See `docs/PEER_PDF_UPDATE_20261002.md`.
The remote V18b campaign is unchanged. Its local waiting finalizer is being re-armed
against this reviewed documentation commit; no GPU worker was stopped.

## Completion pipeline (2026-10-02 18:25 (UTC-5))

The private coordinator now has a single-campaign finalizer for `v27_vllm_v18_panel_001`.
It waits for all eight workers, then runs strict scoring and publishes only aggregate results.
It checks that local/remote branch heads have not moved since it was armed; a changed head,
failed worker, missing pair or nonzero graph counter stops publication for review. It does
not launch another GPU experiment or infer a positive research conclusion. The configured
helper is `vllm_v18_finalize.py` in the private coordination directory; consult its private
status before manually scoring or publishing the same campaign. It will be armed against
this documentation commit after push. Frozen inference remains `8704cd072`.

## Intake and panel preparation (2026-10-02, US Central UTC-5)

- Reviewed remote branch explicitly with `git fetch origin research/humaneval-v27-20261001`:
  remote and local are both `b9e0700d8`. Default fetch only tracks an older branch; use the explicit ref.
- All three H100s verified idle by SSH during intake (0 MiB, no compute processes).
- V18 spec and worker/phase tracker/scorer passed 112 CPU tests and were pushed in `c89472cfb`.
  First 96K qualification failed closed after dense warm-up: external/internal request ID mismatch.
  No qualified record; 241.607 GPU reserved seconds. Fixed with internal ID returned by add_request;
  Attempt 002 exposed an async-counting boundary: scheduler retired 349 denoising forwards,
  but native actually executed 350 (1750 GLOBAL calls, no order errors). The new execution hook
  reads existing vLLM CPU snapshots after boundary sync; no new GPU operation or scheduling change.
  151 CPU tests pass. All GPUs idle; fresh 96K requalification is next. Closed GPU seconds: 755.639.
  See `docs/VLLM_PANEL_V18_20261002.md`. Status changes are recorded here and in STATE.current.
- Fixed stale fastest-dense and blocked-port labels. Historical panel numbers are preserved.

## Situation at handoff (2026-10-02 16:40 UTC−5)

**Running (2026-10-02 16:20 UTC−5):** nothing; all GPUs idle.
- **R17 finished** (`ruler_long_panel_r17/receipts.md`, L1p).
  - Main keeps RULER accuracy at 32K / 64K / 92K (110 / 98 / 91 vs dense 110 / 97 / 91).
  - Fixed 88% / 95% lose only at 32K (`cwe`).
  - Prefix-ranking V ablations are closed as a negative (95% / 32K: mass-only 107 vs rank32 101, p 0.031).
    This does not remove V from tail selection or observation costs; see the audit above.
- **The method now runs inside vLLM** (`experiments/numerical_qk_reuse/vllm_adapter.py`; notes in
  `docs/VLLM_PORT_NOTES_20261002.md`). It runs the unchanged core with the frozen main config; receipts match the panels.
  - Smoke results against vLLM's own default dense serving, per step: **0.96× at 32K, 0.85× at 64K**.
  - The adapter's all-kept arm equals dense.
- **Dense-baseline correction (important).** vLLM's dense GLOBAL call uses FA4's dynamic-causal path with an
  effective 2-way split-KV and is about 1.65× faster than the FA4 configuration our HF panels used as `D_fa4_allkept`.
  - The HF-substrate speed ratios (E4–E15, the result block below) therefore overstate the gain against the
    strongest official dense.
  - Their accuracy results stand. Speed claims for the paper must come from vLLM.
- P16 / P17 finished earlier: the 64K accuracy gap is selection noise; the C gate is not adopted.

**Do the vLLM results agree with our own (HF-substrate) panels?** The direction agrees; the size of the speed gain
is smaller; accuracy, end-to-end time and 96K are not yet measured in vLLM.

| | HF substrate, E13–E15 (18 seeds) | vLLM 0.30.0, smoke (1 item × 2 runs per arm) |
|---|---|---|
| dense reference | FA4, num_splits=1 (`D_fa4_allkept`) | vLLM's own serving (FA4 dynamic-causal, split-KV; ~1.65× faster GLOBAL call) |
| per step, 32K | 0.921 | 0.96 |
| per step, 64K | 0.814 | 0.85 |
| per step, 96K | 0.745 | not run (memory check pending) |
| end-to-end W | 0.950 / 0.867 / 0.818 | not measured; 64K estimate ≈ 0.90 if step counts were equal |
| accuracy | 267/274, 228/238, 93/76; noninferiority unproven | not measured yet (completions saved privately) |

- The same mechanism shows in both: per-step saving grows with context length.
- In vLLM each skipped GLOBAL tile saves less, because vLLM's dense call is faster. The rest of vLLM's step is also
  faster, so GLOBAL attention is a bigger share of a step. The two effects partly offset; the net ratio is a few
  points weaker than on the HF substrate.
- Do not quote the HF speed ratios as the paper's speed result; their dense was not the strongest. The HF accuracy
  results stand.
- The vLLM panel (next step 1) must confirm the vLLM numbers with many items, accuracy and 96K.

**HF-substrate result (18 seeds, E13 + E14 + E15; L1m; dense = FA4 num_splits=1, see the correction above).** M3 R6 DP −ln2 + `carry_first` vs that dense:
- HF generation-call W (prefill included, method setup/cleanup excluded) **0.950 [0.920, 0.980] (32K), 0.867 [0.841, 0.892] (64K), 0.818 [0.731, 0.898] (96K)**;
- generation-only S 0.937 / 0.787 / 0.728, per step 0.921 / 0.814 / 0.745;
- accuracy: 267/274, 228/238, 93/76. Old cell-level p=0.012 at 96K is exploratory;
  11-question exact sign-flip p=0.125. Question-cluster CIs still allow declines at 32K/64K.
  Neither noninferiority nor a robust 96K advantage is established.
- AIME (E15): W 0.951, from fewer steps; accuracy 100 vs 94.

**The most important open issue: the dense baseline is not the fastest official serving system.**
- vLLM 0.30.0's native DiffusionGemma (official, FA4 attention) was run on the same E14 cells and host
  (`vllm_dense_check_1002/README.md`). It is faster than our dense control (`D_fa4_allkept` on the HF-based
  piecewise_v5 substrate):
  - per step 0.76× at 32K, 0.90× at 64K, 0.98× at 96K;
  - prefill 0.44–0.57×.
- The attention kernel is the same, so the gap is MoE kernels, CUDA graphs, sampler and prefill.
- All speed ratios so far hold against our substrate, not against vLLM. Paper-grade claims need the method inside
  vLLM, measured against vLLM dense.

**Historical port blocker (resolved; preserved for context).** `vllm_port_probe_1002/README.md`:
- vLLM's FA4 runs dense attention over the paged cache correctly (page size 64).
- But block-sparse lists over a paged cache read the wrong pages: error about 22 vs the masked reference, while
  contiguous K/V is exact to 5e-4.
- Page size 16 does not run at all.
- Options, in order:
  1. pass a contiguous view when the request's pages are contiguous;
  2. fix the paged block-sparse path in FA4 (translate sparse n-blocks through the page table) and report upstream;
  3. FlashInfer BSR over paged KV (hd 512 speed unverified);
  4. gather kept tiles.

**Expansion setup is ready** (`docs/EXPANSION_PLAN_20261002.md`). On dllm under `/home/exouser/dyh/dlm_models_20261002`:
- SGLang 0.5.21 env (LLaDA2.1-mini official path), the I-DLM bundled SGLang env and the vLLM 0.30.0 env;
- LLaDA2.1-mini and I-DLM-8B weights;
- LongBench Pro data.
- Every item has its HF revision and pip freeze recorded.
- UltraLLaDA is dropped (user rule: old or impractical on one H100).

## Current conclusion

- **Best configuration now: M3 R6 DP −ln2 + `carry_first`.** The pooled 12-seed numbers are in the situation section
  above. The E5 numbers on its 288 cells: 64K W 0.853 [0.802, 0.895], S 0.775; 32K W 0.907 [0.858, 0.952], S 0.878.
- **All speed ratios are against our HF-based dense substrate.** vLLM's official serving is faster; see above.
  - Accuracy 76 vs 75 and 92 vs 86. Versus M3 without carry: 0.971 / 0.963.
  - The `stable1` gate is rejected.
  - 96K (E6 + E6b pooled, 11 fitting items × 6 seeds = 66 cells): W **0.822 [0.703, 0.944]**, S 0.727, per-step
    −25%, accuracy 30 vs 25 (`docs/RESULTS_LEDGER.md` L1c2).
- **Long context: significant end-to-end gain, no accuracy loss.** E4 has 6 never-used seeds, 144 cells per arm per
  bin, on piecewise_v5, against FA4 all-kept dense.
  - **64K:** M3 R6 DP −ln2 request W **0.879 [0.820, 0.926]**; generation-only S 0.807.
  - **32K:** W **0.940 [0.897, 0.980]**; S 0.921.
  - Accuracy is equal or higher (64K 77 vs 75, 32K 86 vs 86). B is similar (0.876 / 0.941).
  - Prefill is not optimized by any arm (P ≈ 1.00).
- **Step count: no significant change, best estimate +1–2% (corrected 2026-10-01 evening).** Per-seed step ratios
  (M3 / dense, 32K, same 24 items and substrate) range 0.85–1.24. The old 3-seed set (101/202/303) sits high and the
  6 new seeds sit low; 13 of 84 random 3/6 splits give a gap at least as large, so both are draws from one
  distribution. Pooled 9 seeds: 32K N 1.024 [0.973, 1.076], 64K 1.015 [0.948, 1.085] (64K pools v4 + v5). E4's
  request W may therefore be about 2% optimistic; per-step costs (S/N) are stable across all batches.
- **Held-out items only** (never used to choose −ln2): 64K M3 0.85 [0.73, 0.96] significant; 32K M3 0.96 n.s.,
  obs2 0.95 [0.91, 0.98].
- **Fan plain M1/M2c/M3 are slower than dense** at every length (1.06–1.34× in E4): the selector and observation cost
  exceeds the skipped attention.
- **AIME has no speed room** (E7, 180 cells per arm, `docs/RESULTS_LEDGER.md` L1d). The best low-overhead variant,
  M3 + `carry_first` + 2K gate, is W 1.005 [0.967, 1.042] with accuracy 99 vs 99. Fan plain M1/M2c/M3 are 7–12%
  slower. No accuracy difference is significant. AIME is an accuracy check only.
- **Novelty is weak.** B ≈ SparseD (no significant speed difference at 64K). **E8 (AIME, fixed 70% sparsity,
  180 cells per arm) finds no evidence that looking at V helps:** no V term beats attention mass alone (all p ≥ 0.30
  vs rank 32), and projected V at rank 32/16/8 scores lowest (86–87 vs dense 99). **E9 (64K, M3 + c0, fixed 88%
  sparsity) and E11 (95%, preview) agree:** mass-only ranking is as accurate (54 vs 51) and as fast (W 0.845) as any V term. The V term is
  not a contribution in these settings and can be dropped (L1e, L1f).
  This conclusion is limited to the tested AIME/LongBench settings. The user relayed a classmate's hypothesis that
  RULER needs V dimensional information; HumanEval is unknown. Actual attention already uses full V. The question
  concerns selector inputs and should be tested on RULER, not generalized from the LongBench negatives.
- **M2 vs M3 under the same optimizations: tie** (E10, L1g): M2c / M3 32K 0.987 [0.914, 1.063], 64K 0.983
  [0.940, 1.024]; accuracy n.s.
- **Skip ratio is not set; it emerges from one fixed threshold.** All GLOBAL layers, heads and lengths share the log
  threshold −3.874 (frozen base −3.180, shifted −ln2). Skipped tiles in sparse calls (diagnostic fidelity_v6, 2
  requests each): AIME about 4%, 32K about 79%, 64K about 88%. Fixed ratios are used only in E8/E9/E11.
- **Within a canvas** (code facts, `docs/RESEARCH_CONTEXT.md` §3): call 1 observes and builds the prefix risk table;
  call 2 decides (map for calls 2–7), call 8 re-decides, and so on. A re-decision reuses the prefix risk table but
  applies the current per-position query sensitivity T = clamp(1 + 3·EMA(argmax flips), 1, 4), which is active in every
  run (corrected 2026-10-02; an earlier note wrongly said T = 1). Measured: re-decisions change 44–71% of the first
  decision's kept tiles (Jaccard 0.59–0.70), so the earlier "≈ select once per canvas" inference was wrong.
- **Per-step time on v5** (one real dense call each, `docs/RESULTS_LEDGER.md` time breakdown): GLOBAL attention 2% /
  19% / 32% of a step at AIME / 32K / 64K; LOCAL 2–3%; MoE experts 27–39%; sampler 10–16%. Single decoder forward
  in one request, dense → best variant: AIME 1.00, 32K 0.85, 64K 0.72.
- **The ceiling at batch 1 is low.** GLOBAL attention is about 21% of a 64K request and 16% at 32K; a step is
  dominated by reading about 46 GB of MoE weights. With `carry_first` the 64K gain (about 15%) is roughly 70% of that
  ceiling.
- **Model.** `google/diffusiongemma-26B-A4B-it` belongs to the Gemma 4 family (Gemma4Processor, gemma4_vision in its
  config); its text model is `diffusion_gemma_text`. Say "DiffusionGemma-26B-A4B", not "DiffusionGemma4".
- **Batching does not raise the attention share at 64K** (B=1→4: 26/22/25% of a forward; keep-0.12 saving about
  20%). See `batch_scaling/README.md`.

## Done (this session, 2026-09-30 to 10-01)

- Substrates piecewise_v4 (static LOCAL shape; fixes the AIME recompile fallback) and v5 (compiled encoder-append
  tail).
- Panels, all scored and in `docs/RESULTS_LEDGER.md`:
  - traj_t1, E1 (cross-canvas carry: rejected), E2 (AIME carry/gates), E3 (32K single-change variants): L11–L14;
  - E4 large-seed confirmation (L1), E5 `carry_first` (L1b), E6 + E6b 96K pooled (L1c, L1c2), E7 AIME (L1d);
  - E8 / E9 / E11 V-term ablations (L1e, L1f, L1h), E10 M2 vs M3 under the same optimizations (L1g);
  - step-count check over 9 seeds (correction: +1–2%, n.s.) and the v5 per-step time breakdown.
- New named variants: `observe_step`, `protect_output` (7786b0ef9); `stable1` (adc2056d7); `carry_first` (56de98fe2);
  `proj_rank`, `risk_value='mass'` (c36b1f933); named keeps k12 (efea33024) and k5 (9d8ae5e), each with unit tests
  (v27 subset passes on dlm2; the stale min_route_keys=4096 test was fixed).
- Agent docs: `AGENTS.md`, `CLAUDE.md`, `docs/RESEARCH_CONTEXT.md`, `docs/DECISIONS.md`, `docs/RESULTS_LEDGER.md`,
  this file, and the `current` block of `STATE.json`.
- Group-meeting deck (2026-10-01): `scripts/v27_build_deck_1001_compact.py` reads the pushed `summary.csv` files and
  the time-breakdown JSONL and writes both `ppt_sample/dlm_sparse_attention_20261001_compact_v6.pptx` (latest) and
  the markdown source `weekly_slides_20261001_compact.md`. `--md-only` refreshes only the markdown (use it when the
  pptx is open in PowerPoint). The older 9-page deck and its source (`weekly_slides_20261001.md`) are superseded.
- Read-only review of group work: Junyu's position protections and value-aware family (branch `ljy/value_aware`);
  their "vector_mean" result files are not committed anywhere we can read.
- Literature check of step and length inflation (SparseD, PulseCol, Focus-dLLM, Lil, LessIsMore, JoT, Prophet).

## Finished on 2026-10-02 (details in `docs/RESULTS_LEDGER.md`, decisions in `docs/DECISIONS.md`)

- E12 HumanEval (L1i): no accuracy loss in any arm; no speed gain (short context).
- E13 q64 (L1j) and E14 (L1k), 12 seeds.
  - q64 (64-row FA4 maps) gives −0.4 to −0.7% per step, significant at 64K, with no end-to-end effect; it stays an
    optional named variant.
  - q64c (64-row carried call-0 map) adds nothing per step over q64 and shows more steps at 96K; not adopted.
- Regrouping is closed as a clean negative:
  - kernel bench (`q64_bench_1002/`): q64r and FA4 split-KV are slower;
  - offline on real need matrices (`regroup_offline_1002/`): no grouping beats natural 64-row tiles by more than
    about 5%. This covers within-block sort, chw's set key, cross-head (GQA) grouping and greedy clustering.
- Call-1 split (`observe_split_1002/`): c01 saves 0.45–1.7 ms per GLOBAL layer per canvas, about 0.5–1% per step at
  kernel level. Our Triton observation-only kernel is slower than FA4 full dense attention, so a faster observation
  kernel is the remaining lever there.
- P15 (`c01_preview_p15/`): the c01 accuracy-first preview passed. c01 / q64c per step is 0.91 on AIME and 0.949 at 64K
  (small n); 64K steps per canvas are 1.107. E15 checks both with many seeds.
- P16 (L1n): the 64K accuracy gap is mostly selection noise; output protection and −2ln2 are not adopted.
- P17 (L1o): the C gate (collaboration) is not adopted; it costs per-step speed and shows no accuracy gain.
- R16 RULER 32K/64K (L1l): accuracy preserved by every arm, including a fixed 88% sparsity; the V term equals
  mass-only (33 vs 33).
- vLLM dense check and paged block-sparse probe: see the situation section.

## Running

- dllm: v27_vllm_v18_panel_001 (formal V18b, 59 items x 4 repeats, 568 timed requests including controls)

## Blockers

- None. The paged block-sparse path is fixed and the method runs inside vLLM.

## Immediate next steps (in order)

1. **vLLM panel (the paper's speed evidence).** Runner: `scripts/v27_vllm_bench_host.sh` (the exact command is in its
   header). The panel plan is in the last section of `docs/VLLM_PORT_NOTES_20261002.md`.
   - LongBench-v2 32K / 64K / 96K, many items × repeats. vLLM cannot fix per-request seeds, so use items × repeats
     and report per-step and request distributions.
   - Arms: vLLM dense (default cudagraphs) vs method (PIECEWISE). All-kept on a subset as the adapter-cost control.
   - Accuracy: score the private completions with the same LongBench scorer as the panels. This needs a small
     scoring script.
   - 96K: memory 0.92; check that the method's score cache fits next to vLLM's full-length KV.
2. **Upstream FA4 reports (waiting for the user's go-ahead and GitHub account).** There are three SM90 issues:
   - paged block-sparse reads logical pages (fixed locally, `patches/`);
   - varlen block-sparse offsets use tile_n 128 (one-line fix, cause confirmed);
   - block-sparse and non-causal split-KV redo the whole range in every split.
3. **New models.** The FlashInfer JIT is fixed (CUDA 13.0 nvcc shim). Next:
   - run the LLaDA2.1-mini SGLang smoke on a GPU;
   - run the I-DLM-8B smoke;
   - time the baseline candidates;
   - write thin adapters (the vLLM adapter shows the pattern: stub hooks around the unchanged core).
4. **Optional HF-substrate rerun.** Make the HF dense control the dynamic-causal split path, and give the HF sparse
   consumer the alias split, if the HF panels are still to be quoted for speed.
5. **Short-prompt positioning and an AIME long-output preview** (unchanged; see DECISIONS).
6. **Datasets:**
   - LongBench Pro official evaluation code and subsets;
   - RULER 8K manifests for the new models.

## Intake audit (2026-10-01, 22:45 UTC−5)

- Three GPUs checked idle by SSH; this review launched no GPU worker, GPU seconds = 0.
- The independent FA4 timing join previously allowed later duplicate records to overwrite first outputs and did
  not recheck scorer/run identity. Added rejection of duplicates, omitted scored executions, mixed host/GPU,
  substrate/protocol/model/source, and unqualified scores. Unknown graph counters are excluded from Wc.
- Audited E4/E5/E6/E6b/E7/E8/E9/E10/E11: 8,826 first runs, same-host cells throughout, no duplicate first outputs,
  no point-estimate or correct-count changes. E7/E8 each retain 3 timed Dynamo new-graph events (use historical Wc);
  these counters do not independently establish CUDA capture counts.
- Final regression: 17 CPU tests pass on the registered dlm2 interpreter; all nine panels pass the final guards.
- Full LongBench-v2 length inventory: 503 inputs, median 107,706, max 5,174,028 rendered tokens. 224 inputs ≤95,074;
  400 fit the configured 262,144-token context with 8,192 output tokens reserved. Chunked prefill can address
  memory, not the 101 inputs whose prompt alone exceeds the context limit. No full-panel run has started.
- HumanEval: 164 Python tasks, 984 generations/arm with six seeds. Not yet a v27 dataset/scorer; proposed as a
  coding quality check. Read-only review of the shared `megakernel` deck was limited to retrievable slide text.
- Slack context: the user subsequently authorized the specified four-person group conversation. Read-only research
  search completed; no one-to-one DM was read. The peer RULER note supports V direction but not a full-rank necessity.
  Aggressive-sparsity step inflation is a collaboration clue, not a new v27 result. A/B refs remain pending.

## Operational notes (still relevant)

- **Cache hygiene (dyh-only):** for SGLang/vLLM runs set `SGLANG_CACHE_DIR`, `SGLANG_JIT_CACHE_DIR`, `TVM_FFI_CACHE_DIR`, `XDG_CACHE_HOME`, `TRITON_CACHE_DIR`,
  `TORCHINDUCTOR_CACHE_DIR`, `CUDA_CACHE_PATH`, `FLASHINFER_WORKSPACE_BASE`, `VLLM_CACHE_ROOT`, `HF_HOME` and `TMPDIR`
  under dyh. SGLang ignores `XDG_CACHE_HOME` and needs `SGLANG_CACHE_DIR`.
- **Hosts** (H100 80GB, user-authorized, write only under `dyh`):
  - dllm `149.165.159.64`;
  - mpk `149.165.151.254` (writes go to `/media/volume/dllm-1/dyh`);
  - dlm2 `149.165.168.28`.

  Interpreters and env are in `E:/dlm/v20_private/hosts.json` plus `E:/dlm/v27_lbfa4_env.json` (torch 2.12 FA4
  overlay; caches pinned inside `dyh`).
- **Panel pipeline.** Coordinator scripts in `E:/dlm`, not in the repo:
  1. Commit the spec in `results/…/specs/`.
  2. Freeze: `python -m scripts.v21_freeze_panel freeze --mode v27_panel --spec … --out-dir E:/dlm/v23_private/<name>_frozen`.
  3. Deploy: `python E:/dlm/v21_deploy_qualify.py --sha <HEAD> --tag v27_<x>_<sha7> --host <alias>`.
  4. Bind: `python E:/dlm/v23_transport.py --action bind --run-dir … --frozen-dir …`.
  5. Launch: `TAG=… RUN=… FROZEN=… STAGES=… bash E:/dlm/v27_lbfa4_host.sh <alias> <ip>` (waits for an idle GPU).
  6. Score: `python E:/dlm/v27_score_lb.py --tag … --run-dir … --label …`.
  7. Summarize: `python -m scripts.v27_fa4_panel_summary …`.

  Scored outputs land in `E:/dlm/v27_private/lb_scoring/<label>/`.
- A killed worker leaves no terminal receipt. Relaunch in a **new** run dir and merge at scoring.
- On Windows, stopping a task can orphan the bash chain. Kill leftover `bash` processes with `v27_` in the command line.
- Diagnostics that must match a frozen config's source hashes run with cwd = that deploy dir.
- Time breakdown of one real decoder call: `scripts/v27_time_breakdown.py --run-dir <bound run dir> --host <ip>
  --gpu-uuid <uuid> --stage <stage> --dataset <dataset> --index 0 --call <k> --arm <dense> --arm <method>`, run in
  that panel's deploy dir with the host env; outputs in `results/…/time_breakdown_v5/`.
- Unit tests on a host: in a deploy dir, `PYTHONPATH=<fa4 overlay>:.:src <python> -m pytest -q -p no:cacheprovider
  tests/test_v27_*.py` (CPU-only tests can run while a GPU job is active).

D2H metadata audit: existing CPU scheduler/sample-count fields cannot replace
the full exact phase/step/sequence-length tuple. CPU length is an upper bound;
next commit state and actual/retired execution identity differ. Async copy plus
a wait at prepare would merely relocate synchronization. No routing optimization
or shadow trace implemented; frozen preview unchanged. See
`docs/V28_CPU_METADATA_AUDIT_20261002.md`.

## Regroup permutation-backend follow-up specified — 2026-10-02 20:37 (UTC-5)

The existing held Torch result remains negative. A separately named Triton
permutation backend is worth testing because generic gather/scatter consumes
more than the consumer saving. Spec: `results/v28_20261002/specs/v28_regroup_triton.json`.
First test exact same-host Torch/Triton permutation; then full gather/FA4/scatter
on one host. No adding timings from different hosts. CPU and GPU equality are
prerequisites; no new performance result yet. Existing request preview untouched.

Triton permutation implementation prepared:10newCPU tests plus8existing held
tests pass. Pure mode100repeats/warm8, full held32/warm8; eachcall output
allocation in both backends. GPU equality/combined timing remain pending.

Triton permutation negative1.62649x Torch with exact GPU equality. Reserved40.2940GPU seconds; full held GPU follow-up gated out before launch. See V28 permutation report.

## V31 AIME26 GLOBAL budget sweep — 2026-10-06

The complete 30-problem × seeds 42/43/44 sweep is complete under
`results/v31_20261006_aime_global_budget_sweep/` at source commit
`1081bf41`. It contains the dense FULL PR #51994 reference and MAGE + LOCAL
arms with GLOBAL budgets 1024, 2048, and 4096 and LOCAL budget 512 using the
compact Triton consumer. All four arms have 90/90 records and official scores.

Pooled exact match is dense 47/90 (52.22%), then 52/90 (57.78%), 47/90
(52.22%), and 50/90 (55.56%) for GLOBAL 1024/2048/4096. Count-weighted
overall GLOBAL+LOCAL tile sparsity is 37.82%, 31.99%, and 22.30%; all-call
LOCAL sparsity is about 25.4% (32.7–32.9% on sparse-only LOCAL calls).
Mean end-to-end dense time is 5.9738 s. The sparse arms are 7.6265 s,
7.8193 s, and 7.3819 s (0.783x, 0.764x, and 0.809x dense speedup); decode-only
speedups are 0.783x, 0.763x, and 0.809x. These are negative end-to-end speed
results despite physical tile skipping.

The pinned native sampler is unchanged, thinking is enabled, and timing uses
the recorded wall time including prefill and decode. `receipt.json`,
`score.summary.json`, `summary.md`, and `final_complete.json` are the source
of truth. Public records are sanitized; completions and logs remain private.
The v31 unit suite passed 73 tests with 25 CUDA-dependent skips.

## V31 AIME26 GLOBAL budget sweep with dense LOCAL — 2026-10-06

The follow-up dense-LOCAL panel is complete at source commit `5fb28bda` under
`results/v31_20261006_aime_global_dense_local_budget_sweep/`. It reruns the same
30 problems at seeds 42/43/44 with one dense FULL PR #51994 reference and MAGE
GLOBAL budgets 1024, 2048, and 4096. Thinking and native adaptive stopping are
unchanged. All 25 LOCAL layers stayed on native dense vLLM attention by leaving
`LOCAL_KV_BUDGET` unset; no LOCAL router or custom LOCAL kernel was installed.

Pooled exact match is 56.67% dense, then 56.67%, 58.89%, and 60.00% for GLOBAL
1024/2048/4096. Denoising forwards are 26,163, 26,081, 25,593, and 25,836;
forwards/canvas are 13.549, 13.368, 13.414, and 13.393. GLOBAL physical
sparsity is 58.05%, 41.33%, and 17.81%; LOCAL is exactly 0%. Geometry-derived
overall GLOBAL+LOCAL sparsity is 22.40%, 15.89%, and 6.86%.

Arithmetic mean end-to-end time is 5.7553 s dense versus 6.2557 s, 6.1797 s,
and 6.2762 s; dense/arm speedups are 0.920x, 0.931x, and 0.917x. Decode-only
speedups are 0.920x, 0.931x, and 0.917x. This confirms no end-to-end benefit
from GLOBAL sparsity when LOCAL layers remain dense. The official scorer,
paired comparison, receipts, and completion marker are in the result directory;
public records are sanitized and raw completions remain private.
