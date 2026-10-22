# Frozen optimized refresh native comparison, 2026-10-22

The user explicitly requests proceeding with the current best qualified kernels.
V2 refresh dense parity is not a prerequisite for this comparison and is not
claimed: at 8K the preserved synthetic result is 427.90 microseconds versus
363.58 plain dense FA4 and 287.47 native paged dense. Kernel tuning stops for
this campaign. The partial 2026-10-21 attempt004 stays paused and immutable.

Use SM90 library build `200354cb919fb6b8`, SHA256
`e6f162a2a02719ef964788b3f5612d9c455d3e4310d68a11aed466fa5d26948c`.
Native V2 uses the qualified fused current-V Gaussian32 projection/packing
wrapper and genuine online conditional full PV. QK/routing remains repeated
across four output-channel groups. V31 uses FA4 score observation and fused
top-k, contiguous below or at 4096 keys and direct paged above that boundary.
Both methods use resident direct-paged held/carry attention. The all-kept arm
uses the same resident consumer with full physical support. Yuhan's inherited
cross-step reuse lifecycle and Junyu's Gaussian32 selector remain distinct.

Compare optimized V2, optimized V31, corrected FULL dense FA4, PIECEWISE dense
FA4, and resident all-kept FA4. AIME26 covers all 30 questions at seeds 42/43/44;
MATH500 covers 500 at seed 42; HumanEval covers 164 at seed 42; LongBench-v2
covers 503 at seed 42. V31 uses K128 on all short benchmarks and K8192 on
LongBench. Keep native thinking-on, canvas 256, maximum 48 denoising calls,
entropy 0.1, confidence 0.005, stability 1, temperature 0.8 to 0.4, dense LOCAL,
prefill and commit semantics. No hidden forwards within a measured request.

Every measured input is preceded by a separate untimed native warm request at
the same seed. This is explicit warmup, not part of the measured trajectory;
its cost remains in worker resource accounting. Record actual CUDA captures and
universal request JIT events on every arm, including dense and all-kept. Reject
clean rows with any event or capture. Audit runs include tile/nonfinite and
event instrumentation; clean runs use request/prefill synchronized boundaries.
Do not combine audit event costs with clean S/W.

Before the full panel require operator parity/changing-input graph tests and
native audit/clean trajectory agreement for all arms and workload context
profiles. Require exact library, selector, budget and runtime receipts, zero
extra forwards, explicitly audited zero LOCAL skips, and zero clean captures
and JIT events. Fields absent from clean audit metadata are not evidence of
zero events; universal counters are mandatory. Model hashes, runtime hashes,
source hashes and input hashes are frozen before generation.

Calibrate V2 independently on four deterministic prompt-length strata per
workload using full native generation budgets. Aggregate held-phase GLOBAL
eligible and kept Q128/KV64 physical tiles and target 40-50% actual sparsity.
The coarse thresholds are 0.05, 0.1, 0.2, 0.4, 0.8, 0.025, 0.0125 and 0;
stop at the first in-range candidate. If needed, refine up to eight times in
the narrowest observed threshold bracket around 45%, using geometric midpoints.
No accuracy is used. Do not launch a panel from out-of-range calibration.
The mandated full coverage includes these transductive items; disclose them.
Report any final-panel drift or V2/V31 work mismatch explicitly.

Report official accuracy and paired/item-clustered uncertainty; N, N/C, T, S/N,
S and W; refresh and held audit latency; GLOBAL/LOCAL/overall physical work;
initial/refresh/held/carried, first-canvas and whole-request tile counts; exact
thresholds and actual sparsity. AIME uncertainty clusters its three seeds by
question. Speed claims require measured clean latency and matched controls.
Synthetic work or denoising call reduction is insufficient.

Preserve new failed preparations and qualification diagnostics under
`results/optimized_refresh_benchmarks_20261022/`. Attempt001 stopped during GPU
identity capture. Attempt002 preserves a validator error comparing an empty
JIT event list to integer zero. Attempt003 preserves 27 diagnostic rows and is
sealed partial after finding audit-only receipt assumptions and a short-canvas
calibration defect. No attempt003 or earlier result is a completed panel.
Every corrected protocol receives a new attempt. The source library stays
unchanged; this final protocol adds full-request calibration, universal timing
counters, per-input warmup and the matched resident all-kept control.

Attempt005 passed 32 native qualification passes and then failed in LongBench
V31 untimed selector warm-up with a device-memory error. It is preserved as
partial, with no workload calibration or full panel launched. Attempt006 keeps
the chosen attention kernels and uses bounded selector storage above 512 KV64
tiles: separate row normalizers and tile scores feed the same deterministic
fixed-budget top-k/list construction. Every launch and temporary allocation is
charged to refresh. Repair scans use at most 128 tiles at once. Warm-up prefix
sizes stop at the actual configured context limit and cache keys exclude the
runtime-only contiguous prefix stride. Short-context fused selection is retained.

The user's subsequent threshold update applies from attempt007: HumanEval uses
the saved K128 work-matched threshold `0.5600716206066294`; LongBench-v2 uses
`0.021`. Neither receives new calibration. Only AIME26/MATH500 calibrate to
40-50% actual held GLOBAL sparsity. Fixed workloads report observed work and
any mismatch without claiming the earlier held-target gate. Source-identical
completed qualification pairs from attempt006 may be referenced only after
verifying identical runtime sources, kernel, model, inputs, budgets and threshold.
Changed-threshold V2 pairs run anew. Full benchmark generations are fresh.

Attempt007 passed all native qualification gates and calibrated AIME26 at
threshold0.059460355750136064 (45.69% held GLOBAL sparsity). MATH500 reached
32.05% at the largest candidate0.8 and therefore stopped before full generation.
Attempt008 preserves and fingerprints that evidence, extends only MATH500's
coarse candidates to1.6/3.2/6.4/12.8, and retains the same eight refinements.
It reuses completed source-identical qualification pairs through explicit
original paths and record/config/receipt/completion hashes. Attention kernels,
sampler, inputs and all previously measured candidates remain unchanged.

Attempt008 establishes a MATH500 calibration ceiling: thresholds3.2/6.4/12.8
have identical output hashes and denoising calls. All70880 retained held tiles
out of115840 eligible equal the mandatory current-canvas/boundary plus first-tile
floor on those trajectories. Maximum measured sparsity is38.8122%, a target
miss. Attempt009 explicitly accepts this demonstrated structural ceiling at3.2
to complete the requested comparison, with a frozen per-canvas count proof.
It does not claim40% attainment or generalize that ceiling to all500 problems.
No further calibration forwards run; all full panel generations are fresh.
