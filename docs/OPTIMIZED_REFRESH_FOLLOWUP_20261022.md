# Optimized refresh follow-up after the current MATH500 panel

The user changes the experiment order: finish the current MATH500 panel, score
it, then stop before LongBench-v2 generation. Preserve achieved HumanEval and
AIME26 results. The old four-workload campaign remains partial. An independent
boundary controller writes the explicit pause marker only after the final MATH
clean worker has passed launch gating; that worker and scoring finish normally.
The unchanged coordinator rejects the next LongBench job before launch.

Use the existing selected V2 online refresh, optimized V31 score observer,
bounded long-context selector and resident held consumer. No kernel tuning or
native sampler change is part of this follow-up. Every new protocol receives a
new frozen attempt. Exact worker/runtime/model/input identity is required before
reusing any old control records; all original paths and hashes are explicit.

First, recalibrate MATH500 V2 downward. Current full-panel records show about
77% GLOBAL sparsity at threshold3.2; the four short-prompt calibration requests
were unrepresentative of long reasoning. Select32 deterministic equal-count
strata of observed full-request physical work from the examined500 cohort.
Use full native budgets, no scoring labels, thresholds0.05/0.1/0.2/0.4/0.8/
0.025/0.0125/0.00625/0 followed by at most10 bracket refinements. Require both
held and whole-request GLOBAL sparsity40-50% from actual eligible/kept tiles.
This calibration is transductive. Run new V2 audit/clean generations over all500
items; retain source-matched V31K128/dense FULL/PIECEWISE/all-kept controls and
the high-sparsity V2 result as supplementary history. Report full-panel drift.

Second, run AIME26 V31K256 on all30 questions and seeds42/43/44. Preserve the
achieved V31K128/V2/dense/all-kept generations. Exact source-matched controls
may be referenced. Add paired question-clustered K256-minus-K128 accuracy and
request/generation/call ratios, alongside the five-arm comparison at K256.

Third, add the existing balanced RULER128K104 subset: eight problems per13
tasks, subset seed6464, panel seed42. Keep its original thinking-off official
answer-prefix prefill and task-specific output budgets. Use V31K8192 and
calibrate V2 on the first sampled problem per task to approximately match
first-canvas absolute GLOBAL kept work (within5%) and kept fraction (within2pp).
The initial threshold candidates are0.021/0.042/0.084/0.168/0.336/0.672/1.344/
2.688 with10 bracket refinements. Report >85% sparsity only if actually measured
on the stated phase/denominator; dense bootstrap may reduce whole-request
sparsity. Use fresh optimized V2/V31, corrected FULL dense, PIECEWISE and all-kept
audit/clean generations. Official RULER scoring retains fractional credit and
uses13-task-clustered paired intervals, not binary accuracy alone.

Every new arm/profile and selected threshold requires native audit/clean
trajectory agreement, finite outputs, exact kernel/library/budget receipts,
zero extra forwards, zero LOCAL skips and zero timed captures/JIT. Preserve
native256-token canvas,48-call maximum and adaptive stopping/temperature.
Warm compilation before timing; clean and audit are separate. Report N,N/C,T,
S/N,S,W, refresh/held latency, physical GLOBAL/LOCAL/overall tiles, phase and
first-canvas work, thresholds, actual sparsity and source identity. Official
scores and completion markers are required before completed-result claims.
Monitor process status every15minutes as requested. Raw inputs/generations stay
private and all previous attempts remain intact.

RULER calibration follow-up: K8192 gives93.5018% held GLOBAL sparsity; V2 at
0.021 gives92.6024% on the13-task calibration cohort. The first-canvas absolute
work objective fails because V31 refresh is dense and trajectories differ;
those negative measurements remain preserved. The final RULER comparison uses
held kept fraction within2pp and both held sparsities above85%, selects0.021,
and explicitly reports absolute held/first-canvas/whole-request work mismatch.
This is approximate sparse-execution matching, not an equal-total-FLOP claim.
