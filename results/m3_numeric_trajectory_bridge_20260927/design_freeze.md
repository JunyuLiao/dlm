# v21 initial diagnostic design (before new GPU outcomes)

Authority: the user's complete v21 specification in this task, 2026-09-27.
Parent452b48d81ee9763a2404dda97894ec1aa3a6da05; v20 producer00a2c4d.
Old campaigns and first outputs are immutable. No v20 generation rerun.

## Budget and ownership

Wall start2026-09-27T22:22:00Z, GPUstop2026-09-28T05:52:00Z,
finalhandoff06:22:00Z. Aggregate GPU-stage receipts <=21600seconds,
generation executions <=520 including captures/parity/failures, max2GPUworkers.
All GPU work supervised with finite outer receipts and exclusive actual GPU
reservation. CPU scoring does not occupy GPU. No old heartbeat revival.
Astra owns decisions/semantics/qualification; Sol owns bounded CPU audit and
explicit kernel prototypes. Prototype presence does not mean promotion.

## CP0

Verify800 published generated records, distinguish400first from400warm.
Split totalcalls = canvases × calls/canvas; use disjoint stop/cap flag categories.
Describe output token common-prefix/chunk divergence without causal inference.
Preserve native/T/B and Dmatched outcomes, both geo and samehost pooled ratios.
Use original qualified scorers for per-cell labels; never infer gold from text.
Old/new G75 grouping/runtime/output contracts audited from executed sources.

## CP1 fixed diagnostic inputs and modes

Select the first two IDs in each v20 frozen protocol dataset list, seed101,
independent of scores or observed delta. First input per task targets canvas0
calls0,1,3; second AIME/LB input targets canvas1 calls0,3,6; second RULER input
targets canvas0 calls0,1,3. At most18 requested actual states, missing native
states remain missing, with no forced denoising call. Record actual prefixes,
cache/mask/positions, native RNG, stopping history and self-conditioning.
Replay establishes M3 history from actual earlier calls in that canvas.

At identical Q/K/V/support compare native dispatch, legacy allkept consumer,
independent FP32 reference, and fp32_scores_bf16_pv prototype. The prototype
removes only the two explicit BF16 QK/scaling roundtrips in final current output;
P->PV remains BF16 and reduction/geometry unchanged. Compare old/new on the SAME
M3 support separately. Trace actual native CUDA backend once per participating
host. Report reference-norm relative, max/tail/nonfinite errors, logits/top1 and
same-history stop/acceptance diagnostics separately from direct timed runs.

Historical observe_scores/cache/selector convention stays EXACTLY unchanged.
If current-output precision is used at A, perform old-cache route_only plus
actual current-QK output recomputation. The extra anchor QK and memory are charged.
No global observe_scores cast removal or synthetic FP32 recovery from BF16.

After operator qualification, bounded natural no-pruning diagnostic:
first two frozen LB IDs ×seeds101/202 ×native/legacy-matched/new-matched=12
executions, all3arms within a GPU, deterministic balanced order; answers not used
to select a precision by correctness. These are separate diagnostic requests.

Promote only ONE named precision if actual-state error/tail or stop-boundary
disturbance improves, guards stay valid, and full-forward cost leaves opportunity.
No universal bitwise-native criterion. Diagnosis target90min from initial design;
negative/inconclusive result ends precision redesign rather than a new sweep.

## CP2 layout-only exact variant

Keep old arithmetic and support; generic Triton writes final output directly
into B,Q,H,D contiguous backing, returned as B,H,Q,D view for the existing adapter.
Apply same explicit output layout to materialized-score PV anchors and preQK D/H.
Old head-major remains selectable. Verify exact elements, strides, GQA, partial
tiles, lifetimes and invalid guards. Actual copy/allocation trace is separate.
Do not change V materialization yet; measure first, no speculative copy saving.
Measure legacy, layout-only, numeric-only and combined complete-forward AND
denoising-step boundaries with native brackets. Model_forward excludes state.begin
as in v20; whole step includes native control. No sum of kernel savings claim.

## Conditional successor

A8/R3/P0/GLOBAL5 remain fixed; native stopping and8192 AIME/LB unchanged.
Qualified precision -> frozen448sevenarm panel per v21, otherwise96 old/newM3
layout comparisons plus tiny RULER parity. Freeze full schedule/identities and
re-estimate aggregate budget BEFORE generation; never launch just to fill quota.
No R/A/tau/earlyphase/geometry/Pprecision sweep, M2 deferred.
