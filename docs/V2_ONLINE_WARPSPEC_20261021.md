# V2 true-online warp-specialized pipeline, 2026-10-21

Continue user-authorized kernel optimization at HEAD755b528c. Preserve ascending
Q128/KV64 fixed-threshold Gaussian32 routing, current full-value projection/RMS,
mandatory first/current-canvas tiles and native held consumer/sampler boundaries.
Synthetic Q256 H16 HK2 D512 screen precedes any native promotion. Retain matched
plain/dynamic dense FA4, previous online and custom all-kept controls. Inherited
Yuhan reuse, Junyu Gaussian32/Hopper and NVIDIA BLASST role/pipeline provenance
remain explicit. No peer or installed source is modified.

New variant: a full-QK/softmax/PZ routing WG and independent output WG per Q64
CTA, two output-channel groups. Shared two-slot P/decision queue allows next QK
and routing to overlap conditional full PV. All duplicated routing is charged;
full PV is skipped on the fly. Router maintains retained-only output scaling,
and a consumer performs BF16 full PV without waiting for the next gate. Explicit
CTA release/acquire publications and acknowledgements protect slot lifetimes;
cluster risk epochs are zero-initialized before an initial cluster barrier.
End barrier protects remote shared-memory lifetime. Projected-PV BF16x3 is a
qualified numerical variant. Fixed synthetic work calibration precedes timing.

Review found the previous channel-router builder looked for output[128] to
insert initial epoch reset but that variant declares output[64]. Its frozen
CUDA lacks explicit epoch initialization and initial cluster rendezvous. Prior
passing examples do not establish race freedom; keep old artifacts unchanged
and treat their timing/qualification as provisional pending a corrected control.
New source does not depend on that text patch. Correct the builder with a
validated insertion anchor and remeasure the corrected previous control.

Checks: independent ordered masks/state/logmass, same-mask FA4 output, sticky,
all-kept, partial dimensions, extreme logits, changing-support graph replay;
clean interleaved CUDA events with actual capture counter0; before/after foreign
client checks. Store exact sources/options/compiler receipts and failures in
results/v2_online_warpspec_20261021. No task accuracy/end-to-end claim from random
QKV. No previous paused study is resumed. Resource preemption authorization
includes positively identified FastWAM and user-approved GR00T sprof_1020 only.


Implementation continuation includes Q128 gates within one CTA (no cluster
vote), BF16 register-source PZ/PV, prepacked full-current Gaussian32 high/residual
values, double-buffered warp-risk slots read by all threads after one barrier,
warpgroup-local V/end synchronization, and base-2 normalizers. Early V prefetch
is an explicit scheduling variant that loads all V tiles but still skips full PV
on threshold rejection. Conditional-V schedule remains separately measured.
Every packing/projection/list launch is charged. FP16 one/two-term numerical
variants, TMA layout failures, spilling QK lookahead, and staged single-QK
probability buffering are diagnostic negatives, not selected semantics.

Final promoted experiment for this continuation is Q128 within one CTA, four
128-channel output groups, BF16 register-source PZ/PV, prepacked Gaussian32
hi/residual operands, one-barrier shared risk vote, WG-local V/end waits,
base2normalization and algebraically unnormalized projected risk. The original
state/mass definition and fixed threshold remain. Early-V schedule2 loads all
full V but conditionally skips PV; schedule3 loads full V only when retained.
Both full-current-V projection and packing are charged. Four output groups
repeat QK/routing four times, disclosed as the main remaining inefficiency.
No native task-generation protocol is frozen or launched in this screen.

All completed random-QKV screens are diagnostics. Parallel producer/buffered
PV and projection/unrolling microprobes lack the full final timing audit and
must not replace it. Candidate failures remain with explicit markers. One
incorrect TMA candidate was accidentally timed before its qualification result
was inspected; the benchmark rejected it and all its numbers are excluded.
One lookahead worker stalled on an unissued K wait; only our positively
identified worker and queued benchmark were stopped and their receipt retained.
Builds now use a unique per-build temporary directory after concurrent nvcc
processes in sandbox PID namespaces collided in /tmp; invalid objects are not
used. Final selected/reference binaries are rebuilt and independently checked.

Later controls: explicitSFU exp2 (smallgain), FP16-PZ lower-bound (one timing mask
flip at4096; diagnostic only), independent routerCTA+twoPVCTAs (correct, much
slower), vectorized DSMhandoff (slower), staged-QK without recompute (correct,
slower). --use_fast_math is an explicit numerical build option, not silently
applied to previous measurements. Source frozen again after final selection.

Final numerical candidate combines explicitSFU exp2 and folded reciprocal/reference
risk arithmetic; the additional fast-math compile option is named and tested.
Do not call it identical to the historical BF16x3 implementation. State/masks
must pass the same tolerance gates and changed-input graph tests. No task-level
accuracy is inherited. Objective remains conditional on measured dense parity.


Measured continuation: final clean screen042 uses frozen/attempt003 and exact
qualified_source/attempt002 binaries (production7300f103369c804d,
diagnostic7a7f517edf6d245d, corrected referenceb93bf2f27e5a7173).
At1024/4096/8192/16384 keys complete online refresh is
84.10/239.20/460.20/892.71us versus plain dense46.72/173.13/361.68/710.40us.
Thus dense parity is still unmet. The corrected previous online implementation
with the same fast lists is104.31/327.51/635.47/1236.88us; time falls19.4-28.3%.
Do not describe this as a native HumanEval improvement. The fastest selected
schedule loads all V but skips rejected full PV; retained-only-V schedule is
separately timed. Both are fixed-threshold, true online attention.

Independent diagnostic and production each pass9 qualification cases; final
13 H100 regression tests pass without failures or skips. Masks agree in these
cases; projected-state max error is1.23e-6, logmass max error9.54e-7, and output
relativeL2 stays below0.003 against same-mask FA4. Explicit fast math is a
numerical variant and requires native qualification before task-level use.
Final timing has9 rotated rounds x30 graph replays per geometry/function and
actual capture count0 during timing. Full-current-V projection/RMS, packed
sketch and output list construction are charged; core timing includes packing.

A last Q128/256-channel variant halves duplicated QK to two groups, aliases V
with packedZ after both WGs finish projectedPV, and uses conditional-V loading.
Reducing QK unroll to4 removes spills (255 registers), yet complete online8K
refresh509.38us loses to four-group460.20us. Keep screen043 as a negative;
qualification051/052/053 validates the wider-group attempt, not a dense speedup.
The promoted four-group schedule2 uses233 registers with zero spills. Compiler
resource reports and isolated controls do not establish hardware stall shares.

One architectural limitation remains: the winning configuration repeats QK
and routing four times across output-channel groups to occupy128 CTAs at
Q256/H16. At50% PV sparsity its compute work is therefore not the ideal
single-QK plus half-PV count. Unique-QK router/consumer DSM designs and the
wider two-group version reduce arithmetic but have measured larger costs;
no claim that online sparse dense parity is theoretically impossible follows.
The next design needs a more efficient unique-QK probability handoff or greater
useful query/head parallelism without changing the ordered threshold policy.
