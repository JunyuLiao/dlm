# Partial pulse/hold development diagnostic — 2026-10-24

The bounded worker stopped at a frozen full-request restoration gate. Two AIME
anchors completed engineering qualification and candidate-bank evaluation; both
banks failed identification. No selected SA/AS/SS factorial branches ran. This
does **not** establish or reject a response mechanism, quality preservation,
shorter native trajectories, or speedup. The experimental stage remains partial;
the partial evidence and handoff are closed out without another GPU launch.

Start: branch `denoising_trajectory_research`, commit `7febadb6`.
Pre-generation checkpoints: `17b8e7ba` (unlaunched attempt001), `c3dd9855`
(attempt002). Frozen protocol SHA256:
`e0223250d0c7b225e7650e43f64fc7ec8ca4c002d65c819e4d2fbd23709ef9e3`.
All 17 pinned generation sources and four scorer sources match at closeout.
Model revision `f7f5b7f5fa82ffc52addd066915886d497f5517b`; all 16 model/config
files were hashed before generation. Runtime/model/scorer identities are in
`protocol.json`, `source_fingerprints.json`, `runtime_fingerprints.json`, and
`model_fingerprints.json`. Original attempt006 sources/results remain unchanged.

## Prospective scope and actual completion

Before generation, the maximum proposed grid was costed at more than 15288
forwards for finalist continuations alone. A smaller **development-only** leaf
was frozen: AIME26/18 and HumanEval/124, seed42, two native reference-relative
first-canvas anchors each. Both identities have prior exposure. No method-holdout
or fresh-benchmark label applies. Long development, seeds43/44 confirmation,
scored finalists, and long replication were deferred before generation. All nine
original over-limit LongBench identities remain blocked without input changes.

| Item / anchor after completed calls | Native L | Remaining native / AA calls | Feasible / bank | Matched / response pairs | Selected factorial branches |
| --- | ---: | --- | --- | --- | ---: |
| aime26/18 after5 | 11 | 6 / 6 | 2 / 8 | 0 / 0 | 0 |
| aime26/18 after9 | 11 | 2 / 2 | 2 / 8 | 0 / 0 | 0 |
| HumanEval/124 after4 | 8 | Unqualified complete-request replay | Not evaluated | Unavailable | 0 |
| HumanEval/124 after6 | 8 | Not reached | Not evaluated | Unavailable | 0 |

The two completed anchors are within ONE question and ONE seed. They are not
independent task samples. Native and AA terminate adaptively at both completed
anchors; neither hits the 48-call cap. No sparse stopping endpoint is available.
Remaining calls exclude the separate native commit forward: after5 has seven
model/engine calls including commit; after9 has three.

## Measurement and restoration

At each completed AIME anchor, 235 storage spans, 276 tensor views, and 13 alias
groups pass byte and live pointer/shape/stride/offset/alias checks. Unrestored
native, two restored native, new measurement disabled, and post-perturbation
native fingerprints/RNG/counters agree exactly through native first-canvas
commit. Repeated and post-perturbation AA controls also agree exactly. The
predeclared maximum-current-error SS support is a restoration stress control,
not a feasible method outcome. Snapshot/config/stream limitations remain; these
certificates cover only the tested AIME anchors before emitted output.

HumanEval/124 after4 failed `RuntimeError: full-request restore parity` at frozen
`code/worker.py:192`. The comparison checks two restored full native results and
the unrestored full result. Its field-level differences were not saved before
raising. The receipt cannot determine which pair or which output/state/RNG/count
field differed, or distinguish restoration failure from a comparison/accounting
defect. **No complete-request certificate** is issued. HumanEval first-canvas
checks executed before that comparison but their individual results were not
persisted, so no separate HumanEval certificate is issued either. Sources and
failure are preserved; the equality requirement was not weakened or retried.

Actual native argmax/history are read and checked against returned scaled logits,
including ties. Across 48 saved repeated-control call records: zero winner
mismatches, 33 tied-position occurrences. Original sampler calls execute once;
observer RNG brackets and measurement-disabled replay pass at completed anchors.
Explicit model forwards equal engine calls in each completed qualified branch.
These measurements add no model forward or native RNG draw in that tested scope.

Acceptance remains **reconstructed on the original device/dtype** from actual
returned scaled logits. Saved controls have zero threshold ties/near cases and
zero CPU-threshold discrepancies given device entropy. This is not a direct
compiled acceptance-buffer capture and cannot justify exact acceptance claims.
`sampled_canvas` names the native emitted buffer, not exposed multinomial
proposals; during denoising use `canvas_after` for actual mutated-canvas changes.
Fixed-reference signed margins, centered logits, KL/JS, history, confidence,
convergence, canvas and self-conditioning comparisons are retained. Old
correction-to-reference labels are unused.

GQA0→KV0 and8→KV1 are asserted and independently checked against current-QKV
FP64 attention for each completed item runtime; first all-kept worst relative
errors are 0.00245 / 0.00230, below the frozen0.02 tolerance. Native PIECEWISE
and paged all-kept FA4 remain distinct consumers: after5 their first two native
winner mismatches are2 and9; after9 they are0 and0. Their remaining call counts
coincide here. Consumer differences are systematic treatments, not subtracted
noise. No FULL-versus-PIECEWISE equivalence or FULL finalist result is claimed.

## Bank, current-call measurements, and missing factorial contrasts

Scope: GLOBAL layer29, heads0/8, queryblocks0/1, Q128×KV64. Each candidate
deletes exactly one wholly-prefix rectangle across four routing units. Here
B=2: 8 eligible, 7 retained. Boundary/current canvas, other heads/layers, LOCAL,
prefill and commit remain dense. This is fixed-total-work allocation, not equal
per-unit sparsity. Candidates are stratified by deleted unit in the analysis.

There are16 candidate records but only8 unique geometries, and4 feasible records
but only2 unique feasible geometries. At each anchor, six candidates fail the
0.1 current-bias guard; four also fail mass≤0.2 and renormalization≤1.25.
Mass/renormalization guards are redundant here. The two feasible candidates
delete in head0's two queryblocks; head8 has no feasible deletion. Neither anchor
has a pair satisfying the frozen5% current-error and0.01 deleted-mass matching
rules; response-identifying pairs are consequently zero. No guard was relaxed.

The four feasible single-call probes provide **current-call** measurements only:

| Anchor | Deleted unit | Native winner / history mismatches | Mutated-canvas mismatches | Self-conditioning L2 | Worst margin erosion |
| --- | --- | --- | ---: | ---: | ---: |
| after5 | head0/qb0 | 0 / 0 | 2 | 0.191 | 0.127 |
| after5 | head0/qb1 | 2 / 2 | 0 | 4.514 | 0.0753 |
| after9 | head0/qb0 | 0 / 0 | 0 | 0.0476 | 1.056 |
| after9 | head0/qb1 | 0 / 0 | 0 | 0.209 | 0.250 |

Neither confidence nor convergence flags change on these four first calls.
These masks do not constitute matched-current-error comparisons; there is no
future-effect ranking result. The ≥4-feasible requirement causes every score
(current error, response, decision, mass, sham) to abstain before finalist
branches. Immutable `selections.json` files contain null choices. Distinct
selected supports=0. SA−AA, AS−AA, SS−SA, and interaction outcomes are **missing**,
not zero. Incremental value over current error or a matched sham is untested.
The response score remains sensitivity of sparse-minus-dense attention bias,
not full denoising contraction; Q/K/V and joint components are separate.

See `analysis_partial002/summary.json`, `method_decisions.json`, and
`bank_identification.png`/`.pdf`. `readout_complete.json` closes only the partial
analysis, not the experimental stage. Complete-stage analysis reader templates
at the campaign root were not executed.

## Work, resources, tests, and failures

Restored AA controls after5 retain6720/6720 GLOBAL and33600/33600 LOCAL
rectangles across six denoising calls. After9:2240/2240 GLOBAL and11200/11200
LOCAL across two calls. Skips=0. Denominator: actual native denoising metadata
at Q128×KV64, not CUDA CTAs. Unrestored-native cumulative counters include
pre-anchor work and must not be used as remaining-branch work. Per-pulse and
stress-control execution metadata were not saved; their intended one-rectangle
support is enforced by source, but separate physical-work receipts are incomplete.
No skipped-work, clean-latency or selector-overhead gain is claimed.

Attempt001: source checkpoint only; pre-generation phase-capture defect detected,
zero GPU work. Attempt002: one H100 worker; exit1 at qualification; allocated
worker1795.510313seconds,640 counted model forwards (208 startup/profile,
432 runtime-path executions),426 engine steps. Six runtime executions occur
outside the explicit engine-step loop. Allocation includes startup, warmup,
qualification and failed continuation work; it is not active GPU time or clean
request latency. 3224 CUDA capture events and one inference JIT event were
observed, so no capture-free whole-run timing claim applies. The budget was
7200seconds/10000forwards; unused resources were not used to bypass the gate.
H100 is idle at closeout:0MiB,0% and no compute processes. No other workload was
preempted, no paused campaign resumed, and no worker remains running.

24 focused CPU tests pass before GPU (`tests/cpu004.xml`) and at closeout
(`tests/cpu005.xml`,1.40seconds). Meaningful cases cover native-winner ties,
signed margins/cancellation, centered logits, absorbing first-call stops,
pre-call phase capture, metadata mutation, equal budgets and common-feasible-set
selection. Independent10-trial FP64 CPU central differences validate the Q/K/V
bias differential (max absolute error2.56e-11 versus tolerance1e-7); they do not
validate native prediction. A CPU-only partial-reader assertion incorrectly
expected a trace from the deliberately measurement-disabled control; its failed
output is preserved as `analysis_partial001/failure.json`, corrected reader
output separately under002. No experimental artifact was changed by that repair.

No official task scores, sparse full requests, accuracy intervals, output-length
comparisons, S/N, S, W or synchronized clean timings exist in this leaf. They are
unmeasured, not inferred from controls or sparse budget. No RESULTS_LEDGER entry
is added for this partial stage.

## Decisions and review handoff

ResponseGuard: **no-go** for proxy/production/promotion; hypothesis untested.
MarginGuard: **no-go** for exhaustive influence oracle; no replicated distinct
decision-score benefit. ValidityReuse remains **parked**. Confirmation, full
benchmarks, kernels and later-canvas work do not launch. This agrees with the
review's diagnosis of insufficient identification and consumer confounding; it
does not validate the proposed beneficial response mechanism or solve the later
LongBench confidence tail. Yuhan reuse, Gaussian32/value-direction, C-gate,
V31/V2 and all earlier negatives retain their separate provenance.

Most informative next bounded work, after review: a **qualification-only** replay
of HumanEval124 after4, persisting each full-branch equality field and the first
divergent aligned call before any assertion. Resolve restoration versus comparator
failure before another support study. Separately, prospectively design a bank
with at least four feasible within-unit alternatives and matched-error coverage;
do not turn this non-identifying result into a mechanism rejection or search
future outcomes until a positive appears. No such next run is authorized by this
handoff itself.

Inspect first: this summary; `closeout001/qualification_failure.json`;
`analysis_partial002/summary.json`; `analysis_partial002/qualification_summary.json`;
`execution001/item000/anchor005/{controls,candidate_bank,selections}.json` and
corresponding anchor009 files; frozen `protocol.json`/`code/worker.py`;
`closeout001/verification.json` and `COMMANDS.md`. Public artifacts contain only
IDs, metadata, hashes and numerical diagnostics. Private inputs/logs/vectors remain
ignored. HANDOFF/DECISIONS updates preserve the existing review edits. No push.
