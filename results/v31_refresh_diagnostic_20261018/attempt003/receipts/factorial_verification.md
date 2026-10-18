# Independent factorial verification — 2026-10-18

The planned factorial data are complete and pass the structural checks. They
contain unexplained trajectory variation, including two A/A discrepancies, so
this verification does not certify deterministic replay or a clean mechanism
promotion. No group was excluded, retried, or selected by its outcome.

This is a CPU-only review. It launched no model calls and changed no execution
source, protocol, completed engine leaf, or report. The companion
`factorial_verification.json` contains the exact marker, input, output, and
comparison bindings.

## Identity and coverage

- Source: `db755c43be763a8af9edfe8a147ec333d8b8542b`.
- Protocol SHA256:
  `6f77c44a6a361f54ea6e124ff7bf5afb746738c3e51d2dab99b098776f4eb693`.
- Engines 9–12 have valid completion markers, all recorded artifact hashes,
  source/protocol identity, private input/schedule hashes, and exact planned
  counts: 304 total requests, including 16 explicit warmups and 288 measured
  requests. The measured set exactly equals all 48 expected six-arm groups.
- LongBench-v2: 8 questions × 2 seeds × 2 engine repetitions. RULER: 4 questions
  × 2 seeds × 2 engine repetitions. Repetitions are technical repeats, not
  additional independent accuracy observations. These are previously examined
  inputs, not held-out confirmation.
- All 1,154 frozen execution-file fingerprints matched at review time. The
  original native worker/adapter path remains intact.
- The entire campaign is `budget_limited_partial`: all-kept profile engine 8
  was skipped. There is no study `final_complete.json`. This does not invalidate
  the independently complete factorial component. The older full503 study is
  still paused.

## Intervention and native-path checks

All 48 groups have equal recorded prefix signatures and entire prefix metadata
across all six arms. All four treatment arms exposed the intended refresh in
GLOBAL layers 5, 11, 17, 23, and 29, with 128 selected prefix tiles of 64 tokens.
The current canvas has 256 query rows; selection units remain 128 query rows.

Every treated layer recorded one original `_mage` call, unchanged RNG, finite
outputs, and the intended current output and subsequent support. Each treatment
added five diagnostic sparse-attention evaluations, but zero model forwards or
sampler calls. Recorded CUDA graph captures are zero. These audit timings include
diagnostic work and are not clean latency measurements.

DF/DH returned identical current-output hashes at all five layers, and SF/SH did
likewise, in every group. Each corresponding pair also had an identical first
post-event recorded state. This verifies that the future-support assignment did
not prematurely alter the current returned output. Fresh/held support hashes
match the intended next assignment and next-call entry. Accounting verifies
800 within-canvas held and 160 cross-canvas carried layer consumptions. The
`held_consumer` flag alone excludes the latter; carried use is verified against
the adapter branch and per-canvas sparse-call counters.

Fresh support is not necessarily different on every layer: RULER Q21, seed42,
layer5 selects the same old/fresh map in both current-output conditions and both
engines. Its other four layers change. All five layers execute the refresh.

DF versus SF changes the first post-event full self-conditioning fingerprint in
48/48 groups, argmax in 34/48, and canvas in 36/48. This is evidence that the
intervention reaches the sampler; it does not establish beneficial or harmful
final behavior.

The frozen configuration uses K8192, carry-first, settle-trigger 0.15, sticky
1.386, and PIECEWISE execution. It must not be described as the separate Slack
comparison with stickiness disabled. All factorial raw records share the same
GPU and runtime identity; the planned maximum model length differs by suite.

## A/A and engine-repeat variation

Outcome comparisons use official score, denoising forwards N, canvases C, output
tokens T, output hash, and finish reason. DF matches control-before on all these
fields in 48/48 groups; it matches control-after in 46/48. The two discrepancies
are in LongBench engine11 and do not change the official score:

| Question / seed | Control-before → control-after N | T | Output hash |
|---|---:|---:|---|
| 125 / 1 | 51 → 60 | 890 → 899 | `f2c6335c3da1ec86` → `70d6284fc5cddcaa` |
| 335 / 42 | 113 → 109 | 2275 → 2279 | `92ba00930c7525e8` → `11a83da26b0ec4ae` |

For Q125/seed1, the entire prefix metadata agrees, including recorded Q/K/V byte
fingerprints, self-conditioning, RNG, maps, and progress counters. The prefix
signature is
`46c75c90cfe13338e578d57c6dad55c49b0e2ea97fe649ebbe1a1005ba74c907`.
At canvas5/call9/layer5, the control anchors differ only in returned-output hash:
`41f5d5da49f5f4744627fe1ad5c01cb7fc9b4d3077e743f5f174445e07b2c7c8`
versus
`4feda779da9711b64acbb7747b9058501609ae8a5bd389cdd4ec6e71c1d2c3d5`.
Argmax and full self-conditioning already differ at the first post-event call10.
For Q335/seed42, both control anchors and all eight recorded post-event states
(calls9–16) agree; the available trace does not locate the divergence.

Across planned engine repetitions, control-before and DF match all outcome
fields for every question/seed. RULER engines10/12 match for all six arms. In
addition to the two control-after cases above, LongBench Q77/seed1 SF changes
N186→219, C11→12, T2597→2991, and output hash
`99867a9e0baca48d`→`b7aa989e2b62d212`, with unchanged official score. Its entire
five-layer event list is identical, canonical SHA256
`9028e8c479804ca85361ffa7ea11dbe630b1bc5990887d2bee1a20f27638702f`,
and all eight recorded post-event states at calls7–14 agree.

Tensor fingerprints hash contiguous bytes. They do not include input shape,
dtype, stride, alignment, scale, stream, or resolved kernel configuration. Full
self-conditioning is hashed only at the first post-event call; later records
contain compact state and RNG fingerprints. Thus these checks do not establish
cloned full-engine state or identical later hidden states. No numerical,
compilation, allocator, or kernel cause has been isolated.

## Qualification comparison is outside the estimator

The original qualification Q422/seed42 differs from factorial engine9 despite
equal recorded prefixes, control anchors, and entire five-layer treatment event
lists. DF changes N515→403, C24→14, T6035→3357, with correct scores in both.
DH changes N2064 at the length limit to N275, remaining incorrect. SF changes
N404→375 and incorrect→correct; SH changes N351→403, remaining incorrect.

The event-list canonical hashes are DF
`1bf9e639407193bdf029b47f1b2aa829d11d3067be9f766a60fad98743bfaae2`,
DH `4b00b6b648ac526f680dceefa433c6560c6411c79cba0ec4e912095c7a9bd64a`,
SF `b85e1a0c89a794696ff0c094486c1dbd96be7fbb1e3f507d6011c6c35a4c70f2`,
and SH `46a052eee830af689fb41334fcafbac9d8c00b1122ca5b95e175d60e80189ab4`.
The first recorded differences are SF call10 (argmax/canvas) and SH call11
(argmax). DF/DH and controls have no difference in the eight logged post-event
states; this is subject to the fingerprint limits above. Qualification is never
pooled with the planned factorial repetitions.

## Report verification and interpretation

All frozen report input/output/source hashes, supplemental analysis bindings,
and terminal-state receipt match. Recomputing the factorial summary with the
frozen report function reproduces every factorial summary field exactly.
`summary.json` SHA256 is
`8d3b3559934e49a0848369098e260758d0956d42282202238f0fdd97b72f7714`;
`analysis_complete.json` SHA256 is
`c4c28f37ad2b482cb68f35e70296cfc56c38ca41209698429ec3161ce8b895f7`.

The supplement correctly retains all groups, question-clustered intervals,
conditional contrasts, seed/engine variation, and adverse/beneficial score
changes. RULER is a mean official partial-credit score, not a count of correct
answers. In each RULER engine, DF/DH/SF mean0.875 comes from six score1 and two
score0.5 cases; SH mean0.8875 replaces one score0.5 with0.6.

The stronger future-support signal is suitable for provisional mechanism triage.
Current-output removal has seed-sign reversals and call increases that can
accompany better answers. Neither a clean causal promotion nor a ≥10% incremental
request-latency benefit is established by these factorial data. The study tests
one refresh, not repeated cheap selection, bootstrap removal, or commit removal.
