# CPU-only design and feasibility review

Reviewed 2026-10-18 against frozen source
`6aab86325990edf6e5ab6bb72bcd909781234e97` and protocol SHA256
`1e7714be3fd4690b41e37727a7b8ae283800c24f579f90eec03d90e37bfb964e`.
This note records a read-only review; it changes no execution source, protocol,
cohort, schedule, or budget and launches no GPU work.

## Verified qualification and schedule

Qualification completed with four explicit warmups and six measured requests,
zero receipt errors, and 471.963 seconds of GPU reservation. Public artifact
hashes match its completion marker. All six measured pre-event signatures and
full prefix metadata match. The four treatments cover all five GLOBAL layers;
both controls and dense/fresh agree exactly in output hash, calls, canvases,
output length, and score. Within each current-output pair, event outputs and
the first post-event state agree; the intended different future maps are
consumed on the next call. RNG checks pass, with no added model or sampler calls.

All four frozen factorial schedule hashes were verified. The factorial contains
48 question/seed/engine groups and 288 measured requests: eight LongBench
questions and four RULER task families, two seeds, two engine repeats, and six
labels per group. Including explicit warmups, it runs 200 LongBench and 104
RULER requests, plus four implicit warmups and engine startups.

## Runtime feasibility

The factorial phase has a fixed 3,900-second cap. Current engine reservation
minus summed public request wall time is 165–198 seconds for completed clean
headroom engines and 279.7 seconds for qualification. This difference includes
initialization, implicit warmup, and other work outside recorded request timing;
it is not a pure model-loading estimate.

Historical control requests for the four selected RULER items average 6.523
seconds, suggesting about 678 seconds for 104 requests. Using 180–280 seconds
of non-recorded overhead per engine leaves a break-even LongBench average of
10.5–12.5 seconds per request. The historical clean64 V31 average of 8.919
seconds implies roughly 3,200–3,600 seconds total, whereas qualification's
six-arm average of 18.416 seconds implies roughly 5,100–5,500 seconds.
These are planning scenarios, not timing claims: the historical controls are
not the current audited intervention workload, and qualification is one item
with a long unsuccessful dense/held trajectory.

Completion is plausible but uncertain; both engine repeats must not be promised.
At 225 seconds of overhead per engine and a 12-second LongBench mean, the
estimated total is 3,978 seconds, cutting the final RULER repeat. At the
qualification mean, the second LongBench engine would be cut and the final
RULER engine skipped. Keep the frozen schedule and cap; do not shorten the
cohort or borrow phase/reserve budget based on observed outcomes.

## Interpretation limits

- The intervention concerns one first eligible progress-refresh episode. It
  does not directly test bootstrap, commit, cross-canvas amortization, repeated
  removal of dense steps, or a deployable cheap selector. RULER non-exposure is
  an outcome, not evidence that refresh is unnecessary.
- Earlier treated layers can change deeper-layer queries and fresh maps.
  Factorial contrasts measure total strategy effects, not direct effects under
  one identical future map across both current-output conditions. Inspect the
  simple within-output fresh/held contrasts before attributing an interaction.
- There are eight and four independent question clusters, respectively;
  seeds and engine repeats test reproducibility. Inputs were previously
  examined, and LongBench selection balances processed length rather than
  task/domain. Both longest selected inputs have processed length 120,016.
- Qualification overlaps LongBench item 422, seed 42. Do not pool it as an
  independent factorial observation or use it to fill missing scheduled data.
  Its only correct treatment is dense/fresh; the lower-call sparse treatments
  are incorrect, while dense/held reaches the output limit. This one-item
  pattern does not establish an aggregate mechanism. Keep score, calls,
  canvases, output length, truncation, and tail sensitivity together.

## Missingness and completion decision

Both LongBench engine schedules progress through length bins from short to
long. A deadline preferentially removes later long-context questions and
repeats. Censoring can also depend on treatment outcomes because slow native
trajectories consume the remaining budget.

Retain all 48 expected groups and 288 expected label slots in coverage reports.
Enumerate missing labels by suite, length bin, question, seed, engine, and
reason. Use complete six-arm groups for the predeclared contrasts while keeping
incomplete arms and the full planned denominator visible. A deadline-killed
request is missing, never score zero; a completed native output-limit request
is an observed adverse outcome, never missing. Do not replace late cells or
describe available complete groups as representative of the full cohort.

The frozen decision rule leaves the causal research conclusion unresolved with
fewer than six distinct matched/exposed LongBench questions or missing seed or
engine coverage. A partial run may support localized descriptive observations,
but cannot satisfy that completion gate.

Sources: `protocol.json`; qualification `records.jsonl`, `scored.json`,
`qualification.json`, and `attempt_complete.json`; completed headroom engine
002–004 records/receipts; historical clean64 control records under
`results/v31_value_e2e_longbench503_20261017/native_attempt001/`; selected RULER
control records under `results/v31_v2_ruler128k_seed42_tau021_20261015/`;
`docs/REFRESH_DIAGNOSTIC_PLAN_20261018.md`.
