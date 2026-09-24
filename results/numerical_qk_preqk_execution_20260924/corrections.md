# Corrections to the v6 recovery reports (records preserved, prose fixed)

Raw receipts, CSVs and published commits are unchanged. This note records
what the earlier *summaries* got wrong and regenerates the affected claims
from the records. Machine-checked table: `correctness_vectors.json`
(`scripts/preqk_correctness_vectors.py`), enforced by
`tests/test_preqk_correctness_vectors.py`.

## A. "Same missed question" was wrong for fresh Junyu T

Per-ID correctness over `[aime26/2, 8, 14, 20]`, read from
`results/numerical_qk_reuse_20260924/request_table.csv` and the new quality
receipts:

| arm | vector | count | notes |
|---|---|---|---|
| native dense (frozen v5) | `[T, T, F, T]` | 3/4 | /14 capped at 8192, unparsed |
| fresh Junyu T (frozen v5) | `[T, T, T, F]` | 3/4 | **/14 correct at 7276 tokens, not capped; /20 wrong at 2777 tokens, not capped** |
| M1 `routing_only_current_output` (new) | `[T, T, F, T]` | 3/4 | /14 capped at 8192 |
| M3 R2 `routing_only_current_output` (new) | `[T, T, F, F]` | 2/4 | /14 and /20 capped + unparsed |

All of dense, T and the new M1 score 3/4, but **they do not miss the same
question**. The new M1 shares a correctness vector with **native dense
only**; against T it is "same count, different questions". The v6 phrase
"matching native dense and fresh Junyu T exactly (same missed question)" is
withdrawn. The sanctioned phrasing is "same observed 3/4 count on four
development questions", and vector identity may only be asserted after an
element-wise check. No population/noninferiority claim follows from four
questions at one seed.

## B. `temperature=0` is a sentinel, not greedy deterministic decoding

`GenerationRequest(temperature=0.0)` in this adapter selects the model's
native 0.8 -> 0.4 annealing schedule (`_generation_kwargs()`; the receipts
record `sampling.native_temperature_schedule = true`). It does **not** mean
greedy/deterministic decoding. The v6 justification for reusing the frozen
dense/T outputs as controls — "deterministic at temperature 0" — is
therefore not valid as stated. Reusing old *quality* requires matching
model/prompt/tokenization/sampler/scorer identities (which do hold for those
frozen receipts, and they stay usable as labeled *development quality
references*); reusing old *timing* does not, and contemporaneous performance
claims need warmed, matched, same-session measurements. Seed control is not
a cross-kernel/cross-environment guarantee.

## C. Phase A2 scope was overstated

The kernel-vs-oracle check covered **12 real samples, two layers, with
`sensitivity` left at the default 1** (uniform T). It supports agreement for
those cases. It does **not** exclude every implementation bug, does not
establish full-decoder or complete A1 equivalence, and does not validate the
nonuniform causal T weights that production actually uses. The v6 wording
"rules out a kernel bug" is narrowed to "no disagreement in the 12 sampled
uniform-T cases at two layers".

## D. Phase B cell definitions and the normalizer

- The diagnostic's **Cell B is all-kept legacy-legal attention**, not the
  fresh-Junyu *selected* support. So `B vs C` measures the effect of pruning
  at all (with current scores), not the incremental effect of choosing the
  support from historical rather than fresh information.
- The diagnostic ran with **default uniform sensitivity**; production causal
  T weights were not applied.
- `rel_l2(x, y) = ||x - y|| / ||y||` divides by the **second** argument. For
  `a_vs_e` and `c_vs_d` the second argument is the *approximate* (cached-score)
  output, not the fresh reference. Values above 1 therefore do **not** mean
  "error larger than the fresh signal"; that v6 gloss is withdrawn. The
  measured quantities are preserved unchanged under their actual definitions.
- `A vs E` and `C vs D` use different supports and different denominators;
  their closeness is an observation, not a mathematical identity.

The qualitative conclusion the v6 round drew — stale final weights are
associated with much larger local mismatch than the support choice is — is
retained as *evidence favouring current retained attention weights*, and the
closed-loop result in section 3 is consistent with it. It is not a complete
causal settlement, and section 6 of this round re-measures the cells with a
common support, real nonuniform T, and both norms stored.

## E. The warm profile is components, not a full-forward profile

`warm_profile.json` times **isolated components at two attention layers** at
Q256/K621. It is not a full-model forward profile and cannot be summed or
multiplied by 30 to produce one.

- The quoted "~7-8%" is the **cold-vs-warm `Sketches` component time**, not a
  full-forward speedup. Its `cold_lease` helper also constructs a new
  `Sketches` object and a new source tensor per repetition, so it includes
  allocation work that old production did not pay; it is an upper-bound-ish
  proxy, not an exact old-production-vs-new-production comparison.
- The "routing_only component total" quoted in the v6 profile report omitted
  V preparation and was arithmetic over separate measurements, not a directly
  measured combined call.

Measured raw values are preserved; only the labels and derived claims change.
Section 7 of this round replaces these with directly measured full
`Attention.__call__` and complete decoder-forward timings.

## F. Execution identity

`STATE.json` carried an inherited `executed_model_source_sha` from the v5
round. Old execution hashes describe old code and are not evidence about the
new code; per-run config receipts (`panel/config.*.json`, each with its own
`fingerprint` and `source_hashes`) are the authority for what actually ran.
STATE.json now separates the reporting SHA from per-run execution identity.
