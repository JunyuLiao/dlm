# Native causal support diagnostics, 2026-10-24

The bounded implementation and diagnostic protocol completed. It supports the earlier warning that current attention accuracy or unchanged current decisions do not guarantee an unchanged next native state. It does **not** establish a ResponseGuard advantage, task-accuracy preservation, shorter denoising trajectories, or speedup. No production selector or benchmark panel is promoted.

Branch: denoising_trajectory_research. Pre-generation specifications: c960142f, a9e5570b, bb8baee8, af0a4a5f. Native restore protocol: fb915b0d419c85a9104a720333486586ce2c263e9c9b3df5ff62b00fa7bfc2f4. Corrected support protocol: 1474ac6ec0ee8f9570f9a4a1f9b6976021c20ffcfc81da691f9b99dd8d39f521. Sources and original private manifest are hash-bound. The corrected adapter was frozen before attempt006 generation; attempt005 remains failed and excluded.

## Implemented and qualified

An in-memory snapshot restores native KV storage, request/scheduler/sampler containers, staging buffers, and CPU/CUDA RNG in place. The first-canvas anchor has emitted no output, allowing the opaque detokenizer stream to be reconstructed from the original private prompt. Model parameters and immutable configuration objects are shared. The implementation is qualified at this boundary; it is not a certificate for arbitrary later canvases or all opaque runtime state.

Attempt004 compares an unrestored dense FULL+fix51994 continuation with two restored complete continuations on HumanEval/53. All three match exactly: output hash, 275 output tokens, native logits, active sampler states, final RNG, 10 remaining native model/engine calls, 12 total sampler calls, 10 denoising calls and two committed canvases. Both immediate post-restore checks match all 235 storage spans (16,868,788,160 bytes). Zero extra forwards and zero measured CUDA captures. The final post-continuation snapshot comparison differs from the anchor, as expected; use the immediate restore receipts for the restoration gate.

Attempt006 implements a diagnostic support bank using current Q/K/V, exact FP64 directional response including moving Q, K and V, fixed physical prefix budgets, mass/near-swap/far-swap/position-sham candidates, deduplication, and deleted-mass/output-bias/renormalization guards. Only GLOBAL layer29/head0/queryblock0 may prune wholly-prefix KV64 tiles. Boundary/current-canvas tiles, other heads/queryblocks/layers, LOCAL, prefill and commit remain native/dense. Held calls read current K/V through native storage aliases; no old K/V or attention output is substituted.

Every common-state branch executes two ordinary native denoising calls. There is no suffix forward, forced continuation or replacement sampler. The observer records downstream logits, margins, entropy/acceptance reconstruction, self-conditioning and native convergence flags. Proposal, re-noising, self-conditioning, native schedule, canvas transitions, stopping thresholds, maximum48, output caps and EOS remain unchanged. These explicit diagnostic branches are not a deployed method with hidden lookahead cost.

## Completed experiments

| Component | Completed scope | Result |
|---|---|---|
| Native dense restoration | HumanEval/53, full continuation, reference + two restores | Exact parity |
| Support GPU smoke | HumanEval/124, five distinct candidate supports | Passed execution/RNG/restore checks |
| Short diagnostics | AIME18/7, HumanEval70/42 | All four complete |
| Long diagnostics | One original LongBench item, 185546 input tokens | Eight candidates complete |
| Aggregate | Six anchors, 23 distinct state/support records, 18 with deletion | All bounded execution checks pass |

The original nine over-limit LongBench items remain blocked without truncation or replacement. AIME/7 has only one eligible prefix tile, so both requested budgets round to all-kept: no sparse predictive evidence. Other short prompts offer only two prefix tiles; the held-out HumanEval state has only two deleting supports and one feasible candidate. This cannot establish a replicated ranking effect across two workload families.

Attempt006 recorded 3779 explicit observed native model executions across smoke/short/long, including dense cleanup, anchors and diagnostic branches. Every two-call fork has exactly two forwards/engine steps and matches dense final CPU/CUDA RNG state. There are zero measured request CUDA captures and one FA4 JIT event per process. All-kept duplicates reproduce the matched all-kept logits exactly. This is audit instrumentation, not clean timing.

## Quantitative findings

**Current agreement is insufficient.** Three feasible short mass masks have zero current argmax and acceptance changes against the matched all-kept consumer, yet diverge on the next call:

| Item | Relative current output error | Prefix tiles kept/eligible | Next-call argmax changes /256 | Next-call acceptance changes /256 |
|---|---:|---:|---:|---:|
| AIME18 | 0.01482 | 1/2 | 41 | 3 |
| HumanEval70 | 0.03206 | 1/2 | 38 | 3 |
| HumanEval42, held out | 0.03348 | 1/2 | 22 | 12 |

Including the smoke, five feasible supports retain current argmax and acceptance agreement but change 17–41 next-call argmax positions. These are reference differences, not correctness labels. They support the diagnostic warning about next-step amplification and reject current agreement as a sufficient safety certificate.

**ResponseGuard adds no demonstrated held-out predictive value.** On the held-out LongBench anchor, response versus next-call logit L2 has Spearman0.500, compared with0.524 for current output error across eight deleting supports. These candidates are correlated within one item and include two budget levels; the values are descriptive, without independent-pair confidence intervals. In all three held-out deleting-budget groups (one HumanEval, two LongBench), response and current-error readouts choose the same support: zero distinct selections. One LongBench group has no feasible support and uses the named mass fallback. There is no identified response-selection treatment advantage.

The earlier inherited-state operator study had held-out response/error correlations0.906/0.917 on AIME and0.869/0.919 on LongBench. The new native result is consistent with the earlier no-go, but measures downstream native logits on dense anchors rather than attention-bias motion on inherited V31 states. These are different endpoints and state distributions.

**Budgets and decisions are not monotone.** LongBench has2899 eligible prefix tiles for the targeted head/queryblock. At K1450 all four candidates fail the frozen guards; mass removes maximum whole-row mass0.28110 and has relative output error0.18684. At K2610 all four pass; mass removes0.03760 and has error0.02761. Mass at K1450 changes6 next-call argmax positions, while mass at K2610 changes15. This does not make the rejected smaller budget safe; it shows that fewer current errors or more tiles do not order every future discrete event.

**Consumer differences matter.** Replacing native dense attention at layer29 with all-kept paged sparse FA4 changes1–6 current argmax positions and4–45 next-call positions across the six anchors. Native and all-kept therefore remain separate controls. Sparse-support effects above use the matched all-kept path; they cannot be attributed wholly to sparsity relative to native FULL. All-kept support duplicates match their consumer control exactly.

**No stopping gain was observed.** None of the23 candidate records changes the native stop flag in either of the two measured calls. These are early states, so this is limited negative evidence, not a test of late-canvas stopping improvement. No sparse branch runs to a final answer or full native trajectory. Paired task accuracy, sparse N/C, S, S/N and clean W are unavailable.

**MarginGuard remains inconclusive.** A post-generation descriptive decision-influence readout at LongBench K2610 chooses a support with6 next-call argmax changes, versus15 for response/current-error, but next-call logit L2 increases from4591.884 to4863.891. The minimum absolute mean-margin-motion readout selects the same mass support as current-error. There is one held-out long state, no correctness labels, no independent confirmation and no full tile-influence/stopping oracle. This mixed exploratory observation does not qualify a MarginGuard proxy. Fields named correction_to_reference/break_from_reference describe movements relative to the dense reference and preceding argmax, not correct/incorrect task outcomes; they are not used as accuracy evidence.

**Renormalization scope is limited.** No current-canvas tile is deleted, so removed current-canvas mass is zero and its deletion-specific factor is1. Whole-attention prefix deletion renormalizes retained canvas attention by1/(1-deleted whole mass), recorded separately as whole_renormalization_max. Prefix-normalized deleted mass is also retained. Actual current-canvas deletion/amplification experiments remain incomplete.

**ValidityReuse is still a negative practical lead for the tested bound.** Prior1481 partial-support norm certificates all fell below0.5 without observed violations. No new validity-triggered refresh method is deployed here, and no certificate crosses a canvas transition.

## Physical work and timing boundaries

| Candidate branches | Target-layer eligible tiles | Kept | Skipped | LOCAL/current-canvas skipped |
|---|---:|---:|---:|---:|
| Smoke, five supports, two calls each | 3200 | 3184 | 16 | 0/0 |
| Short, ten supports, two calls each | 4416 | 4404 | 12 | 0/0 |
| Long, eight supports, two calls each | 1486848 | 1472944 | 13904 | 0/0 |

These Q128xKV64 support rectangles cover the executed target layer across all heads/queryblocks; only head0/queryblock0 is pruned. They are not whole-request or all-GLOBAL-layer work and are not CUDA CTA counters. Long K1450 skips1449/92928 rectangles per target-layer call, while K2610 skips289/92928. Nominal prefix sparsity50%/10% is not whole-model sparsity. No matched full-trajectory sparse work or consumer benchmark was run.

Attempt006 launched worker reservation1547.987s (smoke339.919, short608.075, long599.992), including startup, warmup, snapshots, diagnostics and cleanup. All native-causal attempts combined reserve2660.187s, including failed attempts. These are conservative allocated-worker seconds, not measured active GPU seconds or clean request latency. Selector/packing/snapshot overhead is unoptimized and not separately timed. Three request JIT events also preclude timing interpretation. Final H100 check:0MiB,0%, no compute clients.

## Decisions and incomplete scope

- Native restoration: GO at the tested first-canvas boundary; broader anchors require qualification.
- ResponseGuard: NO_GO for proxy or full benchmark. No added held-out prediction and no distinct guarded support versus current-error.
- MarginGuard: exact decision diagnostics only; no proxy. The exploratory long-state tradeoff is insufficient for promotion.
- ValidityReuse: NO_GO for this norm-bound implementation; preserve the broader hypothesis.
- Native acceleration: UNESTABLISHED. No accuracy-preservation, fewer N/C, latency or novelty claim.

The next justified implementation is a diagnostic MarginGuard tile-influence oracle on middle/late dense native states, with restoration qualified at those boundaries and independently frozen held-out confirmation. Protect native decision margins and record correction-versus-break relative to a controlled final continuation; do not fit a proxy from the present bank. Stronger response directions/candidate banks can remain hypotheses, but cannot be tuned until a positive is found and reported as confirmation. No kernel optimization or full benchmark is justified now.

The broader implementation plan is partial: later-state/multi-layer/head D1 curves, incremental D2 headroom and final-answer continuations, true current-canvas deletion, finite-difference downstream response, reuse-age/certificate interventions, complete MarginGuard oracle, seed replication, task-scored sparse trajectories and FULL/PIECEWISE/all-kept/V31 matched benchmark controls remain undone. The original21-item dense census and its official scores are unchanged. No RESULTS_LEDGER benchmark entry or whole-study completion marker is added.

## Reproducibility and preservation

Nineteen focused CPU tests pass twice, including moving-QKV finite differences, fixed support budget/dedup/RNG, immutable object and native SamplingParams restoration, and native output shape regression. All attempt006 pinned sources still match. Analysis input hashes, method configurations, decisions, per-item controls and numeric candidate traces are saved. See analysis001/summary.json, analysis_complete.json, native_prediction.png/pdf and closeout001/COMMANDS.md.

Attempts001/003 preserve dense references followed by restore setter failures; attempt002 is CPU-only and launched no generation; attempt005 preserves its output adapter failure and source-freeze defect, with no usable sparse result. Attempt006 is a new corrected immutable run. No old campaign was resumed and no unrelated process was killed.

Yuhan's inherited cross-step reuse, V31, Gaussian32/value-direction routing, query-sensitivity/C-gate and V2 retain separate provenance. This diagnostic support bank is a study variant, not a renamed baseline. Private prompts, gold, token IDs, logs and generated text remain local/ignored. Public traces contain IDs, hashes, metadata and numeric aggregates only.
