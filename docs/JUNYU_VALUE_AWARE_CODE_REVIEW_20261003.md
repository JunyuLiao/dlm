# Junyu code-publication update — 2026-10-03

Read-only review of `ljy/value_aware` at
`b890ff49488474c5d476df44019597dc045a5969`, posted 2026-10-03 01:21:57 UTC-5.
The previous reviewed ref was `859c0c8fc2a4509a148a3270915f79961e565dee`.
This updates the publication limitation in the earlier
[review](JUNYU_VALUE_AWARE_REVIEW_20261003.md); its historical observations remain intact.

## What is now published

The two new commits add/change127 files, including the Hopper CUDA kernel,
Torch bridge, loaders, causal query state, evaluation drivers and tests.
All15 previously missing Markdown targets and the additional named report test
are now present. All118 changed Python files pass AST parsing. This is syntax
inspection, not execution of their tests, a build, numerical qualification or
performance validation. No peer implementation was imported or executed.
The review focuses on the core kernel, state and comparison paths, not a claim
of exhaustive review of every added line.

There are no changed `results/` files in this update. New query-adaptive
configuration, attained threshold and run-result records referenced by the
drivers are not in this ref's results tree. Thus the new code resolves the
missing-implementation problem, but does not supply fresh verified C_gate
accuracy, step-count or end-to-end speed evidence.
Machine-readable publication checks are in
[audit.json](../results/v29_20261002/junyu_publication_update001/audit.json).

All peer source locations below refer to this exact commit, not our similarly
named historical files.

## Confirmed interface defect

`csrc/value_direction.h:7` declares ABI4. However `cuda.py:163` rejects every
non-null `debug_scores` argument unless `self.abi_version == 3`.
`qualification.py:29–31` and `tests/test_value_direction_hopper.py:148–149`
pass that argument for same-logit verification. With the newly published ABI4
binary, those calls fail at the Python guard before the diagnostic kernel runs.
The ABI4 parameter structure still contains the debug-score pointer
(`cuda.py:171–174`). The compatibility check and its tests need to cover the
supported ABI4 layout. This finding does not establish a failure of ordinary
production calls without `debug_scores`.

Source links: [guard](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/cuda.py#L163),
[ABI](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/csrc/value_direction.h#L7),
[qualification caller](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/qualification.py#L29).

## What physical skipping saves

The kernel has a real conditional full-dimensional V load/PV path
(`csrc/value_direction.cu:580–607`). Routing still computes current QK,
softmax, rank32 projected PV and the physical-tile vote. It uses sequential
retained-state routing; a skipped candidate does not update the retained state.
This is different from our cached observation/dense-prefix-risk selector,
which can avoid both QK and PV for skipped consumer tiles between observations.
Equal physical skip percentages are therefore not equal compute savings.

The integration caches unchanged-prefix V sketches, not current QK scores
(`integration.py:18–70`). The native-adapter path is separate from our vLLM
FA4 serving adapter. The peer README explicitly says its native dense dispatch
is SDPA and that beating it does not demonstrate FlashAttention/TensorRT speedup
(`README.md:357–361` within the Hopper directory). `native_benchmark.py:31–34`
also constructs SDPA references. A production-style CUDA implementation is not
by itself evidence of parity with the fastest native FA4 kernel or serving stack.

## C_gate: now source-checkable, not a new cure for the prior negative result

For matched inputs and parameters, the new C_gate formula and our historical
port have the same mathematical state update:

- First canvas call uses sensitivity4 (`beta=3`).
- Renoised-mask EMA starts at1 and uses `gamma_q=.65` in the official driver.
- The accepted-run count increments on acceptance and resets on rejection or
  a top-1 flip; `tau=2.5` controls relaxation.
- Confidence uncertainty is `sqrt(max(1-p_top1,0))`.
- Completed sampler outputs affect the next routing call, never the current one.

Peer anchors: `query_adaptive.py:128,173–186,280–308,348–377,420–424`;
`aime_temporal_sweep.py:253–279`. Our historical implementation is
`experiments/value_direction_hopper/query_adaptive.py:57–118` with tests in
`tests/test_v27_cgate.py`. Peer `-expm1` versus our `1-exp` is mathematically
equivalent, not a claim of bitwise equivalence.

The complete systems differ: peer fresh-QK retained-state selection with
uniform local/global thresholds versus our historical-QK DP/R6/carry-first
configuration on five GLOBAL layers. P17 remains a negative result for its
tested port/configuration; it does not reject every peer C_gate system.
Conversely this upload does not overturn P17 or prove that C_gate improves our
main. Our vLLM adapter still explicitly rejects C_gate because its hook does not
yet expose the required accepted-token mask (`vllm_adapter.py:143–149`).

Additional integration qualifications:

- Direct `State('C_gate')` defaults its trajectory EMA from gamma=.5; the
  documented evaluation driver explicitly supplies .65. A port must freeze
  the driver parameters, not rely on the class default.
- Canvas reset assumes `cur_step==48`; changing the schedule needs explicit
  lifecycle qualification. The sampler hook's accepted-mask semantics and
  ownership across compiled multi-request execution must be checked on the
  target runtime; source inspection alone does not prove them.
- `diagnostics=True, collect_all_stats=False` is accepted by the State API,
  but the diagnostics branch accesses omitted margin/entropy/drift statistics
  (`query_adaptive.py:241–278,382–398`). This is a non-default API-combination
  defect, not evidence of default C_gate routing failure.

## Fairness and provenance findings

The dense cross-root cache path in `aime_temporal_sweep.py:333–344` checks only
completion, method, item ID and seed before rewriting the new identity. It does
not establish matching prompt identity, model revision, generation budget,
source or runtime. A changed protocol can therefore accept an old dense record.
This is a missing validation guard; it does not prove that any historical record
actually had mismatched inputs. Keep old controls explicitly imported, or
validate the full original configuration and retain its provenance; regenerate
matched dense controls for new headline comparisons.

The gated driver explicitly discloses overlapping calibration/screen/final
AIME questions (`aime_query_sensitivity_gated.py:60–79,360–366`): its resulting
accuracy is development evidence, not held-out confirmation. Its clean timing
driver reruns dense and finalists with instrumentation disabled, which is useful,
but uses seed42 and reports mean/median request time (`:263–300,342–359`). This
does not replace our question-cluster uncertainty or matched multi-seed W/S/N
reporting. Do not confuse imported historical quality controls with fresh timing.

The HumanEval v2 protocol also explicitly discloses final-cohort tile-budget
adjustment after disjoint calibration (`humaneval_query_sensitivity_v2.py:301–311,
347–354`). No correctness is used for that adjustment, but it is transductive,
not wholly held-out threshold selection. Its seed42/100-question setup and
instrumented timings are not interchangeable with our full164-question,
eight-seed vLLM panel. Preserve these scope distinctions when citing results.

Thresholds are fitted outputs, not transferable constants. The new README
requires attained records and matching configuration/source/kernel identities;
the placeholder zeros are not a usable policy. New vLLM/GLOBAL-only deployment
needs separate development calibration and frozen evaluation, not direct reuse
of an HF local+global target-sparsity threshold.

For a strong engineering reference, peer historical T still computes confidence
and margin even with diagnostics off (`query_adaptive.py:253–264`), whereas its
C_gate production statistics are method-specific (`:265–278`). Our `fast_t`
already omits these unused statistics. Savings against the heavier T statistics
path cannot be attributed to a new gate formula.

## Consequence for our work

Retain official vLLM/FA4 dense, matched native and matched optimized controls.
The immediate sparse-cost/dense-reproducibility/fusion queues remain unchanged.
C_gate is a cooperation candidate for a separately qualified accepted-mask hook
and same-state weight/lifecycle comparison, followed by target-runtime
calibration and accuracy/step/time evaluation. It is not ready to replace main.
Regroup remains a distinct collaboration direction; this review establishes no
new low-cost regroup kernel or evidence that the peer upload solves our current
32K overhead. No peer branch was merged, modified or pushed.
