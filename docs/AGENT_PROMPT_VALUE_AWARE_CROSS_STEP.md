# Delegated implementation task: value-aware selectors for v31 cross-step reuse

You are implementing and evaluating a narrowly scoped research change in the
`/home/exouser/ljy/dlm` repository, on the current `combined_end2end` branch.
Work autonomously through implementation, correctness checks, H100 evaluation,
kernel refinement, and final reporting. Do not stop at a design proposal.

The goal is to test whether better value-direction-aware block selection can
improve the existing v31 cross-step reuse system on tasks where the current
selector has not been satisfactory. The result must be reproducible and
honest about accuracy, selection cost, physical sparsity, and end-to-end time.

## Repository context and attribution

Read these files before editing, in this order:

1. `AGENTS.md`
2. `HANDOFF.md`
3. `docs/V31_BASELINE_ROOT_CAUSE_20261003.md`
4. `docs/RESEARCH_CONTEXT.md`
5. `docs/DECISIONS.md`
6. `docs/RESULTS_LEDGER.md`
7. Relevant `README.md`, `config.json`, `receipt.json`, `summary.md`, and
   `final_complete.json` files under `results/`.

`docs/AGENT_PROMPT.md` and the files under
`docs/peer_references/junyu/` are algorithmic reference material only. Do not
copy collaborator-specific hosts, private paths, coordinator conventions, or
preferences from them. Use this prompt, `AGENTS.md`, and the current branch's
frozen result contracts for execution.

The first cross-step-reuse implementation was Yuhan's inherited work. Preserve
that provenance in names and reports. Gaussian32/value-direction-aware routing
and C_gate/query-sensitivity protection are Junyu's earlier method families.
The variants implemented here are new selectors evaluated inside the existing
v31 reuse pipeline; do not attribute them to Yuhan or Junyu without saying they
are this study's integration.

## Current stack and immutable integration contract

- Model: DiffusionGemma-26B-A4B.
- Runtime: the repository's pinned vLLM 0.30.0 and SM90 FlashAttention-4 path.
- Hardware: one NVIDIA H100 80 GB, CUDA device 0, one worker at a time.
- Native attention geometry: physical `128(or 64, matching the current configuration)` query rows x `64` key/value tokens,
  native GQA, head dimension and structural masks as supplied by the model.
- There are five GLOBAL layers and twenty-five LOCAL sliding-window layers.
  This experiment changes GLOBAL/KV block selection only. Keep LOCAL layers
  native dense, keep prefill dense, keep canvas commits dense, and disable any
  optional `LOCAL_KV_BUDGET` or LOCAL sparse consumer for this study.
- The native LOCAL window is `(1023, 1023)`, but LOCAL sparsity is out of scope.
- vLLM cache pages and physical attention tiles are different concepts. The
  current vLLM cache-page setting may be `32`, while the selector and receipts
  use `64`-token physical KV tiles. Do not conflate them.
- Preserve the official adaptive sampler, proposal/acceptance rule, re-noising,
  self-conditioning, temperature schedule, EOS/stopping behavior, canvas size,
  maximum calls, thinking setting, and output budget from the matched frozen
  protocol. Do not add hidden forwards or replace native stopping by a fixed
  step loop.
- Preserve the existing discovery/refresh clock, first-call behavior, carry
  policy, progress trigger, sticky retention, protected/sink/current-canvas
  tiles, structural masks, GQA/cache semantics, mask format, and fixed budget.
- The selector must return the same kept/skipped mask or index-list interface
  that the current adapter consumes. The later sparse attention executor must
  receive exactly the same type of map as before.

The relevant implementation surface is:

- `experiments/numerical_qk_reuse/vllm_adapter.py`: vLLM adapter, selector
  dispatch, canvas/refresh lifecycle, and current `mage`/`method` arms.
- `experiments/numerical_qk_reuse/integration.py`: inherited online projected-V
  routing state and reuse lifecycle.
- `experiments/numerical_qk_reuse/reference.py`: small trusted routing oracle.
- `experiments/numerical_qk_reuse/cached_executor.py`: CUDA routing/output
  executor and accounting.
- `experiments/numerical_qk_reuse/v31_fa4_observe.py`: FA4 observation path.
- `experiments/numerical_qk_reuse/v31_dp_chunked.py` and
  `v31_logit_stats.py`: existing optimized statistics paths.
- `experiments/numerical_qk_reuse/v31_local_sparse.py` and
  `v31_local_kernel.py`: LOCAL implementation; do not activate or redesign it
  for this experiment.
- `scripts/v31_vllm_paired_bench.py`: matched generation runner.
- `scripts/v31_paired_official.py`, `scripts/v31_ruler_compare.py`,
  `scripts/v31_score_aime.py`, and `scripts/v31_score_humaneval.py`: official
  binding/scoring and comparison tools.
- Existing tests under `tests/test_numerical_reuse_*.py` and
  `tests/test_v31_*.py`.

First inspect the latest handoff and frozen receipt to identify the active
headline selector and its effective settings. The branch has two relevant
implementations: the current fixed-budget qblock selector is exposed through
the adapter's `arm='mage'` path and is called `qblock_max`/`lean` in the v31
documents; the older `arm='method'` path contains the online projected-V
selector. Do not infer semantics from the word `mage`. Record the exact active
control, selector, budget, refresh clock, carry, sticky value, and layer scope
before changing code. Keep the active control unchanged and retain any existing
attention-mass and all-kept controls.

## Scope of the code change

Modify only the block-selector logic used when an initial map or refresh map is
created. The existing reuse mechanism and the actual sparse attention consumer
must remain unchanged.

At an initial/refresh call, the implementation may compute and cache compact
statistics needed to choose a map. The model's actual attention output at that
call must follow the existing path. For V1/V2 in particular, the projected
routing state is internal selector state only; it must never replace the model
attention output. On later held calls, execute the selected physical blocks with
the current-step Q/K/V exactly as the existing consumer does. Never carry a
skipped block's old mass or old value into reused attention.

Do not change layer scope, budgets, sampler behavior, refresh timing, output
renormalization, protections, cache layout, or CUDA graph policy to rescue a
selector. Do not merge a peer branch. Do not rewrite the experiment around a
new local kernel.

## Baselines and named variants

At minimum, implement and report these arms using the same frozen budget and
reuse configuration as the current control:

1. `dense_full_fix51994`: native vLLM FULL dense baseline with the exact
   DiffusionGemma mask fix corresponding to upstream PR #51994.
2. `dense_piecewise`: matched correct dense PIECEWISE reference where the
   sparse adapter runs. Keep both dense references when the existing protocol
   requires them.
3. `allkept_fa4`: the adapter's FA4 path with every eligible block retained;
   this exposes adapter/consumer overhead independently of sparsity.
4. `current_v31_control`: the active qblock/cross-step selector identified from
   the latest frozen receipt, unchanged.
5. The existing attention-mass-only selector/control, if it is separate from
   item 4 in the active configuration.
6. `value_v1_online_discard_mass`.
7. `value_v2_online_preserve_mass`.
8. `value_v3a_singleton_delete`.
9. `value_v3b_greedy_exact`.
10. At least one separately named `value_v3b_approx_*` implementation only if
    it is needed for realistic full evaluation. Never present an approximation
    as exact greedy.

Do not add a projection-family or projection-dimension sweep. Use Gaussian32
for every value-aware variant, with the repository's existing deterministic
projection implementation, pinned projection seed, and recorded matrix hash.
If the effective existing Gaussian32 seed is `1729`, preserve it; do not
silently regenerate matrices per step, canvas, or variant.

## Inspect the current selector before implementing

Write a short source note in the result directory identifying the current
selector's actual formula, aggregation, scan order, budget convention,
mandatory blocks, tie rule, and whether its score uses mass, projected V, or
both. In particular, distinguish:

- MAGE-style mean mass over canvas rows and GQA groups;
- `qblock_max`/worst-row prefix-mass share;
- the inherited online projected-V risk in `reference.py`/`integration.py`;
- any C_gate or row-weight option used only at refresh.

Do not assume that a weighted projected-V mean is the same as an
attention-weighted block contribution. Keep the current selector as a control
even if inspection shows that it is mass-only.

## Shared Gaussian32 statistics

For every native KV head, use the same fixed projection across compatible arms:

$$
z_u = v_u R, \qquad R \in \mathbb{R}^{d_v \times 32}, \qquad
R_{ab} \sim \mathcal{N}(0,1/32).
$$

For query row `i` and physical KV tile `j`, let `s_iu` be the native scaled
attention logit after the existing structural mask. Define the stable tile
mass and within-tile projected mean:

$$
Z_{ij}=\sum_{u\in j}\exp(s_{iu}),\qquad
\mu_{ij}=\frac{\sum_{u\in j}\exp(s_{iu})z_u}{Z_{ij}}.
$$

Compute these with FlashAttention-style max/LSE reductions. Never materialize
unbounded exponentials or a full token attention matrix. `mu_ij` is an
attention-weighted within-block mean. It is not an unweighted pooled value and
not a globally normalized block contribution.

Use the current valid-KV reference scale `nu_i` and the current row
aggregation. The proposed formulas use the maximum over valid rows; if the
active control uses a different fixed aggregation, preserve it across all arms,
record it, and do not tune it per method.

## V1 and V2: faithful online selectors

Scan candidate blocks in the current scan order. Before candidate `j`, maintain
an internal normalized projected output `o_hat_i` and running mass `Z_i`.

$$
eta_{ij}=\frac{Z_{ij}}{Z_i+Z_{ij}},\qquad
rho_j=\operatorname{Agg}_i\left(
\frac{eta_{ij}\|\mu_{ij}-\hat o_i\|_2}
{\max(\nu_i,\epsilon)}\right).
$$

Higher `rho_j` favors retention. Both variants use identical ordering,
aggregation, quota, protected-block policy, and fixed budget.

For V1, a retained block updates:

$$
\hat o'_i=(1-eta_{ij})\hat o_i+eta_{ij}\mu_{ij},\qquad
Z'_i=Z_i+Z_{ij}.
$$

A skipped block leaves both values unchanged. V1 therefore models only retained
support in its routing state.

For V2, a retained block uses the same update. A skipped block updates only the
denominator:

$$
Z'_i=Z_i+Z_{ij},\qquad \hat o'_i=\hat o_i.
$$

If the implementation stores an unnormalized numerator, update it consistently
with unchanged normalized `o_hat_i`; changing only the denominator while
leaving an unnormalized numerator unchanged is wrong. V2 is only a routing
surrogate. It does not make the later masked attention operator mass-preserving.

Force the first valid support before a previous output can approximate a block.
Use the existing budget policy when available. Otherwise use a causal quota:
reserve mandatory slots, apply the original threshold while optional slots
remain, force skip when no optional slot remains, and force keep when all
remaining blocks are needed to fill the quota. Calibrate/freeze thresholds only
with the existing development procedure. Report forced-keep and forced-skip
rates. Do not compute a state-dependent score trajectory and then replace it by
an unrelated retrospective top-k mask while calling it online selection.

## V3: full-support selectors matching masked attention

At each initial/refresh selection, obtain compact statistics over the complete
eligible support. The reference projected output must come from
attention-weighted projected values; do not compute a full-dimensional PV just
to project it afterward except in a separate diagnostic.

For each valid block:

$$
alpha_{ij}=\frac{Z_{ij}}{\sum_l Z_{il}},\qquad c_{ij}=alpha_{ij}\mu_{ij},\qquad
O_i=\sum_j c_{ij}.
$$

For retained set `S`, ordinary masked attention in sketch space is:

$$
O_i(S)=\frac{\sum_{j\in S}c_{ij}}{\sum_{j\in S}alpha_{ij}}.
$$

The fixed-budget objective is:

$$
F(S)=\operatorname{Agg}_i\left(
\frac{\|O_i(S)-O_i\|_2}{\max(\nu_i,\epsilon)}\right).
$$

Translate `|S|=k` through the existing mandatory/protected-block convention.
Always-dense support contributes to both reference and retained statistics even
when it is not charged against the optional prefix budget.

### V3a: singleton deletion ranking

For every removable block, compute its full-support singleton deletion score:

$$
d_j=\operatorname{Agg}_i\left(
\frac{alpha_{ij}\|\mu_{ij}-O_i\|_2}
{(1-alpha_{ij})\max(\nu_i,\epsilon)}\right).
$$

Retain the `k` largest scores, respecting mandatory blocks and row-support
constraints. Compute these scores once; do not recompute them after each
deletion. A deletion that leaves a valid row without support is inadmissible,
not a finite score obtained by denominator clamping.

### V3b: cumulative backward greedy pruning

Start with all valid blocks retained. Precompute:

$$
g_{ij}=c_{ij}-alpha_{ij}O_i,
$$

and maintain:

$$
A_i(S)=\sum_{j\in S}alpha_{ij},\qquad
r_i(S)=\sum_{j\in S}c_{ij}-A_i(S)O_i.
$$

Then `O_i(S)-O_i = r_i(S)/A_i(S)`. For each removable candidate, evaluate the
resulting cumulative error:

$$
G_j(S)=\operatorname{Agg}_i\left(
\frac{\|r_i(S)-g_{ij}\|_2}
{[A_i(S)-alpha_{ij}]\max(\nu_i,\epsilon)}\right).
$$

Remove the candidate with minimum `G_j(S)`, update `A` and `r`, and repeat until
the optional budget is reached. Do not recompute QK or projected PV in this
loop. This is a greedy heuristic and is not a global subset solver; it need not
decrease the error relative to the previous set at every step.

Implement a trusted small exact reference first. For production, a GPU
parallel candidate reduction, a shortlist followed by exact cumulative scoring,
or a bounded batch/drop-and-refine implementation is acceptable only as a new
named approximation. Measure candidate count, selection time, memory, and
quality loss against exact V3b. Do not assume submodularity or use stale-score
lazy-greedy guarantees without proving the relevant condition.

## Kernel and execution requirements

Optimize selector/discovery statistics without changing their definitions:

- Reuse `v31_fa4_observe.py`, existing stable reductions, Gaussian projection
  code, and compatible Triton/CUDA paths where possible.
- Avoid full token attention matrices, Python loops over every candidate,
  per-candidate CPU/GPU synchronization, and host-side mask construction.
- V1/V2's sequential state dependency may be implemented as a GPU scan or a
  fused/batched kernel, but its decisions must match the exact reference.
- V3a should reuse one full-support summary pass.
- V3b must use cached `alpha`, `c`, `g`, `A`, and `r`; each greedy iteration
  scores candidate removals from these small summaries.
- Keep the existing later FA4 sparse consumer unchanged. Do not substitute a
  projected output, a dense-shaped masked PV, or a cached old output for the
  model's native current-step BF16 attention output.
- The initial/refresh call may remain the existing exact observation/output
  call. The value-aware logic records a map for later calls; it must not turn
  the selection step into a second model forward.

## Correctness gates before end-to-end runs

Add focused tests and run the relevant existing tests. At minimum verify:

- V1 and V2 match a simple sequential reference under identical decisions.
- V1 excludes skipped mass; V2 includes skipped mass but leaves its normalized
  routing output unchanged on a skip, including when the running max changes.
- The first valid support, strict threshold ties, mandatory blocks, partial
  tiles, empty rows, invalid scores, GQA mapping, and fixed-k quota are correct.
- Identity projection agrees with a direct full-dimensional formula reference.
- V3a and V3b match direct masked-attention recomputation in sketch space.
- Optimized V3b equals exact V3b when approximation features are disabled.
- Identical masks produce identical downstream outputs regardless of selector
  implementation.
- The following equal-mass case is preserved as a regression test:

  `mu_A = mu_B = a`, `mu_C = b`; skip B and retain C. V2's internal state can
  end at `(2a+b)/3`, while ordinary reused attention over `{A,C}` produces
  `(a+b)/2`. The test must distinguish these values and must not add executor
  compensation.

Run CPU tests first, then one-cell H100 smoke tests for every arm. Check finite
outputs, exact scorer identity, selector/layer receipt, physical tile counts,
no unexpected CUDA graph capture, and no LOCAL routing.

## Evaluation protocol

Use the existing official manifests, prompt hashes, model revision, scorer,
sampler settings, and matched-cell binding. Do not rebuild prompts or change
decoding to make a selector look better. Primary target suites are the previous
weaknesses of the current method:

- RULER official v33 confirmation pool: 13 tasks x 15 items at 32K, 64K, and
  128K (585 cells), thinking off, official RULER scorer.
- LongBench-v2 `0shot_think`: the current official 503-item pool, thinking on,
  the frozen 16,384-token total generation cap and current truncation/
  `max_model_len` contract, official scorer.
- HumanEval: the current official 164-task pool, thinking on, frozen 8,192-token
  generation cap, official `check_program`/pass@1 scorer.

Use the same seed schedule as the corresponding frozen baseline pool. Do not
invent a new seed or use answer labels to select variants. If a small
development subset is needed for thresholds, reserve it before running the
official target pool and report it separately. Freeze thresholds and all
selector parameters before the target run. Evaluate every viable requested
variant and keep failed/negative arms in the report.

Run one worker at a time on the H100 with resumable attempt/shard directories.
Run an audit pass with detailed counters and a separate clean timing pass with
routing traces/scoring instrumentation disabled. Confirm that the timed pass
does not capture new CUDA graphs. Never mix diagnostic timing with headline
latency.

## Required measurements and result files

For every arm, benchmark, and pooled matched comparison, report:

- official accuracy and the per-item/per-seed counts used by the scorer;
- total denoising calls `N`, canvases `C`, `N/C`, output length `T`, and `S/N`;
- GLOBAL physical eligible/kept/skipped tiles and sparsity;
- LOCAL physical sparsity (expected to be zero in this experiment);
- count-weighted overall decoder-attention sparsity with its denominator;
- end-to-end request time `W` including prefill and decode, plus dense/sparse
  speedup `W_dense / W_sparse`;
- decode-only time `S` and speedup, with the same matched cell set;
- initial/refresh/held-call counts, carry/reselection counts, forced-keep and
  forced-skip rates, fallback/invalid-row counts, and C_gate counters if the
  existing control exposes them;
- selection/discovery latency, peak memory, candidate evaluations, and the
  amortized cost under the actual refresh/reuse schedule;
- retained attention mass, mask overlap, fresh-mask versus reused-mask error by
  distance from refresh, and full-dimensional attention-output error measured
  offline after selection. Full-dimensional diagnostics must not be selector
  inputs.

Use speedup only as a measured ratio on the same H100, same cells, same seeds,
same runtime pins, and same warm-up protocol. Never infer speed from a requested
budget or a skipped-tile fraction. Keep the historical AIME V31 `0.951x`
end-to-end result as a diagnostic reference only; it is not evidence that a new
selector is faster.

Write the new study under a new dated directory beneath `results/`, containing
at least:

- `README.md` with setup, arm definitions, provenance, and limitations;
- frozen `config.json` and source/model/manifest hashes;
- `receipt.json` proving the selector, budget, layer scope, projection identity,
  kernel path, and counters actually used;
- per-arm sanitized records and aggregate tables;
- `summary.md` with all requested metrics and paired comparisons;
- `final_complete.json` only after every planned arm/cell is complete and
  validated.

Do not commit prompts, gold answers, token IDs, credentials, raw completions,
or unredacted generated text. Keep raw generations in an ignored user-owned
run directory. Update `HANDOFF.md`, `docs/DECISIONS.md`, and
`docs/RESULTS_LEDGER.md` only for complete, protocol-identified results; label
diagnostics, calibration, partial runs, approximations, and failed attempts.

## Completion standard

Continue fixing and refining until all of the following are true:

1. The selector implementations pass mathematical and integration tests.
2. The receipts prove the intended selector and unchanged reuse contract ran.
3. Every requested viable variant has matched dense/all-kept comparisons on the
   target suites, or the reason for a blocked arm is recorded precisely.
4. Clean timing and audit passes are separate and complete.
5. The final report distinguishes accuracy, full-dimensional error, physical
   sparsity, selection overhead, per-step cost, decode time, and end-to-end
   time, with no claim stronger than the evidence.
6. Source changes, tests, frozen configs, and the final result summary are
   reviewable from the repository and the final commit hash is recorded.

If a run fails, preserve the failed attempt, diagnose the cause, and relaunch
into a new attempt directory. Do not overwrite frozen evidence or silently
relax the protocol.
