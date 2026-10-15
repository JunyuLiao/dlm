# 1. Value-aware V1/V2/V3 selectors with cross-step reuse

Publication checkpoint: 2026-10-15. This document covers method definitions,
integration, implementation changes and completed evidence. Read
[V2 calibration](02_V2_THRESHOLD_CALIBRATION.md) for threshold studies and
[cheap refresh](03_CHEAP_CACHED_REFRESH.md) for the separate stale-map direction.
Sanitized aggregates and original/exported hashes are in
[`results/research_week_20261015`](../../results/research_week_20261015/README.md).

## Provenance and fixed execution contract

Yuhan developed the inherited V31 cross-step-reuse baseline. Junyu's earlier
Gaussian32 value-direction router and query-sensitivity protection are separate
method families. V1/V2/V3 below are this study's integrations into that lifecycle.
V31 uses worst-row normalized attention mass (`qblock_max`); projected V is not
part of its score. The similarly sized normalized V magnitudes should not be
used to reinterpret it as this study's direction-aware selector.

All value arms route only the five GLOBAL layers, 5/11/17/23/29, with physical
Q128 x KV64 rectangles. LOCAL attention, prefill and canvas commits remain dense.
At initial/refresh selection, current V is projected to rank 32 with Gaussian
seed 1729. The resulting state selects a bitmap only. The model always receives
native FA4 attention on current original K/V; a selector's cached output never
replaces it. Held maps are reused, not old attention values.

The original LongBench protocol uses K=8192 tokens (=128 eligible prefix tiles)
for fixed-budget arms, call-1 selection, first-call carry, settledness trigger
0.15 and sticky log bonus 1.386. Protected tail/boundary tiles are separate from
the eligible prefix budget. Native 256-token canvases, acceptance, re-noising,
self-conditioning, stopping, temperature schedule and output parser stay fixed.

## Formulas and distinctions

For query row i and KV tile j, cache raw log mass
`ell_ij = log(sum_{k in j} exp(q_i k_k * scale))`, projected within-tile mean
`mu_ij`, and native-head RMS normalizer `nu_i`. Process candidates in physical
order with running log denominator L and normalized projected output o:

```text
combined = logaddexp(L, ell_j)
eta_i    = exp(ell_ij - combined_i)
rho_j    = max_valid_i eta_i * ||mu_ij - o_i||_2 / nu_i
keep j   iff first-support, mandatory, invalid fallback,
             or log(rho_j) + sticky * previously_kept_j >= log(tau)
on keep: o_i <- (1-eta_i) * o_i + eta_i * mu_ij
```

| Selector | Denominator/state semantics | Budget |
|---|---|---|
| `value_v1_online_discard_mass` | L advances only when a tile is kept; skipping leaves L and o unchanged. | One uniform GLOBAL threshold; no quota. |
| `value_v2_online_preserve_mass` | L advances even on skips; skipping leaves normalized o unchanged, including when the running maximum changes. | One uniform GLOBAL threshold; no quota. |
| `value_v3a_singleton_delete` | Full-support projected output; score each single deletion once, then prune with row-support protection. | Fixed eligible-prefix budget. |
| `value_v3b_approx_batch8` | Re-evaluate cumulative deletion scores between batches of eight, with support cleanup. | Fixed budget; explicitly approximate. |
| `value_v3b_greedy_exact` | Re-evaluate after every deletion. | Reference/retained diagnostics only; excluded from further generation. |

V2's preserve-mass recurrence is a routing surrogate. The later masked FA4
operator renormalizes over selected support and is not mass-preserving.

For V3, let alpha be full-support tile mass, c=alpha*mu, O=sum(c), and
g=c-alpha*O. For a retained set, A=sum(alpha) and r=sum(g). Removing tile j
has row error `||r-g_j|| / ((A-alpha_j)*nu)`. A deletion removing a live row's
last support is inadmissible. V3a ranks the singleton scores once; batch8
re-evaluates a cumulative objective. Singleton errors are not additive.

## Implementation and kernel changes included in this publication

| Component | Responsibility |
|---|---|
| `experiments/numerical_qk_reuse/v31_value_selectors.py` | CPU formula oracles, `Stats`, first support, mandatory tiles, dispatch; experimental mass anchors and fixed-budget hybrids. |
| `v31_value_kernels.py` in the same directory | Triton QK/projected-PV statistics, exact online recurrence, support-aware cached deletion and device-side hybrid ranking. |
| `v31_value_summary.py`, `v31_value_workspace.py` | Full-support summaries, bounded reusable scratch; no model-output substitution. |
| `v31_value_projection.py`, `v31_value_snapshots.py`, `v31_value_native_audit.py` | Projected V, private snapshots and audited phase/canvas accounting. |
| `vllm_adapter.py`, `scripts/v31_vllm_paired_bench.py` | GLOBAL lifecycle, native FA4 consumer, parameter wiring and receipts. |
| `scripts/v31_value_campaign.py` | Fresh immutable attempt launcher, environment isolation and source/runtime receipts. |

The previously uncommitted online kernel now scans at most 128 candidate tiles
per launch and carries the exact FP32 log denominator, output and support state
between chunks. This limits long-prefix kernel code/register pressure without
changing the sequential recurrence. Rank-32 projected statistics, chunked RMS,
workspace reuse and nonfinite checks avoid redundant large temporaries. GPU
ranking/swaps stay on device. Manifest hashing in the launcher now reads each
dataset manifest once instead of once per scheduled item. These are implementation
improvements; no new clean end-to-end speed claim follows from them.

Full-dimensional V is evaluated by the streaming offline diagnostic
`scripts/v31_value_full_dim_diagnostic.py`; the production Triton online state
is still rank 32. A full-information projected score is not true full-V error.

## Completed evidence and limits

Original eight-arm 503-item official audit, seed 1, frozen generation line
`44674c3a` and execution amendment `attempt005_no_exact`:

| Method | Correct/503 | N | GLOBAL sparsity |
|---|---:|---:|---:|
| V31 control | 268 | 140,693 | 83.16% |
| V1, tau=.027 | 247 | 120,301 | 80.04% |
| V2, tau=.027 | 266 | 126,072 | 84.11% |
| V3a | 253 | 137,212 | 83.08% |
| V3 batch8 | 242 | 171,652 | 84.97% |

V2 versus V31 had 228 both-correct, 38 V2-only, 40 V31-only and 197 both-wrong
items. Similar total accuracy therefore means balanced answer flips, not
equivalent outputs or trajectories. V2's paired interval was [-3.78,+3.18] pp.
It also used fewer absolute GLOBAL tiles, which motivated the separate work
calibration. Original full tables remain in
[`V31_VALUE_SELECTOR_RESULTS_20261009.md`](../V31_VALUE_SELECTOR_RESULTS_20261009.md).

Value information did not establish a reuse-era accuracy gain: a current-call
projected residual need not preserve future held-call output, downstream logits,
acceptance or stopping. Different supports can produce comparable task answers,
and lower local operator error need not improve native generation. Mask IoU,
retained mass, full-V error, map age, native N/C/T and paired correctness must
be assessed together; these observations do not prove a single causal explanation.

Later evaluated refinements remain separate variants:

| Variant | Evidence | Conclusion |
|---|---|---|
| `value_v2_online_jcgate` (historical name) | 256/503 vs V31 268; denser GLOBAL support, fewer calls. | No accuracy gain; timing paused. |
| `value_mass_gated_online_swap8` | 264/503 vs 268; paired -0.795 pp, CI [-4.97,+3.38]; mean N 249.86 vs 279.71. | Development gain did not survive full audit. |
| Full-information swaps, phase, fixed-budget C_gate, intersection | 15-item native development and/or frozen operator diagnostics. | No promoted full-panel improvement. |

Fixed-budget C_gate reweights ranking within the same query-tile K; it must not
silently increase K. Historical `jcgate` risk-weighted online arms have a variable
support and are not that balanced-budget method. Strict V31-intersection-V2
support is smaller than either input and is not a matched-work comparison.
Mass-gated/intersection helper APIs are experimental; explicit-mask CUDA parity,
mandatory-prefix cases, tie behavior and candidate-pool semantics require their
own qualification before a new integration. Do not equate registration with
validated deployment. In particular, the prototype pool uses a physical-prefix
restriction, not a general top-M attention-mass-neighbor construction.

## Integration and reproduction

Use DiffusionGemma revision `f7f5b7f5fa82ffc52addd066915886d497f5517b`, the
study's model inventory, vLLM 0.30.0, Torch 2.13.0+cu130 and pinned SM90 FA4.
Set `DLM_PYTHON`, `DLM_MODEL`, `DLM_MANIFESTS` and `DLM_CELLS` to authorized
local copies matching the chosen frozen protocol; none is supplied by Git.
Use a new dated output root under `results/` and an idle H100.

```bash
PYTHONPATH=src:. "$DLM_PYTHON" -m pytest -q tests/test_v31_value_selectors_independent.py
"$DLM_PYTHON" scripts/v31_value_campaign.py \
  --python "$DLM_PYTHON" --model "$DLM_MODEL" \
  --manifests "$DLM_MANIFESTS" --cells "$DLM_CELLS" \
  --result-root "$DLM_STUDY" --attempt "$DLM_STUDY/smoke/attempt001" \
  --arm value_v2_online_preserve_mass --threshold 0.027 --purpose smoke
```

The generic launcher above is the LongBench configuration (block32/chunk16384,
memory .85/maxlen141312). Do not use it unchanged for short suites. The local
threshold driver supplies the correct AIME/MATH geometry; its instructions are
in document 2. An exact historical rerun requires its frozen execution bytes,
not today's changed adapter plus an old control record. A new integration
needs CPU oracle tests, H100 parity, real native smoke/receipts, a new source
freeze, matched controls, full scorer binding and completion markers.

Audit receipts must show `value_output_path=unchanged_fa4`,
`value_state_role=selector_only`, GLOBAL-only scope, zero LOCAL skips, valid rows
and zero new request-time graph captures. Score the private completed generations
with `scripts/v31_score_longbench_official.py`; its `--cells` manifest binding is
mandatory. Sanitized output records alone cannot be rescored: completions and
gold are intentionally private. Reproduce analysis from public aggregate files,
or rerun official scoring with the authorized private inputs.

Additional published tools: full-V and next-iteration diagnostics, supplementary
LongBench reporting, AIME/RULER campaign launcher and deterministic pool/queue
builders. The old quick scorer is historical (string-based MATH checking);
use `v31_score_math500_bound.py` for the current MATH500 study. Reference checkouts
under `reference/` are external dependencies, not bundled peer repositories.
Do not resume a paused queue or exact greedy as part of reproduction by default.
