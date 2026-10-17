# 2026-10-17: evaluated full-value selection and reuse methods

This is the publication record for the three-direction study started on
2026-10-15 and completed on 2026-10-17. It supplements, rather than relabels,
the [October 15 V1/V2/V3 publication](../research_week_20261015/01_VALUE_AWARE_METHODS.md).
Read [results and statistics](02_RESULTS_AND_STATISTICS.md) and
[integration and reproduction](03_INTEGRATION_AND_REPRODUCTION.md) with it.

Yuhan developed the inherited V31 cross-step-reuse baseline. Junyu's earlier
Gaussian32 router and query-sensitivity protection are distinct method families.
The bounded full-V selectors and reconstruction/trust variants below are this
study's later modifications. Residual caching itself is not a novelty claim.

## Shared notation and attention operator

One selection unit is one query head and one Q128 query tile; a KV block has
64 tokens. Let p be the current full-support softmax probabilities. For row i
and eligible prefix block b, define

```text
m_ib = sum_{j in b} p_ij
c_ib = sum_{j in b} p_ij * v_j
D_i  = sum_j p_ij * v_j
S_i(A) = (c_i,mandatory + sum_{b in A} c_ib)
         / (m_i,mandatory + sum_{b in A} m_ib)
```

V has its full 512 dimensions per head. Mandatory boundary/canvas blocks are
included in every evaluated support. Their mass is not silently dropped.
All numerator and denominator changes are recomputed together: a sparse
consumer renormalizes over its selected support.

`mass_anchor` is the actual inherited V31 support, including its sticky and
progress lifecycle. Fresh candidate mass rank uses maximum valid-row
prefix-normalized block mass. It does not use sequence-position neighbors.
For head h, W_h is its slice of the model's real attention O projection.
Single-head selection evaluates `||W_h (S_i-D_i)||²` using `G_h=W_h^T W_h`.
This retains all V dimensions; it is not a rank-32 sketch. Different heads are
selected independently, so this is not a joint cross-head optimum. Whole-layer
replay measures signed head sums after the full O projection.

## Complete evaluated-method catalogue

“Operator” means saved-state attention replay, “decision” means identical-state
model suffix/logit/sampler diagnostics, and “native” means the completed official
four-workload accuracy study. These are different levels of evidence.

| Recorded name(s) | Rule | Furthest evaluated stage and disposition |
|---|---|---|
| `mass_anchor`, `current_v31_control` | Inherited fixed-budget attention-mass map and native FA4 output. | Operator, decision and native reference. |
| `full_swap2`, `full_swap8` | Mass-qualified bounded swaps scored with full-dimensional V and true current dense output. | Cap 2: initial operator screen. Cap 8: whole-layer, decision and native. |
| `online_swap2`, `online_swap8` | Full-V online preserve-mass risks rank additions; the same full-information joint guard admits swaps. | Cap 2: initial operator; cap 8: whole-layer and decision. Not promoted to native. This is a hybrid, not historical V2. |
| `lifetime_swap2`, `lifetime_swap8` | Current loss plus a predicted next-state loss using the preceding query displacement. | Initial operator/prediction screen; rejected. |
| `full_cgate_swap8` | Full swaps with causal row weights from preceding acceptance/change/reversal statistics, clipped to [1,4]. K is unchanged. | Decision diagnostics; no consistent advantage over neutral/shuffled weights. |
| `shuffled_cgate_swap8` | Same weights shuffled across rows, same K. | Decision negative control; no native run. |
| `correction_residual`, native `residual` | Add cached anchor dense-minus-sparse output to the current sparse output. | Operator, decision and native mechanistic baseline. Changes output consumer. |
| `correction_moments`, runtime `moments` | Merge current selected numerator/mass with cached omitted numerator/mass, with consistent softmax scaling. | Operator and decision; rejected for harmful tails/acceptance behavior. No native accuracy result. |
| `correction_tangent` | First-order update of cached omitted moments using the last causal query displacement; fall back to cached positive moments when necessary. | Initial operator screen only; no native result. |
| `residual_direction_guard`, decision `correction_direction_guard` | Direction-dependent coefficient on the cached whole-layer residual. | Operator, decision and native. |
| `residual_norm_guard`, decision `correction_norm_guard` | Shrink residual correction by the magnitude of selected-output movement. | Operator and decision ablation; largely suppresses correction on short tasks. No native run. |
| `full_swap8_direction_guard` | Full swaps plus a residual/guard cache constructed for the changed support. | Decision and native combined arm. |
| `native_control` in suffix reports | Native sparse consumer with its numerical behavior, compared on the identical state. | Decision reference; not a separate new native method. |
| `same_mask_numerical_floor_vs_native` | Same-mask diagnostic implementation versus native consumer. | Numerical-floor control; has a different reference and must not be ranked as a dense-reference treatment. |
| `dense_full_fix51994` | Native FULL decode path with the exact PR #51994 mask fix. | Fresh native headline substrate reference. |
| `dense_piecewise` | Dense PIECEWISE path. | Fresh native matched-substrate reference. |
| `allkept_fa4` | All-kept sparse FA4 consumer. | Fresh native consumer reference. |

Exact greedy deletion remains excluded. Historical V1, rank-32 V2, V3a,
approximate batch8, mass-gated, intersection and cached-refresh variants retain
their own definitions/results in the [October 15 catalogue](../research_week_20261015/01_VALUE_AWARE_METHODS.md),
[threshold/work study](../research_week_20261015/02_V2_THRESHOLD_CALIBRATION.md)
and [refresh study](../research_week_20261015/03_CHEAP_CACHED_REFRESH.md).
The new `full_swap8` must not inherit those older result numbers.

## Full-value swaps: exact implemented procedure

1. Start from the K-block V31 support A. The candidate pool is
   `A union fresh_top_mass(K+32)`.
2. Consider removing the 16 lowest-mass retained blocks and adding the 16
   highest-priority eligible outside blocks: at most 256 pairs.
3. For full swaps, addition priority is mass. For the online hybrid, priority
   is the full-V online risk described below. Both calculate each pair's true
   renormalized output and mean post-O squared error against current dense D.
4. Score all pairs at the unchanged anchor. Choose at most eight individually
   improving, disjoint pairs, with no repeated removal or addition.
5. Re-evaluate the combined support. Require strict mean-loss improvement,
   worst-row post-O error norm no greater than 1.02 times the anchor, and
   maximum per-row retained-mass loss no greater than 0.025 (plus numeric
   tolerances). Try fewer pairs if the joint proposal fails. A no-op is valid.
6. Preserve exactly K eligible prefix blocks. The cap is per head/query tile,
   not eight replacements for the entire layer. The observation call returns
   inherited dense output; subsequent held calls use native FA4 on the new map.
   The deployed map feeds the inherited sticky term at later refreshes.

The plain native arm uses uniform row weights. C_gate is a separately named
diagnostic variant. Full-information admission guarantees only an observed
local objective decrease, not later held-call fidelity, accuracy or shorter N.

## Online, lifetime and C_gate ablations

The online full-V proposal scan traverses physical KV order. Its denominator
advances on every block; its running normalized output advances only when the
block passes the risk threshold (0.021, with first-support protection):

```text
eta_i = block_mass_i / running_full_mass_i
risk_b = max_i eta_i * ||block_value_mean_i - running_output_i|| / V_RMS
```

These risks only rank candidates inside the mass-qualified pool. The final
guard still sees full information. This differs both from the historical
rank-32 variable-budget V2 and from a purely online selector.

The lifetime variant estimates one step using
`dO = sum_j p_j (v_j-O) * (scale * dQ dot k_j)` with the preceding query
displacement. Current and predicted-next losses have equal weight. Future
states only evaluate prediction; they never enter selection. The stored
omitted-moment tangent is clamped to one extrapolation step in replay.

Fixed-budget C_gate uses preceding-call statistics:
`w_i=clip(1 + not_accepted_i + argmax_changed_i + acceptance_reversed_i,1,4)`.
It weights the row loss before tile aggregation and never changes K. It is not
the earlier Gaussian32 risk-multiplier gate or a query-dependent support quota.

## Reconstruction and directional trust

At an observation cache dense output D0, selected output S0, residual r=D0-S0.
At a held call, raw correction returns `St+r`. Directional trust returns

```text
alpha = clip(1 - <W(St-S0), W r> / ||W r||², 0, 1)
output = St + alpha*r
```

Here heads are concatenated before the real O projection; cross-head signed
effects count. Zero residual gives alpha=0. Cache `u=W^T W r` and `d=||Wr||²`
at observation, then compute the held coefficient with a fused row dot/update.
There is no current omitted-key scan or extra held-call O projection.

This minimizes distance to historical D0 along r. It is not a certificate
against the unavailable current dense output or final-answer loss. The
[guard derivation](../V31_CAUSAL_VALUE_RESIDUAL_GUARD_20261015.md) states the
conditional oracle-regret result and its limitations. Norm-only trust replaces
the signed projection by `||W(St-S0)||/||Wr||`.

Omitted-moment reconstruction combines the current selected numerator/mass
with cached omitted numerator/mass under a common log-sum-exp scale; mandatory
tokens belong to the selected side. The native prototype needs a separately
counted retained-only QK observation to obtain selected LSE because the normal
consumer does not expose it. Canvas/support changes invalidate all such caches.

## Frozen experiment setup

| Setting | Value |
|---|---|
| Model | `google/diffusiongemma-26B-A4B-it`, revision `f7f5b7f5fa82ffc52addd066915886d497f5517b`; full model inventory is source-bound. |
| Device | One NVIDIA H100 80GB HBM3; one worker at a time. |
| Runtime | vLLM 0.30.0; Torch 2.13.0, CUDA 13.0; driver 590.48.01; pinned SM90 FA4 source hashes in protocol. |
| Scorers | Official suite scorers; MATH uses math-verify 0.9.0, latex2sympy2-extended 1.11.0, sympy 1.14.0. |
| Physical selector units | Q128 × KV64; 16 query heads, 512-dimensional values. |
| Native intervention scope | GLOBAL layer 29 only. Other GLOBAL layers retain V31; LOCAL stays dense. |
| Native sampler | Canvas 256; max denoising steps 48/canvas; confidence threshold 0.005; entropy bound 0.1; stability threshold 1; temperature endpoints 0.8/0.4; frozen per-input output limits/thinking. |
| V31 lifecycle | Call-1 selection, first-call carry, settledness refresh trigger 0.15, sticky log bonus 1.386. |
| Seed handling | Seed base 31; frozen cell/panel seeds; same input/seed in every candidate/control comparison. |
| Timing | Clean timing paused. Audit wall/generation durations include instrumentation and are not latency or speedup evidence. |

| Workload | Cells | Paired engine shards | Prefix budget | vLLM block / chunk | Max model length | Memory utilization |
|---|---:|---:|---:|---|---:|---:|
| AIME26 | 30 problems × 3 seeds = 90 | 10 | 27 KV64 blocks = 1728 tokens | 64 / 4096 | 9216 | 0.80 |
| LongBench-v2 thinking | 64 | 8 | 128 blocks = 8192 tokens | 32 / 16384 | 141312 | 0.80 |
| MATH | 100 | 10 | 27 blocks = 1728 tokens | 64 / 4096 | 16384 | 0.80 |
| RULER128K non-thinking | 13 tasks × 8 cells = 104 | 13 | 128 blocks = 8192 tokens | 32 / 16384 | 136192 | 0.85 |

The vLLM cache block/chunk settings are distinct from Q128 × KV64 routing
units and the native canvas. If fewer than K prefix blocks exist, all are kept.
Common mandatory/carry additions and dense initial/refresh phases are accounted
separately. A fixed prefix K does not imply a fixed whole-request sparsity.

## Comparison design, source and inference

Each input/seed cell runs in two fresh engines. Each engine runs V31, all four
candidates in Latin-rotated order, then V31 again. Candidate order reverses in
engine 2; engine launch order alternates by shard. All-method warmup is separate
from measured requests. Every candidate has two trials/cell and V31 four.
The estimand is candidate minus mean control bookends within engine, averaged
over engines and cells. This is repeated evaluation, not pass@2 or ensembling.
Three fresh dense/all-kept references run once/cell in separate engines.

Unchanged V31 exhibited within/across-engine variation. The new frozen design
retains every structurally valid outcome, including drift; it does not select
deterministic/favorable cases. Older independent-process and exact-parity
attempts remain held/failed under their own protocols. Runtime root cause is
unresolved. Separate-engine reference contrasts have that limitation.

Intervals use 20,000 fixed-seed bootstrap draws (1729): problem clusters for
AIME/LongBench/MATH, task clusters for RULER, plus a separate engine-shard-block
sensitivity analysis preserving both engine repeats. They are not simultaneous
confidence regions or noninferiority guarantees. Inputs were previously examined
development data. A full503 promotion required no observed score loss or larger
mean N on any workload, at least one score gain, and no >5% p95/cap regression.
No candidate passed; no full503 run was launched.

Method source: `79c994ca`. Frozen native execution: qualified parent `28714b4a`
plus coordinator/test overlay `b485866c`, config commit `09e19e21`. The full
[protocol](../../results/v31_value_decision_reuse_20261015/native_replicated_20261016_attempt001/protocol.json)
records 1,099 execution fingerprints, input bindings, model/runtime identity
and deployment recipe. Earlier operator/decision stages have their own source
pins and completion markers; later code is not substituted into old results.
