# Active-Voter BLASST: implementation and H100 evaluation

## 1. Decision

**No-go. Active-Voter BLASST is not a production-worthy approximation, and
the conditional packed Hopper kernel is not authorized.**

The raw `tau=32` policy has enough theoretical packing opportunity, but it
fails the quality gate by a wide margin. Conservative policies can make local
errors small, but only by reducing candidate work to far below the required
20%. Small local errors also accumulate during denoising: the best sampled-
state margin (`margin=6`) fell to 93.32% masked-token top-1 agreement on a
teacher-forced trajectory step and only 61.63% final agreement with BLASST on
the initially masked positions of the free-running trajectory.

No single policy satisfies all quality and performance conditions. In
accordance with the prescribed gate, this work stops before exception-list
integration, WGMMA/TMA implementation, PTX/SASS claims, or Nsight Compute
profiling.

## 2. Scope and experimental setup

| Item | Setting |
| --- | --- |
| Model | `GSAI-ML/LLaDA-8B-Instruct` |
| GPU | NVIDIA H100 80GB HBM3, compute capability 9.0 |
| Dtype | BF16 model and attention inputs; FP32 online state |
| Attention | Bidirectional MHA, 32 query heads = 32 KV heads, head dimension 128 |
| Sequence | 4096 tokens |
| Physical tile | 128 query rows × 64 KV rows |
| Thresholds | Existing calibrated high/mid/low BLASST schedule |
| Sampled states | Real nested masks at 90%, 50%, and 15% remaining |
| Full trajectory | 16 denoising steps from approximately 90% masks to zero |
| Quality context | One deterministic 4096-token corpus context |
| Contribution corpus | 12 contexts × 3 target states × 32 layers × 32 heads; one query tile and all KV tiles per context/layer |

The contribution corpus contains 2,359,296 sampled physical tiles and
89,174,827 negative-voter row contributions from 751,141 raw `tau=32`
candidate tiles. It uses the previously collected real LLaDA Q/K/V trace, not
synthetic score distributions.

The quality experiment is a complete denoising reconstruction trajectory, not
a task benchmark. Consequently task accuracy is not reported. Final denoised
token agreement, per-step distribution metrics, and free-running divergence
are the applicable end-quality measurements here.

## 3. Implementation and semantic audit

### 3.1 Opt-in behavior

The existing BLASST path remains the default. Active-Voter behavior is enabled
only through the new row-masking arguments or the explicit evaluation flag:

```text
--blasst-row-masking
--row-variant full|output
--max-active-rows 32
--margins ...
--active-layers ...
```

There are two implementations:

1. A correctness-first PyTorch implementation in `blasst/flash_attention.py`.
2. A Triton diagnostic specialization in `blasst/triton_bidirectional.py` used
   to make full-model evaluation tractable. It implements the selected
   recurrence but still performs dense tile matrix products; it is not the
   proposed packed Hopper kernel and makes no speed claim.

### 3.2 Baseline physical decision

For row `i` and physical KV tile `j`, existing BLASST computes

```text
g_ij = local_max_ij - m_i
baseline_vote_ij = g_ij >= log(lambda_tile)
physical_keep_j = any_i(baseline_vote_ij)
```

A physically skipped tile changes no online state. For a retained tile,
Active-Voter uses a separately configurable threshold

```text
active_ij = g_ij >= log(lambda_tile) - margin
```

and applies row masking only when `1 <= sum(active_ij) <= tau`. Tiles above
`tau` fall back to ordinary all-row BLASST processing. This matches the
candidate workload that can potentially be packed.

### 3.3 Variant A: fully row-masked attention

This is the primary and mathematically consistent approximation. For each row,

```text
u_i = physical_keep_j and (active_ij or non_candidate_tile_j)
m'_i = u_i ? max(m_i, local_max_ij) : m_i
alpha_i = u_i ? exp(m_i - m'_i) : 1
P_ij = u_i ? exp(scores_ij - m'_i) : 0
l'_i = alpha_i * l_i + sum(P_ij)
O'_i = alpha_i * O_i + P_ij @ V_j
```

Thus an inactive row excludes the interaction from `m`, `l`, and `O`. The
final result is identical to dense attention with the same explicit row–tile
mask.

### 3.4 Variant B: output-only masking

The diagnostic variant updates `m` and `l` for every row of a retained tile,
but zeros `P @ V` for inactive rows. It therefore includes probability mass in
the denominator without the corresponding value numerator. The inconsistency
is visible empirically:

| Remaining masks | Fully masked top-1 vs BLASST | Output-only top-1 vs BLASST | Output-only logit KL |
| ---: | ---: | ---: | ---: |
| 90% | 90.20% | 91.48% | 0.0203 |
| 50% | 95.21% | 92.21% | 0.0966 |
| 15% | 96.33% | 87.88% | 0.4388 |

Variant B is retained only as a diagnostic and is not a candidate method.

## 4. Correctness validation

The new reference tests cover:

- all rows active, matching ordinary retained-tile BLASST;
- no rows active, matching the existing physical skip;
- mixed voters and per-row running-maximum changes;
- multiple retained/skipped tiles;
- causal invalid rows as a mask stress test;
- bidirectional future-token attention;
- partial query and KV tiles; and
- agreement with a dense reference using an explicitly materialized row–tile
  mask.

The H100 Triton diagnostic recurrence also matches the PyTorch reference on a
mixed-voter case and reports the expected 32 selected and 96 removed rows.
The final repository suite passes 83 tests (including 11 CUDA/Triton tests).

## 5. Quality results

### 5.1 Raw Active-Voter policy

The raw policy uses the physical threshold as the row threshold (`margin=0`,
`tau=32`).

| Remaining masks | Candidate retained work | Masked top-1 vs BLASST | Logit KL vs BLASST | Final hidden relative L2 |
| ---: | ---: | ---: | ---: | ---: |
| 90% | 28.82% | 90.20% | 0.0249 | 11.92% |
| 50% | 56.20% | 95.21% | 0.0248 | 25.34% |
| 15% | 77.78% | 96.33% | 0.0261 | 42.10% |

The approximation is most aggressive at low noise, exactly where its hidden-
state error is largest.

### 5.2 Attention output by layer, head, and token state

All 32 layers and 32 heads were measured for masked and revealed rows at each
target state. The table aggregates the per-layer/head relative L2 values.

| Policy | Masks | Row state | Mean relative L2 | P99 relative L2 | Mean cosine |
| --- | ---: | --- | ---: | ---: | ---: |
| Raw | 90% | masked | 6.99% | 26.29% | 0.99735 |
| Raw | 50% | masked | 22.22% | 52.65% | 0.98005 |
| Raw | 15% | masked | 38.80% | 79.14% | 0.94359 |
| Raw | 15% | revealed | 38.68% | 78.21% | 0.94475 |
| Margin 6 | 90% | masked | 0.00090% | 0.0177% | ≈1.0 |
| Margin 6 | 50% | masked | 0.00336% | 0.0526% | ≈1.0 |
| Margin 6 | 15% | masked | 0.00550% | 0.1029% | ≈1.0 |

For margin 6, Active-Voter-versus-dense attention errors are effectively the
existing BLASST-versus-dense errors; the incremental attention perturbation is
locally tiny. The trajectory results below show that locally tiny is not
sufficient for stable denoising.

### 5.3 Error accumulation across denoising steps

On a teacher-forced BLASST trajectory, every method receives the same token
state at a step. This separates incremental approximation error from
free-running state divergence.

| Policy | Mean step top-1 vs BLASST | Worst step top-1 | Maximum logit KL | Maximum final-hidden relative L2 |
| --- | ---: | ---: | ---: | ---: |
| Raw | 76.95% | 63.04% | 0.2214 | 48.61% |
| Margin 6 | 97.75% | 93.32% | 0.00173 | 9.60% |

Margin 6 clears 99% at the three isolated 90/50/15 states, but fails at 15 of
the 16 actual trajectory states. The worst step occurs near 17% remaining
masks. Its final-hidden relative L2 grows from 1.14% at the first step to 9.60%
at the last step.

### 5.4 Free-running final divergence

Each method independently chooses which tokens to reveal. Agreement is scored
on positions that were masked at the start of the trajectory.

| Method | Final agreement vs BLASST | Final agreement vs dense |
| --- | ---: | ---: |
| Dense | 56.93% | 100% |
| Existing BLASST | 100% | 56.93% |
| Raw Active-Voter | 49.88% | 52.38% |
| Margin-6 Active-Voter | 61.63% | 61.09% |

Existing BLASST already differs from dense, which is why incremental metrics
are reported separately. Relative to its actual BLASST control, neither
Active-Voter policy preserves the denoising result.

## 6. Are negative-voter contributions negligible?

No. For raw `tau=32` candidate tiles:

| Metric over removed negative-voter rows | P50 | P90 | P95 | P99 | P99.9 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Final normalized attention mass | 0.00503 | 0.0278 | 0.0409 | 0.0878 | 0.208 | 0.804 |
| `norm(P_tile @ V_tile)` | 0.0122 | 0.0862 | 0.155 | 0.489 | 1.77 | 26.19 |
| Relative output norm | 0.00727 | 0.0447 | 0.0664 | 0.153 | 0.472 | 1.259 |

The mean relative output norm is 1.84%, and 43.5% of removed contributions
exceed 1% of the current row output norm. These are not uniformly negligible
tails.

Summing contribution norms across removed KV tiles gives a conservative
per-row/layer accumulation bound with median 0.212, P90 0.960, P99 1.53, and
maximum 3.61 relative to the row output. Vector cancellation can reduce this
bound, but the measured full-model divergence shows that cancellation does not
make the policy safe.

The failure is structured:

- mean relative removed contribution rises from 0.70% at 90% masks to 2.95%
  at 15% masks;
- previously revealed, newly revealed, and stable-visible rows have mean
  relative contributions of 2.46%, 2.32%, and 3.09%, versus 1.28% for masked
  rows; and
- early layers 1–7 have several of the largest per-contribution means, while
  full-model hidden error continues accumulating through later layers.

The row gap is therefore not a reliable proxy for final normalized attention
mass or value contribution.

## 7. Training-free policy search

### 7.1 Conservative row-margin sweep

| Margin | Worst target-state top-1 | Candidate work at 90% | At 50% | At 15% |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 90.20% | 28.82% | 56.20% | 77.78% |
| 0.5 | 92.44% | 21.60% | 42.36% | 55.32% |
| 1 | 94.07% | 15.37% | 30.32% | 39.69% |
| 2 | 96.49% | 5.94% | 12.88% | 17.23% |
| 4 | 98.91% | 0.42% | 0.83% | 1.46% |
| 6 | 99.02% | 0.040% | 0.033% | 0.048% |
| 8 | 98.83% | 0.0025% | 0.0008% | 0.0012% |

`margin=2` is the practical Pareto knee: it retains nontrivial work but misses
quality. `margin=6` is the only policy to clear the isolated-state 99% gate,
but it removes essentially no work and fails the full trajectory.

### 7.2 Final-mass and V-aware oracles

The trace was also swept with selectors based on final normalized mass,
`mass × max_norm(V_tile)`, and exact `norm(P_tile @ V_tile)`. These are oracle
analyses, not online implementations.

The final-mass oracle is substantially better than the gap vote:

| Final-mass epsilon | Candidate work | Mean active rows | Mean removed relative norm | Removed rows above 1% |
| ---: | ---: | ---: | ---: | ---: |
| 0.001 | 3.38% | 18.3 | 0.040% | 0.004% |
| 0.003 | 11.88% | 16.0 | 0.14% | 0.82% |
| 0.010 | 33.38% | 12.5 | 0.58% | 19.6% |

At the required ≥20% work region, even this oracle admits a substantial tail
of large removals. The tested cheap V-max bounds remained below 2.4% candidate
work in their safer region. The exact-output oracle reached 16.8% work at its
largest tested threshold, still below the gate and with 38.5% of removed rows
above a 1% relative contribution.

The oracle result shows that final mass is a better statistic, but not enough
to authorize a training-free online predictor. Any online approximation would
also be weaker than the oracle and would add predictor/metadata cost.

### 7.3 Stage- and layer-aware policies

Masking only the first eight layers produced target-state top-1 agreement of
92.84%, 98.32%, and 97.77%. Masking only middle layers 8–23 produced 90.97%,
95.95%, and 96.01%. Neither protects quality. Stage-specific margins reduce to
the same margin trade-off: margins large enough for quality provide at most
about 1.5% candidate work at the target states and much less for margin 6.

Disabling first/last steps or additional sensitive layers cannot restore the
required 20% work coverage once the remaining enabled states use a quality-
safe margin.

## 8. Updated packing and performance gate

The raw mask retains the earlier favorable counterfactual packing result:

| Raw `tau=32`, `PACK_M=64` property | Result |
| --- | ---: |
| Weighted packing utilization | 79.05% |
| Candidate retained P@V work | 48.08% on baseline traces |
| Theoretical V-load factor | 5.25× |
| Extra recomputed QK | 2.60% of baseline QK |
| Complete-pack row share | 76.20% |

Those numbers are useful only if the approximation is quality-valid; the raw
policy is not.

The margin-6 policy is the nearest local-quality candidate, but it covers only
0.033–0.048% of retained P@V row work at the target states. Even assuming free
packing, zero metadata cost, and eliminating all of that candidate P@V work,
its upper-bound complete-attention speedup is below 0.03%, far short of 10%.
It also fails the full-trajectory quality gate, so detailed pack construction
would not change the decision.

| Required gate (one policy must pass all) | Raw | Margin 6 | Decision |
| --- | --- | --- | --- |
| Masked-token top-1 vs BLASST ≥99% over trajectory | Fail | Fail (worst 93.32%) | Fail |
| No material final-token regression | Fail | Fail (61.63% agreement) | Fail |
| Small accumulated hidden/logit error | Fail | Fail (9.60% final-hidden L2) | Fail |
| Packing utilization ≥70% | Pass counterfactually | Not material | — |
| Candidate work ≥20% | Pass | Fail (<0.05%) | Fail |
| Candidate V reduction ≥2× | Pass counterfactually | Irrelevant at this work | — |
| Extra QK ≤20% | Pass (2.60%) | Pass by upper bound | — |
| Estimated complete-attention speedup ≥10% | Plausible only counterfactually | Fail (<0.03%) | Fail |

There is no quality-valid policy for which Phase D packing needs hardware
authorization.

## 9. Hopper phase and production status

The following were intentionally **not** implemented:

- GPU exception count/scan/descriptor kernels;
- persistent packed WGMMA QK or P@V consumers;
- TMA descriptor construction and K/V movement;
- atomic or segmented packed-output accumulation;
- PTX/SASS WGMMA/TMA verification;
- Nsight Compute profiling; and
- production runtime flags or end-to-end speedup claims.

This is the required stop behavior, not missing validation. Building and
profiling those components cannot repair a failed approximation-quality gate.
The default BLASST kernel and its approximation semantics remain unchanged.

## 10. Direct answers

**Are negative-voter contributions negligible?** No. Their median relative
output norm is 0.73%, P99 is 15.3%, and some exceed the complete current row
output norm.

**Does removal accumulate into generation divergence?** Yes. Raw hidden-state
error reaches 48.6%; even margin 6 reaches 9.6% and only 61.6% final agreement
on initially masked tokens.

**Which training-free policy gives the best quality/work trade-off?** The
final-mass oracle dominates the raw gap vote analytically, and margin 2 is the
best simple online Pareto knee. Neither satisfies the combined gate. Margin 6
has the best isolated-state quality but negligible work and unstable full-
trajectory behavior.

**Does any valid policy retain enough rows for efficient Hopper packs?** No.
The packable raw policy is invalid; the locally safest online policy affects
less than 0.05% of retained work.

**Does an integrated WGMMA/TMA implementation improve complete latency?** No
integrated kernel was authorized, so no such claim is made.

**Production-worthy or no-go?** No-go.

## 11. Reproduction and artifacts

```bash
# Reference and Triton correctness
conda run -n ljy_dlm python -m unittest tests.test_active_voter -v
conda run -n ljy_dlm python -m unittest tests.test_blasst_triton_bidirectional -v

# Target-state margin screen
conda run -n ljy_dlm python eval/eval_active_voter_llada.py \
  --blasst-row-masking --ratios 0.90,0.50,0.15 \
  --margins 0,0.5,1,2,4,6,8 \
  --output outputs/active_voter/screening_4096.json

# Paired and free-running trajectories
conda run -n ljy_dlm python eval/eval_active_voter_llada.py \
  --blasst-row-masking --margins 0,6 --trajectory-steps 16 \
  --run-trajectories --paired-trajectory \
  --output outputs/active_voter/trajectory_accumulation_4096.json

# Contribution and oracle-policy analysis
conda run -n ljy_dlm python tools/analyze_active_voter_contributions.py \
  artifacts/veto_traces_h100_v1 \
  --output outputs/active_voter/contribution_policy_analysis.json
```

Primary machine-readable outputs:

- `outputs/active_voter/screening_4096.json`
- `outputs/active_voter/quality_trajectory_4096.json`
- `outputs/active_voter/trajectory_accumulation_4096.json`
- `outputs/active_voter/attention_raw_4096.json`
- `outputs/active_voter/output_only_diagnostic_4096.json`
- `outputs/active_voter/contribution_policy_analysis.json`

All numerical claims in this report are derived from those artifacts or from
the prior `outputs/exception_packing/phase_a_adjacent/summary.json` packing
analysis.
