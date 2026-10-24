# Native end-to-end selector screen — 2026-10-24

The user requests implemented, theoretically motivated selectors and actual
accuracy/native-trajectory comparisons against V31 and corrected dense. The
user explicitly removes campaign GPU time/forward ceilings. This new frozen
screen does not resume any paused campaign or change the nine blocked inputs.
One worker uses one idle H100. A four-hour per-process hang timeout is operational
protection, not an allocated-study budget or permission to expand the cohort.

## Scope frozen before generation

Six short identities (AIME18/9/13; HumanEval124/144/86), seeds42/43:12 cells.
One original185546-token LongBench identity, seed42:1 cell. Full original output
caps8192/16384 and private token identities stay fixed. These are previously
exposed diagnostic inputs, not a fresh benchmark. Long quality remains a single
item; a wrong/capped dense answer cannot establish quality preservation.

Nine arms: corrected native FULL, native PIECEWISE, matched layer29 all-kept
paged sparse FA4, inherited optimized V31, current-error fixed4, response guard
Q-direction, margin-weighted bias, mass-certified reuse, and fixed4 hash sham.
V31 preserves its all5-GLOBAL/K128-short/K8192-long configuration and provenance;
it is an external practical reference, not physically matched to layer29-only
methods. Current-error/sham share the new selector's state-specific bank, scope,
guards and deletion budget. Different trajectories can change future feasibility
and physical work; whole-request work must be reported rather than nominal K.

No restoration is used in evaluated requests. The preceding qualification found
identical complete model-state/logit/RNG/count fingerprints but one256-token
output-list discrepancy. That restored output path remains unqualified. Fresh
from-prompt requests avoid it. Every arm's first complete cell is a smoke gate:
warm full request, audit full request, clean full request, exact output hash and
N/C/T agreement. A failure stops that worker before the next cell. No sparse
method uses lookahead, extra model forwards, forced continuation or altered RNG.

## Theory and implemented hypotheses

For attention o=sum p_j v_j and deleted mass m with contribution c, the exact
current deletion bias is e=(m o-c)/(1-m). All retained attention is recomputed
using current Q/K/V. Prefix/canvas boundary and all current-canvas tiles stay
kept. Native LOCAL, prefill, commit, proposal/acceptance/re-noising, temperature,
self-conditioning, stopping thresholds/cap and EOS/finalization are unchanged.

Each of32 routing units (16heads × two Q128 blocks) enumerates all wholly-prefix
single-KV64 deletions when B≤12. For larger B, at most12 deterministic mass,
position and hash proposals are used. A unit deletes exactly one tile if any
candidate has worst-row whole mass≤0.2 and relative output bias≤0.1; otherwise
that unit is all-kept. Renormalization≤1.25 is the same mass guard, not an
independent safety result. All scores share this feasible set. Units with fewer
than4 alternatives are labeled non-identifying; their execution is exploratory,
not a retrospective passed ranking gate. This explicitly refines the prior
four-unit bank, which had only two feasible supports, without relaxing guards.

1. **current_error_fixed4:** minimize worst-row relative ||e||, refresh every4
   denoising calls after one dense bootstrap, hold support between refreshes.
2. **response_guard_qdirection:** use ∂o/∂q=scale Cov_p(v,k), evaluated on
   current Q/K/V in the previous observed query-motion direction; minimize the
   worst normalized ||J_S u||. Record sparse-minus-dense residual sensitivity
   separately. Reducing local response could reduce perturbation propagation if
   the rest of the native update is stable. This is NOT a proven full-model
   contraction, a future-motion predictor, or evidence that K/V stop changing.
   Current K/V are used; their future motion remains unmodeled. No old K/V or
   attention output is cached. A fixed coordinate probe is used without motion.
3. **margin_guard_weighted_bias:** minimize max_i w_i||e_i||/||o_i||, with
   w_i=1/max(previous native top1 margin,0.1) on previously stable positions;
   use all positions if that protected set is empty. A conditional downstream
   Lipschitz bound makes smaller perturbations relative to a small decision
   margin a plausible protection objective. The unknown downstream Jacobian is
   not estimated: this is a named local surrogate, NOT the proposed exact
   MarginGuard influence oracle or a certified native-margin bound. It may
   protect wrong decisions, so task accuracy is mandatory.
4. **validity_reuse_mass_certificate:** use current-error selection and refresh
   at most every4 calls, or earlier when a retained-mass lower bound falls below
   0.8. Within one native canvas, wholly-prefix keys are fixed. If
   b=scale||q-q_anchor|| max_j||k_prefix,j||, prefix log-partition bounds shift
   by ±b. With CURRENT boundary/canvas logZ_t recomputed exactly:

   retained_mass >= (exp(logZ_kept_anchor-b)+exp(logZ_t)) /
                    (exp(logZ_prefix_anchor+b)+exp(logZ_t)).

   It accounts for changing current-canvas keys instead of silently assuming
   all keys fixed. Reset all certificates at every canvas transition. This is
   a mass certificate only, not output-bias, correctness or contraction safety.
   All-kept units are exactly valid. Prefix values are never substituted.
5. **sham_fixed4:** deterministic hash selection from the identical feasible
   unit bank; no native RNG draws. The sham is guard/budget matched, not claimed
   to satisfy current-error/mass pair tolerances at every state.

These local exact-attention calculations are intentionally unoptimized and all
selector computation is charged. Full QK/PV discovery is extra attention work,
not an extra model forward. Fewer calls alone cannot imply end-to-end speedup.
Yuhan's inherited reuse/V31, Gaussian32/value routing and C-gate remain separate
prior methods. No novelty or production readiness is claimed.

## Measurement and stopping rules

Each cell executes full warm→audit→clean requests with identical prompt/seed,
settings and original sampler. Audit and clean output hashes, N, C and T must
match. Actual native state flags/entropy and per-canvas call intervals are
recorded during audit; no exact compiled-acceptance claim is made. Clean timing
has synchronized request/prefill/end boundaries, no audit entropy/trace/event
instrumentation. New clean captures/JIT invalidate timing rather than becoming a
speed result. Selector operations required for the method remain in clean W.
Measure physical GLOBAL/LOCAL eligible/kept/skipped rectangles, selector refresh,
held/carry, discovery event time separately. Rectangle counts are not CUDA CTAs.

Official AIME exact match, sandboxed HumanEval pass@1 and official LongBench
extraction use private full completions. Report paired item-clustered bootstrap
intervals across the six short identities; masks/seeds are not independent tasks.
Long single-item outcomes are descriptive. No formal noninferiority claim.
No accuracy-based retuning or item replacement in this frozen attempt. Source,
measurement, mask-scope or audit/clean mismatch stops the affected worker; other
independent controls may still complete. Promotion is evaluated only after
complete matched scored evidence; negative/mixed effects remain published.

Implementation/tests, model/runtime/scorer/source hashes, public identity,
launcher receipts, sanitized records, analyses and private ignored generations
are separate artifacts. Frozen files are never edited after launch. Any repair
uses a new immutable attempt and source checkpoint. No push is requested.
