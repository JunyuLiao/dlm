# Why these selectors are plausible, and what they do not prove

This document explains the frozen implementation; it does not change a score,
threshold, budget, input or promotion rule. No new outcome is used for tuning.

Let p=softmax(sQKᵀ), o=Σp_jv_j. Delete a prefix set D with mass m=Σ_Dp_j and
contribution c=Σ_Dp_jv_j, retaining every current-canvas and boundary position.
Renormalized sparse output is o_S=(o-c)/(1-m). Therefore

    e_S=o_S-o=(m o-c)/(1-m).

Current-error selection minimizes the implemented worst-row normalized ||e_S||.
The mass≤0.2 guard bounds renormalization by1.25. It does not bound future model
states, correctness or time to stopping. A low-error current intervention may
still change sampling/acceptance/self-conditioning and future argmax histories.
That issue motivated this study and remains a possible negative outcome here.

For a query perturbation u with current K,V fixed,

    J_A u = s Cov_p(v,k) u,
    J_S u = s Cov_pS(v,k) u.

If dm and dc are directional derivatives, differentiating the deletion bias gives

    de = (dm o + m J_Au - dc)/(1-m) + (m o-c)dm/(1-m)².

The implementation records residual response de=(J_S-J_A)u separately and selects
by ||J_Su||. This distinction repairs the interpretation of the old residual-only
score. The observed direction is the previous query displacement, evaluated with
CURRENT K,V. Those K,V still change across denoising calls, especially inside the
canvas. This method does not estimate their future motion, guarantee that the
previous direction predicts the next, or imply full native-update contraction.
Full Q/K/V directional diagnostics remain a different oracle study; no old K/V
is retained or substituted in this implementation.

Suppose, conditionally, the native update F near a trajectory is contractive with
constantρ<1 and downstream sensitivity to attention bias is bounded byL. Then a
state-error recurrence obeys d_next≤ρd+L||e||. If attention's query dependence is
an appreciable part of that update's sensitivity, a lower local response may
reduce a source of instability. Neitherρ norL is certified here. Even a smaller
state error does not establish fewer calls, because native stopping jointly
requires stable predictions and low whole-canvas entropy. The experiment measures
those native predicates and complete task outcomes instead of assuming success.

For reference winner a, a downstream perturbation δz leaves its argmax unchanged
if z_a-max_b≠a z_b exceeds the worst adverse logit displacement. A sufficient
bound is margin>2||δz||∞. If downstream logits are locally L-Lipschitz in the
attention output, margin>2L||e|| is sufficient. The implemented margin-weighted
surrogate replaces the unknown position-wise L by a common factor and weights
attention error by inverse PREVIOUS native margin on previously stable positions.
This gives a reason to prioritize vulnerable decisions, but no certified bound:
the true downstream layer norm/MLP/residual dynamics and current margins differ.
Protecting a wrong decision is harmful; official task quality is necessary. This
is not the proposed exact MarginGuard influence oracle and is named accordingly.

For reuse, native committed-prefix keys stay fixed within one canvas. Let
b=s||q-q₀||max_prefix||k_j||. By Cauchy-Schwarz, each prefix logit changes by at
mostb, hence Z_prefix(q)≤exp(b)Z_prefix(q₀) and
Z_kept_prefix(q)≥exp(-b)Z_kept_prefix(q₀). Recompute the CURRENT kept tail's
partition Z_tail(q) exactly; do not assume current-canvas keys fixed. Then

    retained_mass(q) ≥ [exp(-b) Z_kept_prefix(q₀)+Z_tail(q)] /
                      [exp(b) Z_prefix(q₀)+Z_tail(q)].

This certificate is conservative and can trigger excessive refresh, increasing
overhead without improving task quality. It only certifies retained mass under
the stated within-canvas prefix assumption. It is not a response-gain or
accuracy certificate. Resetting at each canvas avoids carrying its assumption
through a cache/prefix transition. Current values are always consumed by FA4.

Single- and multi-tile output identities, query directional derivatives and the
retained-mass inequality are independently tested against explicit recomputation
on CPU. Those tests establish implementation algebra, not predictive validity.
Sparse methods use no extra model forward, altered stopping or native RNG draws;
their full selector QK/PV and management costs remain part of request time.

The strongest outcome this exposed small screen could motivate is a separate,
bounded confirmation protocol. Fewer calls with worse scores rejects the tested
objective. Better task fidelity without fewer calls is a safety lead. A locally
distinct score without matched future benefits is not mechanism validation.
Native/all-kept consumer differences and the V31 scope/work mismatch must remain
visible. Prior Yuhan reuse, Gaussian32/value-direction routing, C-gate, V31/V2 and
negative studies are not renamed as contributions of this screen.
