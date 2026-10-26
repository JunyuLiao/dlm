# Why test held versus refreshed support?

For a fixed support S and current native operands,

    o_S(q,K,V) = softmax(sigma q K_S^T) V_S.

Holding the bitmap removes the discrete support-switch term from a local change
in attention. It does not remove Q/K/V evolution or native sampling/schedule
changes. A first-order decomposition for a refreshed operator includes the
Q,K,V differential at fixed support plus an additional change from switching
support. This is a local description, not a denoising contraction theorem.
Dense attention already has a fixed support within a canvas. Therefore held
versus refresh must be tested separately from held versus all-kept/FULL.

For fixed K,V, the query Jacobian is sigma Cov_p(v,k). Response minimizes its
full sparse directional action J_S(q_t-q_prev), while the residual response
(J_S-J_A)(q_t-q_prev) is reported separately. Changing queries, canvas keys,
values, subsequent nonlinear layers and sampling invalidate any inference from
this score alone to shorter native stopping. Prior forecasting/replication
failures and its already-existing fixed-four reuse are retained.

Deleting attention mass m changes the output by (m o - removed_contribution)/(1-m).
Mass<=.2 implies renormalization<=1.25, but a small output bias may still cross
a downstream decision boundary or block necessary correction. Initial guards
are admission checks, not task-quality certificates. Pure held arms are not
silently converted into guarded refresh. Values are never stale substitutes.

The previous-pair projection omits downstream normalization/MLP/MoE Jacobians;
the older margin-weighted bias is a distinct surrogate. Neither is an exact
MarginGuard oracle. Signed output alignment has no guaranteed downstream sign.
Both signs are retained as a paired diagnostic, without choosing a winning sign.

The shadow retained-mass certificate uses query drift and fixed-prefix key-norm
bounds, with a freshly computed current-tail partition. It holds only within
a canvas and never alters these fixed-window policies. Conservatism, violations,
actual retained mass and useful hold ages are measurements, not refresh rules.

Yuhan authored the inherited cross-step-reuse machinery. Gaussian32/value
routing, query-sensitivity/C-gate, V31/V2 and later Response/margin/readout/signed
variants keep their separate provenance. This study tests selector by lifecycle
interactions; it does not claim invention of mask reuse, convergence guarantees,
production latency gains or population accuracy preservation.
