# What Stage A tests

For one query row, let `p = softmax(σqKᵀ)` and `o = pV`. For a fixed support,
its first-order change is

```
δo = Σᵢ pᵢ δvᵢ + Σᵢ pᵢ (vᵢ − o) σ(δq·kᵢ + q·δkᵢ).
```

The query-only component is `∂o/∂q = σ Cov_p(v,k)`. A sparse support changes
the distribution, its normalization, and therefore the response operator as
well as the current output. Equal current-error norms and removed mass need
not give equal vector bias or equal response. This motivates the diagnostic;
it does not prove a shorter denoising trajectory.

The `q_past` diagnostic uses the previous ordinary call's query displacement.
The separately named `qkv_past` includes the previous call's current-canvas key
and value displacement. Within the first canvas, prefix K/V are fixed and
checked by hashes. Captured prior operands enter diagnostic calculations only;
the model's attention always consumes current Q/K/V. Full `J_S u`, dense
`J_A u`, and residual `(J_S−J_A)u` remain separate quantities.

Three failures are distinguishable. Past motion may predict future motion
poorly. A local derivative may approximate a finite native step poorly even
when supplied its actual future displacement. Or a finite candidate bank may
not contain equal-work/error/mass alternatives with sufficient response
separation. The evaluation-only future-motion oracle addresses the second
question; it cannot select a treatment. Direction shams preserve operand-motion
norms while changing their orientation. Support shams use the same guarded
candidate set and match current error/mass.

The primary prediction endpoint concerns the next fixed-support attention
output **along the common all-kept native trajectory**. Candidate recomputation
is an explicit diagnostic operator, not a second model forward. It cannot by
itself show how applying that support changes the next native state. A2 would
test that propagation and repeated intervention with AA/SA/AS/SS after the
identification and predictive gates pass.

Smaller attention motion can preserve an uncertain state or suppress needed
correction. It is therefore distinct from shorter remaining native calls,
official task quality, and clean latency. The native sampler includes random
proposal/re-noising, self-conditioning, a changing temperature and discrete
whole-canvas stopping. None is replaced by an attention contraction model.

Physical deletion is fixed within each comparison, with LOCAL and boundary/
current-canvas support protected. The 0.2 deleted-mass guard implies a maximum
1.25 renormalization factor and is redundant with that factor in this setting.
Guard satisfaction is not accuracy certification. A short prefix may have
fewer than four feasible deletion choices; that is a non-identifying bank.

The common all-kept consumer is installed at all three tested locations in
canvas 1, keeping consumer substitutions constant for the location comparison.
Its differences from native PIECEWISE/FULL are systematic. First-canvas oracle
landmarks are not online scheduling rules, and they do not address every later
confidence-limited tail. Fresh-request equality does not repair the historical
restored-output assembly failure.

This study extends the previous response diagnostic. Yuhan's inherited reuse,
Gaussian32/value-direction routing, query-sensitivity/C-gate, V31 and V2 retain
their separate identities. The old margin prototype remains a weighted-bias
surrogate, and the norm-only validity certificate remains parked. No novelty,
accuracy-preserving acceleration or kernel speed claim follows from these
operator identities alone.
