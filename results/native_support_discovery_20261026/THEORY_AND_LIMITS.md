# Rationale and limits

For current dense attention o = sum(p_i v_i), removing support R of mass alpha
and renormalizing retained current keys/values changes it by

    delta_o = (alpha o - sum_{i in R}(p_i v_i)) / (1-alpha).

This exact real-arithmetic identity motivates diverse value-dependent deletion
sets at equal work. Equal removed mass does not imply equal output bias. A small
scalar maximum output error does not identify the direction/localization of
that bias, its downstream effect, or future denoising safety. Actual native
consumer differences are primary; FP32 formulas are numerical references.

The frozen previous-pair readout proxy maps each current attention bias through
the current output-projection weights and the LM-head difference for the
PREVIOUS native winner/competitor. Previous softcap derivatives and current
temperature are included. The corresponding fixed linear component is an
algebraic projection, not the complete downstream Jacobian. Post-attention RMS,
MLP/MoE, remaining layers and final RMS are omitted. The proxy may have an
incorrect sign for the true downstream margin. Its usefulness is an experiment,
not a theorem. Stable-position projected crossings and past-entropy weighting
are pre-choice heuristics; actual post-pulse entropy/history are outcomes only.

Native stopping requires both whole-canvas confidence and prediction stability,
or the unchanged 48-call cap. Advancing one predicate can leave the other
limiting; premature confidence can also suppress useful correction. Official
complete-request quality and all regressions must accompany any local saving.
Lower N can come from fewer canvases/shorter outputs without improved N/C.
Precomputed diagnostic support branches have audit costs, not deployable clean
selector latency. No speedup follows from deletion or call reduction alone.

Fresh requests qualify common pre-intervention state and RNG. Sparse requests
may legitimately finish with different outputs or RNG positions. If all
persistent state and subsequent policy were identical, with fresh independent
future innovations, a stream-position shift alone would not lower expected
quality. Measured cache/prefix equality does not certify all those assumptions.
Committed-token/cache equality is reference agreement, not task correctness.

The old confidence-trigger gate remains failed. This investigation separately
tests transient confidence delay and unstable states. Historical repeated
Response/A1 forecasts, signed-bias pulses and the previous-margin-weighted
surrogate retain their identities and mixed/negative findings. Yuhan's inherited
cross-step reuse, Gaussian32/value routing, query-sensitivity/C-gate, V31 and V2
are distinct prior work. This study's fixed previous-pair readout pulse is not
an exact MarginGuard oracle, response-contraction proof or reuse certificate.

At most six development questions and six conditional confirmation questions
are used here, with two seeds each. Masks, anchors, seeds and routing units are
not independent benchmark questions. Best-of-bank outcomes are selection-biased
oracle opportunity estimates. No population noninferiority or novelty claim is
supported by this design. Reuse and broader evaluations need separate protocols.
