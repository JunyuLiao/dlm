# Fixed single-pulse hypothesis and boundary interpretation

For removed attention mass alpha, current dense output o and removed weighted
value contribution r, the exact real-valued masked attention bias is

    b = o_sparse - o_dense = (alpha * o - r) / (1 - alpha).

The inherited H bank evaluates this identity on current operands. Its error and
mass guards constrain immediate perturbation; mass<=0.2 also implies the redundant
renormalization bound<=1.25. Neither guard certifies a native decision or quality.
The varied-unit high/low pair matches scalar error/mass and differs in its signed
bias direction. Other units are held to CE. High/low alignment with current dense
attention output is a fixed current-only diagnostic rule, without a claim that
its sign predicts downstream confidence or task usefulness. It is distinct from
Q-only/full-QKV past-motion response, downstream influence, Gaussian32 and C-gate.

Native acceptance, re-noising and stopping are nonlinear functions of logits and
history. A small support bias may cross confidence/stability boundaries without
changing the final argmax canvas. That can change the number of ordinary native
calls and hence the position of the native RNG stream at commit and initialization.
The commit sampler itself still draws Gumbel/uniform noise; fewer calls must not
be compensated by reseeding, replaying or advancing RNG. A next-canvas change is
then possible even with identical emitted prefix. This is a prospective mechanism
hypothesis, not a guarantee of reduced total work or preserved quality.

The competing explanation is persistent active cache/sampler/runtime state. The
boundary observer therefore hashes active logical cache spans, input embeddings,
outputs and sampler state around each branch's OWN commit, and distinguishes
signed zeros from numerical differences. Native masked self-conditioning may be
zero with many negative-zero bytes; byte differences alone are insufficient.
Inactive history/allocation capacity must not count as state. Commit observations
contain model-entry state plus per-layer post-write cache/output; next-first-call
active cache includes that call's stochastic tail. Interpret differences with
those phases rather than equating equal global step numbers.

No RNG mediator is independently manipulated. Even equal measured committed
cache/reset state with different native initialization only supports the RNG-
position explanation within the observed state scope; it does not prove exclusive
mediation or establish expected harm/benefit. The E2E panel is independent of a
favorable boundary result. It retains every missing anchor, bank failure, alias,
wrong/capped answer and correctness regression. No-pulse equality is abstention,
not successful sparse quality preservation. TotalN can improve through fewer
canvases at useful quality; N/C alone is neither necessary nor sufficient.

Original Yuhan reuse provenance, Gaussian32, C-gate, V31/V2, repeated Response
and this single-pulse eligibility revision remain separate. Historical negative
gates stay failed. No reuse/factorial, restoration, production speedup,
noninferiority or novelty claim follows from this bounded study.
