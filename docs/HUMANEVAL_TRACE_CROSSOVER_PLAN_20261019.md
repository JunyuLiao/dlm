# Single-engine observation crossover

Attention attempt003 completed525.730/500.678s. Both targeted observed/unobserved
checks pass and both process traces match:8760 layer events for V3a116/42 and
15300 for V31-control-after116/43, covering all30 layers. Progress, current
Q/K/V, GLOBAL logical K/V, actual maps/observed masses and attention outputs
match. The earlier fault did not recur; no causal operator is localized.

Run one bounded crossover engine in a new attempt004. Preserve the original
clean prefix through the start of the six-arm question116,seed42 group. Repeat
that entire group six times, in this predeclared order:

1. unobserved;
2. V3a sampler-state/logit tracing only;
3. V3a sampler tracing plus attention fingerprints after the entire forward;
4. V3a sampler tracing plus the previous before/after attention fingerprints;
5. V3a sampler tracing only;
6. unobserved.

All other arms remain unobserved. Every group preserves the original six-arm
order and seed. All six groups run the same method/sampler definitions and
budget. Only instrumentation changes per request; source/import order is held
constant in one engine. Full native histories reset per request as usual. There
are38 prefix requests plus36 group requests,74 total, excluding the inherited
implicit warmup. All four observed V3a cases have the first/last unobserved V3a
as bookends. Check every group's V31 bookends too.

The post-only mode adds no fingerprint kernel before or inside the native
attention forward/selector. It reads inputs afterward; if inputs have an
unexpected lifetime race those fingerprints cannot establish what the consumer
actually read. The before/after mode retains actual list/mass probes inside the
selector. Report this distinction. Neither mode changes returned output tensors,
adds a model forward, draws RNG or supplies clean latency. Raw capture stays off.
CPU ordering tests verify native-call preservation and post-only placement;
the unchanged integer kernel retains its H100 oracle qualification.

Compare trajectories and paired logit/attention traces without scoring or
choosing a favorable mode. This tests observation/context dependence, not a
repair. A detected numerical difference still needs actual tensor proof/replay.
If it does not recur, retain inconclusive status; do not release full matrices.

One900-second reservation charges phasek256, currently8622.288s closed of16200s.
Actual original prefix request time126.398s and per-group35.437s support this
bounded single-engine design; startup and tracing remain charged. No allocation
or total43200s expansion. K256/K128 accuracy and the K128 queue stay held.
