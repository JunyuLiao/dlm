# Operator definedness before further native reruns

The fixed-order, single-engine crossover completes510.626s. Its first unobserved
V3a116/42 request hasN274/C20/T5089. Both sampler-only requests, post-only and
before/after attention requests, and the final unobserved request all have
N272/C19/T4787 and the same output. All four observed full sampler/logit traces
match. One other group's V31 bookends differ. Thus tracing alone is not an
established cause; first-use/context dependence remains plausible and unproven.

Before another cold engine, run a bounded operator-only check on the same frozen
selector/observer source and native numerical settings. Apply the same native
FULL/LOCAL mask fixes before importing observer classes. Run the existing three
FA4 observer tests (NaN-prefilled mass outputs, reference masses, native-output
parity) and a new native-dimension workspace test: H16/HK2/D512/Q256/R32,
prefixes2577,8895,4099, capacity9216, fixedK4 for V3a and frozen diagnostic tau
for V2. Reuse the workspace across growth and shrinkage. Surround each retained
scratch allocation with independent canaries, poison owned buffers withNaN,
zero and a finite negative pattern, and compare fields/maps against fresh
allocations. Check earlier returned maps do not alias later scratch.

This tests missing writes, invalid reads and output-buffer overwrites. Passing
does not establish absence of every race or prove native reproducibility.
Failure requires localization and a separate source-bound correction; do not
select a favorable fill pattern or alter a reported method to hide it. These
tests run no model/sampler forward and use synthetic inputs only. No exact-greedy
native evaluation is included.

Freeze source fingerprints and test commands in a new result directory. One
180-second reservation charges existing phasek256; current closed charge
9132.913s of16200s. Keep total43200s and both full-matrix/K128 holds unchanged.
