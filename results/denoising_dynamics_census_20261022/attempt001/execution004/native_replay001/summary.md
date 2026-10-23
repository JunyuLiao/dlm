# Dense native request replay qualification

The logical-state replay gate passes on HumanEval/53 at the original seed 42
and output budget 8192. Five measured native requests each produce 275 output
tokens, 12 sampler calls and 13 model executions. Output/prompt hashes, final
CPU/CUDA RNG and minimal/rich/minimal active sampler-state hashes agree.

Each of the three observed requests records 12 sampler boundaries and 360
layer-boundary fingerprints: all 30 layers at every call. Referenced logical KV,
model input IDs/positions/self-conditioned embeddings, native logits and RNG
match exactly. Physical page-table hashes differ. Native cache strides, page
sizes and alignment are included in the comparison. No old KV, attention output,
logits or sampled tokens are substituted; all requests run from the original
prompt and all forwards are counted. There are zero request-time captures/JIT
events. Two focused CPU tests pass.

This is a one-item, within-engine logical replay certificate. It does not
restore scheduler state, physical page mappings, cache storage or compiled
state in place. It does not qualify support interventions, long-context replay,
causal native transitions or task accuracy preservation. Full-state restoration
and same-support native continuation remain the next engineering gate.

Protocol `0c3356de31cf1f9def4723bdb281b943f7220d6a685e1976cba247247bb0ce68`;
pre-generation spec commit `15bc18d3`. The worker exits 0. Its conservative
reservation is 179.500 seconds including startup/warmup/shutdown, not clean
latency. Exact argv/environment are in launch_receipt.json. Private inputs and
generations remain ignored. Original census/report artifacts are unchanged.
