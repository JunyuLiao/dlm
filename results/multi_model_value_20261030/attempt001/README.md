# Three-model value-direction evaluation

Status: running; no final accuracy result yet. User authorized the complete
campaign without a GPU-hour ceiling, with host GPU monitoring every15minutes.

Specification: `docs/MULTI_MODEL_VALUE_EVALUATION_20261030.md`.
Frozen scientific configuration: `protocol.json`. Generation source binding:
`27d580d661a307f4a669e61fc852b7b3ae2a4914`; later analysis-only changes do not
change running generation. All model revisions and checkpoint/source file
hashes are in `*_model_identity.json`. Runtime packages and raw generations
remain private. A separate CPU monitor appends `health/gpu_health.jsonl`.

Models: LLaDA2.2-mini, UltraLLaDA, TraDo-8B-Instruct. Five arms:
native dense, all-kept FA4, mass top-k reuse4, V2 Gaussian32 reuse4, and V3a
singleton reuse4. Original Gaussian32 is Junyu's method family; reuse lineage
is inherited from Yuhan. Cross-model ports are separately named variants.

Target consumer sparsity48–52%, calibrated by computation only; dense
discovery, prefill/commit, and consumer work are separately accounted.
Native schedules/stopping/editing remain model-specific and fixed.
Accuracy and N/C/T/forward/physical-work statistics are primary; instrumented
times are not clean latency or speedup evidence.

Planned primary requests5110 across40model/task/arm panels:
AIME26 30x3seeds and HumanEval164x1 on all three; RULER128K130x1 on
LLaDA2.2/UltraLLaDA. TraDo128K is unsupported. RULER has thirteen tasks,
ten items/task and tokenizer-specific generator lengths with matched identity
and seed conventions; no prompt truncation or104-item substitution.

Qualification:

- Kernel attempt004 passes independent consumer parity and V2/V3a mask
  oracles; earlier shape/caller failures remain preserved.
- LLaDA2.2 attempt003 passes long synthetic publisher-output parity and50%
  fixed budgets. All-kept FA4 changes calls despite identical final output;
  this consumer difference is retained explicitly.
- TraDo attempt008 passes synthetic publisher-output parity and all arms
  using publisher Transformers4.52.4 in an isolated private target.
- UltraLLaDA attempt002 passes publisher full-horizon sampling-output parity
  and all arms using publisher Transformers4.55.4. Bidirectional prefix is
  never cached; the future-mask horizon and fixed unmasking schedule remain.
- LLaDA2.2 block-causal prefix caching, bounded logits and mask descriptor
  preserve the native editing loop; structural edits invalidate held maps.
- HumanEval OS sandbox preflight passes. Model/sampler import failures and
  qualification attempts are retained rather than overwritten.

Do not treat preparation, calibration, partial shards or audit timing as a
completed benchmark. Publication requires exact coverage, official scoring,
paired question-clustered analysis and a root completion marker.
