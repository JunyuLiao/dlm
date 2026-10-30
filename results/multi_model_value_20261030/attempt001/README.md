# Three-model value-direction evaluation

Status: running; LLaDA2.2 HumanEval/AIME complete, TraDo HumanEval active. User authorized the complete
campaign without a GPU-hour ceiling, with host GPU monitoring every15minutes.

Specification: `docs/MULTI_MODEL_VALUE_EVALUATION_20261030.md`.
Frozen scientific configuration: `protocol.json`. Execution bindings001–004
record the immutable worker sources; current LLaDA2.2 worker is `224f8f11`.
Qualified optimized coordinator `8c0d14b0` passed kernel gate attempt005;
binding006 keeps per-model sources fixed. Analysis-only changes do not change running
generation. All model revisions and checkpoint/source file
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

Complete first model/task panel: HumanEval164 on LLaDA2.2, all five arms,
820 primary requests. Native/all-kept/V31/V2/V3a score141/138/83/10/103.
See `llada22_humaneval_complete_summary.md` and corrected-scoring analysis
`analysis_humaneval_llada22_complete002/`. V3a improves over V31 on this panel
with more denoising calls, while remaining below dense. This is not a
cross-model conclusion. V2's reduced calls accompany severe quality loss.

LLaDA2.2 AIME has five complete arms: native36/90, all-kept29/90, V31mass21/90,
V2online9/90,V3a25/90. V3a's4.44pplead has95%problem CI[-8.89,17.78], including
zero. V31/V3a consumer sparsity is50.0002/49.9967%; V2 drifts to
75.2079% despite development confirmation51.0154%. Its threshold is preserved,
and a matched50%sparsity comparison is explicitly invalid for that panel.
See `llada22_short_complete_summary.md` and
`analysis_llada22_short_complete001/model_scope_complete.json`. Exact-scope
verification passes1270requests/10panels. No reliable AIME win or general
dense-quality-preservation claim is established. RULER and other models remain
unfinished, so there is no campaign-wide completion claim.

Ultra's two-block publisher-model and cached-counter parity gate006 passes.
Its corrected prefix-mask/full-coordinate sampler has independent CPU publisher
oracle coverage. Optional4GiBstatistics-memory gate007 fails strict V3a output
parity; the original memory budget is retained. Binding007 records the selected
sources and both outcomes. TraDo HumanEval computation-only calibration confirms
50.2730%sparsity at threshold0.17782794100389226before its benchmark launch.
