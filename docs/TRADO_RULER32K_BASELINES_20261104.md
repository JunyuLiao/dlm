# Complete TraDo RULER32K dense/V31/V3a baselines

User-authorized four panels on the exact completed V2 RULER32K cohort:
130problems,10/task across13official RULER tasks,seed42. Arms:native dense SDPA,
all-kept FA4,V31 mass top-k with four-call reuse,V3a singleton top-k with the
same reuse. No V2 generation or calibration. Preserve previous V2 evidence.

Original TraDo model revision2d37bd3ebb743a62a3bfecc4cd27b92cf396c5a7,
source snapshot8c0d14b0,compatTransformers4.52.4; same H100,BF16,batch1.
Native four-token blocks,four denoising steps,temperature1,top-p1,top-k0,
confidence0.9,EOS unchanged. Cap128. Same tokenizer-specific RULER data,prompt
range29798–32472 within32768 including output reserve; no truncation.
All36 denoising layers eligible;32Q/8KV,D128,Q128/KV64; dense prefill/commit.

For V31 and V3a keep max(round(J/2),mandatory_count) tiles per query head/block,
including the current mandatory tile, ties-to-even rounding and existing stable
tie rules. Physical budget follows effective context length at each refresh.
No threshold calibration is needed. Retain full-dimensional current V output,
existing FA4 sparse consumer,Gaussian32 seed1729+layer for V3a,refresh every
four calls,reset on canvas,no carry. Publish achieved sparsity48–52%, with
per-selection exact budget/mandatory/eligibility assertions and counters.

Freeze committed spec/source/model/data/scorer bindings before generation.
CPU tests,independent CPU selector/FP32 masked-attention CUDA oracle checks,
exact historical all-kept HumanEval control before benchmark. Four disjoint
shards/arm,immutable receipt+completion. One worker,900second GPU health checks;
new attempt for any interruption. Official pinned RULER fractional scoring;
report task scores,full-credit counts,N,C,N/C,T,EOS/caps,actual geometric tile
sparsity and paired problem-bootstrap intervals. Reuse/recheck existing
current-tile V2 scores,never regenerate it or overwrite earlier reports.
Instrumented times are accounting only; no clean latency claim. Privacy:raw
prompts/gold/tokens/completions/private paths remain local. New report closes
the missing four cells and references the unchanged six-panel prior report.
