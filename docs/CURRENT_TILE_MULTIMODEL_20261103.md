# Current-tile V2 across three models and benchmarks

User-authorized nine-panel campaign, one H100, no GPU-hour restriction.
LLaDA2.2-mini, TraDo-8B-Instruct, DiffusionGemma-26B-A4B-it; only new V2
current-tile/current-canvas initialization arms. Preserve every old result and
failed attempt. AIME26:30 questions x seeds42/43/44; HumanEval164 seed42;
RULER130 seed42,10 samples per13 official tasks. RULER length128K for LLaDA
and DiffusionGemma,32K for TraDo. The previous Gemma RULER panel contained104
items (8/task); the new130 panel and new TraDo32K panel have no matched complete
historical baseline. Existing baseline numbers stay tied to their own cohorts.

HF models use their original source-pinned numerical runtime, all denoising
layers, Q128/KV64, Gaussian32(seed1729+layer), four-call reuse and no carry.
Current mandatory tile is processed once before tile0..J-2. Native LLaDA32
and TraDo4 samplers, temperatures, acceptance/editing, stopping and output caps
remain as originally configured. Preserve source/method identities and raw data.

Gemma uses original deterministic vLLM/FA4/native sampler and GLOBAL layers
5,11,17,23,29 only;25 LOCAL layers, prefill and commits remain dense. The entire
mandatory current256-token canvas (including its KV boundary tile) initializes
V2 before scanning optional chronological prefix tiles. Preserve original
select-step1, carry-first, progress-trigger0.15 and sticky1.386 lifecycle.
Use existing optimized Triton statistics/online chunks and FA4 current-full-V
sparse consumers, including at refresh. The selector implementation is a new
port, not a bitwise substitution for the old fused CUDA refresh. Record source,
kernel and actual physical-work differences; no latency superiority claim.
Gemma AIME/HumanEval thinking on and8192 output cap; RULER thinking off with
original official per-task output budgets and answer prefix. Native256 canvas,
48 maximum steps, temperatures0.8->0.4, entropy-bound0.1,confidence0.005,
stability1. Existing corrected dense baselines stay intact.

Before generation require CPU tests and CUDA oracle checks covering4/32/256
queries,D128/D512,multiple current tiles,128-tile chunk boundaries and Gemma's
sticky rule. Each HF panel replays original V2 and all-kept HumanEval controls
with exact hashes/counts; Gemma has separate native control/receipt checks.
Freeze committed spec and source-bound per-panel protocol before generation.
Use immutable per-panel attempt directories and four completed shards.

Calibrate near50% actual eligible denoising-consumer tile sparsity, including
mandatory and dense/carry calls; exclude selector discovery,prefill,commits.
For short benchmarks use six fixed prompt-length-stratified benchmark prompts
seed42, never all90 calibration requests. This is transductive computation-only
calibration: neither gold nor scores may choose thresholds. Retrieval initially
uses13 separately generated development prompts,one/task,seed142. At most10
small calibration candidates per attempt; choose closest to50% (48–52% target).
Benchmark runs all requested questions/seeds once per selected threshold.
Accept achieved40–60% inclusive. Outside that range, retain the complete result,
recalibrate at a new threshold and rerun in a new named attempt. Narrow the
threshold search in the indicated direction using only achieved physical work;
no accuracy-guided tuning. No GPU budget ceiling or silently accepted outside
range panel. Recalibration supersedes the earlier no-post-benchmark-retuning
instruction. The prior LLaDA HumanEval61.35% variant remains historical.

Record nvidia-smi, process health and progress every900seconds until all nine
panels and official scores/audits finish. One worker per GPU. Report full
accuracy,N,C,N/C,T,EOS/caps,achieved sparsity and all attempts/calibrations.
Use official scorers and paired question-clustered uncertainty only where input,
seed,model,sampler and source identities support it. Clearly label new cohorts.
Keep model/task-specific output caps. Publish sanitized aggregates/receipts only.
