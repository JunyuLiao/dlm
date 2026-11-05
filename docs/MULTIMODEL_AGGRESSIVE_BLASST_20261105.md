# Aggressive BLASST follow-up after LongBench v2

User-authorized 2026-11-05. Wait for the current two-model LongBench v2 campaign
to finish generation, official scoring and audit before starting any GPU work
for this follow-up. Do not alter its frozen sources or queues.

Six new arms/panels: LLaDA2.2-mini and TraDo-8B-Instruct on AIME26 (30 problems,
seeds 42/43/44), HumanEval (all 164, seed 42), and RULER (130, 10/task across
13 tasks, seed 42), using LLaDA128K / TraDo32K. Reuse the exact current prompts,
model/source/runtime pins, native samplers, output limits and official scorers.
This adds only the aggressive BLASST baseline; other completed methods remain
unchanged and are used as matched historical references.

Port the repository's **preceding seen maximum** aggressive BLASST extension,
not literal lambda>1 substitution into updated-maximum paper pseudocode.
For each valid query row and KV64 tile j, let b_j be the maximum scaled QK logit
in that tile, and m_previous the maximum over every preceding tile, including
skipped tiles. Skip only if all active rows have prior support and
b_j - m_previous < log(lambda). Exact ties are kept. First support and the
current tile are mandatory. Then update the seen maximum even when skipping.
Keep the established inverse-length policy lambda(L)=exp(log_kappa)/L; finite
lambda may exceed 1. This permits more aggressive pruning, without an accuracy
guarantee. It retains its own first-tile initialization; it is not BLASST-CTI.

Match the other methods' Q128/KV64 physical tiles, all denoising layer scope,
full-D128 FA4 consumer, four-call cross-step map reuse and no cross-canvas carry.
Dense prefill/commit work is preserved. Record dense QK discovery separately;
do not claim that selector discovery was removed or infer latency from sparsity.
Credit the existing aggressive BLASST rule and Yuhan's inherited reuse separately.

Calibrate a fresh log_kappa for each model/benchmark, never importing V2
thresholds. Follow the V2 small count-only protocol: six fixed prompt-length
strata of AIME/HumanEval (transductive, seed 42), or the existing 13 independent
RULER development prompts (seed 142). Use native full generation on these small
samples only, with the existing benchmark output caps; never use full 90/164/130
panels as calibration probes. Target 48–52% pooled achieved tile sparsity,
select closest from at most ten candidates, no accuracy/gold-based selection.
Freeze before final generation. Accept full-panel achieved sparsity in 40–60%;
outside this range, recalibrate in a new attempt using only the small set.
Retain all attempts and disclose drift/calibration exposure.

Before generation: committed spec/source → immutable source/data/model protocol
→ bind → focused CPU tests → independent CPU/CUDA mask checks (including
lambda>1, ties, invalid rows, padding and first/current protections) → full-D128
consumer checks → exact historical all-kept generation/count replay. Record
selector, lambda range, current/first support, physical eligible/kept counts,
refresh/reuse, model and kernel identities. One worker per H100; fresh idle
check before launch, nvidia-smi every 900 seconds, no GPU-hour ceiling.

Report official accuracy, N, C, T, N/C, calls and canvases per request, EOS/cap
rates, measured sparsity, lambda range, discovery work and GPU seconds. Paired
intervals use questions as clusters for AIME's three seeds. No clean latency
claim. Source/protocol/shard/private-token-hash/scorer and fairness audits gate
completion. Raw prompts, gold, tokens and completions remain private.
