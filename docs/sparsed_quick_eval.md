# SparseD quick valuation

This is a deliberately small feasibility check, not a reproduction of the
paper's full MMLU/GSM8K/HumanEval/RULER evaluation.

The paper compares the original dense DLM (FlashAttention) with SparseD and
reports task accuracy plus single-sample latency. Its SparseD defaults are full
attention for the first 20% of denoising steps, then a reused head-specific
sparse pattern. For short-context tasks it uses selection ratio 50% and block
size 32; for RULER it uses selection ratio 30% and block size 128. The quick
evaluator preserves that dense-vs-sparse paired design but replaces expensive
benchmark accuracy with output fidelity to the dense run. Optional reference
answers in the prompt file provide a basic task-level exact/contains check.

## Setup

Use the environment pinned by the upstream implementation (PyTorch 2.6 and
Transformers 4.46.2 are the known-good versions):

```bash
mkdir -p reference
git clone --depth 1 https://github.com/INV-WZQ/SparseD.git reference/SparseD
python3 -m pip install -r reference/SparseD/requirements.txt
```

The upstream implementation also imports FlashAttention and uses CUDA
FlexAttention. Run this on a CUDA GPU; the first sparse call compiles kernels.
The evaluator warms up each prompt and mode before timing.

## Small first run

```bash
python3 scripts/sparsed_quick_eval.py \
  --sparsed-repo reference/SparseD \
  --prompts data/prompts_heterogeneous.jsonl \
  --max-prompts 2 \
  --warmup 1 \
  --repeats 2 \
  --outdir outputs/sparsed_smoke
```

For a speed signal, long context is more informative. The upstream repository
ships named `4k`, `8k`, and longer prompts:

```bash
python3 scripts/sparsed_quick_eval.py \
  --sparsed-repo reference/SparseD \
  --prompts reference/SparseD/prompts.json \
  --prompt-ids 4k \
  --max-prompts 1 \
  --select 0.3 \
  --sparse-block-size 128 \
  --warmup 1 \
  --repeats 3 \
  --outdir outputs/sparsed_4k
```

Use only lengths supported by the selected model. Start with one 4k prompt;
larger lengths and a full parameter sweep are later-stage experiments.

## Outputs and interpretation

- `measurements.csv`: every timed repeat, including latency, generated tokens/s,
  peak allocated GPU memory, and decoded output.
- `comparisons.csv`: paired prompt results: speedup, token agreement, normalized
  text exact match, and text similarity.
- `summary.json`: aggregate efficiency and fidelity metrics plus the full run
  configuration.

The go/no-go signal is simple: SparseD should produce a speedup above 1 on the
target context length without a material drop in token/text agreement. Agreement
with dense output is only a regression proxy. If the idea passes this check, the
next step is a small real benchmark slice (for example 50 GSM8K examples with
the paper's 4-shot prompt and answer extraction), followed by longer RULER cases.

Prompt JSONL may include `reference`, `answer`, `references`, or `answers`:

```json
{"id":"arithmetic-1","prompt":"What is 17 + 25?","reference":"42"}
```
