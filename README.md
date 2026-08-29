# dLLM Inference and Evaluation Framework

This repository is a model-independent research framework for inference and
evaluation of diffusion-based large language models (dLLMs). Its primary code
is the `dllm` Python package in [`src/dllm`](src/dllm): model adapters expose a
common generation interface, attention backends can be selected independently
of the model family, and reproducible benchmark workflows cover long-context
recall, mathematical reasoning, and serving-system behavior.

## What the framework provides

- A lazy model-adapter registry with one generation contract for multiple text
  dLLM families.
- Dense attention and a drop-in `blasst-reference` attention backend across all
  registered adapters.
- Reproducible evaluation tracks for exact-count NVIDIA RULER, NeMo-Skills
  MATH500 reasoning, and an isolated SGLang FDFO/GSM8K system benchmark.
- Official task scoring, per-example seeds, provenance hashes, resumable
  predictions, run-fingerprint validation, and dense-versus-BLASST comparisons.
- Scalar command-line entry points plus explicit scripts for context-length,
  block-size, lambda, attention-distribution, and pruning experiments.

> [!IMPORTANT]
> `blasst-reference` materializes dense QK scores before masking attention
> tiles. It reports reference correctness and theoretical sparsity statistics;
> runtime acceleration requires an optimized sparse kernel.

## Evaluation and benchmark tracks

| Track | Focus | Entry point |
|---|---|---|
| MATH500 | Mathematical reasoning, sampling diversity, and majority/self-consistency scoring through NVIDIA NeMo-Skills | [`benchmarks/diffusion_gemma_math500`](benchmarks/diffusion_gemma_math500) |
| NVIDIA RULER | Exact-count, tokenizer-validated long-context retrieval and aggregation | `dllm-prepare-ruler`, `dllm-eval-ruler`, and [`scripts/ruler`](scripts/ruler) |
| SGLang FDFO | Scheduler-level serving behavior on GSM8K with LLaDA2.1-mini | [`benchmarks/sglang_fdfo`](benchmarks/sglang_fdfo) |
| Research experiments | Attention distributions, lambda sweeps, block-max/quantile pruning, and model-specific BLASST studies | [`scripts/ruler`](scripts/ruler) and [`benchmarks`](benchmarks) |

These tracks share model adapters and attention instrumentation where useful,
but each retains the upstream prompt format, scorer, provenance, and reporting
appropriate to its task.

## Supported text adapters

| Adapter | Default checkpoint | Native decoding path | BLASST integration |
|---|---|---|---|
| `fast_dllm_v1_llada` | `GSAI-ML/LLaDA-8B-Instruct` | LLaDA diffusion generation | Direct local SDPA/FlashAttention dispatch |
| `fast_dllm_v1_dream` | `Dream-org/Dream-v0-Base-7B` | Dream `diffusion_generate` | Direct Dream SDPA dispatch |
| `fast_dllm_v2` | `Efficient-Large-Model/Fast_dLLM_v2_7B` | Fast-dLLM block sampler | Hugging Face attention registry |
| `llada2_1_mini` | `inclusionAI/LLaDA2.1-mini` | Native block-diffusion/token-editing decoder | Native model attention registry |
| `diffusion_gemma` | `google/diffusiongemma-26B-A4B-it` | Native encoder/decoder block-diffusion sampler | Decoder canvas self-attention only; cached prompt KV stays dense |

Model-specific loading, tokenization, prompt handling, and denoising stay in
the adapter. Evaluation and attention selection remain model-independent.

## Installation

Python 3.10 or newer is required. Install a CUDA-enabled PyTorch build suitable
for the host first, then install the root package:

```bash
pip install -e .
```

The root environment pins `transformers==5.11.0`. This is the first released
Transformers wheel that contains `DiffusionGemmaForBlockDiffusion`, the
Gemma4 `AutoProcessor` mapping, its native block-diffusion `generate()`, and
the attention registry used by BLASST; the 5.10.4 wheel contains no
DiffusionGemma package. The root dependencies also include Accelerate and the
vision-side processor dependencies required to construct `Gemma4Processor`.
No moving `main` checkout is needed.

Install the optional dependencies needed by RULER generation and scoring, or
the test suite, with:

```bash
pip install -e '.[eval]'
pip install -e '.[test]'
```

The project-specific Fast-dLLM, Fast-dVLM, and Fast-dDrive packages have
separate dependency instructions in their own READMEs. Isolated virtual
environments are recommended because their dependencies may differ from the
root package. MATH500 evaluation additionally uses a pinned external
[NVIDIA NeMo-Skills](https://github.com/NVIDIA-NeMo/Skills) checkout; see the
benchmark instructions below.

## Programmatic inference

All registered models implement the same `ModelAdapter` interface while
preserving their native decoder:

```python
from dllm.models import GenerationRequest, create_adapter

adapter = create_adapter(
    "llada2_1_mini",
    "inclusionAI/LLaDA2.1-mini",
    device="cuda",
    precision="bfloat16",
).load()

result = adapter.generate(
    GenerationRequest(
        prompt="Explain diffusion language modeling in two sentences.",
        max_new_tokens=128,
        block_size=32,
        threshold=0.5,
        seed=42,
    )
)
print(result.text)
```

DiffusionGemma uses the same interface. A positive `temperature` applies a
constant sampling temperature, and `extra["top_p"]` enables nucleus sampling;
this disables the checkpoint's native temperature schedule so the requested
values are not compounded. Generic `threshold` is unused, and `block_size` is
reported but does not override the model's native 256-token canvas:

```python
from dllm.models import GenerationRequest, create_adapter

adapter = create_adapter(
    "diffusion_gemma",
    "google/diffusiongemma-26B-A4B-it",
    device="cuda",
    precision="bfloat16",
).load()
result = adapter.generate(
    GenerationRequest(
        prompt="Explain diffusion language modeling in two sentences.",
        max_new_tokens=64,
        temperature=0.6,
        seed=42,
        extra={"top_p": 0.95, "thinking": False},
    )
)
print(result.text, result.metadata)
```

The checkpoint defaults are 48 maximum denoising steps, `t_max=0.8`,
`t_min=0.4`, entropy bound `0.1`, confidence threshold `0.005`, and stability
threshold `1`. `GenerationRequest.steps` maps to `max_denoising_steps`.
Supported `extra` keys are `max_denoising_steps`, `t_max`, `t_min`,
`entropy_bound`, `confidence_threshold`, `stability_threshold`, and boolean
`thinking` (the latter maps to the official template's `enable_thinking`).
`top_p` is also supported when `temperature` is positive. The native canvas
remains 256 tokens; `block_size` is reported but is not presented as an
effective native override.

The full BF16 checkpoint fits on one 80 GB H100 without CPU offload. The
recorded smoke at resolved model revision
`f7f5b7f5fa82ffc52addd066915886d497f5517b` peaked at 53.70 GB
(50.01 GiB) allocated for native generation and about 53.82 GB (50.12 GiB)
for the RULER/BLASST runs. These are measured framework allocations, not a
guarantee for unrelated processes sharing the GPU.

No Hugging Face credential is stored by the adapter. If access is denied,
accept any checkpoint terms and authenticate through the standard
`hf auth login` flow; the adapter reports that instruction for 401, 403, or
gated-repository failures.

The adapter API provides in-process inference. Optional dInfer and SGLang
integrations are available under [`third_party`](third_party), independently of
the core package.

## MATH500 reasoning evaluation

[`benchmarks/diffusion_gemma_math500`](benchmarks/diffusion_gemma_math500)
evaluates DiffusionGemma on all 500 MATH500 problems through NVIDIA
NeMo-Skills. The matched protocol uses the zero-shot `generic/math` prompt,
symbolic grading, temperature `0.6`, top-p `0.95`, ten samples per problem,
and majority/self-consistency aggregation. Runs are append-only and resumable.

Prepare a pinned NeMo-Skills checkout and its MATH500 data as described in the
[benchmark README](benchmarks/diffusion_gemma_math500/README.md), then run
BLASST with local/global lambda `0.9/0.6`:

```bash
PYTHONPATH=src python benchmarks/diffusion_gemma_math500/run_experiment.py \
  --nemo-skills-root /path/to/Skills \
  --output-dir results/math500/blasst_l0p9_g0p6
```

Run the matched native dense baseline by adding `--dense-baseline` and using a
fresh output directory:

```bash
PYTHONPATH=src python benchmarks/diffusion_gemma_math500/run_experiment.py \
  --nemo-skills-root /path/to/Skills \
  --output-dir results/math500/dense \
  --dense-baseline
```

The completed reference comparison is summarized below:

| Mode | NeMo majority@10 | Deterministic majority@10 | Average pass@1 | Oracle pass@10 | Physical sparsity (local/global) |
|---|---:|---:|---:|---:|---:|
| Dense | 92.91% | 93.20% | 76.54% | 96.20% | 0% / 0% |
| BLASST, λ=0.9/0.6 | 93.23% | 93.20% | 71.38% | 96.20% | 28.62% / 19.97% |

The deterministic majority result is unchanged, while BLASST reduces
individual-draw accuracy and increases unextractable answers. See the
[combined dense-versus-BLASST report](results/math500/dense_vs_blasst.md)
for paired confidence intervals, subject/difficulty breakdowns, prompt and
sequence lengths, tile counts, and the reference-backend timing caveat. The
comparison can be regenerated with
[`compare_results.py`](benchmarks/diffusion_gemma_math500/compare_results.py).

## Universal RULER evaluation

### 1. Pin the official benchmark

The pipeline accepts only the verified NVIDIA RULER commit below and rejects a
modified checkout:

```bash
git clone https://github.com/NVIDIA/RULER.git /path/to/RULER
git -C /path/to/RULER checkout c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
```

Follow RULER's upstream setup for its data-generation resources. If those
dependencies live outside the active environment, pass
`--ruler-dependency-path` and `--nltk-data` when preparing the manifest.

### 2. Prepare an exact-count manifest

`--num-samples` is mandatory. The command balances that exact total across the
selected tasks, validates each prompt with the target adapter's tokenizer, and
writes `manifest.json` plus `samples.jsonl`:

```bash
dllm-prepare-ruler \
  --model-adapter fast_dllm_v2 \
  --tokenizer-path Efficient-Large-Model/Fast_dLLM_v2_7B \
  --ruler-root /path/to/RULER \
  --context-length 8192 \
  --num-samples 100 \
  --seed 42 \
  --output-dir results/ruler/manifests/fast_dllm_v2/8192
```

By default, long-context manifests use `niah_multikey_1`, `niah_multivalue`,
`niah_multiquery`, `vt`, and `fwe`. Use a comma-separated `--tasks` value to
choose a different subset.

For DiffusionGemma, manifest preparation calls only `AutoProcessor`; it does
not load the 26B weights. Prompt-affecting adapter options can be supplied as
an inline JSON object or JSON file and are recorded in the manifest:

```bash
dllm-prepare-ruler \
  --model-adapter diffusion_gemma \
  --tokenizer-path google/diffusiongemma-26B-A4B-it \
  --ruler-root /path/to/RULER \
  --context-length 512 --num-samples 1 --tasks fwe \
  --generation-config-json '{"thinking": false}' \
  --output-dir results/ruler/manifests/diffusion_gemma/512
```

### 3. Evaluate one model and attention configuration

Run a dense baseline:

```bash
dllm-eval-ruler \
  --model-adapter fast_dllm_v2 \
  --model-path Efficient-Large-Model/Fast_dLLM_v2_7B \
  --manifest results/ruler/manifests/fast_dllm_v2/8192/manifest.json \
  --ruler-root /path/to/RULER \
  --context-length 8192 \
  --num-samples 100 \
  --attention-backend dense \
  --block-size 16 \
  --output-dir results/ruler/fast_dllm_v2/8192/dense
```

Run the same manifest with BLASST sparsity analysis:

```bash
dllm-eval-ruler \
  --model-adapter fast_dllm_v2 \
  --model-path Efficient-Large-Model/Fast_dLLM_v2_7B \
  --manifest results/ruler/manifests/fast_dllm_v2/8192/manifest.json \
  --ruler-root /path/to/RULER \
  --context-length 8192 \
  --num-samples 100 \
  --attention-backend blasst-reference \
  --blasst-lambda 0.003 \
  --block-size 16 \
  --collect-attention-stats \
  --stats-level summary \
  --output-dir results/ruler/fast_dllm_v2/8192/blasst_0.003
```

The evaluator verifies that the manifest, runtime tokenizer, adapter, context
length, and exact example count agree before generation. A run writes:

- `run_config.json`: full configuration, environment metadata, and run
  fingerprint.
- `predictions.jsonl`: append-only per-example outputs used for safe resume.
- `summary.json`: official overall and per-task RULER accuracy plus runtime
  totals.
- `attention_stats/`: BLASST tile, row, and valid-element sparsity statistics
  when collection is enabled.

An existing output directory can resume only the identical fingerprinted run.
Use `--no-resume` to deliberately restart it.

Adapter-specific sampler values use the same `--generation-config-json`
option. They participate in the run fingerprint and are stored verbatim in
`run_config.json`:

```bash
# Dense
dllm-eval-ruler \
  --model-adapter diffusion_gemma \
  --model-path google/diffusiongemma-26B-A4B-it \
  --manifest results/ruler/manifests/diffusion_gemma/512/manifest.json \
  --ruler-root /path/to/RULER --context-length 512 --num-samples 1 \
  --attention-backend dense --steps 4 \
  --output-dir results/ruler/diffusion_gemma/512/dense

# BLASST reference plus theoretical sparsity statistics
dllm-eval-ruler \
  --model-adapter diffusion_gemma \
  --model-path google/diffusiongemma-26B-A4B-it \
  --manifest results/ruler/manifests/diffusion_gemma/512/manifest.json \
  --ruler-root /path/to/RULER --context-length 512 --num-samples 1 \
  --attention-backend blasst-reference --steps 4 \
  --collect-attention-stats --verify-dense-after-blasst \
  --output-dir results/ruler/diffusion_gemma/512/blasst-reference
```

DiffusionGemma's decoder attention concatenates cached prompt KV and canvas
KV. The adapter selects only `DiffusionGemmaDecoderTextAttention`, and the
model-independent binding marks the entire cached prompt prefix dense; BLASST
decisions are eligible only for complete canvas-to-canvas KV tiles. Encoder,
vision, and cached cross-attention KV are never sparsified. The original SDPA
registry entry is restored when the binding closes. As elsewhere,
`blasst-reference` still materializes dense QK scores and must not be treated
as a kernel-speedup result.

### Shell workflows and sweeps

[`scripts/ruler`](scripts/ruler) keeps scalar CLI behavior separate from sweep
policy:

| Script | Purpose |
|---|---|
| `prepare.sh` | Prepare one exact-count manifest from environment variables. |
| `eval_one.sh` | Evaluate one adapter, manifest, and attention configuration. |
| `eval_fast_dllm_v1_llada.sh` | Apply the LLaDA v1 defaults. |
| `eval_fast_dllm_v1_dream.sh` | Apply the Dream v1 defaults. |
| `eval_fast_dllm_v2.sh` | Apply the Fast-dLLM v2 defaults. |
| `eval_llada2_1_mini.sh` | Apply the LLaDA2.1-mini defaults. |
| `eval_diffusion_gemma.sh` | Apply DiffusionGemma's BF16/native-canvas defaults. |
| `compare_dense_blasst.sh` | Run dense and `blasst-reference` into sibling directories. |
| `sweep_blasst_lambda.sh` | Sweep the explicit BLASST lambda matrix. |
| `sweep_block_size.sh` | Sweep the explicit decoder block-size matrix. |
| `sweep_context.sh` | Evaluate prebuilt manifests across context lengths. |
| `smoke_h100.sh` | Run one short dense and BLASST example for every adapter. |
| `smoke_diffusion_gemma_h100.sh` | Run native, exact-count dense, BLASST-statistics, and post-cleanup dense DiffusionGemma checks. |

For example:

```bash
export RULER_ROOT=/path/to/RULER
export MODEL_ADAPTER=fast_dllm_v2
export MODEL_PATH=Efficient-Large-Model/Fast_dLLM_v2_7B
export TOKENIZER_PATH="$MODEL_PATH"
export CONTEXT_LENGTH=8192
export NUM_SAMPLES=100

export OUTPUT_DIR=results/ruler/manifests/fast_dllm_v2/8192
scripts/ruler/evaluation/prepare.sh

export MANIFEST="$OUTPUT_DIR/manifest.json"
export OUTPUT_DIR=results/ruler/fast_dllm_v2/8192/compare
scripts/ruler/blasst/compare_dense_blasst.sh
```

Each sweep point invokes `eval_one.sh` independently, which keeps the exact
configuration and output fingerprint visible for every result.

## Adding another dLLM

1. Implement `dllm.models.ModelAdapter` under
   [`src/dllm/models/adapters`](src/dllm/models/adapters).
2. Keep model loading, tokenizer behavior, native decoding, special-token
   discovery, and generation metadata inside the adapter.
3. Declare the adapter's attention integration (`registry` or direct dispatch)
   and its attention class names.
4. Register it lazily in
   [`src/dllm/models/registry.py`](src/dllm/models/registry.py).
5. Add adapter-contract, dense-attention, BLASST, and benchmark-facing tests.

The RULER runner and BLASST algorithm should not require model-specific
branches for a new adapter.

## SGLang FDFO benchmark

[`benchmarks/sglang_fdfo`](benchmarks/sglang_fdfo) is a separate, reproducible
system benchmark for LLaDA2.1-mini on exactly 100 GSM8K examples. It launches a
real SGLang HTTP server to exercise the request scheduler and FDFO behavior.
This benchmark has its own environment and runner.

Use a dedicated environment because its pinned SGLang 0.5.16 stack can conflict
with the root or vendored environments:

```bash
python -m venv .venv-sglang-fdfo
source .venv-sglang-fdfo/bin/activate
python -m pip install -r benchmarks/sglang_fdfo/requirements.txt

CUDA_VISIBLE_DEVICES=0 python benchmarks/sglang_fdfo/run.py \
  --model inclusionAI/LLaDA2.1-mini \
  --model-revision 20e64e2ad21644d0e5248586ed9c942cdd45de0f \
  --num-examples 100 \
  --scheduler-caps 4 16 \
  --repetitions 3 \
  --output-dir results/systems/sglang_fdfo/runtime
```

The benchmark pins its model revision and GSM8K data, runs parity and lifecycle
smoke tests, interleaves baseline and FDFO arms in fresh server processes, and
records prompts, raw requests, telemetry, server logs, environment metadata,
machine-readable comparisons, and `report.md`. Use
`--cuda-graph-batch-sizes 1 2 4 8 16` or `--disable-cuda-graphs` if graph
capture exceeds available memory; the same fallback is applied to both arms.

## Repository layout

```text
.
├── src/dllm/                 # Primary model, attention, and evaluation package
│   ├── models/               # Adapter contract, registry, and implementations
│   ├── attention/            # Shared attention integration and BLASST reference
│   ├── evaluation/ruler/     # Pinned generation, manifests, runner, and scoring
│   └── cli/                  # dllm-prepare-ruler and dllm-eval-ruler
├── tests/                    # Framework unit, integration, and CUDA tests
├── scripts/ruler/            # Evaluation, BLASST, pruning, and diagnostics
├── results/                  # Organized outputs and experiment summaries
├── benchmarks/diffusion_gemma_math500/ # NeMo-Skills reasoning evaluation
├── benchmarks/sglang_fdfo/   # Isolated scheduler-level FDFO benchmark
├── fast_dllm_v1/             # Fast-dLLM v1 research implementation
├── fast_dllm_v2/             # Fast-dLLM v2 research implementation
├── fast_dvlm/                # Fast-dVLM research implementation
├── fast_ddrive/              # Fast-dDrive research implementation
└── third_party/               # Vendored project integrations
```

## Validation

Run the root test suite with:

```bash
pytest
```

To exercise the complete adapter-to-RULER path on one CUDA GPU, including both
attention backends for every registered adapter:

```bash
RULER_ROOT=/path/to/RULER scripts/ruler/evaluation/smoke_h100.sh
```

The smoke test downloads or loads the adapters' default checkpoints, so it
requires model access and sufficient local cache space. Its default output root
is `/tmp/dllm-ruler-smoke`; override it with `SMOKE_ROOT` if needed.

The universal RULER pipeline is text-only. DiffusionGemma deliberately loads
through `AutoProcessor` so future typed image/video requests remain possible,
but image paths are not accepted through the untyped `GenerationRequest.extra`
mapping and multimodal evaluation is outside this integration.

## Project-specific research code

These directories provide project-specific training, demo, and evaluation
workflows. Their READMEs describe the corresponding environments and entry
points:

| Project | Scope | Documentation | Paper |
|---|---|---|---|
| Fast-dLLM v1 | Training-free text-dLLM acceleration | [`fast_dllm_v1/README.md`](fast_dllm_v1/README.md) | [arXiv:2505.22618](https://arxiv.org/abs/2505.22618) |
| Fast-dLLM v2 | Block-diffusion LLM training and inference | [`fast_dllm_v2/README.md`](fast_dllm_v2/README.md) | [arXiv:2509.26328](https://arxiv.org/abs/2509.26328) |
| Fast-dVLM | Block-diffusion vision-language model | [`fast_dvlm/README.md`](fast_dvlm/README.md) | [arXiv:2604.06832](https://arxiv.org/abs/2604.06832) |
| Fast-dDrive | Block-diffusion vision-language-action driving model | [`fast_ddrive/README.md`](fast_ddrive/README.md) | [arXiv:2605.23163](https://arxiv.org/abs/2605.23163) |

## Contributing

Issues and pull requests are welcome. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for repository guidelines.

## License and attribution

This repository is licensed under the Apache License 2.0. See
[`LICENSE`](LICENSE).

If you use a project-specific research implementation, cite the corresponding
paper listed above. Complete citation entries are available in each
subproject's README. The framework also builds on LLaDA, Dream, Qwen2.5,
Qwen2.5-VL, LMFlow, NVIDIA RULER, Hugging Face Transformers, dInfer, and SGLang;
please follow their licenses and citation guidance where applicable.
