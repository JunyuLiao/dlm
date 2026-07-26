# Official SparseD versus BLASST on LLaDA

`scripts/llada_blasst_vs_sparsed.py` executes the authors' SparseD artifact
directly and compares it with this repository's fused BLASST kernel on the same
model, prompt, generated length, denoising steps, block length, batch size,
sampling policy, dtype, and GPU process.

The validated SparseD artifact is
[`INV-WZQ/SparseD`](https://github.com/INV-WZQ/SparseD) at commit
`52155b4aa78368695f96744cade06d2e0865d085`. It is kept under ignored
`reference/SparseD` so third-party source is not copied into this repository.

```bash
git clone --depth 1 https://github.com/INV-WZQ/SparseD.git reference/SparseD
```

## Matched protocol

The default is the official repository's LLaDA long-context latency profile:

| Setting | Value |
| --- | --- |
| Model | `GSAI-ML/LLaDA-1.5` |
| Batch size | 1 |
| Prompt inputs | official `4k,8k,16k,32k,64k` prompts |
| Generated tokens | 128 |
| Denoising steps | 128 |
| Generation block length | 32 |
| Temperature / CFG | 0 / 0 |
| Remasking | `low_confidence` |
| Dense early attention | PyTorch SDPA forced to FlashAttention |
| SparseD | skip `0.2`, select `0.3`, selection block 128 |
| Dtype | BF16 |
| Timing | batch-1 complete generation, one warmup then one CUDA-event sample |

Dense, SparseD, and BLASST use the artifact generator's denoising schedule and
token-transfer semantics. SparseD receives the authors' `SparseD_param`
dictionary and therefore uses their full-attention prefix, isolated block
selection, cached per-head pattern, and compiled FlexAttention path without a
local attention reimplementation.

The driver deliberately avoids projecting prompt hidden states through the
126,464-token vocabulary head. LLaDA only uses logits at the 128 generated
positions: prompt positions are fixed, excluded by the mask, and never selected
for transfer. At 64k, projecting every prompt position creates an otherwise
unused BF16 logits tensor of about 16.6 GiB and can OOM an 80 GiB GPU. A forward
hook slices the final normalized hidden state to the generated suffix immediately
before the unchanged vocabulary head. Every transformer layer still processes
the complete sequence, so attention work, SparseD masks, BLASST masks, generated
tokens, and the timed algorithm settings are unchanged.

SparseD's cached `fine_mask` and FlexAttention `block_mask` are cleared between
methods and prompt lengths, followed by allocator cache release. This cleanup is
outside timed regions; each timed method still has its own full warmup.

BLASST uses the same artifact model. Its existing `flash_attn_func` layer hooks
are replaced by the fused 128x64 bidirectional BLASST kernel. The BLASST
generation loop follows the official generator while updating lambda from the
remaining generated-token mask ratio before each model forward.

The checked-in BLASST noise schedule was calibrated at length 4096. At other
lengths the driver applies BLASST's paper rule `lambda proportional to 1/L`,
records the exact derived thresholds, and retains the calibrated noise bucket
boundaries and four-warp/two-stage launch.

## Run the paper latency sweep

The complete sweep is expensive, especially at 64k:

```bash
conda run -n ljy_dlm python scripts/llada_blasst_vs_sparsed.py
```

For a fast integration check:

```bash
conda run -n ljy_dlm python scripts/llada_blasst_vs_sparsed.py \
  --prompts short_context --seq-len 128 --steps 128 \
  --sparsed-select 0.5 --sparsed-block-size 32
```

To reproduce the paper's 64k step-count endpoint:

```bash
conda run -n ljy_dlm python scripts/llada_blasst_vs_sparsed.py \
  --prompts 64k --steps 1024
```

The report is written incrementally to
`outputs/llada_official_sparsed_vs_blasst.json` after every completed prompt.
It contains actual tokenized prompt and total sequence lengths, median latency,
speedup versus dense, method-to-method speedup, peak allocated memory, decoded
answers, expected-answer correctness, and generated-token agreement with the
dense run. `--collect-blasst-stats` adds an untimed generation for physical
BLASST counters and can substantially extend long-context runs.

## Accuracy scope

The artifact ships one example prompt per length, not the evaluation harness or
datasets used for paper Table 1. The driver scores those supplied prompts—the
long prompts all ask whether Scott Derrickson and Ed Wood share a nationality,
whose expected answer is “yes”—and reports token agreement with dense output.
This is a matched correctness check, not a reproduction of Table 1.

Reproducing Table 1 additionally requires the authors' external MMLU, GSM8K,
HumanEval, and RULER harness with the paper's task-specific few-shot, generation
length, step, block-length, and batch-size settings. The JSON labels this
limitation explicitly rather than presenting example-prompt correctness as
benchmark accuracy.
