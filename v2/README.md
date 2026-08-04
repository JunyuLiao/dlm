# Fast-dLLM v2: Efficient Block-Diffusion Large Language Model

[![Project](https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages)](https://nvlabs.github.io/Fast-dLLM/v2)
[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2509.26328)
[![Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_7B)

Fast-dLLM v2 is a carefully designed block diffusion language model (dLLM) that efficiently adapts pretrained autoregressive (AR) models into dLLMs for parallel text generation, requiring only approximately 1B tokens of fine-tuning. This represents a **500x reduction** in training data compared to full-attention diffusion LLMs while preserving the original model's performance.

## 🎬 Demo
https://github.com/user-attachments/assets/f2e055f5-3a44-41ca-9ef8-c84cf3ac2951

## 🎯 Key Features

### 1. **Block Diffusion Mechanism**
- Novel training recipe combining block diffusion with complementary attention masks
- Enables blockwise bidirectional context modeling 
- Token shift mechanism to retain autoregressive characteristics

<div align="center">
  <img src="asset/training_recipe.png" alt="Training Recipe" width="700"/>
  <p><em>Block-wise causal attention mask and complementary training strategy</em></p>
</div>

### 2. **Hierarchical Caching System**
- **Block-level cache**: Stores historical context representations across blocks
- **Sub-block cache**: Enables efficient parallel generation within partially decoded blocks

### 3. **Parallel Decoding Pipeline**
- Achieves up to **2.5x speedup** over standard AR decoding
- Real-time visualization of the denoising process
- Maintains generation quality while delivering state-of-the-art efficiency

<div align="center">
  <img src="asset/visualization_animation.gif" alt="Generation Process Visualization" width="700"/>
  <p><em>Block-level autoregressive generation with sub-block parallelization</em></p>
</div>

## 🚀 Performance

### Throughput Comparison
Fast-dLLM v2 significantly outperforms baselines in both efficiency and accuracy:
- **2.54× higher throughput** than Qwen2.5-7B-Instruct
- **5.2% accuracy improvement** over Fast-dLLM-LLaDA

<div align="center">
  <img src="asset/throughput.png" alt="Throughput Comparison" width="700"/>
  <p><em>Throughput and accuracy comparison across different model variants</em></p>
</div>

### Benchmark Results
Comprehensive evaluation across diverse tasks:

| Model Size | Model | HumanEval-Base | HumanEval-Plus| MBPP-Base | MBPP-Plus |  GSM8K | Math | IFEval | MMLU | GPQA | Average |
|------------|-------|-----------|--|---|---|-------|------|--------|------|------|---------|
| **1B-scale** | Fast-dLLM v2 (1.5B) | 43.9 | 40.2 | 50.0 | 41.3 | 62.0 | 38.1  | 47.0 | 55.1 | 27.7 | **45.0** |
| **7B+ scale** | Fast-dLLM v2 (7B) | 63.4 | 58.5 | 63.0 | 52.3 | 83.7 | 61.6 | 61.4 | 66.6 | 31.9 | **60.3** |

<div align="center">
  <img src="asset/benchmark_results.png" alt="Benchmark Results" width="800"/>
  <p><em>Comprehensive benchmark comparison across diverse tasks</em></p>
</div>


## 🏋️ Training

### Environment Setup
First, create and activate a conda environment:

```bash
conda create -n lmflow python=3.9 -y
conda activate lmflow
conda install mpi4py
```

### Installation
Install the package in development mode:

```bash
pip install -e .
```

### Data Preparation
Download the training data (e.g., Alpaca dataset):

```bash
cd data
bash download.sh alpaca
```

### Fine-tuning
Run the fine-tuning script:

```bash
bash train_scripts/finetune_alpaca.sh
```

This will start the training process using the Alpaca dataset with the optimized block diffusion training recipe.

## 🎮 Quick Start

### Interactive Chatbot
Launch the Gradio-based web interface:

```bash
python app.py
```

This will start a web server at `http://localhost:10086` with:
- Real-time conversation interface
- Live visualization of the denoising process
- Adjustable generation parameters (block size, temperature, threshold)
- Performance metrics display

### Command Line Chat
For a simple command-line interface:

```bash
python run_chatbot.py
```

Commands:
- Type your message and press Enter
- `clear` - Clear conversation history
- `exit` - Quit the chatbot


## 📊 Evaluation

### Run Benchmark Evaluation
Execute the evaluation script for comprehensive benchmarking:

```bash
bash eval_script.sh
```

This script evaluates the model on:
- **MMLU**: Massive Multitask Language Understanding
- **GPQA**: Graduate-level Google-Proof Q&A
- **GSM8K**: Grade School Math 8K
- **Minerva Math**: Mathematical reasoning
- **IFEval**: Instruction following evaluation

### Custom Evaluation
For custom evaluation with specific parameters:

```bash
accelerate launch eval.py \
    --tasks gsm8k \
    --batch_size 32 \
    --num_fewshot 0 \
    --model fast_dllm_v2 \
    --model_args model_path=Efficient-Large-Model/Fast_dLLM_v2_7B,threshold=0.9
```

### Reference 2D-BLASST experiment

The repository includes an opt-in, accuracy-only implementation of the
two-dimensional BLASST tile decision. It keeps dense QK computation and
emulates skipped softmax/PV tiles with a score mask; it is not a CUDA/Triton
kernel and does not claim runtime speedup. The existing dense SDPA path is
called directly when the feature is disabled.

The reference experiment uses diffusion `block_size=16` by default. This is
independent of the fixed BLASST Q/KV physical tile sizes of 128/64. Sub-block
splitting and the associated dual block-cache path remain disabled unless
explicitly requested; ordinary read-only KV caching remains available.

Run ordinary-cache generation in dense and sparse modes:

```bash
python scripts/run_blasst_2d.py \
    --output-dir ../results/blasst_2d/dense

python scripts/run_blasst_2d.py \
    --enable-blasst-2d \
    --blasst-lambda 0.5 \
    --collect-blasst-stats \
    --output-dir ../results/blasst_2d/lambda_0p5
```

Run paired same-state and end-to-end evaluation:

```bash
python scripts/eval_blasst_2d.py \
    --blasst-lambda 0.5 \
    --collect-blasst-stats \
    --output-dir ../results/blasst_2d_lambda_0p5
```

The evaluation exports `summary.json`, `per_step.csv`, `per_layer.csv`,
`per_head.csv`, `run_config.json`, paired accuracy artifacts, and `report.md`.
Recorded BLASST query length equals the configured diffusion block size.

The older controlled valid-context experiment is retained for historical
comparison:

```bash
python scripts/sweep_blasst_controlled.py \
    --model-path /path/to/Fast_dLLM_v2_7B \
    --output-dir ../results/blasst_controlled_sweeps
```

That script uses artificial long-context construction and is not the basis for
the RULER claims below.

Run paired labeled sanity metrics on that same manifest:

```bash
python scripts/eval_blasst_mixed_tasks.py \
    --model-path /path/to/Fast_dLLM_v2_7B \
    --max-new-tokens 512 \
    --output-dir ../results/blasst_controlled_sweeps
```

This reports GSM8K extracted-answer exact match and resource-limited,
execution-based HumanEval pass@1, plus agreement with dense generation.
Agreement is a behavioral diagnostic, not task accuracy. With only eight
examples per benchmark these scores are regression sanity checks; Wilson
intervals are included and the full benchmark is still required for an
accuracy claim. Both official source files are checksum validated.

### Reproducible RULER evaluation

The primary long-context evaluation is a separate pipeline using NVIDIA's
official RULER generators and synthetic-task scorers at pinned commit
`c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`:

```bash
python scripts/eval_blasst_ruler.py \
    --phase all \
    --model-path /path/to/Fast_dLLM_v2_7B \
    --ruler-root /path/to/NVIDIA-RULER-at-the-pinned-commit \
    --output-dir ../results/blasst_ruler
```

The pipeline uses model-tokenizer-verified prompts from NIAH multi, variable
tracking, and frequent-word extraction. It caches deterministic samples and
states, sweeps context lengths 512–16384 and diffusion block/query lengths
1–64 over eight λ values, and runs paired dense/sparse official RULER scoring
on 100 balanced 8K examples at λ=0.003 and block/Q=16. The block sweep exactly
reuses the 8K context samples. Sub-block splitting and the dual block cache
remain disabled; physical Q/KV tiles remain 128/64.

Outputs include aggregate, per-task, and per-example CSVs; sample manifests
with seeds, tasks, actual lengths, source hashes, and generator commands;
correctness JSON; and publication-quality PNG/PDF plots. Sparsity is always
aggregated from total integer counts. This is dense-QK observation of
theoretical skipped softmax/PV work and does not measure runtime speedup.

### DualCache sub-block experiment

The controlled DualCache experiment reuses the exact cached set of 100
balanced 8K RULER examples above. It fixes λ=0.003 and physical Q/KV tiles
128/64. The DualCache curve fixes outer block size 64 and sweeps sub-block
sizes 4, 8, 16, and 32; the standalone comparison curve disables DualCache
and sets both the outer/query block to 4, 8, 16, and 32:

```bash
PYTHONPATH=/path/to/transformers:/path/to/NVIDIA-RULER \
python scripts/eval_blasst_dualcache_ruler.py \
    --phase all \
    --model-path /path/to/Fast_dLLM_v2_7B \
    --ruler-root /path/to/NVIDIA-RULER-at-the-pinned-commit \
    --source-results-dir ../results/blasst_ruler \
    --output-dir ../results/blasst_dualcache_ruler
```

On the fixed manifest, official RULER accuracy was 73.43% dense and 71.47%
for sparse generation without DualCache. DualCache results for sub-block
sizes 4/8/16/32 were 75.20%/72.92%/74.50%/75.97%, with globally aggregated
physical tile sparsity 53.85%/55.33%/54.48%/49.69% versus 44.28% for the
no-DualCache sparse reference. These 100-example point estimates are not
uncertainty bounds.

For standalone no-DualCache block sizes 4/8/16/32, accuracy was
86.40%/84.02%/78.00%/77.83% and global physical sparsity was
68.32%/61.59%/55.90%/49.13%. Thus DualCache did not improve the paired
accuracy–sparsity result over decoding directly at the same update size on
these measurements; at size 32 it gained 0.57 percentage points of physical
sparsity while losing 1.86 points of accuracy. This is a system-level
comparison rather than a pure cache on/off ablation because the standalone
curve changes the outer diffusion block boundary.

The output records every denoising attention forward and every layer/head,
runtime-observed query lengths, forward-weighted and globally
count-aggregated sparsity, per-query-length decompositions, official
per-task/per-example accuracy, correctness checks, a Markdown report, and
PNG/PDF plots. It remains a dense-QK reference evaluation and makes no
runtime-speedup claim.

## 🏗️ Architecture

### Training Recipe
- **Token Shift Mechanism**: Each masked token is predicted using the logit of its preceding token
- **Block-wise Causal Attention**: Access to all clean tokens from previous blocks and noisy tokens within current block
- **Complementary Masks**: Alternate masking patterns ensure every token position is learned

### Generation Process
1. **Block-level Generation**: Autoregressive at the block level
2. **Sub-block Parallelization**: Parallel decoding within blocks for efficiency
3. **Hierarchical Caching**: Block and sub-block level caching for speed optimization

## 📁 File Structure

```
v2/
├── app.py                    # Gradio web interface
├── run_chatbot.py           # Command-line chatbot
├── eval.py                  # Evaluation harness integration
├── eval_script.sh           # Benchmark evaluation script
├── generation_functions.py  # Core generation algorithms
├── index.html              # Project webpage
├── asset/                  # Visual assets
│   ├── demo.mp4
│   ├── benchmark_results.png
│   ├── throughput.png
│   ├── training_recipe.png
│   └── visualization_animation.gif
└── README.md               # This file
```

## 🎨 Visualization Features

The web interface provides real-time visualization of:
- **Denoising Process**: Watch tokens being unmasked in real-time
- **Generation Progress**: Visual feedback of the generation pipeline
- **Performance Metrics**: Live throughput and timing information
- **Slow Motion Replay**: Detailed step-by-step visualization

## 🔬 Technical Details

### Model Architecture
- Based on Qwen2.5 architecture with block diffusion modifications
- 7B parameter model with efficient parallel decoding capabilities
- Custom attention mechanisms for block-wise processing

### Optimization Techniques
- Block-level KV caching for reduced computation
- Sub-block parallel processing for improved throughput
- Confidence-aware token unmasking for quality preservation

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](../CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](../LICENSE) file for details.

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{wu2025fastdllmv2efficientblockdiffusion,
      title={Fast-dLLM v2: Efficient Block-Diffusion LLM}, 
      author={Chengyue Wu and Hao Zhang and Shuchen Xue and Shizhe Diao and Yonggan Fu and Zhijian Liu and Pavlo Molchanov and Ping Luo and Song Han and Enze Xie},
      year={2025},
      eprint={2509.26328},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.26328}, 
}
```

## 🙏 Acknowledgements

We thank [Qwen2.5](https://github.com/QwenLM/Qwen2.5) for the base model architecture
