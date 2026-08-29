# Fresh Sol-Attn-style tile routing

This package contains two related protocols:

1. the historical distribution/Math500 threshold-modeling CLI (`collect`,
   `fit`, `eval-math500`, and `report`); and
2. the canonical RULER-16K fresh-routing study (`eval-ruler-routing`,
   `report-ruler`, and `canonical-report`).

The RULER study uses post-normalization, post-RoPE Q/K, the model's real GQA
mapping, native structural masks, and 64x64 logical tiles. For every eligible
query/KV tile it computes

```text
Mean(Q_i) Mean(K_j)^T * native_attention_scale
```

and recomputes the keep/drop mask on every denoising call. Dropped tiles
contribute neither logits nor values. There is no frozen-mask reuse and no
Sol-Attn approximate correction.

## Canonical RULER-16K sweep

Prepare the fixed tokenizer-specific 50-example manifests first, then run the
same named conditions for each adapter:

```bash
PYTHONPATH=src python -m experiments.diffusion_attention_threshold_modeling \
  eval-ruler-routing \
  --adapter diffusion_gemma \
  --model-path MODEL_SNAPSHOT \
  --revision PINNED_REVISION \
  --manifest-path results/attention/routing/ruler16k/manifests/diffusion_gemma/manifest.json \
  --ruler-root /tmp/NVIDIA-RULER \
  --output-dir results/attention/routing/ruler16k/diffusion_gemma \
  --num-samples 50 \
  --threshold-model results/attention/routing/ruler16k/profiled_threshold_model.json
```

Without `--conditions`, this runs dense plus Gaussian, profiled, exact-density
oracle, and deterministic random controls at 75%, 50%, 25%, and 10%, both for
all-KV routing and the prefix-only ablation. Every condition is resumable.

Use `--conditions NAME ...` to run a subset. The canonical block size is
`--q-block-size 64 --kv-block-size 64`; 32x64 and 128x64 are supported for the
small block-size ablation.

## Physical reference and timing

`--routing-execution logical` is the exact dense-masked behavioral reference.
`--routing-execution physical` invokes the retained-tile two-pass QK/AV
reference, which does not materialize dropped QK tiles. Physical timing runs
should use a separate output directory and explicit repetitions, for example:

```bash
PYTHONPATH=src python -m experiments.diffusion_attention_threshold_modeling \
  eval-ruler-routing ... \
  --routing-execution physical \
  --conditions dense gaussian_rho75 \
  --performance-warmups 1 \
  --performance-repeats 3
```

CUDA is synchronized around every measured generation. Per-sample latency is
the median over repeats; model loading and prompt validation are outside the
timed region. Reports expose end-to-end speedup only for physical conditions.
Logical and physical tile densities are never described as speedups.

## Reports

```bash
PYTHONPATH=src python -m experiments.diffusion_attention_threshold_modeling \
  report-ruler \
  --output-dir results/attention/routing/ruler16k \
  --ruler-root /tmp/NVIDIA-RULER

PYTHONPATH=src python -m experiments.diffusion_attention_threshold_modeling \
  canonical-report \
  --bundle-root results/attention/routing/ruler16k
```

The canonical bundle audits prompt/seed identity, paired RULER accuracy,
bootstrap confidence intervals, dense-correct failures/recoveries, logical and
physical density, prefix/canvas density, fallback frequency, retained dense
attention mass, output error, fresh-mask overlap, threshold validation, and
distribution shape. The read-only distribution shards retain native-dense
parity evidence for the first two examples per adapter.
