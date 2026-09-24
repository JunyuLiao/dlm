# Query-sensitivity-aware value-direction block skipping (v3)

This snapshot contains the Gaussian-32 query-sensitivity implementation and
the completed `query_adaptive_v3` RULER4K study. It is a research result, not
a production-speedup claim. The 130 evaluation prompts were previously
examined during method development, so the measured gains are exploratory.

## Router and decoder contract

The underlying value-direction router computes the attention-mass-weighted,
centered change to its retained projected output for query row `i` and KV
block `J`:

```text
rho[i,J] = alpha[i,J] * ||mu_projected[i,J] - o_projected[i]||_2
           / max(valid_KV_RMS[i], epsilon)
```

`mu_projected` comes from attention-weighted projected token values `Z = VR`,
not a projection of a full-dimensional block output. The temporal (`T`)
variant multiplies **each row's** risk by its own sensitivity before the
physical tile reduction:

```text
u[i,t] = 0.5 * u[i,t-1] + 0.5 * 1[top1[i,t] != top1[i,t-1]]
s[i,t+1] = 1 + beta * u[i,t], beta = 3
skip(I,J) iff max_{i in valid query rows of I}(s[i] * rho[i,J]) < tau_type
```

`top1` is deterministic argmax, not the sampled token. Only predictions from
completed iterations affect the next iteration; sensitivity is one before
flip history exists (the first two iterations). `u` resets for each canvas.
The kernel implements the comparison in log space as
`log(rho) + log(s) < log(tau_type)`. It retains ties and first-support blocks.
Skipped blocks do not update retained softmax/projected/output state. Retained
blocks use original full-dimensional V and ordinary renormalized attention.
The native sampler, reversible acceptance/renoising, self-conditioning,
temperature annealing, and stopping check remain unchanged.

`beta = 3` here is the **query-sensitivity strength**, not a Gaussian
tile-proxy quantile. Other frozen settings: EMA `gamma = 0.5`; Gaussian value
projection rank 32 and seed 1729; BF16 DiffusionGemma-26B-A4B-it; FP32
routing arithmetic; physical 128-query × 64-KV tiles; one 256-position canvas;
48 maximum denoising iterations; native temperature 0.8→0.4; generation seed
42. Local and global thresholds are separately calibrated but shared across
layers and heads of each type.

## Frozen T policies

The following are **log thresholds**, in local/global order. The frozen policy
file is [`results/query_adaptive_v3/configs/frozen_policies.json`](../../results/query_adaptive_v3/configs/frozen_policies.json).

| Target physical sparsity | `log tau_local` | `log tau_global` | `beta` | `gamma` |
|---:|---:|---:|---:|---:|
| 50% | -1.0099318266 | -3.1366905212 | 3 | 0.5 |
| 70% | -0.0844297796 | -2.1409701347 | 3 | 0.5 |

Each candidate threshold pair was tested by complete native adaptive
generation on a disjoint 26-prompt calibration set (two per task). Calibration
measured pooled skipped/eligible physical tiles over all decoder calls,
separately for whole-model, local, and global attention. A bounded search
tested at most seven full-generation points, adjusting local/global log
thresholds independently with bracket midpoints or ±0.35 directional steps.
It selected the tested pair minimizing the worst deviation from target across
the three sparsity measurements, stopping when each was within two percentage
points. Accuracy and denoising-step count did **not** select thresholds. The
130 evaluation prompts did not influence threshold selection. See the frozen
per-condition traces in `results/query_adaptive_v3/configs/thresholds/` and
`results/query_adaptive_v3/calibration_traces/`.

## RULER4K result (13 tasks × 10 prompts)

Physical sparsity pools actual skipped/eligible tile counts over all realized
denoising calls; prefix encoding is excluded. Accuracy is the repository's
RULER score. `kernel_dense` is the matched-kernel numerical control; native
dense uses the original SDPA path and is not numerically identical.

| Method | Actual overall/global/local sparsity | Accuracy | Mean calls/canvas |
|---|---:|---:|---:|
| Native dense | 0% / 0% / 0% | 90.3% | 4.00 |
| Matched-kernel dense | 0% / 0% / 0% | 89.4% | 4.08 |
| Unweighted Gaussian-32, target 50% | 49.0% / 48.8% / 49.2% | 90.0% | 4.73 |
| T, target 50% | 49.5% / 49.8% / 49.4% | 90.0% | 4.45 |
| Unweighted Gaussian-32, target 70% | 70.3% / 71.4% / 69.6% | 86.6% | 16.45 |
| T, target 70% | 69.8% / 70.9% / 69.1% | 89.7% | 6.78 |

On the same prompts with generation seeds 43 and 44, T70 scored 89.0% and
90.5%, with 7.15 and 7.26 mean calls. The full
[`results/query_adaptive_v3/report.md`](../../results/query_adaptive_v3/report.md)
contains all method variants, per-task results, paired intervals, diagnostics,
and plots. These results do **not** establish hardware speedup: separate
uninstrumented timing measured T70 at 0.840 s/example versus 0.584 s/example
for native dense on this setup.

## Reproduction and archived data

The source includes the H100 CUDA/ATen kernel, model integration, optional
instrumentation, calibration/evaluation/reporting scripts, and tests. The
snapshot includes the exact v3 configuration, calibration/final manifests,
thresholds, projection metadata, summary tables, per-example/per-canvas and
per-step measurements, timing summaries, diagnostic summaries, and plots.
The three minimal input files under `results/value_direction_sparsity_steps_v2/`
are included because the v3 preparation script reads that source manifest and
base policy. Build the CUDA kernel and ATen bridge for the local environment;
compiled `.so` files and machine-specific absolute paths in archived configs
are not portable. See
[`results/query_adaptive_v3/configs/reproduce.md`](../../results/query_adaptive_v3/configs/reproduce.md)
for the original commands and build prerequisites.

The archived configuration retains its original source checksums. The
published *current* `query_adaptive_study.py` driver has a different checksum
from the original run-time driver, although the archived checksums for
`query_adaptive.py`, `integration.py`, `cuda.py`, the projection module, and
the three source-manifest files match this snapshot. Do not edit the archived
checksum to disguise this distinction. A new run should use a new output root
and record its own provenance.

Redundant per-candidate JSON shard trees and worker logs are not committed:
the frozen directory is 709 MB, while the included per-example/per-step tables
and traces retain the principal measurements. The archived report and outputs
are preserved verbatim; rerun into a new root rather than overwriting them.
