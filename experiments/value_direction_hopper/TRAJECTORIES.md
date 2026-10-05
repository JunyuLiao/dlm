# Denoising diagnostics

Run from the repository root with the existing `ljy_dlm` environment and
`PYTHONPATH=src:.`. The canonical output is
`results/value_direction_trajectory_v2`. The initial v1 traces are preserved;
their top-token tie handling was corrected before collecting v2.

This experiment uses the frozen 873c3073e54a37b8 kernel, its existing ATen
bridge, frozen Gaussian32/BLASST thresholds, and the original 130 RULER4K
examples. It does not compile or optimize a kernel. All normal inference paths
are unchanged. Observation is enabled only by the scoped `trajectory.observe`
context in the diagnostic CLI:

```bash
python -m experiments.value_direction_hopper.trajectory_experiment smoke --root results/value_direction_trajectory_v2
python -m experiments.value_direction_hopper.trajectory_launch --root results/value_direction_trajectory_v2
python -m experiments.value_direction_hopper.trajectory_experiment report --root results/value_direction_trajectory_v2
```

The detached worker has an exclusive lock, writes each complete sample
atomically, verifies trace hashes on resume, and records failures without
discarding independent samples. Launch only one GPU worker. Check its status
and launch log approximately every 15 minutes. Completed adaptive and fixed
phases automatically regenerate the analysis from raw shards. Report writes
should not overlap one another.

## Controls

- `native_dense`: the exact original dense backend and local-mask convention.
- `kernel_dense`: all tiles retained using the same kernel/mask/arithmetic path
  as value routing. This is the matched structural/numerical dense control.
- `blasst_capped`: existing BLASST online-max voting with the frozen
  length-dependent threshold capped at lambda=1.
- `kernel_gaussian32`: frozen value-aware Gaussian32 s70.
- `kernel_blasst`: original s70-calibrated aggressive BLASST, permitting
  lambda>1. This is not labelled official-range BLASST.

All five run adaptively on all130 examples. The fixed512 comparison runs
kernel_dense, kernel_gaussian32 and kernel_blasst on all130. It forces exactly
512 decoder calls by observing the real stopper but suppressing its return.
All examples require one256-token canvas under their official output budgets.
The native sampler, random draws and temperature schedule remain in use.
The512 schedule stretches annealing relative to the original48-step maximum;
the fixed experiment is not solely a stopping intervention across regimes.

## Native decoding semantics

DiffusionGemma initializes the canvas with random vocabulary tokens. Each
iteration selects tokens using a cumulative entropy budget, excluding the
largest selected entropy, then renoises positions not selected. Selection is
recomputed and can revoke previously accepted positions. There is no monotone
permanent unmasking count. We save accepted positions, newly selected positions,
revocations, unique-ever acceptance, and remaining positions selected for
renoising. The native stopper requires stable argmax across all256 positions
and mean processed-logit entropy below0.005.

Observation wraps the existing native denoising step, sampler and stopper;
it does not copy their control flow. The native argmax tie rule is verified on
each step. Confidence and margins use FP32 normalization of raw and processed
logits; entropy uses the native Categorical implementation/dtype because this
is what controls actual acceptance and stopping. No full logits are persisted.
Raw traces contain per-position predictions, probabilities, margins, entropy,
acceptance, instability and the true/would-stop conditions.

The smoke test repeats two existing prompts through all five methods with and
without instrumentation. It requires identical output IDs and realized steps.
Three original methods also require exact archived-output/step reproduction.
The full report checks all available archived pairs and halts phase progression
if reproduction fails.

## Attention and time measurements

On every sparse/unpruned-kernel step, layers0/5/29, heads0/8 and queries
0/31/63/127/191/255 are compared with dense attention on identical QKV and
the same structural mask. Store squared error, dense norm and cosine sums;
aggregate L2 as sqrt(sum error_sq / sum dense_sq). This samples operator
error along each method's own trajectory. It is not a shared generation state
after trajectories diverge, and it is not all-layer error coverage.

Instrumentation includes reductions, device/host transfers and diagnostic
attention. Its wall and CUDA-event timings must not be presented as clean
production latency. Original uninstrumented timings remain authoritative for
the observed slowdown; wall/call ratios include fixed costs and do not prove a
kernel-level saving.

## Outputs and interpretation

Each shard references a compressed trace and SHA256. Reports are regenerated
without loading the model. CSVs contain per-step, per-position, sampled
attention, aligned disagreement, per-sample and aggregate results. Alignment
uses equal steps and nearest accepted fraction; acceptance is nonmonotone, so
the latter is descriptive. Token disagreement is not necessarily incorrectness.
Draft scores permit checking whether later iterations actually improve the
official answer metric. Reused examples are not fresh held-out evidence for a
new stopping rule.

Final analysis addresses all eleven research questions and reports paired
prompt-bootstrap accuracy intervals. Any proposed dynamic sparsity or earlier
stopping remains a future experiment, not a validated conclusion from confidence
curves alone.
