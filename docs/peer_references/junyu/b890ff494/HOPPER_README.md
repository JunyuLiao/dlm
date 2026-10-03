> Peer reference snapshot, author/source: JunyuLiao, `ljy/value_aware` at `b890ff49488474c5d476df44019597dc045a5969`.
> Original: https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/README.md
> Relative links point to the pinned original; private absolute paths are redacted.
> Peer claims are quoted reference material, not our verified results or contribution.

# Hopper value-direction router

This directory contains the current integrated implementation of
value-direction-aware sparse attention for DiffusionGemma. It targets one
NVIDIA Hopper GPU (SM90/H100), BF16 Q/K/V tensors, native grouped-query
attention, and physical `128 × 64` query/KV tiles.

The main method is **Gaussian32**. It sketches each native KV-head value into
32 FP32 dimensions, estimates the candidate block's attention-output change,
and skips a tile when all valid query rows have low estimated impact. **C_gate**
uses the same value router and multiplies its per-query risk by a causal
sensitivity coefficient. Dense and BLASST are retained as controls.

This is an inference and systems research path. It preserves the native
DiffusionGemma sampler and decoder; a sparse run may still use more denoising
calls, and a physical skip count is not a wall-clock speedup claim.

## Routing contract

For each native layer and KV head, the projection cache stores

$$
z_k=v_kR,\qquad R_{ab}\sim\mathcal{N}(0,1/32),
$$

with a fixed projection seed (`1729`). For query row `i` and candidate tile
`J`, the router computes a within-tile softmax, the projected weighted
direction $\mu_i^J$, and candidate mass $\alpha_i^J$. If $o_i$ is the
projected output from earlier retained tiles, the centered candidate change is

$$
\Delta_i^J=\alpha_i^J(\mu_i^J-o_i).
$$

The physical decision is the maximum risk over the valid rows in a query tile.
The Hopper implementation compares a normalized log magnitude of this change
against `log_threshold`; this is monotone with the squared magnitude used in
the method description. Retained attention uses the original BF16 values, and
skipped tiles leave the retained softmax/projected state unchanged.

The kernel still computes QK, within-block softmax, projected routing
statistics, and the conservative row vote for candidate tiles. A skipped tile
omits the retained full-dimensional PV contribution. The BLASST control uses a
separate mass-based route and can omit block-softmax work under its own
convention.

## Methods

| Method | Implementation | Threshold parameter |
|---|---|---|
| `native_dense` / `dense` | Native DiffusionGemma SDPA; all eligible tiles retained | None |
| `kernel_dense` | Hopper kernel in dense mode; useful for backend parity | `-inf`/all retained |
| `kernel_gaussian32` / `gaussian32` | Value-direction route with rank-32 Gaussian sketches | local/global `log_threshold` |
| `C_gate` | Gaussian32 plus causal query sensitivity | local/global `log_threshold` |
| `kernel_blasst` / `blasst_aggressive` | BLASST mass/block-max route | local/global `log_scale` |

The BLASST parameter is converted inside the integration to a threshold with
the valid KV length. The aggressive control allows values outside the original
`lambda <= 1` range. It remains a baseline for comparison, not the primary
value-direction method.

### C_gate state and kernel interface

The query sensitivity is a contiguous FP32 `B × Q` tensor passed through ABI
v4. On call 1, every query receives `1 + beta`. After the sampler completes a
call, the next call consumes only completed outcomes:

```text
u_conf       = sqrt(max(1 - top1_probability, 0))
q            = EMA(previous_renoised_mask), initialized to 1
stable_run   = consecutive accepted calls, reset by a top-1 flip
gate         = 1 - exp(-stable_run / gate_tau)
h            = 1 - gate * (1 - q) * (1 - u_conf)
s            = clip(1 + beta * h, 1, 1 + beta)
```

The CUDA router adds `log(s)` to each active row's log risk before taking the
physical tile maximum. `State.observe_logits` is called from the native
sampler hook after acceptance, so current-call logits cannot influence the
current routing decision. `UniformThresholdState` installs the same local and
global threshold for call 1, call 2, and all later calls.

Other causal formulas (`M_prior`, `C_prior`, `T_prior`, `T_smooth`,
`T_hybrid`, `T_run`, the other stable-run gates, and the historical `M/C/T`
composites) are available in `query_adaptive.py` for controlled studies.

The causal variants use the same bounded coefficient `s = clip(1 + beta*h,
1, 1 + beta)`:

| Variant | Base signal | Causal completion |
|---|---|---|
| `M_prior` | `u_M = m_ref / (margin + m_ref)` | `h = q + (1 - q) * u_M` |
| `C_prior` | `u_C = sqrt(max(1 - confidence, 0))` | `h = q + (1 - q) * u_C` |
| `T_prior` | EMA of top-1 flips | `h = q + (1 - q) * flip_ema` |
| `T_smooth` | EMA of top-1 flips | Same as `T_prior`, with a slower `q` EMA |
| `T_hybrid` | Flip EMA plus confidence drift | Noisy-OR of `q`, flip EMA, and drift |
| `T_run` | Flip EMA | Replaces `q` with `exp(-stable_run / 2)` |
| `M_gate`, `C_gate`, `T_gate` | Margin, confidence, or flips | Stable-run gate delays relaxation |

`M_gate_norm`, `C_gate_soft`, `C_tail`, `H_anchor`, and `C_soft` are additional
diagnostic variants. All of them are updated only after a completed native
sampler call and can be evaluated under the same uniform-threshold adapter.

### Frozen hyperparameters and threshold artifacts

The calibration driver freezes every value that can change routing or the
denoising trajectory in `configuration.json`. The current AIME26 uniform
protocol uses these defaults:

| Parameter | Value | Scope |
|---|---:|---|
| `beta` | `3.0` | Sensitivity range is `1` to `1 + beta = 4`; call 1 always uses 4 |
| `gamma` | `0.5` | Completed-call top-1 flip EMA and the default trajectory EMA |
| `trajectory_gamma` | `0.5` | `M_prior`, `C_prior`, `T_prior`, and `T_hybrid` |
| `smooth_trajectory_gamma` | `0.8` | `T_smooth` |
| `gate_trajectory_gamma` | `0.65` | `M_gate`, `M_gate_norm`, `C_gate`, `T_gate`, and `C_gate_soft` |
| `gate_tau` | `2.5` | Stable-run gate time constant for gate variants |
| `m_ref` | loaded from the frozen parent configuration | Margin reference for `M_*`; it is not the Gaussian32 value-RMS reference |
| Gaussian rank | `32` | Value sketch dimension |
| Projection seed | `1729` | One projection per native layer and KV head |
| Physical tile | `128 × 64` | Query rows × key/value rows |
| Canvas / step cap | `256` / `48` | Native DiffusionGemma request schedule |

The integration defaults that affect execution are
`precision=tf32x3_register`, `projection=fused`, `allocation=normal`,
`bootstrap=false`, and `tma=true` for unmasked value-router calls. The
masked path disables TMA; BLASST TMA is off unless `blasst_tma=true` is
explicitly selected. Calibration and audit runs keep `collect=true` and
diagnostics enabled, while clean timing runs must disable routing/profile
instrumentation separately.

`T_run` uses `exp(-stable_run / 2)` for its causal trajectory completion;
`gamma=0.5` still controls its completed-call flip EMA.

The native temperature processor remains
`0.4 + 0.4 * (remaining_schedule_step / 48)`. Calibration uses the fixed
seed-42 six-question manifest `aime26/{2,8,14,20,23,30}`. The standard final
development runs use seeds `42,43,44`; set `AIME_SEEDS` or
`AIME_FINAL_SEEDS` explicitly when reproducing a different seed set.

The archived AIME/HumanEval configurations currently use
`m_ref = 14.258454322814941`; a new parent configuration must be treated as a
new protocol if this value changes. Optional diagnostic variants additionally
record `anchor_lambda=0.90`, `soft_prior_alpha=0.25`, `tail_tau=0.75`,
`tail_lambda=0.5`, `tail_trajectory_gamma=0.65`, and
`soft_gate_lambda=0.2`. `M_gate_norm` has one extra fitted value,
`margin_scale`; it is the frozen robust scale
`max(IQR(log1p(raw_margin))/1.349, 0.05)` from dense calibration trajectories.
The other gate variants do not consume `margin_scale`.

Gaussian32's value-RMS reference is recomputed from the valid K/V values for
each native layer and call by `integration.Sketches`; it is not a second
user-supplied threshold or a fixed constant.

The threshold files are protocol outputs, not universal constants. For a value
router, `thresholds/<method>_s<target>.json` contains an attained record such
as:

```json
{
  "status": "attained",
  "target": 50,
  "policy": {
    "call1": {"local": {"log_threshold": 0.0}, "global": {"log_threshold": 0.0}},
    "call2": {"local": {"log_threshold": 0.0}, "global": {"log_threshold": 0.0}},
    "late":  {"local": {"log_threshold": 0.0}, "global": {"log_threshold": 0.0}}
  },
  "max_error": 0.0
}
```

The zeros above are schema placeholders, not recommended thresholds. Use the
full-precision values from the selected record. A uniform run must satisfy
`policy.call1 == policy.call2 == policy.late`; the runtime passes the
`policy.late` local/global mapping to `UniformThresholdState`, which returns
that same pair on every call. The calibration acceptance test is pooled whole,
local, and global physical sparsity within ±2 percentage points of the target;
the selected feasible point then minimizes mean canvas steps. A threshold
record is valid only with its matching configuration fingerprint, source
hashes, model revision, kernel provenance, manifest, and `status="attained"`.

BLASST uses a different field: `log_scale`, not `log_threshold`. The
integration converts it to the per-call kernel threshold by subtracting
`log(valid_kv_length)` and applies `cap_one` when present. Do not copy a value
router threshold into a BLASST policy or vice versa.

For a direct integration, the executable reference is
[`aime_temporal_sweep._run_one`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/aime_temporal_sweep.py). It loads the frozen
record, calls `install(...)` with the local/global policy, constructs
`UniformThresholdState` with the values in `configuration.json`, and wraps the
native sampler with `observe(...)`. This preserves causal updates: completed
sampler masks and logits affect only the next routing call.

## Implementation files

The integration order is:

1. [`src/dllm/models/adapters/diffusion_gemma.py`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/src/dllm/models/adapters/diffusion_gemma.py)
   loads the gated DiffusionGemma checkpoint with native `generate()` and
   exposes the attention module selector.
2. [`integration.py`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/integration.py) installs a request-local attention
   override through the model's registry. It owns structural masks, sketch
   leases, prefix invalidation, boundary/canvas refresh, and deferred tile
   accounting.
3. [`projection.py`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/projection.py) refreshes Gaussian32 sketches and native
   value RMS references. The fused path is optional; the Torch projection is a
   separate control.
4. [`csrc/value_direction.cu`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/csrc/value_direction.cu) implements the SM90
   kernel. [`csrc/value_direction.h`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/csrc/value_direction.h) defines the
   versioned ABI, including the optional sensitivity pointer.
5. [`cuda.py`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/cuda.py) builds and loads the shared library. [`torch_build.py`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/torch_build.py)
   and [`csrc/torch_bridge.cpp`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/csrc/torch_bridge.cpp) provide the explicit
   ATen bridge used by the PyTorch model.
6. [`query_adaptive.py`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/query_adaptive.py) implements sensitivity formulas,
   state initialization, causal sampler observation, and diagnostics.
7. [`query_sensitivity_uniform.py`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/query_sensitivity_uniform.py) adapts the
   state to a uniform local/global threshold pair.

Reference and calibration code under
[`../diffusion_gemma_jl_output_aware`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/diffusion_gemma_jl_output_aware) and
[`../diffusion_gemma_value_aware`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/diffusion_gemma_value_aware) is retained
for operator checks and historical comparisons. It is not the current H100
integration entry point.

## Environment and installation

From the repository root, use Python 3.10 or newer, a CUDA-enabled PyTorch
environment, and the checked-in `transformers==5.11.0` dependency:

```bash
cd <PRIVATE_PATH_REDACTED>
pip install -e .
pip install -e '.[eval]'
pip install -e '.[test]'
```

The production path requires an SM90 GPU and BF16 DiffusionGemma. The model
adapter loads `google/diffusiongemma-26B-A4B-it`; accept the checkpoint terms
and run `hf auth login` if Hugging Face access is gated.

## Build

Build the CUDA library explicitly. The build uses `/usr/local/cuda/bin/nvcc`
and writes a content-hashed binary and build proof under `results/`:

```bash
PYTHONPATH=src:. python -m experiments.value_direction_hopper.cuda \
  --fast-sfu --inline-roles
```

Then compile the ATen bridge against exactly that binary and the active
PyTorch version:

```bash
PYTHONPATH=src:. python -m experiments.value_direction_hopper.torch_build \
  --kernel /absolute/path/to/value_direction_HASH.so
```

The bridge writes `provenance.json`. `cuda.Kernel` checks the ABI version,
binary hash, and PyTorch version before loading it. Do not JIT-build or change
the kernel during a generation run. A C ABI or the Python loader alone is not
a TensorRT-LLM integration.

## Reproduction workflows

Use a new output directory for every changed build, protocol, threshold, or
manifest. Existing run directories contain immutable fingerprints and cannot
be mixed with another configuration.

### Kernel/reference smoke tests

```bash
VD_KERNEL=/absolute/path/to/value_direction_HASH.so \
VD_TORCH_KERNEL=/absolute/path/to/value_direction_torch_HASH.so \
PYTHONPATH=src:. python -m pytest -q \
  tests/test_value_direction_hopper.py tests/test_value_direction_hopper_report.py
```

The CPU-side query state suite is independent of the GPU binary:

```bash
PYTHONPATH=src:. python -m pytest -q tests/test_query_adaptive.py
```

### RULER dense/Gaussian32/BLASST controls

```bash
PYTHONPATH=src:. python -m experiments.value_direction_hopper.experiment run \
  --root results/my_ruler_run \
  --library /absolute/path/to/value_direction_HASH.so \
  --torch-library /absolute/path/to/value_direction_torch_HASH.so

PYTHONPATH=src:. python -m experiments.value_direction_hopper.experiment report \
  --root results/my_ruler_run
```

`experiment.py` uses the frozen RULER manifest and policy sources, records
source hashes, resumes completed shards under an exclusive worker lock, and
reports pooled physical eligible/skipped tiles. Its methods are
`native_dense`, `kernel_gaussian32`, and `kernel_blasst`.

### AIME uniform thresholds

The baseline driver calibrates Gaussian32 and aggressive BLASST on the six
existing AIME calibration IDs, then copies one local/global pair to all calls:

```bash
PYTHONPATH=src:. python -m experiments.value_direction_hopper.aime_uniform_baselines \
  calibrate --root results/aime_uniform_baselines
PYTHONPATH=src:. python -m experiments.value_direction_hopper.aime_uniform_baselines \
  run --root results/aime_uniform_baselines
PYTHONPATH=src:. python -m experiments.value_direction_hopper.aime_uniform_baselines \
  report --root results/aime_uniform_baselines
```

Before `run`, verify `calibration_complete.json` has `"passed": true` and
that every requested threshold has `status="attained"`. `search_incomplete`
records retain the nearest measured point for diagnosis, but they are not
calibrated evaluation policies.

The causal driver supports the same uniform threshold contract:

```bash
export AIME_OUTPUT_ROOT=/absolute/path/to/results/aime_query_uniform
export AIME_UNIFORM_METHODS=C_gate,T_prior,T_smooth
export AIME_TARGETS=50
PYTHONPATH=src:. python -m experiments.value_direction_hopper.aime_query_sensitivity_uniform prepare
PYTHONPATH=src:. python -m experiments.value_direction_hopper.aime_query_sensitivity_uniform calibrate
AIME_FINAL_SEEDS=42,43,44 PYTHONPATH=src:. \
  python -m experiments.value_direction_hopper.aime_query_sensitivity_uniform run
PYTHONPATH=src:. python -m experiments.value_direction_hopper.aime_query_sensitivity_uniform \
  --root "$AIME_OUTPUT_ROOT" report
```

For the C_gate screen/finalist workflow, use
`aime_query_sensitivity_gated.py prepare`, `run`, and `report` with an explicit
`--root`. It records each arm's threshold pair, causal parameters, screen
selection, per-seed outcomes, and final disagreement table.

These AIME calibration IDs are included in the historical final manifest.
Reports must label the resulting accuracy as development evidence rather than
held-out validation.

## Accounting and integration limits

Physical sparsity is computed as

```text
sum(skipped eligible tiles) / sum(eligible tiles)
```

and is reported separately for local, global, and all attention types. Dense
prefill is excluded from this accounting. The native adapter uses a 256-token
canvas and its native denoising schedule; `block_size` is reported by the
adapter but does not override that canvas.

The current adapter loads native DiffusionGemma attention with SDPA. A result
that beats an SDPA fallback is not evidence of a FlashAttention or TensorRT
speedup. Report synchronized end-to-end timing, denoising calls, output length,
physical sparsity, and the dense dispatch separately.

The request-local sketch cache invalidates on encoder entry and on changed
prefix object/version/shape. Boundary and canvas values refresh each call.
Concurrent requests sharing one bound model are unsupported by this
integration.
