> Peer reference snapshot, author/source: JunyuLiao, `ljy/value_aware` at `b890ff49488474c5d476df44019597dc045a5969`.
> Original: https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/README.md
> Relative links point to the pinned original; private absolute paths are redacted.
> Peer claims are quoted reference material, not our verified results or contribution.

# Value-direction-aware sparse attention for diffusion LLMs

This repository implements physical-tile sparse attention for the native
DiffusionGemma block-diffusion decoder. The main method is **Gaussian32**: it
uses a fixed rank-32 Gaussian sketch of the value vectors to estimate the
change that a candidate key/value block would make to the attention output.
**C_gate** is a query-adaptive protection variant of the same router. It gives
queries that are still changing, or have not yet demonstrated stable accepted
tokens, a larger routing sensitivity coefficient.

The implementation preserves the model's native proposal, acceptance,
renoising, self-conditioning, stopping rule, temperature schedule, and output
attention computation. Sparsity is applied only to eligible physical
`128 query × 64 key/value` tiles in decoder self-attention. The project is an
H100/SM90 research implementation; skipped-tile counts do not by themselves
establish end-to-end speedup.

## Method

For a value vector $v_k$, Gaussian32 constructs a fixed sketch

$$
z_k = v_k R, \qquad R\in\mathbb{R}^{d_v\times32},
$$

where the entries of $R$ are independent $\mathcal{N}(0,1/32)$ samples.
The projection is generated once per native layer and KV head (seed `1729`)
and is reused for the request.

For query $i$ and candidate KV tile $J$, the router computes the
within-tile softmax weights $w^J_{ik}$, the projected weighted value
direction

$$
\mu^J_i = \sum_{k\in J} w^J_{ik}z_k,
$$

and the candidate attention mass $\alpha^J_i$ from the running softmax
statistics. If $o_i$ is the projected output accumulated from previously
retained tiles, the estimated candidate contribution is

$$
\Delta^J_i = \alpha^J_i(\mu^J_i-o_i).
$$

This keeps blocks with large attention mass and a distinct value direction,
while allowing low-mass, redundant, or internally cancelling blocks to be
skipped. The physical tile is skipped only when every valid query row in the
`128 × 64` tile is below the threshold. The production kernel compares an
equivalent monotone log magnitude, normalized by a native-head value RMS
reference; its `log_threshold` is therefore the logarithm of the routing
threshold. Original BF16 values are used for the retained attention output.

### C_gate query-sensitivity protection

Gaussian32 can receive one FP32 sensitivity coefficient per query row. The
kernel applies it to the value-routing risk before the tile maximum is taken:

$$
\rho_i \leftarrow \rho_i + \log s_i.
$$

The C_gate state is causal. At the first denoising call, every query receives
the maximum coefficient $s_i=1+\beta$. After a call finishes, the state is
updated from the completed sampler mask and completed logits only:

$$
u^C_i=\sqrt{\max(1-p_i,0)},\qquad
q_i\leftarrow\gamma_q q_i+(1-\gamma_q)\,\mathbf{1}[i\text{ was renoised}],
$$

$$
g_i=1-\exp(-r_i/\tau),\qquad
h_i=1-g_i(1-q_i)(1-u^C_i),\qquad
s_i=\mathrm{clip}(1+\beta h_i,1,1+\beta).
$$

Here $p_i$ is the completed-call top-1 probability and $r_i$ is the
completed stable-acceptance run, reset by a top-1 flip. Thus a query remains
protected until it has both stopped being renoised and accumulated stable
history. C_gate does not introduce a phase-specific threshold: one calibrated
local threshold and one calibrated global threshold are copied to call 1,
call 2, and every later call.

The full causal family is implemented in
[`query_adaptive.py`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/query_adaptive.py).
It includes `M_prior`, `C_prior`, `T_prior`, `T_smooth`, `T_hybrid`, `T_run`,
the stable-run gates (`M_gate`, `M_gate_norm`, `C_gate`, `T_gate`), and the
additional confidence/entropy controls used by the studies. The historical
`M`, `C`, `T`, `MT`, and `CT` formulas remain available for comparison.

## Methods and baselines

| Name | Router | Purpose |
|---|---|---|
| **Dense** (`native_dense`) | Native DiffusionGemma SDPA; all eligible tiles retained | Accuracy and trajectory reference |
| **Gaussian32** (`kernel_gaussian32`, `gaussian32`) | Value-direction-aware rank-32 Gaussian projection and centered output-risk estimate | Main sparse method |
| **C_gate** | Gaussian32 plus causal per-query sensitivity protection | Query-adaptive Gaussian32 variant |
| **BLASST** (`kernel_blasst`, `blasst_aggressive`) | Attention-mass/block-max routing using the BLASST convention | Mass-based sparse baseline |

`blasst_aggressive` permits the calibrated parameter to exceed the original
BLASST `lambda <= 1` range when a target sparsity requires it. The Hopper
integration stores this baseline as a `log_scale` and subtracts the valid KV
length logarithm inside the router. Dense and BLASST are controls; Gaussian32
is the primary value-direction-aware method and C_gate is its query-protection
variant.

## Repository map

The following files are the implementation and reproduction surface for this
method:

| File | Role |
|---|---|
| [`src/dllm/models/adapters/diffusion_gemma.py`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/src/dllm/models/adapters/diffusion_gemma.py) | Loads `google/diffusiongemma-26B-A4B-it`, keeps native `generate()`, and exposes the DiffusionGemma attention registry. |
| [`experiments/value_direction_hopper/csrc/value_direction.cu`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/csrc/value_direction.cu) | H100/SM90 CUDA router and retained-output kernel for dense, Gaussian32, and BLASST modes. |
| [`experiments/value_direction_hopper/csrc/value_direction.h`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/csrc/value_direction.h) | Versioned allocation-free CUDA ABI (currently v4), including the optional `B × Q` sensitivity input. |
| [`experiments/value_direction_hopper/cuda.py`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/cuda.py) | Explicit CUDA build and ABI-checked shared-library loader. |
| [`experiments/value_direction_hopper/csrc/torch_bridge.cpp`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/csrc/torch_bridge.cpp) | ATen operator bridge used by the native PyTorch model. |
| [`experiments/value_direction_hopper/torch_build.py`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/torch_build.py) | Builds the hash- and PyTorch-version-pinned ATen bridge. |
| [`experiments/value_direction_hopper/projection.py`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/projection.py) | Optional fused FP32/TF32x3 Gaussian32 sketch and value-norm refresh. |
| [`experiments/value_direction_hopper/integration.py`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/integration.py) | Installs the attention override, request-local sketch leases, structural masks, threshold selection, and deferred tile accounting. |
| [`experiments/value_direction_hopper/query_adaptive.py`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/query_adaptive.py) | Sensitivity formulas, causal state, completed-call updates, stable-run tracking, and sampler hook. |
| [`experiments/value_direction_hopper/query_sensitivity_uniform.py`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/query_sensitivity_uniform.py) | Adapter that enforces one local/global log-threshold pair for every denoising call. |
| [`experiments/value_direction_hopper/aime_uniform_baselines.py`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/aime_uniform_baselines.py) | AIME26 Gaussian32 and BLASST uniform-threshold calibration/evaluation. |
| [`experiments/value_direction_hopper/aime_query_sensitivity_uniform.py`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/aime_query_sensitivity_uniform.py) | AIME26 calibration/evaluation driver for causal query-sensitivity methods. |
| [`experiments/value_direction_hopper/aime_query_sensitivity_gated.py`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/aime_query_sensitivity_gated.py) | Causal stable-run gate screen and finalist workflow, including C_gate. |
| [`experiments/value_direction_hopper/experiment.py`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/experiment.py) | Frozen RULER4K generation for native dense, Gaussian32, and BLASST kernel modes. |
| [`experiments/value_direction_hopper/README.md`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/README.md) | Detailed Hopper build, integration, calibration, and audit instructions. |
| [`tests/test_query_adaptive.py`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/tests/test_query_adaptive.py) | CPU-side formula, causal-history, tile-reduction, and sampler-hook tests. |
| [`tests/test_value_direction_hopper.py`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/tests/test_value_direction_hopper.py) | H100 kernel, ABI, mask, cache, projection, ATen bridge, and CUDA graph tests. |

Historical reference implementations under
[`experiments/diffusion_gemma_jl_output_aware`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/diffusion_gemma_jl_output_aware)
and [`experiments/diffusion_gemma_value_aware`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/diffusion_gemma_value_aware)
are useful for the PyTorch/reference operator and calibration audits. The
Hopper path is the current integrated implementation described here.

## Installation

Use Python 3.10 or newer and a CUDA-enabled PyTorch environment. The checked-in
package requires the DiffusionGemma support in `transformers==5.11.0`:

```bash
cd <PRIVATE_PATH_REDACTED>
pip install -e .
pip install -e '.[eval]'
pip install -e '.[test]'
```

The checkpoint is gated on Hugging Face. Accept its terms and authenticate with
`hf auth login` when the adapter reports a 401/403 or gated-repository error.
The CUDA implementation requires an NVIDIA Hopper GPU (SM90, such as H100),
BF16 Q/K/V tensors, and native head dimensions 256 or 512.

## Build and integrate the Hopper path

Build the CUDA library explicitly before inference. Do not compile during a
generation run:

```bash
cd <PRIVATE_PATH_REDACTED>
PYTHONPATH=src:. python -m experiments.value_direction_hopper.cuda \
  --fast-sfu --inline-roles
```

The command prints a hash-named `value_direction_*.so` under
`results/value_direction_hopper_v1/build`. Build its ATen bridge with the exact
same kernel and active PyTorch version:

```bash
PYTHONPATH=src:. python -m experiments.value_direction_hopper.torch_build \
  --kernel /absolute/path/to/value_direction_HASH.so
```

The bridge writes a provenance file and refuses a mismatched kernel or PyTorch
runtime. `integration.install(...)` binds a request-local router to the native
DiffusionGemma attention registry; it does not replace the decoder or sampler.
The integration owns sketch-cache invalidation and refreshes the boundary and
canvas values on every call.

## Reproduce a frozen RULER run

After building the two libraries, run the explicit Hopper experiment with a
new output directory:

```bash
PYTHONPATH=src:. python -m experiments.value_direction_hopper.experiment run \
  --root results/my_ruler_run \
  --library /absolute/path/to/value_direction_HASH.so \
  --torch-library /absolute/path/to/value_direction_torch_HASH.so

PYTHONPATH=src:. python -m experiments.value_direction_hopper.experiment report \
  --root results/my_ruler_run
```

The experiment records immutable configuration/source hashes, resumes verified
completed shards, and writes physical eligible/skipped tile counts. Its method
controls are `native_dense`, `kernel_gaussian32`, and `kernel_blasst`; the
frozen manifest, model revision, projection seed, precision, masks, and
threshold policies are captured in the run directory.

## Reproduce uniform-threshold AIME studies

The AIME drivers use the six existing calibration IDs and copy one local/global
pair to every denoising call. Use a fresh result directory for each protocol:

```bash
PYTHONPATH=src:. python -m experiments.value_direction_hopper.aime_uniform_baselines \
  calibrate --root results/aime_uniform_baselines
PYTHONPATH=src:. python -m experiments.value_direction_hopper.aime_uniform_baselines \
  run --root results/aime_uniform_baselines
PYTHONPATH=src:. python -m experiments.value_direction_hopper.aime_uniform_baselines \
  report --root results/aime_uniform_baselines
```

For the causal query methods, select the methods and targets through the
driver's environment variables, then prepare, calibrate, freeze, run, and
report:

```bash
AIME_OUTPUT_ROOT=/absolute/path/to/results/aime_query_uniform \
AIME_UNIFORM_METHODS=C_gate,T_prior,T_smooth \
AIME_TARGETS=50 \
PYTHONPATH=src:. python -m experiments.value_direction_hopper.aime_query_sensitivity_uniform prepare

AIME_OUTPUT_ROOT=/absolute/path/to/results/aime_query_uniform \
AIME_UNIFORM_METHODS=C_gate,T_prior,T_smooth \
AIME_TARGETS=50 \
PYTHONPATH=src:. python -m experiments.value_direction_hopper.aime_query_sensitivity_uniform calibrate

AIME_OUTPUT_ROOT=/absolute/path/to/results/aime_query_uniform \
AIME_FINAL_SEEDS=42,43,44 \
PYTHONPATH=src:. python -m experiments.value_direction_hopper.aime_query_sensitivity_uniform run

PYTHONPATH=src:. python -m experiments.value_direction_hopper.aime_query_sensitivity_uniform \
  --root /absolute/path/to/results/aime_query_uniform report
```

The gated finalist workflow has the same `prepare`, `run`, and `report`
stages and accepts `--root`; its calibration and screen policy are recorded in
the root's JSON files. Threshold fitting uses tile counts and trajectory
metadata only. Calibration IDs overlap the historical AIME manifest, so those
runs should be reported as development evidence rather than held-out results.

The numeric configuration and threshold-file contract is documented in the
[Hopper frozen-configuration section](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/experiments/value_direction_hopper/README.md#frozen-hyperparameters-and-threshold-artifacts).
Load the full-precision pair from the selected `thresholds/*.json` record and
keep its `configuration.json` fingerprint, source hashes, model revision, and
kernel provenance together with the run. Thresholds are protocol-specific;
they should be recalibrated when the model, manifest, kernel, tile geometry,
projection, sensitivity hyperparameters, or runtime changes.

## Verification

Run the CPU formula suite in every environment. The H100 suite additionally
requires `VD_KERNEL` and checks the real SM90 library, mask packing, GQA,
projection refresh, ABI v4 sensitivity, ATen bridge, and CUDA graph behavior:

```bash
PYTHONPATH=src:. python -m pytest -q tests/test_query_adaptive.py

VD_KERNEL=/absolute/path/to/value_direction_HASH.so \
VD_TORCH_KERNEL=/absolute/path/to/value_direction_torch_HASH.so \
PYTHONPATH=src:. python -m pytest -q \
  tests/test_value_direction_hopper.py tests/test_value_direction_hopper_report.py
```

Keep raw runs and reports under `results/`. Do not infer wall-clock gains from
physical sparsity alone; report the dense backend, physical tile sparsity,
denoising call count, and synchronized end-to-end timing separately.

## License

The repository is released under the Apache-2.0 license. See
[`LICENSE`](https://github.com/coconight01/dlm_test/blob/b890ff49488474c5d476df44019597dc045a5969/LICENSE).
