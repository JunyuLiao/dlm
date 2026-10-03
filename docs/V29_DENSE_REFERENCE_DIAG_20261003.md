# V29 dense-reference reproducibility diagnostic — prepared, not GPU-run

The new worker is `scripts/v29_dense_reference_diag.py`. It reuses the original
V28 qualification wrapper, v27 panel loop, PhaseTracker and native sampler.
It does not change the running formal panel or the closed002/003 deployments.
No new GPU evidence exists yet.

Three committed-spec candidates are under `results/v29_20261002/specs/`:

- `v29_dense_reference_diag_default_dense.json`: original dense, requested default.
- `v29_dense_reference_diag_piecewise_dense_nohook.json`: original dense, requested PIECEWISE.
- `v29_dense_reference_diag_piecewise_native_hook.json`: official dense with existing native adapter hooks, PIECEWISE.

They freeze the same32K index0, engine seeds28001/28002, one full-inventory warm
and one timed request per fresh engine. Repeat each seed twice in a fresh
engine:12 workers total. The retained V28 declarations include all four
original worker arms because its validator requires them, but the launch list
contains only the three reference aliases. No all-kept/main run is added here.

`make_spec(base, variant, indices, seeds=(28001,28002))` returns a new dictionary.
It removes old cost-profile/event/trace settings, sets `cuda_events=false`, and
records that reseeding/restoring RNG and profiling are disabled. The private
coordinator must compare the full generated spec with the corresponding public
spec, not just accept matching aliases. Input/config rebinding remains strict:
validate the old binding/source bytes, preserve source hashes, use the existing
`rebind_config`, and validate the effective config before GPU work.

Example after root commits the source, deploys a new own directory, binds every
alias separately and the queued predecessors finish:

```sh
python -m scripts.v29_dense_reference_diag \
  --binding NEW_PRIVATE_BINDING --variant default_dense \
  --block 0 --engine-repeat 0 --run-dir NEW_PRIVATE_RUN
```

Use the other alias bindings and blocks0/1, repeats0/1 unchanged. Runtime
inventory and seeds come only from the frozen binding; `--index` and
`--engine-seeds` are allowed only with `--print-spec`. The coordinator owns
fresh-process scheduling and idle checks. No worker is launched by this module's
CPU spec helpers or tests.

Pin the new diagnostic worker, original panel/metrics/V28 wrapper/adapter/JIT
receipt module, and its two reused helper sources:
`scripts/v29_vllm_cost_profile.py` (CPU input validation only) and
`scripts/v29_component_timing.py` (existing monitor activation validation only).
The old frozen deployments are not edited. Process-local monkeypatches are
restored on exit/errors and output files use exclusive creation; failures are
preserved, not cleaned away or overwritten.

The worker adds a CPU observer of each actual `prepare_attn` runtime mode. It
pairs these submissions with the PhaseTracker's existing immutable async CPU
sample-count snapshots after the existing request-boundary synchronization.
It records prefill/denoise/commit graph-mode histograms and per-execution rows,
including unused terminal denoising. Initialization graph configuration is
recorded separately. There is no new per-step GPU scalar read or synchronization.
The native arm retains its pre-existing exact metadata read.

Warm output lengths are read as scalar lengths from the original final
RequestOutput; token IDs/text are never copied into these diagnostic receipts.
Every warm/timed row stores actual phase counts and output length, plus original
prefill/decode and request boundary spans. These include diagnostic overhead and
are not formal speed evidence. Original zero-timed-JIT/capture, adapter coverage
and clock checks remain active; no new math kernel or numerical oracle is claimed.

`dense_reference_diagnostic.json` contains safe counts, durations, graph modes,
ordinals and boundary equality booleans. `rng_boundaries.private.json` contains
only request ordinals and private SHA256 digests of RNG state. Do not publish
that file. States are read using `torch.cuda.get_rng_state()` at the existing
synchronized request boundaries; no seed/reset/restore/random operation is
introduced. This covers the default generator on the current CUDA device only.
It does not establish complete compiled/CUDA-graph internal RNG state, identical
random inputs or identical trajectories. Differences are preserved as observations,
not screened as qualification failures.

After each worker closes, the coordinator can run CPU validation with CUDA hidden:

```sh
CUDA_VISIBLE_DEVICES='' python -m scripts.v29_dense_reference_diag \
  --binding NEW_PRIVATE_BINDING --variant default_dense \
  --block 0 --engine-repeat 0 --run-dir CLOSED_PRIVATE_RUN --validate-run
```

`validate_run(run_dir,binding,variant,block,engine_repeat)` is the same CPU API.
The CLI first verifies immutable binding hashes; the API validates closed
terminal/request/graph/RNG coverage, original timed zero-JIT receipts and actual
phase accounting. Its stdout contains safe status fields only, with
`diagnostic_validation_passed=true`. Unequal random states/lengths are valid
diagnostic outcomes; missing/inconsistent evidence fails closed.

CPU command: `python -m unittest tests.test_v29_dense_reference_diag
 tests.test_v29_native_passthrough -v`. Twenty-one tests pass, including actual
native-wrapper argument/return contracts and diagnostic async/partial-commit,
missing-evidence, warm-output, RNG-chain, no-overwrite and restoration behavior.
The GPU runtime instrumentation itself remains unqualified until the new run.
