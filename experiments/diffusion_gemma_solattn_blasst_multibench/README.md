# Controlled DiffusionGemma multibench comparison

This entry point uses `controlled_*` modules. The older `dataset.py`, `runner.py`,
and `report.py` scaffold is retained for provenance, but is not the evaluation CLI.
The controlled runner reuses its installation/request/calibration helpers.

From the repository root, using the `ljy_dlm` Python environment:

```bash
export PYTHONPATH=src:.
python -m experiments.diffusion_gemma_solattn_blasst_multibench prepare
python -m experiments.diffusion_gemma_solattn_blasst_multibench smoke
python -m experiments.diffusion_gemma_solattn_blasst_multibench run
python -m experiments.diffusion_gemma_solattn_blasst_multibench grade-livecodebench
python -m experiments.diffusion_gemma_solattn_blasst_multibench report
```

Default bundle: `results/diffusion_gemma_solattn_blasst_multibench_controlled`.
`run --conditions dense` or a subset of the eight sparse condition names is
supported. A matching smoke audit is mandatory. Atomic per-sample JSON shards
are immutable, fingerprinted, and skipped on resume. Errors are logged per sample
without stopping independent examples. Re-running retries failed/missing samples.

122 shared samples × nine conditions = 1,098 final generations. See
`manifest_audit.json` for exact task counts and `manifest.jsonl` for source IDs,
prompts, hashes, budgets, seeds, and truncation. Dense is run first. A single
dense generation records the existing eight routing policies plus λ=1 masks on
each actual dense Q/K state; dense QK/probabilities are reused within that call.
Only sufficient statistics are cached. Sparse-trajectory mass is stored separately.

Decoding uses the pinned native sampler. This adapter's request `temperature=0`
is a sentinel, not greedy decoding: the model's 0.4–0.8 schedule, maximum 48
denoising steps, confidence 0.005, stability 1, and entropy bound 0.1 remain active.
Seeds are reset per sample/condition. The native canvas is 256; the requested
block size does not override it. See `decoding_protocol.json` in the bundle.

Sparsity always divides summed skipped eligible physical 64×64 tiles by summed
eligible physical tiles, for whole decoder/local/global. Both prefix and canvas
participate, with globally aligned indivisible tiles. The existing current BLASST
implementation executes whole-tile masks (`physical_tile_v1`), not historical
row-granular masks. Its λ is length-adjusted by the existing calibration relation;
unattainable calibration targets use λ=1, not a new fit.

The official pinned LongBench source supplies prompts, generation budgets and
metrics. LiveCodeBench official code executes only in network-disabled, read-only,
nonroot resource-limited Docker containers, with only the scorer/input mounted.
Do not invoke `grade_worker.py` on the host. `grading_smoke.json` covers a known
correct, wrong and timeout program. Grading artifacts are separate from generations.

`launch_controlled.sh` runs the complete pipeline. `monitor_controlled.py PID ROOT`
logs state every 900 seconds. There are no latency/throughput/speedup claims: these
are reference-mask accuracy experiments, not optimized attention kernels.

Tests:

```bash
PYTHONPATH=src:. python -m pytest -q tests/test_multibench_controlled.py tests/test_diffusion_gemma_solattn_math500.py tests/test_blasst_mask_migration.py
```
