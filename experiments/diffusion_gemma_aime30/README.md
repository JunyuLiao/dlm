# AIME2024: all 30 problems, zero-shot and fixed five-shot CoT

Two prompt modes × 30 problems × 9 conditions = **540 final generations**.
Every condition sees exactly the same frozen prompt tokens, seed 42 and 2,048
output-token budget within each prompt mode. Dense is included in both modes.
The five-shot prompt prepends the same five worked AIME2025 solutions, in a fixed
order, to every test question. Six separate AIME2025 problems calibrate BLASST.
Test, demonstration and calibration problem hashes are audited for overlap.
Rows without a worked solution are ineligible as demonstrations; their use as
calibration questions does not require a reference rationale.

Conditions: dense; Sol Gaussian at targets 25/50/75/90%; BLASST at the same targets.
Both use existing 64×64 physical-tile masks with prefix **and canvas** eligible.
Sol betas are fixed analytic Gaussian quantiles. BLASST uses one calibrated local
and one global scalar per target and prompt mode, shared across layers, heads and
steps. The scalar policy makes this a new calibration experiment, rather than
another transfer of the old RULER length fit.

Calibration records dense online-softmax margins on six held-out questions per
mode. The maximum valid-query-row margin per physical tile exactly determines
whether that entire tile can skip. This sufficient statistic permits offline
evaluation of the existing lambda grid, plus lambda=1 and 32 log-bisection points.
Select the closest achieved target. If the lambda=1 ceiling is below the target,
deploy lambda=1 and label it unattainable. Export the old exponential-fit relation
as a diagnostic only. Verify each selected policy on the six sparse calibration
generations and report any trajectory-dependent mismatch; never tune on AIME2024.

The pinned BF16 DiffusionGemma and its native sampler are reused: seed 42,
temperature schedule 0.4–0.8, max 48 denoising steps, confidence 0.005, stability 1,
entropy bound 0.1, canvas 256, thinking prompt flag false. Prompt text asks for
worked reasoning in both modes. The adapter request temperature=0 preserves the
native schedule and does not mean greedy decoding. Token-budget terminations are
reported, and the budget is never changed based on observed answers.

Scoring extracts the final numeric answer independently of the gold: last boxed
answer, then final-answer marker, then last number. Leading zeros, decimal and
scientific forms are accepted with absolute tolerance 1e-6. It does not scan the
reasoning for any occurrence of the gold answer. Raw generations and extracted
answers permit scoring audits. Primary sparsity is summed skipped/eligible
physical tiles, with global/local/whole-decoder splits. Retained dense mass on
the current trajectory, positional token agreement, termination and extraction
failure counts are secondary diagnostics. These are accuracy experiments with
reference masks; no speedup is claimed.

Run from the repository root with the existing `ljy_dlm` Python:

```bash
export PYTHONPATH=src:.
python -m experiments.diffusion_gemma_aime30.run prepare
python -m experiments.diffusion_gemma_aime30.run run
python -m experiments.diffusion_gemma_aime30.run report
```

Default bundle: `results/diffusion_gemma_aime30_zero_vs_five_shot/`.
`run` checks exact eager-dense/zero-prune parity in both prompt modes, confirms
prefix/canvas skipping and local/global/layer coverage, calibrates, verifies,
then runs dense and BLASST before the more costly Sol reference loop.
It writes atomic sample shards and skips compatible completed samples on resume.
Failed final samples are recorded and independent samples continue. Re-running
retries missing samples. The inherited `monitor_controlled.py PID ROOT` records
status every 900 seconds. Reports regenerate without GPU inference.

Artifacts: frozen manifests and demonstrations; setup, code and model provenance;
smoke audit; compressed calibration margins; full threshold grid/search traces;
sparse calibration verification; raw generations and per-layer counts; numerical
accuracy and weighted actual sparsity CSV/JSON; Markdown report; standalone plots.
