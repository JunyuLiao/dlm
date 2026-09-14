# Pre-current-QK router study

Separate namespace; reuses the established DiffusionGemma adapter, exact attention
preparation, pinned native decoding, dataset rows, and verified dense cache.

Run from the repository root using the `ljy_dlm` Python environment:

```bash
PYTHONPATH=src python -m pytest tests/test_diffusion_gemma_preqk.py -q
PYTHONPATH=src python -m experiments.diffusion_gemma_preqk.run prepare
PYTHONPATH=src python -m experiments.diffusion_gemma_preqk.run screen-pipeline
PYTHONPATH=src python -m experiments.diffusion_gemma_preqk.run report
```

If the shared GPU is occupied, use `python -m
experiments.diffusion_gemma_preqk.launch` with the same environment. It waits for
an idle GPU and logs status every 900 seconds, without stopping other processes.

The protocol reserves 16 development prompts (the old oracle subset) and 20
additional held-out IDs (10 LongBench, 10 AIME) with two paired seeds. No held-out
inference is performed by the screening command. Quality tolerances are fixed at
3 percentage points LongBench and 5 AIME before held-out evaluation.

Stage 1 independently tests 20 candidates at two budgets per benchmark against
four diagnostic oracle rankings. It observes genuinely consecutive steps 0–7 of
the first canvas, all layers/heads/query tiles; temporal analyses exclude cold
step 0. Full official-budget generation must exactly reproduce cached dense
tokens. This is deliberately an EARLY-STEP screen, not evidence of later-step
generalization. Each candidate chooses before current dense QK; reference dense
history is updated only afterward. Dense-history last-mask reuses the previous
exact mass top-k mask. That acquisition cost is not claimed to be cheap online.
Binary frequency counts observations whose tile-mean mass is at least
`1 / eligible_tile_count`; it does not count retention decisions. Observation
counts and age are tracked separately. Sparse history supports both conditional
mass and a previous-coverage correction, which still requires empirical checking.

Value RMS/max summaries come from one charged encoder-cache initialization scan
and a canvas `v_norm` producer hook. Later decisions read scalar summaries, not
full V. The current mean-Q/mean-K proxy explicitly pays a full K summary scan in
this reference screen; online caching and measured costs must be evaluated if
it advances. Prefix-only, canvas-only and boundary-straddling tiles form three
disjoint populations, preserving exactly the whole-model physical denominator.

Raw NPZ files preserve candidate × batch × head × query-tile numerators and
denominators. Q0 additionally stores raw rankings, norm variability and exact
single-block deletion-risk summaries; these diagnostic labels never feed current
masks. The report reads only completed sample shards. Partial traces from failed
samples do not enter aggregates. Existing valid completed shards resume by
fingerprint; source changes require a new namespace or re-verification.

Stage 2 online-history rollouts, stage 3 conditional allocation and stage 4 frozen
validation/performance are not launched blindly. They follow analysis of the
screen under the decision rules in `protocol.json`. No sparse speedup or final
mechanism is established by dense-emulated screening.

## Evidence-gated online development rollouts

After reviewing the complete screen, write a selection artifact containing a
`hypothesis`, `decision_rationale`, and 2–6 `entries` with `benchmark`, `predictor`
and `target_sparsity`. Both development benchmarks and a same-budget `last_mass`
control are required. Only screened predictor names and budgets are accepted.
The CLI automatically pairs dense-reference and sparse-history observations;
everything else is identical within the pair. Optional `online_policy` overrides
are limited to refresh interval, exploration count and renormalization treatment
and must be justified as a separate frozen development iteration.

```bash
PYTHONPATH=src python -m experiments.diffusion_gemma_preqk.online_run freeze \
  --selection results/diffusion_gemma_preqk/stage2_selection.json
PYTHONPATH=src python -m experiments.diffusion_gemma_preqk.online_run pipeline
PYTHONPATH=src python -m experiments.diffusion_gemma_preqk.online_run report
```

`freeze` reconstructs and validates stage-one evidence before accepting a
shortlist; it refuses incomplete screening. No selection artifact has been
created yet. This CLI cannot run held-out IDs. A fresh CUDA smoke is mandatory,
but a matching passing smoke is reused on resume. Completed sample shards are
immutable; independent generation failures are preserved and do not stop others.

Online `last_mask` means the previous **estimated-important** mask: it is acquired
after the previous step from permitted observations, consistent with the screen's
previous mass top-k mask. It is not a retained-bit frequency. Its additional
history-side selection is charged explicitly. Dense-reference observations are
dense counterfactuals on the candidate's own sparse trajectory, not a replay of
the original dense trajectory. Sparse-history updates never read dropped logits.

Online diagnostic output error uses direct FP32 dense/masked PV rather than
materializing all block-contribution vectors; regression tests verify equality
with the existing oracle. Diagnostic PV costs are counted separately. Outputs
include prompt-paired task-macro scores, bootstrap intervals (seeds grouped within
prompt), token-match numerators/denominators, output lengths, budget-hit rates,
exact decoder-call counts, physical-work counts and cost ledgers. No actual sparse
backend, allocation policy or measured net saving is claimed by these commands.
# Refinement and reference-cost profiling

## Frozen held-out quality confirmation and final audit

After development/cost evidence, only the plain LongBench history signal receives
a quality-only confirmation; reference deployment remains rejected for cost.
The frozen set contains two additional IDs from each of qasper, hotpotqa,
gov_report, trec and passage_retrieval_en, each with seeds314159 and271828.
This requires20 new dense baselines and20 sparse runs. AIME fails the development
quality gate; allocation fails the total-work gate. Neither is further tuned.
RULER/LiveCodeBench expansion is conditional on a deployment finalist; none exists.

```bash
PYTHONPATH=src python -m experiments.diffusion_gemma_preqk.validation freeze
PYTHONPATH=src python -m experiments.diffusion_gemma_preqk.launch --pipeline validation \
  --output results/diffusion_gemma_preqk/heldout_quality_v1 --interval 900
PYTHONPATH=src python -m experiments.diffusion_gemma_preqk.validation report
PYTHONPATH=src python -m pytest tests/test_diffusion_gemma_preqk.py tests/test_diffusion_gemma_oracle.py \
  --junitxml=results/diffusion_gemma_preqk/test_results.xml
PYTHONPATH=src python -m experiments.diffusion_gemma_preqk.final_report
```

The storage ID includes the seed, while `prompt_id` retains the original ID for
within-prompt seed averaging and task-stratified paired bootstrap. Dense caches
have separate hashes and cannot alter the immutable selection. Epsilon is the
previously declared .03 LongBench / .05 AIME; held-out outcomes cannot tune it.
The final report command re-audits raw masks/metrics, archived sources, complete
shards, exact cached development tokens, local/global/head coverage, region/count
sums, held-out source/seed disjointness and the saved regression test results.
It writes report.md, quality_results.csv, physical_work.csv, compact_ablations.csv,
quality_sparsity.png, raw_integrity.json and completion_audit.json. Preserve the
held-out harness overlay with `snapshot.preserve_validation(stage)` after its
matching CUDA smoke, before changing inference files.

The completed `online_v1` evidence gates a bounded, development-only refinement:

```bash
PYTHONPATH=src python -m experiments.diffusion_gemma_preqk.refinement
PYTHONPATH=src python -m experiments.diffusion_gemma_preqk.launch --pipeline online --output results/diffusion_gemma_preqk/refinement_v1 --interval 900
```

This freezes six conditions / 44 generations: LongBench plain last-mass regression
control and one paired layer allocation (10 prompts each); AIME plain s40 with
paid dense-reference vs own history (6 each); and two single-call s50 perturbations
at the development-error-maximizing local/global layers (6 each). Only the first
matching layer/step in the entire request is perturbed, not every canvas.

Allocation moves one retained tile per head/query block from five lower-error
local layers to five higher-error local layers, and from one lower-error global
layer to one higher-error global layer. Changes are jointly clipped for feasibility.
Every forward is audited for exact zero-sum retention within attention type;
initialization, refresh and exploration remain counted. Uniform shadow masks use
the same causal history and current diagnostic Q/K/V. They never affect generation.
Equal per-forward retention does **not** imply equal total rollout work when the
denoising duration changes; see `trajectory_work.csv` and `allocation_same_state.csv`.
These layer groups are based on observational development error, not an assertion
of causally established layer sensitivity. Held-out samples remain excluded.

`performance.py` is a separate reference-backend profiling command, to run only
after the selected stage completes and when the H100 is free. For example:

```bash
PYTHONPATH=src python -m experiments.diffusion_gemma_preqk.performance \
  --stage results/diffusion_gemma_preqk/refinement_v1 \
  --output results/diffusion_gemma_preqk/reference_cost_v1 \
  --conditions longbench__last_mass_control_s50__sparse aime24__last_mass_s40__sparse
```

It measures repeated full-budget end-to-end reference runs, with dense diagnostics
disabled, against the same eager dense baseline, and requires exact token parity
with completed artifacts. New dense runs here are necessary timing repetitions,
not recomputation of benchmark quality. Warmups are excluded; order alternates.
Separate capped passes measure nested host-wall/CUDA-stream component spans;
these are not additive, not isolated GPU kernel timings, and not part of headline
unprofiled timing. No compatible pre-QK sparse execution backend is integrated:
executed tile QK/PV avoidance remains zero. Profiling implementation alone is not
a performance result; inspect its completed audit before making empirical claims.
