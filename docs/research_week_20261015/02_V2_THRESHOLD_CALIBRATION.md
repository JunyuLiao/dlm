# 2. V2 threshold calibration and absolute-work comparisons

Publication checkpoint: 2026-10-15. This study always uses
`value_v2_online_preserve_mass`: the exact online recurrence, one uniform
GLOBAL threshold, no fixed quota or mass anchor, rank32/seed1729 for selection,
native FA4 model output and dense LOCAL attention. See
[method definitions](01_VALUE_AWARE_METHODS.md) and the
[sanitized publication](../../results/research_week_20261015/README.md).

## Why absolute work replaced sparsity matching

The original 503-item audit compared V31 at 83.16% GLOBAL sparsity with V2 at
84.11%. V2 generated fewer denoising calls, canvases and output tokens, so its
eligible-work denominator also shrank:

| Original arm | Correct/503 | N | Eligible GLOBAL tiles | Kept GLOBAL tiles |
|---|---:|---:|---:|---:|
| V31 | 268 | 140,693 | 34.600B | 5.826B |
| V2 tau=.027 | 266 | 126,072 | 30.566B | 4.856B |

V2 performed 16.65% less retained GLOBAL work. Equal sparsity is therefore not
equal physical work. Define `K_total=sum(global_kept_tiles)` over actual native
calls. Define `K_C` over the first C actual denoising canvases on a matched item
cohort. Read these counts from adapter receipts, not selected-prefix counts or
requested K. Selector discovery/ranking is additional work, separately reported.

## Completed 11-point development sweep

The original 64-cell development panel and approved seeds were used before the
new official comparison. No official-503 accuracy selected these thresholds.
Totals below are unaffected by the later canvas-axis reporting correction.

| Threshold | Correct/64 | Kept GLOBAL tiles | GLOBAL sparsity | Mean N |
|---:|---:|---:|---:|---:|
| V31 | 28 | 835,928,960 | 83.25% | 279.02 |
| .001 | 27 | 2,790,937,975 | 28.47% | 231.55 |
| .002 | 30 | 2,298,289,627 | 51.94% | 273.22 |
| .005 | 25 | 1,243,895,220 | 68.78% | 233.83 |
| .009 | 28 | 959,011,334 | 76.73% | 235.12 |
| .015 | 32 | 843,406,939 | 82.42% | 277.81 |
| .021 | 27 | 711,431,684 | 80.03% | 213.33 |
| .027 | 32 | 643,306,001 | 84.60% | 240.56 |
| .033 | 33 | 698,850,450 | 85.62% | 285.92 |
| .045 | 25 | 660,568,341 | 86.16% | 276.52 |
| .0675 | 26 | 637,188,730 | 86.27% | 288.67 |
| .108 | 23 | 616,029,375 | 86.18% | 265.06 |

The realized curve is non-monotonic: threshold changes alter online state,
reused support and the native trajectory. Tau .015 minimized total-work mismatch
on development (+0.89%). Tau .021 was selected by the historical early-work
calculation; that selection must not be described as a validated optimum under
the corrected canvas axis below.

## Source-matched official audit and reporting correction

The completed equal-work campaign is `results/v31_value_v2_equal_work_20261011/`,
source commit recorded as `fce50a8b`, with a new matched control. Its control
must not be replaced by the 268/503 control from another source snapshot.

| Arm | Tau | Correct/503 | N | Kept GLOBAL tiles | Total work/V31 |
|---|---:|---:|---:|---:|---:|
| Matched V31 | — | 256 (50.89%) | 134,521 | 5,691,974,080 | 1.0000 |
| `v2_work_match_total` | .015 | 263 (52.29%) | 124,671 | 5,879,477,757 | 1.0329 |
| `v2_work_match_early` | .021 | 274 (54.47%) | 131,786 | 5,248,339,635 | 0.9221 |

The paired differences are +1.392 pp (bootstrap [-2.187,+4.970], McNemar
p=.5203) and +3.579 pp ([0,+7.157], p=.0630). These are exploratory results,
not established superiority or noninferiority. The nominal arm names describe
the calibration target, not exact realized matching: +3.29% and -7.79% official
total-work mismatches exceed the recommended +/-2% total tolerance.

**Canvas-axis correction:** adapter lifecycle IDs advance during encoder/prefill
phases and are not denoising-canvas ordinals. The historical sweep/official
reports used incorrect zero-based ranges and lifecycle-ID cohort eligibility.
Their early-work columns and the old +13.29% / +1.19% C=20 claims are superseded.
Accuracy and entire-run work are unchanged. The corrected official receipt is
[`progress_axis_correction.json`](../../results/research_week_20261015/v2_calibration/progress_axis_correction.json):

| Historical arm | Common items reaching 20 visible canvases | V31 kept through ordinal 20 | V2 kept | Relative difference |
|---|---:|---:|---:|---:|
| tau .015 total arm | 50 | 819,186,400 | 882,592,957 | +7.74% |
| tau .021 early arm | 45 | 680,459,200 | 701,344,984 | +3.07% |

The cohorts differ between arms, so this is not a same-cohort ranking of the two
thresholds. Tau .021 also narrowly exceeds +/-3% early tolerance. Development
early-work optimization has not been rerun on the corrected axis; do not claim
that a uniform threshold is proven to match both corrected criteria. Public
historical curve exports omit the invalid early columns and preserve original
file hashes. Frozen originals are unchanged.

Correct implementation: sort actual receipt canvas IDs into ordinal order;
check that their kept-work sum equals total GLOBAL work. At prefix C, require
both arms' visible canvas counts and observed ordinal counts to reach C. Report
cohort size, per-item differences/quantiles and cumulative eligible/kept work.
Visible output C and terminal EOS observations can differ; report definitions.

## Local accuracy sweep around .021: completed and pending scope

`results/v31_v2_local_threshold_sweep_20261014_attempt002/` froze 886 execution
files at recorded commit `ab05d324`, with 24 input hashes. Initial thresholds
are .018/.0195/.021/.0225/.024. Workloads are official LongBench-v2 503/seed1,
AIME26 30 problems at seeds42/43/44, and a fixed MATH500 sample of 100/seed1.
All references are fresh per-suite V31/FULL/PIECEWISE/all-kept controls.

| Threshold | AIME correct/90 | Mean N | Work/V31 | MATH correct/100 | Mean N | Work/V31 |
|---|---:|---:|---:|---:|---:|---:|
| V31 | 53 | 287.51 | 1.000 | 95 | 62.02 | 1.000 |
| .018 | 45 | 284.38 | 1.547 | 92 | 60.64 | 1.037 |
| .0195 | 52 | 284.69 | 1.520 | 94 | 64.21 | 1.127 |
| .021 | 60 | 278.68 | 1.428 | 95 | 61.80 | 1.035 |
| .0225 | 55 | 284.14 | 1.462 | 95 | 59.79 | .980 |
| .024 | 50 | 287.77 | 1.481 | 92 | 59.77 | .981 |

MATH sampling sorts source IDs, samples with `Random(20261011+100)`, then sorts
again; effective seed is 20261111. The current score is bound symbolic
math-verify 0.9.0 with the official final-response parser, not the legacy quick
string scorer. The 100 selected inputs were independently checked against the
full 500-item source. AIME uses 30 problem clusters for bootstrap uncertainty,
keeping its three seeds together. Tau .021's AIME delta is +7.78 pp with
interval [-2.22,+17.78]; its higher accuracy also uses 42.8% more GLOBAL work.

LongBench controls completed: V31 268/503, FULL and PIECEWISE 247/503, all-kept
252/503. The FULL/PIECEWISE outputs and N/T/C match on all 503 items in this
campaign; that is not a universal cross-mode guarantee for another campaign.

The user paused after controls. All five full LongBench V2 arms and both finer
thresholds across all three workloads are **pending and must not auto-resume**.
There are 22 verified reports / 3,722 scored cells, zero invalid rows, no missing
scores, zero LOCAL skips and zero unexpected request graph captures. The scoped
`controls_report_complete.json` is not a whole-campaign completion marker.

The frozen plan would select the equal-weight three-workload macro-accuracy
winner, using normalized N then kept work only for exact accuracy ties; then
test best +/- .00075 and report workload winners separately. This is full-panel
exploratory tuning, not held-out calibration or a global-optimum claim. No shared
winner has been selected while LongBench thresholds remain pending.

## Reproduction and integration instructions

Read-only public analysis: `results/research_week_20261015/v2_calibration/` and
the existing local-sweep `summary.md`/`curve.csv`. Reproducing generations needs
the authorized frozen manifests, seeds, scorer inputs and model/runtime pins.

For a **new authorized** equal-work reproduction, the published driver now takes
explicit portable paths instead of private machine constants:

```bash
PYTHONPATH=src:. "$DLM_PYTHON" scripts/v31_value_v2_equal_work.py \
  --study "$DLM_STUDY" --python "$DLM_PYTHON" --model "$DLM_MODEL" \
  --manifests "$DLM_MANIFESTS" --reference-study "$DLM_REFERENCE_STUDY" \
  --development-cells "$DLM_DEV_CELLS" --target-cells "$DLM_TARGET_CELLS" \
  --stage sweep
```

The reference study supplies its model inventory; use a new dated output root.
This archived equal-work runner retains the old canvas-axis implementation: its
total-work outputs are usable, but **do not use its early selector/reporting
columns to choose a new early-matched threshold**. Integrate the corrected
`ordinal_canvas()` and common-visible-cohort implementation from
`scripts/v31_v2_local_threshold_sweep.py` and
`scripts/v31_monitor_local_threshold_sweep.py` into a separately named, tested
driver and freeze it before any new early-work generation study. Do not mutate
the old frozen protocol or retrofit a quota into natural V2.

For the paused local sweep, read its protocol, scope amendment and pause marker
first. Its committed entrypoint is `scripts/v31_v2_local_threshold_sweep.py`,
`--study PATH --stage run`; it skips validated completed shards and creates a new
leaf for partial attempts. It relies on its private frozen execution copy and
local pinned runtime/model constants. Moving hosts requires explicit path binding
and a new protocol/control, not editing frozen bytes. Launching is currently
unauthorized; these are future reproduction instructions only.

Runtime geometry: LongBench block32/chunk16384/memory.85/maxlen141312/K8192;
AIME block64/chunk4096/memory.80/maxlen9216/K1728/output8192; MATH block64/
chunk4096/memory.80/maxlen16384/K1728/output4096. K applies to V31, never a V2
quota. The old matched-sparsity matrix additionally requires short thresholds
for AIME and long thresholds for RULER/LongBench; its provisional misrouted
queue is diagnostic and remains paused.

Validation: `PYTHONPATH=src:. "$DLM_PYTHON" -m pytest -q
tests/test_v31_v2_local_threshold_sweep.py`. Check exact selector/threshold,
source/runtime/input fingerprints, official scorer coverage, FA4 output path,
phase sums and completion markers. Keep physical FA4 work, selector discovery,
selection, N/C/T and audit wall time separate. Equal sparsity, equal retained
tiles, equal native trajectory and equal clean throughput are four distinct
claims. A fixed-budget V2 hybrid is a separate future method, not calibration.
