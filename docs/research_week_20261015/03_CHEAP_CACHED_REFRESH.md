# 3. Cheap cached-statistics refresh, stable updates and factor isolation

Publication checkpoint: 2026-10-15. This direction repairs stale V31 support
between dense observations. It is distinct from
[V1/V2/V3 value selection](01_VALUE_AWARE_METHODS.md) and
[V2 threshold calibration](02_V2_THRESHOLD_CALIBRATION.md).
Yuhan's inherited V31 lifecycle remains the attributed baseline. Sanitized
evidence is under
[`results/research_week_20261015/cached_refresh`](../../results/research_week_20261015/cached_refresh/).

## Idea and execution boundary

At an existing dense selection, save raw per-row log masses over all eligible
GLOBAL prefix tiles. On a cheap event, recompute QK statistics only for current
retained tiles (and the protected tail as required by the implementation).
For each row, combine fresh retained statistics with cached discarded statistics:

```text
hybrid_ij = fresh_ij if tile j is currently selected else cached_ij
score_j   = max_valid_i (hybrid_ij - logsumexp_over_all_prefix_tiles(hybrid_i))
```

These are raw log masses normalized together; normalizing fresh selected mass
separately destroys comparability with stale outside candidates. Rank the hybrid
scores to prepare a new fixed-K map. The current attention call still returns
native FA4 on its current map; the new map is installed for the next call. A
cheap refresh introduces no model forward, dense FA4 output or selector PV.
Dense initial/canvas observations and optional progress-triggered dense refresh
remain explicit, separate events.

Both raw/global-top-K and bounded updates preserve a fixed eligible-prefix
budget per query-head/Q128 unit, capped by available prefix support. Tail tiles
are mandatory outside that budget. Physical FA4 work and additional selected/
tail QK statistics must be counted separately; duplicated QK remains real work.

## Three iterations and evidence boundaries

### Initial cached-refresh panel: 2026-10-12

`results/v31_cached_refresh_20261012/` completed 16 runnable arms x 154 cells:
64 seed-1 LongBench-v2 problems and 30 AIME26 problems x seeds42/43/44. Primary
bounded-M256 is attempt009; the matched matrix is attempt010. It includes
global-top-K and bounded candidate pools M=160/256/512, B=8 bounded swaps,
outside-statistic decay, phase/C_gate ablations and periods 1/2/4.

M is the **total** bounded candidate pool, not M extra outside tiles. The decay
arm uses lambda=log(2)/2 and floor=.25 outside the retained map; retained ages
reset to zero and discarded ages increment with int32 intermediates before
int16 saturation. Decay is a separately named heuristic, not fresh evidence.

The initial source was later found to have two important defects: raw
`[H,QB,PT,128]` statistics were viewed as `[H,QB*128,PT]` without the required
permutation, and the CUDA bounded path could drop a retained block when no
outside candidate existed. Preserve its completed historical panel, but do not
interpret it as a clean test of a corrected selector. Its native operator-error
fields were absent; synthetic replay is not native output-quality evidence.
The raw normalized drift ratio also became ill-conditioned near zero; use
absolute log-mass shifts, common-shift residuals or normalized distribution
drift with age-conditioned checks instead.

Online/full-value tie-break and strict/filled intersection in this first study
were operator diagnostics; native variants were not wired separately. Registered
launcher names do not prove those methods executed. No arm was promoted and
clean timing stayed paused.

### Corrected stable follow-up: 2026-10-13

`results/v31_cached_refresh_stable_20261013/`, completed native publication
`a136b40d`, fixes the layout and budget invariants and adds conservative update
guards. Raw re-ranks the full hybrid prefix support. Stable uses:

| Parameter | Frozen value | Meaning |
|---|---:|---|
| Hysteresis | 0 | Additional cheap-refresh margin, independent of inherited sticky. |
| Persistence | 2 events | Candidate must remain preferred over two cheap observations. |
| Event cap | 2 swaps | At most two replacements per query tile/event. |
| Anchor-epoch cap | 8 swaps | Cumulative replacements since a dense anchor. |
| Row removal tolerance | .025 | Rowwise mass-removal protection. |
| Outside probes | 0 | Probe4 remained conditional and was not promoted. |

No-change/no-op is valid. Force an audited dense fallback at the next allowed
point on invalid state; do not return an old selector output as attention.
The stable panel completed 13 arms x 154 = 2,002 native cells with fresh controls.
Representative LongBench results are:

| Arm | Correct/64 | Mean N | N/V31 |
|---|---:|---:|---:|
| Progress V31 hold | 34 | 240.27 | 1.000 |
| Raw + progress, every 2 | 33 | 238.89 | .994 |
| Stable + progress, every 2 | 31 | 271.86 | 1.132 |
| No-progress hold | 33 | 262.69 | 1.093 |
| Stable, no progress, every 1 | 32 | 218.64 | .910 |
| Stable, no progress, every 4 | 32 | 273.38 | 1.138 |
| Stable, no progress, every 8 | 35 | 293.83 | 1.223 |

The s8 +1 answer over V31 accompanies +22.29% denoising calls. The lower-N s1
arm also changes output lengths. These development signals justify measured
contrasts, not a speed or accuracy promotion. AIME is almost all-kept at this
study's K=128 prefix tiles: cheap-update effects are weakly exposed and no-
progress cheap arms share the hold correctness vector. Do not substitute the
K1728-token short-suite V31 configuration from the separate V2 sweep.

### Factor isolation and the later full503 campaign: 2026-10-14/15

`results/v31_refresh_factor_isolation_20261014/` completed paired analysis of
existing native data and independent raw/stable x progress-on/off x intervals
1/2/4/8 operator replay. Only 8 of 16 settings have native development evidence;
the operator factorial does not fill in their missing native scores.

Replay covered 620 real snapshots / 670 files, five GLOBAL layers and both
workloads. All 2,550 cheap-event maps matched an independent NumPy oracle;
fresh-QK and score errors were <=1.91e-6 and <=1.64e-6, with zero map, budget,
cap or capture errors. Raw's more frequent updates reduced common-call operator
error; stable greatly reduced churn but had slightly higher error at matched
settings. Progress reduced common-call error by .89–1.57% conditional on the
imposed source trajectory. It does not predict native accuracy or stopping.

Stable + progress + interval8 made **zero swaps**: its two-event persistence
could not be reached before dense reset. Zero churn/p95 therefore did not prove
a superior active selector. The 64-item native contrasts remain noisy,
incomplete and exploratory; they do not isolate every stabilizer component.

The subsequently authorized `results/v31_full503_refresh_20261014/` is a new,
source-frozen campaign with 13 x 503 = 6,539 seed-1 cells: seven selected raw/
stable/progress/frequency arms, two held controls, a fresh every-step V31
selection baseline, and FULL/PIECEWISE/all-kept controls. Recorded generation
freeze is `98a37a14`; the first complete 13-arm round (1,638 cells) was validated
at `53859740`. The publication bundle captures a later queue checkpoint if
available. **This is generation progress, not a final official accuracy result.**
Its active worker, frozen execution tree, queue and source-study artifacts are
not modified by this documentation/publication task.

The every-step baseline uses current full-support V31 selection and returns
same-call sparse FA4 output, discarding the dense observer output. It disables
reuse/carry/previous-map bonus/cached statistics for that arm only. Count its
full observer QK/PV separately from sparse consumer work and duplicated tail QK.
No-progress hold still initializes/carries per canvas; it is not a single
once-per-problem map.

The FULL runtime also qualified persistent int32 causal-metadata aliasing with
28,410 checks and zero alias/dtype errors, plus exact repeated within-FULL N/T/C
and output matches. This is a correctness-qualified runtime integration, not a
clean speed measurement. Cross-mode free-running FULL/PIECEWISE equality is not
required; consumer/control differences stay visible.

## Integration surfaces

| File | Role |
|---|---|
| `experiments/numerical_qk_reuse/v31_cached_refresh.py` | CPU hybrid normalization, fixed top-K/bounded swaps, decay/ages, state and diagnostics. |
| `v31_stable_refresh.py` | Raw/stable update kernels, exact budget checks, persistence/caps/removal guard. |
| `v31_every_step_refresh.py` | Fresh current-step baseline support selection. |
| `vllm_adapter.py` | `_cached_prepare/_cached_refresh`, `_stable_*` lifecycle, raw-layout conversion, dense-reset/carry state, next-map installation, receipts. |
| `scripts/v31_vllm_paired_bench.py` | Constructor and native runtime/sampler wiring. |
| `scripts/v31_stable_refresh_campaign.py` | Arm names, stable parameters, source-matched immutable native leaf launcher. |
| `scripts/v31_stable_snapshot_replay.py` | Real frozen-Q/K/V replay before native generation. |
| `scripts/v31_refresh_factorial_replay.py` | Full factorial plus independent NumPy oracle. |
| `scripts/v31_full503_refresh_campaign.py` | New full-panel arm configurations and every-step baseline. |
| `scripts/v31_full503_refresh_study.py`, `v31_full503_qualified_resume.py` | Frozen orchestration, qualification, resource admission and resumable shards. |

The older `scripts/v31_value_campaign.py` exposes historical
`CACHED_REFRESH_*` arms. Use the stable/full503 launcher for corrected raw/stable
experiments; those are separate configurations, not aliases for the first panel.
The standalone `scripts/v31_cached_refresh_replay.py` is synthetic CPU diagnostic
code, not a substitute for real model snapshots or official accuracy.

The stable configuration is passed through `STABLE_REFRESH_CONFIG` to the native
bench. Check the arm table in `v31_stable_refresh_campaign.py`, the frozen
`specs/parameters.json`, and adapter validation together. Integrate the raw
row/tile permutation before normalization; retain protected tail and exact K;
reset caches, persistence and caps at their named dense/canvas boundaries.
Verify no-outside-candidate events leave the map unchanged. Value tie-breaks or
probes are separate variants requiring new source/receipt qualification.

## Reproduction instructions

Use the model/runtime/FA4 pins and authorized private manifests described in
document 1. Always run CPU checks first; CUDA/FA4 tests and real model smoke
require an idle H100 and a newly named run family. Do not launch a second GPU
worker alongside the active full503 campaign.

```bash
PYTHONPATH=src:. "$DLM_PYTHON" -m pytest -q \
  tests/test_v31_cached_refresh.py tests/test_v31_stable_refresh.py \
  tests/test_v31_refresh_factorial.py tests/test_v31_every_step_refresh.py

# A new authorized native smoke; DLM_CELLS contains a bound small smoke schedule.
PYTHONPATH=src:. "$DLM_PYTHON" scripts/v31_stable_refresh_campaign.py \
  --python "$DLM_PYTHON" --model "$DLM_MODEL" \
  --manifests "$DLM_MANIFESTS" --cells "$DLM_CELLS" \
  --result-root "$DLM_STUDY" --attempt "$DLM_STUDY/smoke/attempt001" \
  --arm stable_global_progress_dense_s2 --purpose smoke \
  --stable-parameters "$DLM_PARAMETERS" --max-model-len 141312 \
  --gpu-memory-utilization 0.80
```

Check the period/progress flag and K=128 tiles in the resulting receipt rather
than inferring them from an arm name. Follow with matching replay, fresh
V31/dense/all-kept controls, native generation, bound official scoring and
problem-clustered comparisons. Private replay snapshots and manifest hashes are
required; they are not bundled in Git. The factorial replay CLI takes
`--source-study`, `--study` and `--attempt` (see `--help` for additional flags).

The active full503 coordinator is already running. For future relocation or
resumption, inspect its `specs/protocol.json`, execution copy, `queue_state.json`,
qualification receipts and completion hashes first; use
`v31_full503_qualified_resume.py --study PATH --require-reservation` only when
that continuation is authorized and no coordinator is alive. Completed leaves
must be identity-validated, failed leaves retained, and new attempts numbered.
Do not run a report/analysis tool into an existing frozen result directory.

Publish N/C/T, per-problem paired correctness, actual physical eligible/kept work,
cheap/discovery statistics, accepted/no-op/fallback/cap counters, churn and
operator-error distributions together. Current selectors duplicate selected-
tile QK and maintain extra state; smaller N or sparser support is not measured
throughput. Clean timing remains paused throughout these studies.
