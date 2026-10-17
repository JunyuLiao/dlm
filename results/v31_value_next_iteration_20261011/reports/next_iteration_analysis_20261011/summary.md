# V31 online value-aware next iteration — accuracy report

Status: complete accuracy panel; clean timing remains paused and resumable. Source
commit recorded in each attempt is `fce50a8b110f29c47b01cd5d2dd42d3986df328a`. The
new selector code is uncommitted in this worktree because the repository's git
metadata is read-only; the dirty user prompt file was preserved.

## What was tested

The implementation keeps the inherited V31 reuse lifecycle and the native sparse
consumer unchanged. At every initial/refresh map, the exact native attention-mass
map supplies the same per-(KV head, 128-query-row block) prefix budget as the
control. A value score may exchange tiles only within that budget. CUDA ranking
and bounded swaps remain on-device; `C_gate` is a ranking weight with no change
to K. The fixed-budget path has a CPU oracle, H100 synthetic CPU/CUDA parity,
real-adapter smoke validation, and 34 passing unit tests (28 CUDA-dependent
skips in the non-CUDA test process).

The five requested ideas were evaluated as follows:

| idea | arms | evidence and gate |
|---|---|---|
| fixed-budget mass-gated value selection | online swap8/swap16; projected full swap16 | six saved full-dimensional snapshots: online error 0.1949/0.1927 vs mass control 0.1977 at 8/16 swaps; projected full 0.1990, so no full-information promotion |
| online versus full information | online and full swap16; full-V true omission score in the offline diagnostic | true full-V score did not beat the matched mass map; live full arm is explicitly projected-summary information and is not called true full-V |
| balanced query sensitivity (`C_gate`) | online C_gate swap16 | fixed K was preserved, but development accuracy was 6/15 versus 7/15 control; reject |
| phase/drift protection | phase swap16, plus age/phase and denoising-call receipts | phase arm was 8/15 and raised mean calls to 183.9 versus 146.9 control; no promotion; drift evidence is retained for a later trigger-only experiment |
| V31∩V2 support | online and full intersection | six-snapshot intersection kept about 30–39 of 128 tiles and had error about 0.366–0.372; both 15-item arms were 7/15 and had about 232 calls; reject as a production map |

## Development score

The official LongBench scorer on the frozen 15-item seed-1 schedule gave:

| arm | correct | overall | mean denoising forwards | realized GLOBAL sparsity |
|---|---:|---:|---:|---:|
| current V31 control | 7/15 | 46.7% | 146.9 | 76.45% |
| online swap8 | 9/15 | 60.0% | 137.8 | 79.16% |
| online swap16 | 8/15 | 53.3% | 147.0 | 78.37% |
| online phase swap16 | 8/15 | 53.3% | 183.9 | 81.44% |
| online C_gate swap16 | 6/15 | 40.0% | 132.7 | 80.38% |
| projected full swap16 | 7/15 | 46.7% | 323.6 | 85.97% |
| V31∩V2 online | 7/15 | 46.7% | 232.2 | 86.71% |
| V31∩V2 full | 7/15 | 46.7% | 232.2 | 86.71% |

The 15-item gain of swap8 is a development signal only. It did not survive the
full paired audit.

## Full official audit

`value_mass_gated_online_swap8` completed all 503 target cells in four resumable
shards. The official scorer reports 264/503 (52.5%) versus 268/503 (53.3%) for
the matched V31 control: -4 items, -0.795 percentage points. Discordant pairs
are 43 control-only and 39 candidate-only; exact two-sided McNemar p=0.741 and
paired item bootstrap 95% CI [-4.97, +3.38] percentage points. The candidate's
mean denoising forwards are 249.86 versus 279.71 for control (-29.84), and its
realized GLOBAL sparsity is 82.50% versus 83.16%. The lower call count is real,
but no accuracy or timing gain is established.

The negative full-audit result is consistent with a trajectory effect rather
than a clean selector win: candidate-only correct items used about 351 fewer
forwards than control, while control-only items used about 105 more forwards
than control. Output lengths differ by the same direction. Therefore the
shorter trajectory cannot be treated as a speed benefit without a matched clean
timing run, and the 15-item positive result was sampling noise or task-specific.
By strata, swap8 is +1.29 points on Hard, +1.40 on Medium, but -4.17 on Easy,
-2.78 on Long, and -2.22 on Short.

## Decision and next refinement

Do not resume timing and do not promote swap8, swap16, C_gate, phase, or
intersection. The strongest remaining direction is a **mass-anchored,
confidence-gated online swap**: keep the exact V31 support envelope and allow a
small swap only when (a) the online residual margin over the mass candidate is
large, and (b) the held-map age/drift certificate is below a refresh threshold.
Otherwise retain the mass map. This directly addresses the observed failure:
value ranking can be locally better on a frozen snapshot but can perturb the
native adaptive stopping trajectory when the map is reused. The gate must be
per-qblock and fixed-budget, with a no-swap fallback; its first test should be
an audit-only 15-item arm with swap8, swap16, and no-swap strata, followed by a
full audit only if the paired accuracy interval is nonnegative and denoising
calls do not fall through trajectory truncation.

A true full-dimensional-V kernel is not justified by these data: its matched
operator error was worse than mass control, and the live projected-full arm
raised calls. The intersection diagnostic is useful as a support-identification
measurement, but its low retained support makes it unsuitable as the selected
map.
