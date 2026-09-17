# JL-inspired output-aware physical-tile routing

This isolated v1 study extends the existing DiffusionGemma native-generation,
calibration and reporting code. Prior completed workers, configurations and
result bundles are not modified. No FlashAttention/hardware-speedup claim is
made. The target is downstream accuracy at comparable measured tile sparsity.

## Frozen sample and decoding scope

- Preserve the exact historical 30 AIME26 and 50 LongBench v2 IDs.
- AIME keeps its original prompts, scorer, seed42 and 2048-token output budget.
  Compatible dense and baseline generations/policies can be imported with
  source hashes and identity checks.
- The user approved upgrading the same LongBench IDs to the actual pinned
  NeMo-Skills prompt renderer/MCQ scorer and a 4096-token output budget.
  Therefore a new LongBench dense baseline is required. The domain counts are
  17 single-document QA, 17 multi-document QA, 16 code repositories; these are
  the requested historical IDs, not a newly proportional six-domain sample.
- Input limit32K; context-only head/tail truncation affects25/50 LongBench
  inputs. All80 final examples have been examined previously. Full AIME30
  includes the six historical calibration questions; noncalibration24 is
  reported separately and is not called fresh confirmation.
- Native BF16, canvas256, maximum48 denoising steps, thinkingFalse and the
  existing0.4–0.8 sampling schedule remain unchanged. Requested temperature0
  is the existing sentinel, not greedy decoding. All compared final runs use
  identical prompts, seeds and budgets.

## Predeclared candidates and controls

Both targets50/75% are requested skipped128-query×64-KV physical tiles.
Prefix, canvas and mixed boundary tiles remain eligible under native masks.

The primary grid contains all six Gaussian/random-sign × rank8/16/32 centered
routers. The fixed primary projection seed is1729. Seeds2718 and31415 are
prespecified sensitivity checks on shared calibration states and development
generation, not candidates from which to choose a favorable primary seed.
Every layer/native KV head gets an independent CPU-generated FP32 matrix,
with derived seed and SHA256 recorded. Streams also differ across dimensions;
the rank comparison is not a nested-column projection experiment.

Controls: exact-mass routing under the same retained-state convention; both
families' uncentered projected contributions at rank16; full-dimensional
attention-weighted centering; the historical unweighted-centered-plus-radius
heuristic; historical value-vector-mean norm, mass-bound and mean-norm risk
methods; and original/aggressive BLASST. Rank16 control choices are specified
as the middle dimension, not selected by accuracy.

Two type-aware controls, Gaussian16 and sign16, use a second independent
rank16 sketch only to safeguard suspected cancellation: if primary projected
κ<0.25 and exact online α≥0.1, use the larger centered risk from the two
sketches. Otherwise use the primary centered risk. Both projected outputs are
updated from the same retained blocks. This heuristic is not a certified JL
or cumulative-error bound. The plain centered versions remain ablations.

No method is eliminated for a modest development-score difference or better
operator error. The intended complete sweep is all18 sparse candidates at
both targets plus dense:37 conditions ×80 examples =2960 final outputs.
Failures remain visible; no unsuccessful generation is silently dropped.

Historical operator inventory additionally identified the token-aligned
value-norm control. It is prespecified in `aligned_addendum.py`, queued after
the primary final audit/report, preserving the already-running worker and
its frozen 37-condition configuration. It adds both targets on the same80
examples, reusing all dense outputs and compatible AIME results. The final
consolidated scope is39 conditions ×80 =3120 unique final results. Its scalar
local/global calibration reuses the same captured calibration states and
native aligned risk implementation; no JL final score selected this control.

## Calibration and diagnostics

Use the same six calibration question IDs per benchmark with explicitly
recorded512-token calibration rollouts. Obtain shared-state score quantiles,
then reuse the existing local/global scalar-CDF refinement, at most three
verified sparse-trajectory points per candidate/target/benchmark. This reduced
fitting budget is not the final generation budget; final sparsity transfer
errors are reported without retuning. Compatible AIME baseline policies keep
their historical full-budget verification and are explicitly distinguished.
BLASST keeps its established inverse-valid-KV-length λ convention. Original
caps λ at1 and requires measured boundary evidence; aggressive permits λ>1.

Shared-state diagnostics preserve the candidate's retained support while
replaying its full-dimensional centered risk, so adaptive-prefix
underestimation is measured rather than invoking a JL guarantee. Type labels
overlap. Record projected/full classifications, mixed-query tiles, dropped
update energies, risk distortion and full-dimensional local output error.
Full-dimensional data are diagnostic-only, except the explicit full control.

The actual router computes projected means from token sketches, not from a
full block PV. FP32 projection, within-block softmax and retained state are
checked against a trusted reference. Original V is used in the native final
attention output. Complete unchanged prefix tiles reuse sketches only after
exact value/validity equality checks; boundary/canvas entries refresh.

## Accounting and reproducibility

Physical sparsity is sum(skipped)/sum(eligible), independently overall/global/
local. Retained mass is exact dense probability mass at each sparse query
state; positional token agreement continues after the first divergence and
counts missing/extra positions as disagreements. Operator error on shared QKV
is separate from generated-sequence divergence.

QK, block softmax and projected PV are still computed. The final native BF16
PV currently remains a dense-shaped matmul with zeroed skipped probabilities;
physical deletion is not claimed to elide that hardware arithmetic. Full
block PV is absent from sketch-based routing, but full-dimensional diagnostic
outputs are computed separately. Projection multiply-adds, stored sketch
bytes, value-comparison/copy overhead and refresh/reuse counts are recorded.
Raw per-layer records are losslessly compressed with hashes, retaining exact
numeric values and allowing CPU-only audit/report regeneration.

Run from the repository root with the existing `ljy_dlm` Python environment:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONPATH=src:. python -m experiments.diffusion_gemma_jl_output_aware.protocol
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=src:. python -m experiments.diffusion_gemma_jl_output_aware.validation
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=src:. OMP_NUM_THREADS=8 python -m experiments.diffusion_gemma_jl_output_aware.workflow launch
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONPATH=src:. python -m experiments.diffusion_gemma_jl_output_aware.workflow report
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONPATH=src:. python -m experiments.diffusion_gemma_jl_output_aware.verify
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=src:. OMP_NUM_THREADS=8 python -m experiments.diffusion_gemma_jl_output_aware.aligned_addendum launch
```

The subsequent workflow uses one resumable GPU worker and approximately
15-minute health checks. Development source versions and unsuccessful
validation evidence are archived, not overwritten.
