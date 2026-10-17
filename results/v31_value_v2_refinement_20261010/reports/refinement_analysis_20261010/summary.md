# V2 online refinement analysis (2026-10-10)

This report separates the completed official 503-item audit from a new 15-item development panel. Timing is paused; all new runs are audit/development runs with zero timed CUDA captures.

## What the audit shows

The official scorer on 503 LongBench-v2 `0shot_think` items gives the current V31 attention-mass control 268/503 (53.3%) at 83.1611% GLOBAL sparsity. V2 preserve-mass gives 266/503 (52.9%) at 84.1121% GLOBAL sparsity. The paired difference is -0.3976 percentage points, with a paired p-value of 0.90995 and interval [-3.78, +3.18] percentage points. V2 is therefore accuracy-comparable on this audit, but no positive gain is established.

Every sparse arm changed all 503 output hashes. V2 used 29.07 fewer denoising forwards on average. The trajectory effect is asymmetric: on 38 items correct only for V2, V2 used 198.87 fewer forwards on average; on 40 items correct only for the control, V2 used 204.25 more. V2 is changing the sampler basin, not merely stopping early.

## Why V2 has low mask IoU

V31 control ranks each query block by native attention mass and retains a fixed 128 prefix tiles per unit. V2 scans tiles sequentially and keeps a tile when its projected value-direction update exceeds the online risk threshold. These objectives disagree even when both see the same Q/K masses. Cross-step reuse magnifies the difference because V2 holds an alternative map while Q and V change.

The six-snapshot full-dimensional diagnostic shows projected V2 and full-dimensional V2 maps have mean IoU 0.858, while full-dimensional V2 and the mass control have mean IoU 0.156. This rules out rank32 projection as the main explanation for low control IoU. The online value objective itself selects a different support, often with much fewer tiles but a higher masked-operator error than the mass control at the same threshold.

## What shortens V2's trajectory

The V2 map changes attention outputs enough to alter acceptance and re-noising. On V2-only correct items, that change commonly reaches a stable answer in fewer canvases and forwards; on control-only items it can instead prolong the trajectory. This explains why fewer calls and comparable accuracy can coexist, and why a call-count reduction cannot be treated as a speedup claim until clean end-to-end timing is resumed.


## Complete jcgate audit

The official scorer covers all 503 cells for both arms. V31 control is 268/503 (53.3%); V2+jcgate is 256/503 (50.9%), a paired change of -2.39 percentage points. The discordant pairs are 44 control-only versus 32 jcgate-only (203 both wrong, 224 both correct); the two-sided exact McNemar p-value is 0.207, so this is not a significant loss, but it is clearly not an accuracy gain. Difficulty scores move from 58.3 to 52.1 on Easy and 42.6 to 37.0 on Long, while Hard is unchanged at 50.2. Parser coverage does not explain the loss: jcgate has 28 unparsed and 11 capped items versus 30 and 18 for control.

The candidate used 122,023 denoising forwards (242.59/item), versus 140,693 (279.71/item) for control, a mean reduction of 37.12. It also retained more GLOBAL work: 78.07% GLOBAL sparsity versus 83.16% for control. Its shorter trajectory therefore came with a denser selector and still lower accuracy; this is a sampler-basin effect, not evidence of a speedup.

This result rejects the current jcgate formulation. The weight is multiplicative on the online value risk, so unsettled rows raise the threshold-crossing score and protect extra tiles. That can improve per-row protection, but it changes the support and global work budget instead of reallocating a fixed budget. The development signal of fewer calls did not survive as an accuracy improvement on the full audit.

## Refinements tested

* **Mass anchors.** A new selector guard adds the top native-mass tiles as extra protections while preserving V2's running log mass and projected value state. In a six-snapshot operator sweep, 0→64 anchors improved control IoU 0.2928→0.4852 and full-dimensional error 0.3186→0.2585, but retained work rose from 2,827→3,850 tiles per unit. The live anchor32 smoke and 15-item development run were valid. Development accuracy matched the control at 7/15, while mean denoising forwards increased from 146.9 (control) to 159.7 and GLOBAL sparsity fell to 81.61%.

* **Settledness/J-C-gate protection.** V2 re-selections now weight unsettled query rows using the existing progress history. On 15 development items, jcgate matched the control at 7/15, reduced mean denoising forwards to 122.9, and had zero graph captures, but GLOBAL sparsity fell to 71.68%; total kept GLOBAL tiles were higher than the control. This is a promising accuracy-first trajectory arm, not a speed claim.

* **Combined jcgate + 16 mass anchors.** This addressed both failure modes. It remained 7/15 overall, flipped one control-only and one method-only item, and increased mean forwards to 180.3. It is rejected for the next panel.

* **Full-dimensional V.** A streaming full-V diagnostic avoids storing a Q×K×512 tensor. At threshold 0.027, six snapshots had mean normalized error 0.3298 versus 0.2467 for the mass control; projected/full-V map IoU was 0.8584. On one initial snapshot, lowering the threshold to 0.012 (137.6 tiles versus the control's 128) still gave error 0.2622; threshold 0.010 reached 166.8 tiles and error 0.2453. Full-dimensional V alone therefore does not buy accuracy at matched work and does not justify a fused kernel yet.

## Decision and next experiment

Keep the original V2 and V31 control intact. The complete jcgate audit rejects the row-weighted variant before timing: it is 2.39 percentage points below control and has lower GLOBAL sparsity. Do not resume timing. The next accuracy experiment, if requested, should be a fixed-budget mass-gated value selector: use native mass to define the control-sized candidate/support envelope, then use value residuals only to choose swaps within that envelope. This directly tests the objective-mismatch hypothesis without paying the uncalibrated work increase of anchors or jcgate. Full-dimensional V remains gated because it showed no matched-work operator advantage.

All new development outputs, scorer files, receipts, and diagnostics are under `results/v31_value_v2_refinement_20261010/`. The anchor64 smoke attempts are retained with `interrupted_oom` receipts and are resumable; no timing worker was launched.
