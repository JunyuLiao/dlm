# EOS-confidence masking on LLaDA2.2-mini HumanEval

## Result

Setting predicted EOS confidence to zero, while retaining native logits, forced progress and EOS stopping, partially improves V2 but does not fix HumanEval accuracy.

| Arm | Correct /164 | Mean T | Median T | N/request | C/request | N/C | Consumer sparsity | Syntax errors | Substantive functions | Cap hits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense native | 141 (85.98%) | 243.1 | 224 | 28.8 | 8.57 | 3.357 | 0% | 2 | 162 | 0 |
| Dense + EOS-confidence zero | 142 (86.59%) | 244.9 | 224 | 44.7 | 8.63 | 5.179 | 0% | 1 | 163 | 0 |
| V2, original | 10 (6.10%) | 62.9 | 58 | 14.2 | 3.07 | 4.634 | 49.22% | 98 | 28 | 0 |
| V2 + EOS-confidence zero | 27 (16.46%) | 139.4 | 126 | 55.9 | 5.41 | 10.322 | 61.53% | 89 | 65 | 0 |

V2 + EOS-confidence zero gains 17 tasks versus the original V2, +10.37 percentage points, paired 95% problem-bootstrap CI [+4.88, +15.85], with 20 wins, 3 losses and 141 ties. Mean output length more than doubles, but remains well below dense. All164 outputs still end in EOS; none reaches the1024-token cap. The intervention delays EOS selection in the trajectories; it does not disable EOS termination.

The dense control changes by one task:142/164 versus141/164, +0.61pp CI[0.00,+1.83]. This suggests the measured gain is specific to the V2 trajectory, but V2 + EOS-confidence zero realizes61.53% sparsity rather than its original49.22% or the48-52% target. It also performs9166 forwards versus2331 in original V2, and paired calls/canvas rises substantially. Consequently the accuracy gain is not established at matched sparsity or matched attention work; the confidence intervention changed the trajectory and V2's realized routing.

The method still trails V31 (83/164,50.61%) by34.15pp, paired CI[-43.90,-24.39], and dense by69.51pp, CI[-76.22,-62.20]. Among V2 + mask's72 parseable functions with the required entry point,65 have substantive bodies, but89 outputs still fail syntax parsing; only27 tasks pass the official tests. This is a partial mitigation, not an acceptable V2 HumanEval operating point.

## Validation

Both arms use all164 official tasks and seed42. The intervention only changes confidence passed into native confidence-based selection when the predicted token is EOS; model logits, sampled token predictions, V2 selector maps/threshold, forced-progress schedule and stopping rule remain unchanged. V2 threshold is fixed at0.8944271909999159; no recalibration or benchmark-score tuning was done. Dense + mask controls for a general sampler effect.

Official HumanEval scoring passes in the pinned isolated scorer. The paired audit verifies all item identities, prompts, generation caps, public/private output hashes, physical-work counters, source bindings, EOS confidence-zero counts, and full coverage for the four new shards per arm. Seven focused CPU tests pass. Detailed paired outputs and full accounting are in `analysis_eos_conf0_20261102_attempt003/`.

The 48-52% sparsity target misses under the frozen V2 threshold. Do not recalibrate on these HumanEval scores or claim a matched-sparsity win. A next confirmation should calibrate this combined policy on separate long-code development prompts using both physical sparsity and code-completeness/EOS diagnostics, then test on a fresh code-completion cohort. This HumanEval panel is already examined.
