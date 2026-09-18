# RULER8K: empirical interpretation

All1170 outputs passed the final audit and independent raw-only report regeneration.19 CPU/CUDA tests and the two-input actual-model smoke passed. Version13’s initial bookkeeping failure is preserved; version14 completed without failed calibration or final evaluations. The experiment used130 questions,10 per task, with26 disjoint calibration examples and fixed seeds/settings. Previously used source pool and one generation/projection seed: these are not fresh, multi-seed confirmation results.

## Main results

All entries below are percentages. Score is official RULER equal-task macro with fractional answer credit, not an integer correct-question count. Physical sparsity uses summed skipped/eligible tiles across the complete runs.

| Method | Target | Actual | Global | Local | RULER score | Retained mass | Token agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
|dense|0.00|0.00|0.00|0.00|89.23|100.00|100.00|
|blasst|50.00|52.27|69.81|28.67|92.23|89.25|83.49|
|mass|50.00|49.55|49.55|49.55|93.08|75.86|78.86|
|full_centered|50.00|50.93|51.35|50.36|91.92|74.11|77.57|
|jl_gaussian_r2|50.00|50.77|51.05|50.39|91.92|72.88|74.99|
|blasst|75.00|75.19|75.02|75.43|3.27|38.55|5.82|
|mass|75.00|75.23|75.49|74.90|21.27|40.53|12.40|
|full_centered|75.00|75.89|76.26|75.40|49.78|39.92|27.36|
|jl_gaussian_r2|75.00|75.64|75.20|76.24|2.50|38.67|7.24|

## What the comparison supports

1. **Around50% sparsity, Gaussian2 matches the full-dimensional score; directional routing has no clear accuracy advantage over mass-only.** Full and Gaussian2 both score91.92%; their physical sparsities differ by only0.16pp. The paired full-minus-Gaussian2 score interval is−3.85 to+3.85pp. Mass-only scores93.08%, but its+1.15pp difference from either centered method is inconclusive (95%CI−1.92 to+4.62pp).

2. **At75%, full-dimensional direction clearly helps, but rank2 does not preserve that benefit.** Actual physical sparsities are75.19–75.89%, with local/global gaps below1.34pp between all methods. Full-dimensional centered scores49.78%, versus21.27% mass-only,3.27% aggressive BLASST and2.50% Gaussian2. Full-minus-mass is+28.51pp (paired95%CI+20.13 to+36.82); full-minus-Gaussian2 is+47.28pp (+39.86 to+54.40). These are comparisons at closely matched actual budgets. Even the full-dimensional reference remains39.45pp below dense. The result supports useful directional information, not a claim that a two-dimensional sketch is sufficient at high sparsity.

3. **RULER’s8K input does not make local BLASST sparsity easy.** At the joint λlocal=λglobal=1 ceiling, calibration achieved60.18% overall,83.67% global and28.42% local. The50% policy therefore stays capped: final λlocal=1 and λglobal≈0.479–0.618, with28.67% local and69.81% global sparsity. That layer allocation differs materially from the roughly50/50 local/global allocation of the other methods, so the50% BLASST comparison is not layer-budget matched. The75% ceiling test permits aggressive mode: λlocal≈18.869 while λglobal≈0.590–0.760 remains below1. Only the local thresholds actually exceed1. Local masks see the native limited window, not the whole8K prompt.

4. **Attention mass alone does not explain accuracy.** At75%, all methods retain about38.5–40.5% of dense probability mass, yet scores range from2.50% to49.78%. Full-dimensional centered retains slightly less mass than mass-only but has much better accuracy. Its trajectory-local normalized operator error is0.556 versus0.726 for mass-only,0.670 for Gaussian2 and0.830 for BLASST. Token agreement is27.36% for full versus12.40%,7.24% and5.82%, respectively. These relationships are descriptive, not a causal or cumulative-error guarantee.

## Important dense-baseline limitation

All10/10 dense variable-tracking (`vt`) outputs hit the official30-token budget and score zero. The raw outputs show an unfinished introductory sentence before the requested variable names. Some50% sparse outputs answer more tersely and fit within the same budget. Consequently, their headline improvements over dense are partly a formatting/budget effect—not clean evidence that pruning improves reasoning. The official budgets were kept identical across methods and were not changed after seeing scores.

As a **post-hoc sensitivity check only**, excluding VT gives dense96.67%, BLASST-50 96.58%, mass-50 98.33%, full-50 96.25% and Gaussian2-50 96.25%. The registered13-task results above remain primary; the alternative12-task numbers do not replace them. A longer-output VT rerun would require a separately identified experiment.

## Calibration and diagnostic limits

All eight policies passed calibration within2pp of their declared whole-model and attainable per-type goals. Final BLASST-50 drifted to52.27% overall (calibration50.25%); it was not retuned. Other50% final points lie49.55–50.93%;75% points lie75.19–75.89%.

Dense proposal snapshots cover780 sampled states:26 calibration prompts×30 layers, all at denoising step0. The prespecified optional steps4/12/24 did not occur on the first two calibration prompts. The existing shared-operator selector chooses one source per layer/step, resulting here in30 states from a single CWE calibration prompt and240 method/target evaluations. It is therefore limited diagnostic evidence, not representative coverage of all13 tasks or later denoising. Full-generation routing/mass/error counters separately cover every executed layer/head/step.

On that shared probe at75%, local normalized operator errors are0.629 full,0.717 Gaussian2,0.846 mass-only and0.881 BLASST. Gaussian2 disagrees with same-threshold full-risk decisions on69/480 local physical tiles, using its own retained history. It severely underestimates10,868/56,640 supported local row-block updates (projected centered risk below half the true risk, true risk at least0.05). This is consistent with projection distortion affecting local selection, but the restricted probe cannot establish the sole cause of generation failure. Baseline risk-agreement fields are not applicable; aggregate zeros in those legacy fields must not be interpreted as perfect agreement.

## Bottom line

The50% operating point preserves RULER quality for all tested methods, with no clear directional-method winner. At matched75% budgets, full-dimensional centered selection is substantially better than mass-only and BLASST, while Gaussian2 fails to retain that advantage. The strongest follow-up question is whether a less distorted sketch and/or a less aggressive local-layer budget preserves the full-dimensional benefit; this run does not answer it. No hardware-speedup claim is made.

See [canonical report](report.md), [per-task results](per_task.csv), [thresholds](thresholds.csv), [paired comparisons](comparisons.csv), and [tradeoff plots](figures/tradeoffs.png).
