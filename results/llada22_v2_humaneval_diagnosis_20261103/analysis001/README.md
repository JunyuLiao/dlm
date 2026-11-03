# Why LLaDA2.2-mini V2 reaches EOS early on HumanEval

The diagnostic identifies a support-selection failure in the frozen V2 operating
point. Its threshold leaves almost every head with the compulsory first KV64
tile and the current-block tile, ignoring useful middle context. On identical
token states and prefix KV, this support changes the model's prediction to EOS.
The native sampler then accepts those predictions and can refine a block into
an EOS-dominated stable state. EOS handling itself was not changed.

## Direct native evidence

Four previously examined canonical HumanEval items (0–3), seed42, were replayed
under original V2, V31, V3a and all-kept FA4. All16 observed outputs and native
forward/canvas counts match both saved benchmark records and uninstrumented
clean bookends:48 complete replays total. These are descriptive mechanism
diagnostics, not a new benchmark or held-out validation.

| Item | Original V2 output tokens | First EOS-writing call in its block | Initial EOS writes | EOS tokens in terminal block | Final stop |
|---|---:|---:|---:|---:|---|
| HumanEval/0 |110|1|18|18|Fully unmasked and unchanged block|
| HumanEval/1 |39|1|1|32|Fully unmasked and unchanged block|
| HumanEval/2 |9|1|3|32|Fully unmasked and unchanged block|
| HumanEval/3 |115|1|7|20|Fully unmasked and unchanged block|

All first EOS writes are mask-to-token writes on a **fresh first forward**;
no held map has yet been reused in those blocks. In items0/2/3, the initial
EOS writes exceed native confidence threshold0.5. In item1, no position exceeds
that threshold; native forced progress chooses one EOS prediction with
confidence0.355. It subsequently expands to32 EOS tokens. The editing rule can
rewrite generated tokens, and a fully unmasked unchanged block ends refinement.
Outer generation stops after a completed block contains EOS. Entropy is an
observed diagnostic; it is not LLaDA's block-stop criterion. No step cap causes
these terminations.

## Same-state causal route comparisons

The captured input tokens and all layer-prefix KV are held identical. Alternative
routes run diagnostic-only forwards; their predictions never enter generation.
The29 positions below are exactly the positions selected for the first EOS write
across four V2 trajectories. They are selected failure positions, not29
independent tasks.

| Route on identical input and prefix KV | EOS argmax positions /29 | EOS confidence >0.5 /29 | Mean EOS probability |
|---|---:|---:|---:|
| Original V2 support replay |29|28|0.76751|
| Fresh V2, unchanged threshold |29|28|0.76751|
| All-kept FA4 |0|0|0.000178|
| V31, ordinary half-budget |4|0|0.04978|
| V3a, ordinary half-budget |7|0|0.07451|
| Mass ranking, exact V2 count in every layer/head |12|4|0.21247|
| Replace only the first tile when a higher-mass outside tile exists, same counts |12|4|0.21191|

Original-map replays reproduce the captured full logit vectors **bit-for-bit**.
The equal-count controls keep exactly the same tile count in every layer/head;
this isolates tile identity from allocation of work across heads. The first-tile
swap changes at most one tile/head in497 of1280 layer/head cases across the four
states. It removes17 of29 EOS argmaxes and24 of28 high-confidence EOS predictions.
The aggregate effect is large but incomplete: item3 keeps7 EOS argmaxes under
both equal-count controls, though all seven fall below0.5 confidence. No complete
continuation or accuracy improvement is established by these local probes.

V2 with the prior AIME threshold0.012489663701836472 also removes all29 EOS
argmaxes, but is nearly dense on these short states. It is an uncalibrated
sensitivity control, not a proposed setting or matched-work repair.

## What is wrong with the selected support

At the EOS-writing states, **1219/1280 (95.23%)** layer/head cases retain only
the first KV64 tile and mandatory current-block tile. V2 always keeps tile0
because its online recurrence forces the first support; it then scans in token
order. With the frozen HumanEval threshold0.8944271909999159, almost all later
optional updates are rejected. Mandatory current-block protection supplies the
other tile. This looks approximately50% sparse when the short context has about
four tiles, while selecting a very different support from a mass budget.

The direct same-Q/K/V operator audit covers two snapshots/item (first forward
and first EOS-writing forward), all20 layers:160 snapshot/layer cases. At the
EOS-writing snapshots, means below weight each of the80 layer cases equally
and average over its16 heads and32 query rows. Sparse outputs use full-dimensional
V and exact retained-mass renormalization.

| Support on identical Q/K/V at first EOS-writing states | Kept tiles | Retained probability mass | Mean output error / value RMS | Mean output error / dense-output norm |
|---|---:|---:|---:|---:|
| Actual V2 |2626|58.06%|0.18254|0.57754|
| Fresh V2 |2626|58.06%|0.18254|0.57754|
| Mass ranking, same count in each head |2626|67.05%|0.12957|0.46029|
| V31 ordinary half-budget |2560|66.01%|0.13557|0.48094|
| Full128-dimensional V online selector, same threshold |2622|57.87%|0.18402|0.58110|

Value RMS means `sqrt(mean_key ||V_key||^2)` per head. The equal-count mass
support raises retained mass by8.99 percentage points and reduces the
value-RMS-normalized error by29.0%. For item2, the three KV tiles carry
33.87%,36.95%,29.18% mean attention mass, respectively. V2 keeps the first and
last in every head, but keeps the **middle tile in only3.44%** of layer/head
cases. Its9-token answer therefore coexists with omitting the highest-mass
context tile almost everywhere. In item3, the penultimate tile carries25.61%
mean mass but is kept in only5.94% of cases.

## Implementation and projection checks

- Independent FP32 CPU QK/projected-value statistics and online recurrence agree
  with CUDA on all160 masks:zero mismatched tile decisions. Maximum log-mass
  difference1.53e-5; projected-mean difference1.60e-4.
- Actual FA4 outputs agree with the mathematical output on their selected
  full-dimensional support to at most0.001821 relative L2 error. The dominant
  perturbation is the selected support, not a large consumer arithmetic error.
- Replacing Gaussian32 with full128-dimensional V yields mean mask IoU0.99163
  across160 cases and essentially the same mass loss/output error. Increasing
  projection rank alone does not address this failure on the observed states.
- Native SDPA clean/observed runs match within each process but showed a
  process-history-dependent historical mismatch (24 versus25 forwards on item0).
  Attempts002/004 preserve it; attempt003 reproduces the historical control.
  The complete mechanism panel uses all-kept FA4 as its matched dense control
  and keeps strict historical plus clean-bookend parity for every included arm.

## Interpretation and limits

The evidence now goes beyond a correlation between short output and EOS:
changing attention support alone changes the EOS decision, and a one-tile swap
at identical per-head work removes most high-confidence EOS predictions. The
key problem is a high-threshold online selector collapsing to its compulsory
support, followed by native confidence/forced-progress and token refinement
reinforcing the wrong state. Cross-step reuse may affect earlier context or
subsequent refinement, but it is not required for these first EOS triggers.

This also corrects an earlier report: completed matched LLaDA AIME V2 emits
**422632/90=4695.9 tokens/request**, not52.18. Output length alone therefore
cannot explain the HumanEval-specific collapse. The AIME, HumanEval and RULER
V2 thresholds differ materially (0.01249,0.89443,0.00007499). Same aggregate
sparsity does not ensure the same quality of support across these geometries.
This diagnostic identifies the failure in the frozen LLaDA HumanEval setting;
it does not prove an intrinsic MoE defect or a general failure of value-aware
code generation, nor does it establish a repaired benchmark accuracy.

The appropriate next target is V2 support selection with a meaningful fixed
budget and mass protection, preserving native EOS/acceptance/edit rules.
No EOS mask, threshold retuning, calibration, or full benchmark rerun was done.

Canonical sources: native replay `attempt007/receipt.json`, operator audit
`attempt008/metrics.json` and `oracle.json`, matched-count causal probes
`attempt009/receipt.json`; `summary.json`, `receipt.json` and
`final_complete.json` beside this report verify their identities and accounting.
