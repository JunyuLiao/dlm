# LLaDA2.2 HumanEval EOS-confidence validation

Output token IDs and logits are unchanged by the intervention; only confidence-based selection zeros the confidence for predicted EOS. Native forced progress and EOS termination remain enabled.

| Arm | Correct /164 | Mean T | Median T | Max T | EOS finished | Cap hits | N/request | C/request | N/C | Consumer sparsity | Syntax errors | Parsed funcs | Substantive funcs | EOS confidence zeros |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_native | 141.0/164 (85.98%) | 243.1 | 224.0 | 783 | 164 | 0 | 28.8 | 8.57 | 3.357 | 0.00% | 2 | 162 | 162 | 0 |
| allkept_fa4 | 138.0/164 (84.15%) | 249.2 | 228.5 | 1024 | 163 | 1 | 31.4 | 8.77 | 3.581 | 0.00% | 2 | 162 | 162 | 0 |
| mass_topk_reuse_r4 | 83.0/164 (50.61%) | 218.9 | 193.0 | 788 | 164 | 0 | 36.8 | 7.84 | 4.692 | 49.98% | 31 | 128 | 121 | 0 |
| value_v2_gaussian32_reuse_r4 | 10.0/164 (6.10%) | 62.9 | 58.0 | 155 | 164 | 0 | 14.2 | 3.07 | 4.634 | 49.22% | 98 | 61 | 28 | 0 |
| value_v3a_singleton_reuse_r4 | 103.0/164 (62.80%) | 245.3 | 219.5 | 1024 | 163 | 1 | 42.3 | 8.67 | 4.881 | 49.99% | 21 | 140 | 135 | 0 |
| dense_native_eos_conf0 | 142.0/164 (86.59%) | 244.9 | 224.0 | 786 | 164 | 0 | 44.7 | 8.63 | 5.179 | 0.00% | 1 | 163 | 163 | 59049 |
| value_v2_gaussian32_reuse_r4_eos_conf0 | 27.0/164 (16.46%) | 139.4 | 126.0 | 375 | 164 | 0 | 55.9 | 5.41 | 10.322 | 61.53% | 89 | 72 | 65 | 109766 |

Paired accuracy contrasts (difference and 95% problem-bootstrap CI):
- `dense_native_eos_conf0_vs_dense_native`: +0.61 percentage points [+0.00, +1.83], 1 wins / 0 losses / 163 ties.
- `value_v2_gaussian32_reuse_r4_eos_conf0_vs_value_v2_gaussian32_reuse_r4`: +10.37 percentage points [+4.88, +15.85], 20 wins / 3 losses / 141 ties.
- `value_v2_gaussian32_reuse_r4_eos_conf0_vs_dense_native_eos_conf0`: -70.12 percentage points [-76.83, -62.80], 0 wins / 115 losses / 49 ties.
- `value_v2_gaussian32_reuse_r4_eos_conf0_vs_mass_topk_reuse_r4`: -34.15 percentage points [-43.90, -24.39], 14 wins / 70 losses / 80 ties.
- `value_v2_gaussian32_reuse_r4_eos_conf0_vs_dense_native`: -69.51 percentage points [-76.22, -62.20], 0 wins / 114 losses / 50 ties.

The EOS-confidence-zero column counts EOS predictions whose confidence value was zeroed across all model sampling calls; it is not the number of EOS tokens emitted in final answers.

V2 changes from 10/164 at 62.9 mean tokens, 14.2 N/request, and 49.22% sparsity to 27/164 at 139.4 tokens, 55.9 N/request, and 61.53% sparsity. The paired improvement is 10.37pp, but generation work rises substantially and realized sparsity moves outside the 48-52% target. This partially rescues V2 on this fixed threshold; it is not a matched-sparsity result or a complete repair.
Dense changes from 141/164 to 142/164 (+0.61pp), with mean output length 243.1 vs 244.9. The larger gain is specific to the tested V2 trajectory, though the V2 sparsity/work shift prevents attributing it to EOS confidence alone at equal attention work.

The confidence mask leaves EOS logits and native stopping enabled. EOS still ends all164 requests, so it delays/changes EOS selection rather than removing the stop token. Follow-up should calibrate this combined V2+mask policy on separate non-benchmark long-code prompts to restore 48-52% physical sparsity, then test on a fresh code-completion cohort or explicitly label any repeated HumanEval use as transductive. Include syntax/entry-point/completeness checks so sparsity calibration cannot select a short-answer failure.

Intervals resample HumanEval problems (10,000 draws); one seed/problem means they quantify item uncertainty, not seed variability. The no-retuning, fixed-threshold panel is exploratory; do not claim matched-work superiority.
