# LongBench v2: LLaDA2.2-mini and TraDo-8B-Instruct

Seed 42; official zero-shot prompt and exact official answer extraction. Invalid/unparsed answers count wrong. No chance compensation.

| Model | Length groups | Method | Correct | Accuracy | Mean output tokens | Steps/request | Canvases/request | Steps/canvas | Actual sparsity | EOS / caps | Unparsed |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llada22 | short + medium | Native dense | 125/395 | 31.65% | 47.3 | 20.89 | 2.56 | 8.153 | 0.000% | 288 / 107 | 67 |
| llada22 | short + medium | All-kept FA4 | 130/395 | 32.91% | 48.7 | 21.33 | 2.61 | 8.179 | 0.000% | 285 / 110 | 69 |
| llada22 | short + medium | V31 K512 | 145/395 | 36.71% | 22.9 | 8.24 | 1.83 | 4.506 | 99.443% | 360 / 35 | 32 |
| llada22 | short + medium | V3a K512 | 138/395 | 34.94% | 23.1 | 8.72 | 1.83 | 4.757 | 99.441% | 358 / 37 | 35 |
| llada22 | short + medium | V2-CTI | 134/395 | 33.92% | 26.9 | 10.26 | 1.93 | 5.326 | 98.504% | 346 / 49 | 46 |
| trado | short | Native dense | 70/180 | 38.89% | 15.8 | 10.59 | 4.67 | 2.269 | 0.000% | 176 / 4 | 0 |
| trado | short | All-kept FA4 | 73/180 | 40.56% | 15.6 | 10.76 | 4.60 | 2.338 | 0.000% | 176 / 4 | 1 |
| trado | short | V31 K512 | 76/180 | 42.22% | 14.8 | 10.34 | 4.40 | 2.350 | 97.988% | 175 / 5 | 4 |
| trado | short | V3a K512 | 80/180 | 44.44% | 14.9 | 10.23 | 4.42 | 2.316 | 97.971% | 175 / 5 | 3 |
| trado | short | V2-CTI | 79/180 | 43.89% | 14.8 | 10.41 | 4.39 | 2.371 | 98.351% | 174 / 6 | 3 |

Official length-group accuracy:

| Model | Group | Native dense | All-kept FA4 | V31 K512 | V3a K512 | V2-CTI |
|---|---|---:|---:|---:|---:|---:|
| llada22 | short | 39.44% | 39.44% | 42.22% | 41.11% | 41.11% |
| llada22 | medium | 25.12% | 27.44% | 32.09% | 29.77% | 27.91% |
| trado | short | 38.89% | 40.56% | 42.22% | 44.44% | 43.89% |

Input/truncation accounting (identical across methods within each model):

- llada22: 395 problems, 99 middle-truncated; actual prompt range 10352–130936 tokens. Context limit 131072, including the 128-token output reserve.
- trado: 180 problems, 64 middle-truncated; actual prompt range 10150–32633 tokens. Context limit 32768, including the 128-token output reserve.

Setup and interpretation:

Each model retains its existing chat template and native sampler. LLaDA uses 32-token blocks, 32 steps, threshold 0.5, editing threshold 0, max post steps 16, temperature 0 and native early EOS. TraDo uses 4-token blocks, 4 steps, temperature 1, top-p 1, top-k 0 and confidence 0.9 with native EOS. BF16, batch 1, one H100.

All denoising layers are eligible; prefill and commit forwards stay dense. Native dense uses SDPA; all-kept FA4 controls for the sparse consumer. V31 ranks attention mass; V3a uses Gaussian32 full-information singleton risk. K512 means eight KV64 tiles, including mandatory current tiles, per query head/Q128 block. Four-call map reuse, no cross-canvas carry. Junyu selectors and current-tile initialization are distinct from Yuhan inherited map reuse.

Sparsity is one minus pooled actual kept/eligible denoising tiles, including mandatory tiles and selection/reuse consumer calls; selector discovery, dense prefill and commits are outside that denominator. Discovery QK pairs and projected-value work appear in summary.json. Sparsity does not imply whole-model speedup. Timings are instrumented accounting only.

V2-CTI thresholds are selected after both fixed-budget runs, on six count-only prompt-length strata per requested length group. This is transductive calibration: no correctness, gold or parsing metric is used. Exact reference and achieved calibration points are in each selected_threshold.json. Achieved full-panel sparsity is reported above; it is not assumed equal from requested budgets.

Paired item-bootstrap differences in percentage points (10,000 resamples):

- llada22, V31 K512 − Native dense: +5.06 pp, 95% CI [+1.77, +8.10].
- llada22, V31 K512 − All-kept FA4: +3.80 pp, 95% CI [+0.51, +6.84].
- llada22, V3a K512 − Native dense: +3.29 pp, 95% CI [+0.25, +6.33].
- llada22, V3a K512 − All-kept FA4: +2.03 pp, 95% CI [-1.01, +5.06].
- llada22, V3a K512 − V31 K512: -1.77 pp, 95% CI [-3.54, -0.25].
- llada22, V2-CTI − Native dense: +2.28 pp, 95% CI [-0.25, +4.81].
- llada22, V2-CTI − All-kept FA4: +1.01 pp, 95% CI [-1.77, +3.80].
- llada22, V2-CTI − V31 K512: -2.78 pp, 95% CI [-5.06, -0.51].
- trado, V31 K512 − Native dense: +3.33 pp, 95% CI [-1.11, +7.78].
- trado, V31 K512 − All-kept FA4: +1.67 pp, 95% CI [-2.78, +6.11].
- trado, V3a K512 − Native dense: +5.56 pp, 95% CI [+1.11, +10.00].
- trado, V3a K512 − All-kept FA4: +3.89 pp, 95% CI [-0.56, +8.89].
- trado, V3a K512 − V31 K512: +2.22 pp, 95% CI [-0.56, +5.56].
- trado, V2-CTI − Native dense: +5.00 pp, 95% CI [+1.11, +9.44].
- trado, V2-CTI − All-kept FA4: +3.33 pp, 95% CI [-1.11, +7.78].
- trado, V2-CTI − V31 K512: +1.67 pp, 95% CI [-0.56, +4.44].

Single-seed exploratory comparison; these intervals do not estimate seed variability or establish noninferiority. Official word-length groups do not guarantee token fit; truncated and untruncated, difficulty and domain slices are retained in slices.json. Prompts, gold, token IDs and completions stay private.
