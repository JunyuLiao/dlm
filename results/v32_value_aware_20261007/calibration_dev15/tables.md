# v32 aggregate tables

| arm | cells | acc | N | C | N/C | T | S/N | prompt tok | GLOBAL sparsity | LOCAL sparsity | overall | wall s | decode s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_full_fix51994 | 15 | - | 4,940 | 333 | 14.835 | 5556.5 | 0.03110 | 72424 | - | - | - | 13.1321 | 10.2406 |
| dense_piecewise | 15 | - | 4,940 | 333 | 14.835 | 5556.5 | 0.03196 | 72424 | - | - | - | 13.4192 | 10.5242 |
| allkept_fa4 | 15 | - | 4,863 | 293 | 16.597 | 4859.1 | 0.03259 | 72424 | 0.0000 | - | 0.0000 | 13.4610 | 10.5653 |
| current_v31_control | 15 | - | 4,457 | 242 | 18.417 | 3988.7 | 0.02443 | 72424 | 0.8760 | - | 0.8760 | 10.1487 | 7.2595 |
| value_v1_t0.005 | 15 | - | 3,541 | 224 | 15.808 | 3702.6 | 0.34447 | 72424 | 0.0651 | - | 0.0651 | 84.1884 | 81.3186 |
| value_v2_t0.005 | 15 | - | 3,420 | 223 | 15.336 | 3685.6 | 0.04438 | 72424 | 0.0571 | - | 0.0571 | 13.0039 | 10.1179 |
| value_v1_t0.01 | 15 | - | 3,850 | 243 | 15.844 | 4033.4 | 0.35740 | 72424 | 0.0638 | - | 0.0638 | 94.6014 | 91.7315 |
| value_v2_t0.01 | 15 | - | 3,795 | 232 | 16.358 | 3815.0 | 0.04400 | 72424 | 0.0673 | - | 0.0673 | 14.0196 | 11.1322 |
| value_v1_t0.02 | 15 | - | 3,620 | 232 | 15.603 | 3841.0 | 0.34558 | 72424 | 0.0591 | - | 0.0591 | 86.2705 | 83.3999 |
| value_v2_t0.02 | 15 | - | 3,948 | 240 | 16.450 | 3966.3 | 0.04430 | 72424 | 0.0684 | - | 0.0684 | 14.5464 | 11.6584 |
| value_v1_t0.05 | 15 | - | 4,441 | 225 | 19.738 | 3724.4 | 0.27034 | 72424 | 0.0916 | - | 0.0916 | 82.9125 | 80.0378 |
| value_v2_t0.05 | 15 | - | 4,233 | 256 | 16.535 | 4247.4 | 0.04414 | 72424 | 0.0676 | - | 0.0676 | 15.3462 | 12.4574 |
| value_v3a | 15 | - | 3,831 | 232 | 16.513 | 3835.3 | 0.04460 | 72424 | 0.0672 | - | 0.0672 | 14.2761 | 11.3910 |
| value_v3b | 15 | - | 4,488 | 241 | 18.622 | 3980.1 | 0.03143 | 72424 | 0.8776 | - | 0.8776 | 12.2894 | 9.4026 |
| value_v3b_drop | 15 | - | 7,887 | 264 | 29.875 | 4393.6 | 0.04681 | 72424 | 0.1001 | - | 0.1001 | 27.5021 | 24.6145 |
| value_v3b_short | 15 | - | 3,824 | 232 | 16.483 | 3825.1 | 0.04804 | 72424 | 0.0672 | - | 0.0672 | 15.1328 | 12.2479 |

## selection counters and the sketch-space objective

| arm | selector | candidates | evaluations | forced keep | forced skip | threshold keeps | blocked | objective_max mean | retained mass mean |
|---|---|---|---|---|---|---|---|---|---|
| dense_full_fix51994 | None | - | - | 0 | 0 | 0 | - | - | - |
| dense_piecewise | None | - | - | 0 | 0 | 0 | - | - | - |
| allkept_fa4 | None | - | - | 0 | 0 | 0 | - | - | - |
| current_v31_control | None | - | - | 0 | 0 | 0 | - | - | - |
| value_v1_t0.005 | v1 | 3034730.0000 | 0.0000 | 504002 | 36480 | 31518 | 0 | - | - |
| value_v2_t0.005 | v2 | 3004630.0000 | 0.0000 | 519918 | 36352 | 15602 | 0 | 0.014956 | 0.9909 |
| value_v1_t0.01 | v1 | 3425870.0000 | 0.0000 | 515357 | 39040 | 20035 | 0 | - | - |
| value_v2_t0.01 | v2 | 3189350.0000 | 0.0000 | 527676 | 37504 | 7844 | 0 | 0.016805 | 0.9863 |
| value_v1_t0.02 | v1 | 3104950.0000 | 0.0000 | 527487 | 37504 | 8033 | 0 | - | - |
| value_v2_t0.02 | v2 | 3354530.0000 | 0.0000 | 532376 | 38528 | 3144 | 0 | 0.024624 | 0.9844 |
| value_v1_t0.05 | v1 | 2885330.0000 | 0.0000 | 533223 | 36608 | 2297 | 0 | - | - |
| value_v2_t0.05 | v2 | 3591010.0000 | 0.0000 | 535592 | 39552 | 952 | 0 | 0.027430 | 0.9830 |
| value_v3a | v3a | 3187135.0000 | 101988320.0000 | 0 | 0 | 0 | 0 | 0.006028 | 0.9965 |
| value_v3b | v3b | 0.0000 | 0.0000 | 0 | 0 | 0 | 2405 | - | - |
| value_v3b_drop | v3b_drop | 97125600.0000 | 131859394560.0000 | 0 | 40416 | 0 | 0 | 0.363995 | 0.8819 |
| value_v3b_short | v3b_shortlist | 94013280.0000 | 0.0000 | 0 | 36704 | 0 | 0 | 0.006167 | 0.9961 |

## physical tile accounting (with denominators)

Substrate pins are asserted per arm; see `substrate` in aggregate.json.

| arm | GLOBAL eligible | GLOBAL kept | LOCAL eligible | LOCAL kept | overall denominator | overall sparsity |
|---|---|---|---|---|---|---|
| dense_full_fix51994 | - | - | - | - | - | - |
| dense_piecewise | - | - | - | - | - | - |
| allkept_fa4 | 1096254560 | 1096254560 | - | - | 1096254560 | 0.0000 |
| current_v31_control | 990841760 | 122874080 | - | - | 990841760 | 0.8760 |
| value_v1_t0.005 | 819006400 | 765707260 | - | - | 819006400 | 0.0651 |
| value_v2_t0.005 | 770397120 | 726381120 | - | - | 770397120 | 0.0571 |
| value_v1_t0.01 | 898306880 | 840972480 | - | - | 898306880 | 0.0638 |
| value_v2_t0.01 | 890660160 | 830724800 | - | - | 890660160 | 0.0673 |
| value_v1_t0.02 | 830975680 | 781851840 | - | - | 830975680 | 0.0591 |
| value_v2_t0.02 | 937805120 | 873616960 | - | - | 937805120 | 0.0684 |
| value_v1_t0.05 | 1109495360 | 1007892160 | - | - | 1109495360 | 0.0916 |
| value_v2_t0.05 | 1007666880 | 939561280 | - | - | 1007666880 | 0.0676 |
| value_v3a | 902755520 | 842051520 | - | - | 902755520 | 0.0672 |
| value_v3b | 1004552320 | 122932000 | - | - | 1004552320 | 0.8776 |
| value_v3b_drop | 2182901440 | 1964351360 | - | - | 2182901440 | 0.1001 |
| value_v3b_short | 901384800 | 840806240 | - | - | 901384800 | 0.0672 |

## paired differences (ref minus arm, common cells)

| pair | metric | n | mean | median | wins/losses |
|---|---|---|---|---|---|
| dense_full_fix51994_minus_dense_piecewise | denoise_forwards | 15 | 0.00000 | 0.00000 | 0/0 |
| dense_full_fix51994_minus_dense_piecewise | output_tokens | 15 | 0.00000 | 0.00000 | 0/0 |
| dense_full_fix51994_minus_dense_piecewise | decode_s | 15 | -0.28363 | -0.18150 | 0/15 |
| dense_full_fix51994_minus_dense_piecewise | wall_s | 15 | -0.28714 | -0.18595 | 0/15 |
| dense_full_fix51994_minus_dense_piecewise | prefill_s | 15 | -0.00353 | -0.00019 | 6/9 |
| dense_full_fix51994_minus_allkept_fa4 | denoise_forwards | 15 | 5.13333 | 10.00000 | 9/6 |
| dense_full_fix51994_minus_allkept_fa4 | output_tokens | 15 | 697.40000 | 191.00000 | 9/5 |
| dense_full_fix51994_minus_allkept_fa4 | decode_s | 15 | -0.32475 | 0.16319 | 8/7 |
| dense_full_fix51994_minus_allkept_fa4 | wall_s | 15 | -0.32891 | 0.16015 | 8/7 |
| dense_full_fix51994_minus_allkept_fa4 | prefill_s | 15 | -0.00410 | -0.00107 | 5/10 |
| dense_full_fix51994_minus_current_v31_control | denoise_forwards | 15 | 32.20000 | 22.00000 | 10/5 |
| dense_full_fix51994_minus_current_v31_control | output_tokens | 15 | 1567.86667 | 409.00000 | 10/5 |
| dense_full_fix51994_minus_current_v31_control | decode_s | 15 | 2.98101 | 1.51762 | 11/4 |
| dense_full_fix51994_minus_current_v31_control | wall_s | 15 | 2.98339 | 1.51710 | 11/4 |
| dense_full_fix51994_minus_current_v31_control | prefill_s | 15 | 0.00227 | 0.00272 | 10/5 |
| dense_full_fix51994_minus_value_v1_t0.005 | denoise_forwards | 15 | 93.26667 | 25.00000 | 11/4 |
| dense_full_fix51994_minus_value_v1_t0.005 | output_tokens | 15 | 1853.93333 | 268.00000 | 12/2 |
| dense_full_fix51994_minus_value_v1_t0.005 | decode_s | 15 | -71.07800 | -34.33043 | 0/15 |
| dense_full_fix51994_minus_value_v1_t0.005 | wall_s | 15 | -71.05628 | -34.29930 | 0/15 |
| dense_full_fix51994_minus_value_v1_t0.005 | prefill_s | 15 | 0.02148 | 0.02135 | 15/0 |
| dense_full_fix51994_minus_value_v2_t0.005 | denoise_forwards | 15 | 101.33333 | 57.00000 | 10/5 |
| dense_full_fix51994_minus_value_v2_t0.005 | output_tokens | 15 | 1870.93333 | 315.00000 | 10/4 |
| dense_full_fix51994_minus_value_v2_t0.005 | decode_s | 15 | 0.12269 | 0.28839 | 8/7 |
| dense_full_fix51994_minus_value_v2_t0.005 | wall_s | 15 | 0.12817 | 0.28076 | 8/7 |
| dense_full_fix51994_minus_value_v2_t0.005 | prefill_s | 15 | 0.00514 | 0.00508 | 12/3 |
| dense_full_fix51994_minus_value_v1_t0.01 | denoise_forwards | 15 | 72.66667 | 7.00000 | 9/6 |
| dense_full_fix51994_minus_value_v1_t0.01 | output_tokens | 15 | 1523.13333 | 244.00000 | 9/5 |
| dense_full_fix51994_minus_value_v1_t0.01 | decode_s | 15 | -81.49091 | -34.58481 | 0/15 |
| dense_full_fix51994_minus_value_v1_t0.01 | wall_s | 15 | -81.46928 | -34.55724 | 0/15 |
| dense_full_fix51994_minus_value_v1_t0.01 | prefill_s | 15 | 0.02143 | 0.02012 | 15/0 |
| dense_full_fix51994_minus_value_v2_t0.01 | denoise_forwards | 15 | 76.33333 | 57.00000 | 10/5 |
| dense_full_fix51994_minus_value_v2_t0.01 | output_tokens | 15 | 1741.53333 | 278.00000 | 10/4 |
| dense_full_fix51994_minus_value_v2_t0.01 | decode_s | 15 | -0.89161 | 0.39221 | 8/7 |
| dense_full_fix51994_minus_value_v2_t0.01 | wall_s | 15 | -0.88753 | 0.39315 | 8/7 |
| dense_full_fix51994_minus_value_v2_t0.01 | prefill_s | 15 | 0.00384 | 0.00285 | 13/2 |
| dense_full_fix51994_minus_value_v1_t0.02 | denoise_forwards | 15 | 88.00000 | 7.00000 | 9/6 |
| dense_full_fix51994_minus_value_v1_t0.02 | output_tokens | 15 | 1715.53333 | 172.00000 | 9/5 |
| dense_full_fix51994_minus_value_v1_t0.02 | decode_s | 15 | -73.15932 | -35.05122 | 0/15 |
| dense_full_fix51994_minus_value_v1_t0.02 | wall_s | 15 | -73.13843 | -35.01603 | 0/15 |
| dense_full_fix51994_minus_value_v1_t0.02 | prefill_s | 15 | 0.02074 | 0.01878 | 15/0 |
| dense_full_fix51994_minus_value_v2_t0.02 | denoise_forwards | 15 | 66.13333 | 57.00000 | 10/5 |
| dense_full_fix51994_minus_value_v2_t0.02 | output_tokens | 15 | 1590.26667 | 244.00000 | 10/4 |
| dense_full_fix51994_minus_value_v2_t0.02 | decode_s | 15 | -1.41780 | 0.42693 | 8/7 |
| dense_full_fix51994_minus_value_v2_t0.02 | wall_s | 15 | -1.41429 | 0.43077 | 8/7 |
| dense_full_fix51994_minus_value_v2_t0.02 | prefill_s | 15 | 0.00331 | 0.00457 | 12/3 |
| dense_full_fix51994_minus_value_v1_t0.05 | denoise_forwards | 15 | 33.26667 | 40.00000 | 11/4 |
| dense_full_fix51994_minus_value_v1_t0.05 | output_tokens | 15 | 1832.13333 | 244.00000 | 11/3 |
| dense_full_fix51994_minus_value_v1_t0.05 | decode_s | 15 | -69.79721 | -34.22372 | 0/15 |
| dense_full_fix51994_minus_value_v1_t0.05 | wall_s | 15 | -69.78040 | -34.19783 | 0/15 |
| dense_full_fix51994_minus_value_v1_t0.05 | prefill_s | 15 | 0.01682 | 0.01707 | 15/0 |
| dense_full_fix51994_minus_value_v2_t0.05 | denoise_forwards | 15 | 47.13333 | 7.00000 | 9/6 |
| dense_full_fix51994_minus_value_v2_t0.05 | output_tokens | 15 | 1309.13333 | 244.00000 | 9/5 |
| dense_full_fix51994_minus_value_v2_t0.05 | decode_s | 15 | -2.21686 | -0.19581 | 7/8 |
| dense_full_fix51994_minus_value_v2_t0.05 | wall_s | 15 | -2.21411 | -0.19005 | 7/8 |
| dense_full_fix51994_minus_value_v2_t0.05 | prefill_s | 15 | 0.00264 | 0.00291 | 13/2 |
| dense_full_fix51994_minus_value_v3a | denoise_forwards | 15 | 73.93333 | 57.00000 | 10/5 |
| dense_full_fix51994_minus_value_v3a | output_tokens | 15 | 1721.26667 | 244.00000 | 11/3 |
| dense_full_fix51994_minus_value_v3a | decode_s | 15 | -1.15041 | -0.20169 | 7/8 |
| dense_full_fix51994_minus_value_v3a | wall_s | 15 | -1.14403 | -0.19766 | 7/8 |
| dense_full_fix51994_minus_value_v3a | prefill_s | 15 | 0.00638 | 0.00393 | 13/2 |
| dense_full_fix51994_minus_value_v3b | denoise_forwards | 15 | 30.13333 | 12.00000 | 9/6 |
| dense_full_fix51994_minus_value_v3b | output_tokens | 15 | 1576.46667 | 409.00000 | 10/5 |
| dense_full_fix51994_minus_value_v3b | decode_s | 15 | 0.83800 | 0.61207 | 9/6 |
| dense_full_fix51994_minus_value_v3b | wall_s | 15 | 0.84271 | 0.61314 | 9/6 |
| dense_full_fix51994_minus_value_v3b | prefill_s | 15 | 0.00456 | 0.00439 | 14/1 |
| dense_full_fix51994_minus_value_v3b_drop | denoise_forwards | 15 | -196.46667 | 5.00000 | 8/7 |
| dense_full_fix51994_minus_value_v3b_drop | output_tokens | 15 | 1162.93333 | 244.00000 | 9/6 |
| dense_full_fix51994_minus_value_v3b_drop | decode_s | 15 | -14.37394 | -0.34767 | 6/9 |
| dense_full_fix51994_minus_value_v3b_drop | wall_s | 15 | -14.37003 | -0.34562 | 6/9 |
| dense_full_fix51994_minus_value_v3b_drop | prefill_s | 15 | 0.00419 | 0.00215 | 9/5 |
| dense_full_fix51994_minus_value_v3b_short | denoise_forwards | 15 | 74.40000 | 57.00000 | 10/5 |
| dense_full_fix51994_minus_value_v3b_short | output_tokens | 15 | 1731.40000 | 244.00000 | 11/3 |
| dense_full_fix51994_minus_value_v3b_short | decode_s | 15 | -2.00732 | -0.16420 | 6/9 |
| dense_full_fix51994_minus_value_v3b_short | wall_s | 15 | -2.00075 | -0.16081 | 6/9 |
| dense_full_fix51994_minus_value_v3b_short | prefill_s | 15 | 0.00653 | 0.00478 | 13/2 |
