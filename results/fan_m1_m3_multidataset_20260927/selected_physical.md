# Selected001 physical work and phase evidence

140 qualified arm/canvas-sequence/boundary rows; 32 missing-sequence rows (RULER N16). RULER requested calls 0 and 2 alias the same N4 replay beginning at call0 and are counted once. The two boundaries replay the same captured state and are not independent requests.

Whole/static-prefix/current-canvas and LOCAL/GLOBAL legal-pair counts are in the JSON and CSV. GLOBAL_ONLY_NATIVE_LOCAL counts cover five routed GLOBAL layers, not all 30 layers. Operator errors and prepared-consumer timings cover only observed layer 0/5 QKV and support; they exclude route, projection, model forward and generation.

- ruler_selected_001 dllm: profile `216b28db57caa7d143d6a947e895c86997be11f2162f686010c9a65f7c76b032`, GPU `GPU-fc12ad5c-5334-5509-8fc6-465498fd3915`, source commits 00a2c4d3c93a7f685a4f75fba8ea47991fa5c383.
  - `v20_counter.py` `6812f1e103837ab7a99aff2479b584affb094f56f7127aeb602f1c239b2e3fea`
  - `replay_harness.py` `60b8c09e98636b0cdd449dbdb4fb54f1cfd834b0e9648e068ac42b0c0881e879`
  - `v20_operator_probe.py` `89d8344ee40b41bb50996e55f15f2ccdde1a706e2de1fe094785a80fdb815f78`
  - `v20_profile.py` `974f9dadf74180b178126b68535a9b7917a457e2a9e9a403b0a0360b8059980c`
  - `v9_step_replay_profile.py` `31266adc0cfee09e865cb4b347b2f5e15eaf3c1e4f1d12e0b125ee28a57afc13`
- ruler_selected_001 mpk: profile `522b73aaed26c7e57a83c0e2c09d067c7304175765533e54d45cddf13e8affa3`, GPU `GPU-6139046a-b005-8fe5-a837-f8270472ab72`, source commits 00a2c4d3c93a7f685a4f75fba8ea47991fa5c383.
  - `v20_counter.py` `6812f1e103837ab7a99aff2479b584affb094f56f7127aeb602f1c239b2e3fea`
  - `replay_harness.py` `60b8c09e98636b0cdd449dbdb4fb54f1cfd834b0e9648e068ac42b0c0881e879`
  - `v20_operator_probe.py` `89d8344ee40b41bb50996e55f15f2ccdde1a706e2de1fe094785a80fdb815f78`
  - `v20_profile.py` `974f9dadf74180b178126b68535a9b7917a457e2a9e9a403b0a0360b8059980c`
  - `v9_step_replay_profile.py` `31266adc0cfee09e865cb4b347b2f5e15eaf3c1e4f1d12e0b125ee28a57afc13`
- selected_001 dllm: profile `3bb85aa1584de9e07c465d97c45b2588b06abd184b0dfe7858e2d27b481268f5`, GPU `GPU-fc12ad5c-5334-5509-8fc6-465498fd3915`, source commits 00a2c4d3c93a7f685a4f75fba8ea47991fa5c383.
  - `v20_counter.py` `6812f1e103837ab7a99aff2479b584affb094f56f7127aeb602f1c239b2e3fea`
  - `replay_harness.py` `60b8c09e98636b0cdd449dbdb4fb54f1cfd834b0e9648e068ac42b0c0881e879`
  - `v20_operator_probe.py` `89d8344ee40b41bb50996e55f15f2ccdde1a706e2de1fe094785a80fdb815f78`
  - `v20_profile.py` `974f9dadf74180b178126b68535a9b7917a457e2a9e9a403b0a0360b8059980c`
  - `v9_step_replay_profile.py` `31266adc0cfee09e865cb4b347b2f5e15eaf3c1e4f1d12e0b125ee28a57afc13`
- selected_001 mpk: profile `b299725104289f06495e56fa13a4a678e3cd87d0b71f1e3b1b78ccc68eeed9d8`, GPU `GPU-6139046a-b005-8fe5-a837-f8270472ab72`, source commits 00a2c4d3c93a7f685a4f75fba8ea47991fa5c383.
  - `v20_counter.py` `6812f1e103837ab7a99aff2479b584affb094f56f7127aeb602f1c239b2e3fea`
  - `replay_harness.py` `60b8c09e98636b0cdd449dbdb4fb54f1cfd834b0e9648e068ac42b0c0881e879`
  - `v20_operator_probe.py` `89d8344ee40b41bb50996e55f15f2ccdde1a706e2de1fe094785a80fdb815f78`
  - `v20_profile.py` `974f9dadf74180b178126b68535a9b7917a457e2a9e9a403b0a0360b8059980c`
  - `v9_step_replay_profile.py` `31266adc0cfee09e865cb4b347b2f5e15eaf3c1e4f1d12e0b125ee28a57afc13`

| Stage | Host | Dataset | Canvas/window | Boundary/N reached | Arm | Scope | A/D/H layers | Numeric age | Decision age | Whole legal QK/PV skipped | Operator only L0/L5 max abs error, prepared consumer ms | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ruler_selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | model_forward/N4 4/4 | D_native | NATIVE | 0/0/0 | N/A | N/A | N/A/N/A | N/A | unavailable |
| ruler_selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | model_forward/N4 4/4 | G75L30_nativeQ128 | ALL_NATIVE_LEGAL | 60/0/60 | 0..2 | 0..2 | 20.1%/30.1% | L0:e=0.03917,t=0.17..0.175ms/L5:e=0.07146,t=0.323..0.323ms | qualified |
| ruler_selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | model_forward/N4 4/4 | D_matched | GLOBAL_ONLY_NATIVE_LOCAL | 20/0/0 | N/A | N/A | 0.0%/0.0% | N/A | qualified |
| ruler_selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | model_forward/N4 4/4 | T_scope | GLOBAL_ONLY_NATIVE_LOCAL | 20/0/0 | 0..0 | 0..0 | 0.0%/59.0% | N/A | qualified |
| ruler_selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | model_forward/N4 4/4 | M1_R1_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 5/15/0 | 0..3 | 0..0 | 43.3%/58.3% | L5:e=0.07014,t=0.323..0.328ms | qualified |
| ruler_selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | model_forward/N4 4/4 | M3_R2_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 5/5/10 | 0..3 | 0..1 | 43.1%/58.1% | L5:e=0.07502,t=0.32..0.351ms | qualified |
| ruler_selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | model_forward/N4 4/4 | M3_R3_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 5/5/10 | 0..3 | 0..2 | 44.2%/59.2% | L5:e=0.07014,t=0.321..0.341ms | qualified |
| ruler_selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | model_forward/N4 4/4 | B_A8_matched | GLOBAL_ONLY_NATIVE_LOCAL | 5/0/15 | 0..3 | 0..3 | 45.0%/59.9% | L5:e=0.07014,t=0.322..0.326ms | qualified |
| ruler_selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | denoising_step/N4 4/4 | D_native | NATIVE | 0/0/0 | N/A | N/A | N/A/N/A | N/A | unavailable |
| ruler_selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | denoising_step/N4 4/4 | G75L30_nativeQ128 | ALL_NATIVE_LEGAL | 60/0/60 | 0..2 | 0..2 | 20.1%/30.1% | L0:e=0.03917,t=0.172..0.173ms/L5:e=0.07146,t=0.317..0.324ms | qualified |
| ruler_selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | denoising_step/N4 4/4 | D_matched | GLOBAL_ONLY_NATIVE_LOCAL | 20/0/0 | N/A | N/A | 0.0%/0.0% | N/A | qualified |
| ruler_selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | denoising_step/N4 4/4 | T_scope | GLOBAL_ONLY_NATIVE_LOCAL | 20/0/0 | 0..0 | 0..0 | 0.0%/59.0% | N/A | qualified |
| ruler_selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | denoising_step/N4 4/4 | M1_R1_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 5/15/0 | 0..3 | 0..0 | 43.3%/58.3% | L5:e=0.07014,t=0.316..0.325ms | qualified |
| ruler_selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | denoising_step/N4 4/4 | M3_R2_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 5/5/10 | 0..3 | 0..1 | 43.1%/58.1% | L5:e=0.07502,t=0.319..0.353ms | qualified |
| ruler_selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | denoising_step/N4 4/4 | M3_R3_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 5/5/10 | 0..3 | 0..2 | 44.2%/59.2% | L5:e=0.07014,t=0.317..0.342ms | qualified |
| ruler_selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | denoising_step/N4 4/4 | B_A8_matched | GLOBAL_ONLY_NATIVE_LOCAL | 5/0/15 | 0..3 | 0..3 | 45.0%/59.9% | L5:e=0.07014,t=0.321..0.324ms | qualified |
| ruler_selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | model_forward/N4 4/4 | D_native | NATIVE | 0/0/0 | N/A | N/A | N/A/N/A | N/A | unavailable |
| ruler_selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | model_forward/N4 4/4 | G75L30_nativeQ128 | ALL_NATIVE_LEGAL | 60/0/60 | 0..2 | 0..2 | 19.6%/29.4% | L0:e=0.03932,t=0.18..0.185ms/L5:e=0.0806,t=0.342..0.349ms | qualified |
| ruler_selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | model_forward/N4 4/4 | D_matched | GLOBAL_ONLY_NATIVE_LOCAL | 20/0/0 | N/A | N/A | 0.0%/0.0% | N/A | qualified |
| ruler_selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | model_forward/N4 4/4 | T_scope | GLOBAL_ONLY_NATIVE_LOCAL | 20/0/0 | 0..0 | 0..0 | 0.0%/50.0% | N/A | qualified |
| ruler_selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | model_forward/N4 4/4 | M1_R1_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 5/15/0 | 0..3 | 0..0 | 31.7%/43.2% | L5:e=0.073,t=0.475..0.476ms | qualified |
| ruler_selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | model_forward/N4 4/4 | M3_R2_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 5/5/10 | 0..3 | 0..1 | 31.1%/42.7% | L5:e=0.07665,t=0.475..0.544ms | qualified |
| ruler_selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | model_forward/N4 4/4 | M3_R3_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 5/5/10 | 0..3 | 0..2 | 33.4%/44.9% | L5:e=0.07764,t=0.476..0.541ms | qualified |
| ruler_selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | model_forward/N4 4/4 | B_A8_matched | GLOBAL_ONLY_NATIVE_LOCAL | 5/0/15 | 0..3 | 0..3 | 34.6%/46.1% | L5:e=0.073,t=0.477..0.479ms | qualified |
| ruler_selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | denoising_step/N4 4/4 | D_native | NATIVE | 0/0/0 | N/A | N/A | N/A/N/A | N/A | unavailable |
| ruler_selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | denoising_step/N4 4/4 | G75L30_nativeQ128 | ALL_NATIVE_LEGAL | 60/0/60 | 0..2 | 0..2 | 19.6%/29.4% | L0:e=0.03932,t=0.183..0.184ms/L5:e=0.0806,t=0.345..0.345ms | qualified |
| ruler_selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | denoising_step/N4 4/4 | D_matched | GLOBAL_ONLY_NATIVE_LOCAL | 20/0/0 | N/A | N/A | 0.0%/0.0% | N/A | qualified |
| ruler_selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | denoising_step/N4 4/4 | T_scope | GLOBAL_ONLY_NATIVE_LOCAL | 20/0/0 | 0..0 | 0..0 | 0.0%/50.0% | N/A | qualified |
| ruler_selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | denoising_step/N4 4/4 | M1_R1_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 5/15/0 | 0..3 | 0..0 | 31.7%/43.2% | L5:e=0.073,t=0.48..0.482ms | qualified |
| ruler_selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | denoising_step/N4 4/4 | M3_R2_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 5/5/10 | 0..3 | 0..1 | 31.1%/42.7% | L5:e=0.07665,t=0.479..0.547ms | qualified |
| ruler_selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | denoising_step/N4 4/4 | M3_R3_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 5/5/10 | 0..3 | 0..2 | 33.4%/44.9% | L5:e=0.07764,t=0.483..0.547ms | qualified |
| ruler_selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | denoising_step/N4 4/4 | B_A8_matched | GLOBAL_ONLY_NATIVE_LOCAL | 5/0/15 | 0..3 | 0..3 | 34.6%/46.1% | L5:e=0.073,t=0.475..0.48ms | qualified |
| selected_001 | dllm | aime26 | 0/6724e8 (requested 0) | model_forward/N16 8/16 | D_native | NATIVE | 0/0/0 | N/A | N/A | N/A/N/A | N/A | unavailable (native_shortened) |
| selected_001 | dllm | aime26 | 0/6724e8 (requested 0) | model_forward/N16 8/16 | G75L30_nativeQ128 | ALL_NATIVE_LEGAL | 60/0/180 | 0..6 | 0..6 | 0.0%/0.0% | L0:e=0.03927,t=0.113..0.115ms/L5:e=0.07416,t=0.146..0.156ms | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 0/6724e8 (requested 0) | model_forward/N16 8/16 | D_matched | GLOBAL_ONLY_NATIVE_LOCAL | 40/0/0 | N/A | N/A | 0.0%/0.0% | N/A | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 0/6724e8 (requested 0) | model_forward/N16 8/16 | T_scope | GLOBAL_ONLY_NATIVE_LOCAL | 40/0/0 | 0..0 | 0..0 | 0.0%/1.4% | N/A | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 0/6724e8 (requested 0) | model_forward/N16 8/16 | M1_R1_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 5/35/0 | 0..7 | 0..0 | 1.6%/2.1% | L5:e=0.06748,t=0.15..0.152ms | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 0/6724e8 (requested 0) | model_forward/N16 8/16 | M3_R2_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 5/15/20 | 0..7 | 0..1 | 1.6%/2.0% | L5:e=0.09812,t=0.148..0.152ms | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 0/6724e8 (requested 0) | model_forward/N16 8/16 | M3_R3_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 5/10/25 | 0..7 | 0..2 | 2.0%/2.4% | L5:e=0.06748,t=0.147..0.149ms | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 0/6724e8 (requested 0) | model_forward/N16 8/16 | B_A8_matched | GLOBAL_ONLY_NATIVE_LOCAL | 5/0/35 | 0..7 | 0..7 | 3.0%/3.5% | L5:e=0.06748,t=0.148..0.149ms | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 0/6724e8 (requested 0) | denoising_step/N16 8/16 | D_native | NATIVE | 0/0/0 | N/A | N/A | N/A/N/A | N/A | unavailable (native_shortened) |
| selected_001 | dllm | aime26 | 0/6724e8 (requested 0) | denoising_step/N16 8/16 | G75L30_nativeQ128 | ALL_NATIVE_LEGAL | 60/0/180 | 0..6 | 0..6 | 0.0%/0.0% | L0:e=0.03927,t=0.12..0.121ms/L5:e=0.07416,t=0.149..0.149ms | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 0/6724e8 (requested 0) | denoising_step/N16 8/16 | D_matched | GLOBAL_ONLY_NATIVE_LOCAL | 40/0/0 | N/A | N/A | 0.0%/0.0% | N/A | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 0/6724e8 (requested 0) | denoising_step/N16 8/16 | T_scope | GLOBAL_ONLY_NATIVE_LOCAL | 40/0/0 | 0..0 | 0..0 | 0.0%/1.4% | N/A | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 0/6724e8 (requested 0) | denoising_step/N16 8/16 | M1_R1_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 5/35/0 | 0..7 | 0..0 | 1.6%/2.1% | L5:e=0.06748,t=0.147..0.15ms | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 0/6724e8 (requested 0) | denoising_step/N16 8/16 | M3_R2_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 5/15/20 | 0..7 | 0..1 | 1.6%/2.0% | L5:e=0.09812,t=0.146..0.148ms | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 0/6724e8 (requested 0) | denoising_step/N16 8/16 | M3_R3_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 5/10/25 | 0..7 | 0..2 | 2.0%/2.4% | L5:e=0.06748,t=0.148..0.151ms | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 0/6724e8 (requested 0) | denoising_step/N16 8/16 | B_A8_matched | GLOBAL_ONLY_NATIVE_LOCAL | 5/0/35 | 0..7 | 0..7 | 3.0%/3.5% | L5:e=0.06748,t=0.146..0.151ms | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 8/a0a615 (requested 2) | model_forward/N16 7/16 | D_native | NATIVE | 0/0/0 | N/A | N/A | N/A/N/A | N/A | unavailable (native_shortened) |
| selected_001 | dllm | aime26 | 8/a0a615 (requested 2) | model_forward/N16 7/16 | G75L30_nativeQ128 | ALL_NATIVE_LEGAL | 60/0/150 | 0..5 | 0..5 | 22.1%/26.5% | L0:e=0.03832,t=0.172..0.173ms/L5:e=0.07074,t=0.289..0.291ms | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 8/a0a615 (requested 2) | model_forward/N16 7/16 | D_matched | GLOBAL_ONLY_NATIVE_LOCAL | 35/0/0 | N/A | N/A | 0.0%/0.0% | N/A | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 8/a0a615 (requested 2) | model_forward/N16 7/16 | T_scope | GLOBAL_ONLY_NATIVE_LOCAL | 35/0/0 | 0..0 | 0..0 | 0.0%/20.2% | N/A | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 8/a0a615 (requested 2) | model_forward/N16 7/16 | M1_R1_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 5/30/0 | 0..6 | 0..0 | 21.1%/25.8% | L5:e=0.07104,t=0.339..0.349ms | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 8/a0a615 (requested 2) | model_forward/N16 7/16 | M3_R2_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 5/15/15 | 0..6 | 0..1 | 21.4%/26.1% | L5:e=0.07104,t=0.339..0.432ms | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 8/a0a615 (requested 2) | model_forward/N16 7/16 | M3_R3_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 5/10/20 | 0..6 | 0..2 | 21.9%/26.6% | L5:e=0.09182,t=0.347..0.464ms | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 8/a0a615 (requested 2) | model_forward/N16 7/16 | B_A8_matched | GLOBAL_ONLY_NATIVE_LOCAL | 5/0/30 | 0..6 | 0..6 | 28.2%/32.9% | L5:e=0.07104,t=0.342..0.347ms | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 8/a0a615 (requested 2) | denoising_step/N16 7/16 | D_native | NATIVE | 0/0/0 | N/A | N/A | N/A/N/A | N/A | unavailable (native_shortened) |
| selected_001 | dllm | aime26 | 8/a0a615 (requested 2) | denoising_step/N16 7/16 | G75L30_nativeQ128 | ALL_NATIVE_LEGAL | 60/0/150 | 0..5 | 0..5 | 22.1%/26.5% | L0:e=0.03832,t=0.173..0.182ms/L5:e=0.07074,t=0.288..0.291ms | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 8/a0a615 (requested 2) | denoising_step/N16 7/16 | D_matched | GLOBAL_ONLY_NATIVE_LOCAL | 35/0/0 | N/A | N/A | 0.0%/0.0% | N/A | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 8/a0a615 (requested 2) | denoising_step/N16 7/16 | T_scope | GLOBAL_ONLY_NATIVE_LOCAL | 35/0/0 | 0..0 | 0..0 | 0.0%/20.2% | N/A | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 8/a0a615 (requested 2) | denoising_step/N16 7/16 | M1_R1_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 5/30/0 | 0..6 | 0..0 | 21.1%/25.8% | L5:e=0.07104,t=0.339..0.349ms | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 8/a0a615 (requested 2) | denoising_step/N16 7/16 | M3_R2_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 5/15/15 | 0..6 | 0..1 | 21.4%/26.1% | L5:e=0.07104,t=0.342..0.433ms | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 8/a0a615 (requested 2) | denoising_step/N16 7/16 | M3_R3_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 5/10/20 | 0..6 | 0..2 | 21.9%/26.6% | L5:e=0.09182,t=0.342..0.465ms | qualified (native_shortened) |
| selected_001 | dllm | aime26 | 8/a0a615 (requested 2) | denoising_step/N16 7/16 | B_A8_matched | GLOBAL_ONLY_NATIVE_LOCAL | 5/0/30 | 0..6 | 0..6 | 28.2%/32.9% | L5:e=0.07104,t=0.347..0.348ms | qualified (native_shortened) |
| selected_001 | dllm | longbench_v2 | 0/c1a48b (requested 0) | model_forward/N16 16/16 | D_native | NATIVE | 0/0/0 | N/A | N/A | N/A/N/A | N/A | unavailable |
| selected_001 | dllm | longbench_v2 | 0/c1a48b (requested 0) | model_forward/N16 16/16 | G75L30_nativeQ128 | ALL_NATIVE_LEGAL | 60/0/420 | 0..14 | 0..14 | 52.7%/56.4% | L0:e=0.04621,t=0.171..0.173ms/L5:e=0.0715,t=1.13..1.13ms | qualified |
| selected_001 | dllm | longbench_v2 | 0/c1a48b (requested 0) | model_forward/N16 16/16 | D_matched | GLOBAL_ONLY_NATIVE_LOCAL | 80/0/0 | N/A | N/A | 0.0%/0.0% | N/A | qualified |
| selected_001 | dllm | longbench_v2 | 0/c1a48b (requested 0) | model_forward/N16 16/16 | T_scope | GLOBAL_ONLY_NATIVE_LOCAL | 80/0/0 | 0..0 | 0..0 | 0.0%/70.2% | N/A | qualified |
| selected_001 | dllm | longbench_v2 | 0/c1a48b (requested 0) | model_forward/N16 16/16 | M1_R1_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/70/0 | 0..7 | 0..0 | 61.8%/71.4% | L5:e=0.06825,t=0.924..0.927ms | qualified |
| selected_001 | dllm | longbench_v2 | 0/c1a48b (requested 0) | model_forward/N16 16/16 | M3_R2_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/30/40 | 0..7 | 0..1 | 61.5%/71.1% | L5:e=0.06825,t=0.93..1.61ms | qualified |
| selected_001 | dllm | longbench_v2 | 0/c1a48b (requested 0) | model_forward/N16 16/16 | M3_R3_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/20/50 | 0..7 | 0..2 | 62.2%/71.8% | L5:e=0.06825,t=0.929..1.84ms | qualified |
| selected_001 | dllm | longbench_v2 | 0/c1a48b (requested 0) | model_forward/N16 16/16 | B_A8_matched | GLOBAL_ONLY_NATIVE_LOCAL | 10/0/70 | 0..7 | 0..7 | 67.1%/76.6% | L5:e=0.06825,t=0.923..0.935ms | qualified |
| selected_001 | dllm | longbench_v2 | 0/c1a48b (requested 0) | denoising_step/N16 16/16 | D_native | NATIVE | 0/0/0 | N/A | N/A | N/A/N/A | N/A | unavailable |
| selected_001 | dllm | longbench_v2 | 0/c1a48b (requested 0) | denoising_step/N16 16/16 | G75L30_nativeQ128 | ALL_NATIVE_LEGAL | 60/0/420 | 0..14 | 0..14 | 52.7%/56.4% | L0:e=0.04621,t=0.171..0.173ms/L5:e=0.0715,t=1.13..1.13ms | qualified |
| selected_001 | dllm | longbench_v2 | 0/c1a48b (requested 0) | denoising_step/N16 16/16 | D_matched | GLOBAL_ONLY_NATIVE_LOCAL | 80/0/0 | N/A | N/A | 0.0%/0.0% | N/A | qualified |
| selected_001 | dllm | longbench_v2 | 0/c1a48b (requested 0) | denoising_step/N16 16/16 | T_scope | GLOBAL_ONLY_NATIVE_LOCAL | 80/0/0 | 0..0 | 0..0 | 0.0%/70.2% | N/A | qualified |
| selected_001 | dllm | longbench_v2 | 0/c1a48b (requested 0) | denoising_step/N16 16/16 | M1_R1_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/70/0 | 0..7 | 0..0 | 61.8%/71.4% | L5:e=0.06825,t=0.925..0.925ms | qualified |
| selected_001 | dllm | longbench_v2 | 0/c1a48b (requested 0) | denoising_step/N16 16/16 | M3_R2_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/30/40 | 0..7 | 0..1 | 61.5%/71.1% | L5:e=0.06825,t=0.926..1.61ms | qualified |
| selected_001 | dllm | longbench_v2 | 0/c1a48b (requested 0) | denoising_step/N16 16/16 | M3_R3_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/20/50 | 0..7 | 0..2 | 62.2%/71.8% | L5:e=0.06825,t=0.92..1.84ms | qualified |
| selected_001 | dllm | longbench_v2 | 0/c1a48b (requested 0) | denoising_step/N16 16/16 | B_A8_matched | GLOBAL_ONLY_NATIVE_LOCAL | 10/0/70 | 0..7 | 0..7 | 67.1%/76.6% | L5:e=0.06825,t=0.924..0.931ms | qualified |
| selected_001 | dllm | longbench_v2 | 4/1fa38c (requested 2) | model_forward/N16 16/16 | D_native | NATIVE | 0/0/0 | N/A | N/A | N/A/N/A | N/A | unavailable |
| selected_001 | dllm | longbench_v2 | 4/1fa38c (requested 2) | model_forward/N16 16/16 | G75L30_nativeQ128 | ALL_NATIVE_LEGAL | 60/0/420 | 0..14 | 0..14 | 53.2%/57.0% | L0:e=0.03747,t=0.172..0.175ms/L5:e=0.06649,t=1.18..1.18ms | qualified |
| selected_001 | dllm | longbench_v2 | 4/1fa38c (requested 2) | model_forward/N16 16/16 | D_matched | GLOBAL_ONLY_NATIVE_LOCAL | 80/0/0 | N/A | N/A | 0.0%/0.0% | N/A | qualified |
| selected_001 | dllm | longbench_v2 | 4/1fa38c (requested 2) | model_forward/N16 16/16 | T_scope | GLOBAL_ONLY_NATIVE_LOCAL | 80/0/0 | 0..0 | 0..0 | 0.0%/53.3% | N/A | qualified |
| selected_001 | dllm | longbench_v2 | 4/1fa38c (requested 2) | model_forward/N16 16/16 | M1_R1_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/70/0 | 0..7 | 0..0 | 44.8%/52.1% | L5:e=0.06675,t=1.08..1.09ms | qualified |
| selected_001 | dllm | longbench_v2 | 4/1fa38c (requested 2) | model_forward/N16 16/16 | M3_R2_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/30/40 | 0..7 | 0..1 | 44.6%/51.9% | L5:e=0.07326,t=1.09..1.77ms | qualified |
| selected_001 | dllm | longbench_v2 | 4/1fa38c (requested 2) | model_forward/N16 16/16 | M3_R3_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/20/50 | 0..7 | 0..2 | 45.6%/52.8% | L5:e=0.06501,t=1.09..1.82ms | qualified |
| selected_001 | dllm | longbench_v2 | 4/1fa38c (requested 2) | model_forward/N16 16/16 | B_A8_matched | GLOBAL_ONLY_NATIVE_LOCAL | 10/0/70 | 0..7 | 0..7 | 51.0%/58.3% | L5:e=0.06501,t=1.08..1.09ms | qualified |
| selected_001 | dllm | longbench_v2 | 4/1fa38c (requested 2) | denoising_step/N16 16/16 | D_native | NATIVE | 0/0/0 | N/A | N/A | N/A/N/A | N/A | unavailable |
| selected_001 | dllm | longbench_v2 | 4/1fa38c (requested 2) | denoising_step/N16 16/16 | G75L30_nativeQ128 | ALL_NATIVE_LEGAL | 60/0/420 | 0..14 | 0..14 | 53.2%/57.0% | L0:e=0.03747,t=0.172..0.172ms/L5:e=0.06649,t=1.17..1.18ms | qualified |
| selected_001 | dllm | longbench_v2 | 4/1fa38c (requested 2) | denoising_step/N16 16/16 | D_matched | GLOBAL_ONLY_NATIVE_LOCAL | 80/0/0 | N/A | N/A | 0.0%/0.0% | N/A | qualified |
| selected_001 | dllm | longbench_v2 | 4/1fa38c (requested 2) | denoising_step/N16 16/16 | T_scope | GLOBAL_ONLY_NATIVE_LOCAL | 80/0/0 | 0..0 | 0..0 | 0.0%/53.3% | N/A | qualified |
| selected_001 | dllm | longbench_v2 | 4/1fa38c (requested 2) | denoising_step/N16 16/16 | M1_R1_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/70/0 | 0..7 | 0..0 | 44.8%/52.1% | L5:e=0.06675,t=1.08..1.08ms | qualified |
| selected_001 | dllm | longbench_v2 | 4/1fa38c (requested 2) | denoising_step/N16 16/16 | M3_R2_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/30/40 | 0..7 | 0..1 | 44.6%/51.9% | L5:e=0.07326,t=1.09..1.77ms | qualified |
| selected_001 | dllm | longbench_v2 | 4/1fa38c (requested 2) | denoising_step/N16 16/16 | M3_R3_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/20/50 | 0..7 | 0..2 | 45.6%/52.8% | L5:e=0.06501,t=1.09..1.82ms | qualified |
| selected_001 | dllm | longbench_v2 | 4/1fa38c (requested 2) | denoising_step/N16 16/16 | B_A8_matched | GLOBAL_ONLY_NATIVE_LOCAL | 10/0/70 | 0..7 | 0..7 | 51.0%/58.3% | L5:e=0.06501,t=1.08..1.09ms | qualified |
| selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | model_forward/N16 N/A/N/A | D_native | NATIVE | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | model_forward/N16 N/A/N/A | G75L30_nativeQ128 | ALL_NATIVE_LEGAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | model_forward/N16 N/A/N/A | D_matched | GLOBAL_ONLY_NATIVE_LOCAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | model_forward/N16 N/A/N/A | T_scope | GLOBAL_ONLY_NATIVE_LOCAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | model_forward/N16 N/A/N/A | M1_R1_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | model_forward/N16 N/A/N/A | M3_R2_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | model_forward/N16 N/A/N/A | M3_R3_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | model_forward/N16 N/A/N/A | B_A8_matched | GLOBAL_ONLY_NATIVE_LOCAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | denoising_step/N16 N/A/N/A | D_native | NATIVE | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | denoising_step/N16 N/A/N/A | G75L30_nativeQ128 | ALL_NATIVE_LEGAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | denoising_step/N16 N/A/N/A | D_matched | GLOBAL_ONLY_NATIVE_LOCAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | denoising_step/N16 N/A/N/A | T_scope | GLOBAL_ONLY_NATIVE_LOCAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | denoising_step/N16 N/A/N/A | M1_R1_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | denoising_step/N16 N/A/N/A | M3_R2_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | denoising_step/N16 N/A/N/A | M3_R3_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | dllm | ruler4k | 0/988ec8 (requested 0,2) | denoising_step/N16 N/A/N/A | B_A8_matched | GLOBAL_ONLY_NATIVE_LOCAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | mpk | aime26 | 0/dc3a96 (requested 0) | model_forward/N16 12/16 | D_native | NATIVE | 0/0/0 | N/A | N/A | N/A/N/A | N/A | unavailable (native_shortened) |
| selected_001 | mpk | aime26 | 0/dc3a96 (requested 0) | model_forward/N16 12/16 | G75L30_nativeQ128 | ALL_NATIVE_LEGAL | 60/0/300 | 0..10 | 0..10 | 1.0%/1.1% | L0:e=0.04684,t=0.125..0.125ms/L5:e=0.0754,t=0.156..0.162ms | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 0/dc3a96 (requested 0) | model_forward/N16 12/16 | D_matched | GLOBAL_ONLY_NATIVE_LOCAL | 60/0/0 | N/A | N/A | 0.0%/0.0% | N/A | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 0/dc3a96 (requested 0) | model_forward/N16 12/16 | T_scope | GLOBAL_ONLY_NATIVE_LOCAL | 60/0/0 | 0..0 | 0..0 | 0.0%/1.2% | N/A | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 0/dc3a96 (requested 0) | model_forward/N16 12/16 | M1_R1_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/50/0 | 0..7 | 0..0 | 0.8%/1.0% | L5:e=0.07391,t=0.158..0.16ms | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 0/dc3a96 (requested 0) | model_forward/N16 12/16 | M3_R2_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/20/30 | 0..7 | 0..1 | 0.8%/1.1% | L5:e=0.08,t=0.16..0.169ms | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 0/dc3a96 (requested 0) | model_forward/N16 12/16 | M3_R3_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/15/35 | 0..7 | 0..2 | 0.9%/1.2% | L5:e=0.07391,t=0.161..0.165ms | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 0/dc3a96 (requested 0) | model_forward/N16 12/16 | B_A8_matched | GLOBAL_ONLY_NATIVE_LOCAL | 10/0/50 | 0..7 | 0..7 | 1.8%/2.1% | L5:e=0.07391,t=0.167..0.173ms | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 0/dc3a96 (requested 0) | denoising_step/N16 12/16 | D_native | NATIVE | 0/0/0 | N/A | N/A | N/A/N/A | N/A | unavailable (native_shortened) |
| selected_001 | mpk | aime26 | 0/dc3a96 (requested 0) | denoising_step/N16 12/16 | G75L30_nativeQ128 | ALL_NATIVE_LEGAL | 60/0/300 | 0..10 | 0..10 | 1.0%/1.1% | L0:e=0.04684,t=0.127..0.135ms/L5:e=0.0754,t=0.156..0.158ms | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 0/dc3a96 (requested 0) | denoising_step/N16 12/16 | D_matched | GLOBAL_ONLY_NATIVE_LOCAL | 60/0/0 | N/A | N/A | 0.0%/0.0% | N/A | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 0/dc3a96 (requested 0) | denoising_step/N16 12/16 | T_scope | GLOBAL_ONLY_NATIVE_LOCAL | 60/0/0 | 0..0 | 0..0 | 0.0%/1.2% | N/A | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 0/dc3a96 (requested 0) | denoising_step/N16 12/16 | M1_R1_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/50/0 | 0..7 | 0..0 | 0.8%/1.0% | L5:e=0.07391,t=0.158..0.169ms | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 0/dc3a96 (requested 0) | denoising_step/N16 12/16 | M3_R2_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/20/30 | 0..7 | 0..1 | 0.8%/1.1% | L5:e=0.08,t=0.162..0.171ms | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 0/dc3a96 (requested 0) | denoising_step/N16 12/16 | M3_R3_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/15/35 | 0..7 | 0..2 | 0.9%/1.2% | L5:e=0.07391,t=0.158..0.173ms | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 0/dc3a96 (requested 0) | denoising_step/N16 12/16 | B_A8_matched | GLOBAL_ONLY_NATIVE_LOCAL | 10/0/50 | 0..7 | 0..7 | 1.8%/2.1% | L5:e=0.07391,t=0.154..0.155ms | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 8/f79e8d (requested 2) | model_forward/N16 14/16 | D_native | NATIVE | 0/0/0 | N/A | N/A | N/A/N/A | N/A | unavailable (native_shortened) |
| selected_001 | mpk | aime26 | 8/f79e8d (requested 2) | model_forward/N16 14/16 | G75L30_nativeQ128 | ALL_NATIVE_LEGAL | 60/0/360 | 0..12 | 0..12 | 26.1%/28.3% | L0:e=0.0384,t=0.177..0.179ms/L5:e=0.05575,t=0.307..0.309ms | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 8/f79e8d (requested 2) | model_forward/N16 14/16 | D_matched | GLOBAL_ONLY_NATIVE_LOCAL | 70/0/0 | N/A | N/A | 0.0%/0.0% | N/A | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 8/f79e8d (requested 2) | model_forward/N16 14/16 | T_scope | GLOBAL_ONLY_NATIVE_LOCAL | 70/0/0 | 0..0 | 0..0 | 0.0%/15.2% | N/A | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 8/f79e8d (requested 2) | model_forward/N16 14/16 | M1_R1_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/60/0 | 0..7 | 0..0 | 13.4%/15.9% | L5:e=0.06414,t=0.381..0.388ms | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 8/f79e8d (requested 2) | model_forward/N16 14/16 | M3_R2_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/25/35 | 0..7 | 0..1 | 13.6%/16.0% | L5:e=0.07176,t=0.383..0.447ms | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 8/f79e8d (requested 2) | model_forward/N16 14/16 | M3_R3_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/15/45 | 0..7 | 0..2 | 13.8%/16.2% | L5:e=0.06414,t=0.383..0.461ms | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 8/f79e8d (requested 2) | model_forward/N16 14/16 | B_A8_matched | GLOBAL_ONLY_NATIVE_LOCAL | 10/0/60 | 0..7 | 0..7 | 15.8%/18.2% | L5:e=0.06414,t=0.382..0.39ms | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 8/f79e8d (requested 2) | denoising_step/N16 14/16 | D_native | NATIVE | 0/0/0 | N/A | N/A | N/A/N/A | N/A | unavailable (native_shortened) |
| selected_001 | mpk | aime26 | 8/f79e8d (requested 2) | denoising_step/N16 14/16 | G75L30_nativeQ128 | ALL_NATIVE_LEGAL | 60/0/360 | 0..12 | 0..12 | 26.1%/28.3% | L0:e=0.0384,t=0.18..0.186ms/L5:e=0.05575,t=0.307..0.309ms | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 8/f79e8d (requested 2) | denoising_step/N16 14/16 | D_matched | GLOBAL_ONLY_NATIVE_LOCAL | 70/0/0 | N/A | N/A | 0.0%/0.0% | N/A | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 8/f79e8d (requested 2) | denoising_step/N16 14/16 | T_scope | GLOBAL_ONLY_NATIVE_LOCAL | 70/0/0 | 0..0 | 0..0 | 0.0%/15.2% | N/A | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 8/f79e8d (requested 2) | denoising_step/N16 14/16 | M1_R1_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/60/0 | 0..7 | 0..0 | 13.4%/15.9% | L5:e=0.06414,t=0.383..0.386ms | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 8/f79e8d (requested 2) | denoising_step/N16 14/16 | M3_R2_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/25/35 | 0..7 | 0..1 | 13.6%/16.0% | L5:e=0.07176,t=0.381..0.452ms | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 8/f79e8d (requested 2) | denoising_step/N16 14/16 | M3_R3_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/15/45 | 0..7 | 0..2 | 13.8%/16.2% | L5:e=0.06414,t=0.388..0.453ms | qualified (native_shortened) |
| selected_001 | mpk | aime26 | 8/f79e8d (requested 2) | denoising_step/N16 14/16 | B_A8_matched | GLOBAL_ONLY_NATIVE_LOCAL | 10/0/60 | 0..7 | 0..7 | 15.8%/18.2% | L5:e=0.06414,t=0.379..0.385ms | qualified (native_shortened) |
| selected_001 | mpk | longbench_v2 | 0/5c7381 (requested 0) | model_forward/N16 16/16 | D_native | NATIVE | 0/0/0 | N/A | N/A | N/A/N/A | N/A | unavailable |
| selected_001 | mpk | longbench_v2 | 0/5c7381 (requested 0) | model_forward/N16 16/16 | G75L30_nativeQ128 | ALL_NATIVE_LEGAL | 60/0/420 | 0..14 | 0..14 | 49.7%/53.3% | L0:e=0.04228,t=0.181..0.182ms/L5:e=0.07162,t=0.905..0.906ms | qualified |
| selected_001 | mpk | longbench_v2 | 0/5c7381 (requested 0) | model_forward/N16 16/16 | D_matched | GLOBAL_ONLY_NATIVE_LOCAL | 80/0/0 | N/A | N/A | 0.0%/0.0% | N/A | qualified |
| selected_001 | mpk | longbench_v2 | 0/5c7381 (requested 0) | model_forward/N16 16/16 | T_scope | GLOBAL_ONLY_NATIVE_LOCAL | 80/0/0 | 0..0 | 0..0 | 0.0%/53.9% | N/A | qualified |
| selected_001 | mpk | longbench_v2 | 0/5c7381 (requested 0) | model_forward/N16 16/16 | M1_R1_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/70/0 | 0..7 | 0..0 | 49.6%/57.2% | L5:e=0.07644,t=0.848..0.857ms | qualified |
| selected_001 | mpk | longbench_v2 | 0/5c7381 (requested 0) | model_forward/N16 16/16 | M3_R2_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/30/40 | 0..7 | 0..1 | 49.8%/57.5% | L5:e=0.07644,t=0.845..0.902ms | qualified |
| selected_001 | mpk | longbench_v2 | 0/5c7381 (requested 0) | model_forward/N16 16/16 | M3_R3_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/20/50 | 0..7 | 0..2 | 50.5%/58.1% | L5:e=0.07644,t=0.849..1.09ms | qualified |
| selected_001 | mpk | longbench_v2 | 0/5c7381 (requested 0) | model_forward/N16 16/16 | B_A8_matched | GLOBAL_ONLY_NATIVE_LOCAL | 10/0/70 | 0..7 | 0..7 | 53.6%/61.3% | L5:e=0.07644,t=0.853..0.855ms | qualified |
| selected_001 | mpk | longbench_v2 | 0/5c7381 (requested 0) | denoising_step/N16 16/16 | D_native | NATIVE | 0/0/0 | N/A | N/A | N/A/N/A | N/A | unavailable |
| selected_001 | mpk | longbench_v2 | 0/5c7381 (requested 0) | denoising_step/N16 16/16 | G75L30_nativeQ128 | ALL_NATIVE_LEGAL | 60/0/420 | 0..14 | 0..14 | 49.7%/53.3% | L0:e=0.04228,t=0.194..0.195ms/L5:e=0.07162,t=0.903..0.905ms | qualified |
| selected_001 | mpk | longbench_v2 | 0/5c7381 (requested 0) | denoising_step/N16 16/16 | D_matched | GLOBAL_ONLY_NATIVE_LOCAL | 80/0/0 | N/A | N/A | 0.0%/0.0% | N/A | qualified |
| selected_001 | mpk | longbench_v2 | 0/5c7381 (requested 0) | denoising_step/N16 16/16 | T_scope | GLOBAL_ONLY_NATIVE_LOCAL | 80/0/0 | 0..0 | 0..0 | 0.0%/53.9% | N/A | qualified |
| selected_001 | mpk | longbench_v2 | 0/5c7381 (requested 0) | denoising_step/N16 16/16 | M1_R1_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/70/0 | 0..7 | 0..0 | 49.6%/57.2% | L5:e=0.07644,t=0.846..0.854ms | qualified |
| selected_001 | mpk | longbench_v2 | 0/5c7381 (requested 0) | denoising_step/N16 16/16 | M3_R2_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/30/40 | 0..7 | 0..1 | 49.8%/57.5% | L5:e=0.07644,t=0.849..0.892ms | qualified |
| selected_001 | mpk | longbench_v2 | 0/5c7381 (requested 0) | denoising_step/N16 16/16 | M3_R3_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/20/50 | 0..7 | 0..2 | 50.5%/58.1% | L5:e=0.07644,t=0.847..1.08ms | qualified |
| selected_001 | mpk | longbench_v2 | 0/5c7381 (requested 0) | denoising_step/N16 16/16 | B_A8_matched | GLOBAL_ONLY_NATIVE_LOCAL | 10/0/70 | 0..7 | 0..7 | 53.6%/61.3% | L5:e=0.07644,t=0.85..0.852ms | qualified |
| selected_001 | mpk | longbench_v2 | 4/3316cd (requested 2) | model_forward/N16 16/16 | D_native | NATIVE | 0/0/0 | N/A | N/A | N/A/N/A | N/A | unavailable |
| selected_001 | mpk | longbench_v2 | 4/3316cd (requested 2) | model_forward/N16 16/16 | G75L30_nativeQ128 | ALL_NATIVE_LEGAL | 60/0/420 | 0..14 | 0..14 | 50.6%/54.2% | L0:e=0.03951,t=0.185..0.189ms/L5:e=0.05984,t=0.937..0.947ms | qualified |
| selected_001 | mpk | longbench_v2 | 4/3316cd (requested 2) | model_forward/N16 16/16 | D_matched | GLOBAL_ONLY_NATIVE_LOCAL | 80/0/0 | N/A | N/A | 0.0%/0.0% | N/A | qualified |
| selected_001 | mpk | longbench_v2 | 4/3316cd (requested 2) | model_forward/N16 16/16 | T_scope | GLOBAL_ONLY_NATIVE_LOCAL | 80/0/0 | 0..0 | 0..0 | 0.0%/55.0% | N/A | qualified |
| selected_001 | mpk | longbench_v2 | 4/3316cd (requested 2) | model_forward/N16 16/16 | M1_R1_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/70/0 | 0..7 | 0..0 | 49.3%/56.3% | L5:e=0.06338,t=1.02..1.03ms | qualified |
| selected_001 | mpk | longbench_v2 | 4/3316cd (requested 2) | model_forward/N16 16/16 | M3_R2_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/30/40 | 0..7 | 0..1 | 49.4%/56.4% | L5:e=0.06824,t=1.03..1.3ms | qualified |
| selected_001 | mpk | longbench_v2 | 4/3316cd (requested 2) | model_forward/N16 16/16 | M3_R3_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/20/50 | 0..7 | 0..2 | 49.3%/56.3% | L5:e=0.0678,t=1.02..1.42ms | qualified |
| selected_001 | mpk | longbench_v2 | 4/3316cd (requested 2) | model_forward/N16 16/16 | B_A8_matched | GLOBAL_ONLY_NATIVE_LOCAL | 10/0/70 | 0..7 | 0..7 | 49.1%/56.1% | L5:e=0.06338,t=1.03..1.03ms | qualified |
| selected_001 | mpk | longbench_v2 | 4/3316cd (requested 2) | denoising_step/N16 16/16 | D_native | NATIVE | 0/0/0 | N/A | N/A | N/A/N/A | N/A | unavailable |
| selected_001 | mpk | longbench_v2 | 4/3316cd (requested 2) | denoising_step/N16 16/16 | G75L30_nativeQ128 | ALL_NATIVE_LEGAL | 60/0/420 | 0..14 | 0..14 | 50.6%/54.2% | L0:e=0.03951,t=0.181..0.184ms/L5:e=0.05984,t=0.932..0.94ms | qualified |
| selected_001 | mpk | longbench_v2 | 4/3316cd (requested 2) | denoising_step/N16 16/16 | D_matched | GLOBAL_ONLY_NATIVE_LOCAL | 80/0/0 | N/A | N/A | 0.0%/0.0% | N/A | qualified |
| selected_001 | mpk | longbench_v2 | 4/3316cd (requested 2) | denoising_step/N16 16/16 | T_scope | GLOBAL_ONLY_NATIVE_LOCAL | 80/0/0 | 0..0 | 0..0 | 0.0%/55.0% | N/A | qualified |
| selected_001 | mpk | longbench_v2 | 4/3316cd (requested 2) | denoising_step/N16 16/16 | M1_R1_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/70/0 | 0..7 | 0..0 | 49.3%/56.3% | L5:e=0.06338,t=1.02..1.03ms | qualified |
| selected_001 | mpk | longbench_v2 | 4/3316cd (requested 2) | denoising_step/N16 16/16 | M3_R2_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/30/40 | 0..7 | 0..1 | 49.4%/56.4% | L5:e=0.06824,t=1.03..1.3ms | qualified |
| selected_001 | mpk | longbench_v2 | 4/3316cd (requested 2) | denoising_step/N16 16/16 | M3_R3_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 10/20/50 | 0..7 | 0..2 | 49.3%/56.3% | L5:e=0.0678,t=1.03..1.41ms | qualified |
| selected_001 | mpk | longbench_v2 | 4/3316cd (requested 2) | denoising_step/N16 16/16 | B_A8_matched | GLOBAL_ONLY_NATIVE_LOCAL | 10/0/70 | 0..7 | 0..7 | 49.1%/56.1% | L5:e=0.06338,t=1.02..1.02ms | qualified |
| selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | model_forward/N16 N/A/N/A | D_native | NATIVE | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | model_forward/N16 N/A/N/A | G75L30_nativeQ128 | ALL_NATIVE_LEGAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | model_forward/N16 N/A/N/A | D_matched | GLOBAL_ONLY_NATIVE_LOCAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | model_forward/N16 N/A/N/A | T_scope | GLOBAL_ONLY_NATIVE_LOCAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | model_forward/N16 N/A/N/A | M1_R1_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | model_forward/N16 N/A/N/A | M3_R2_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | model_forward/N16 N/A/N/A | M3_R3_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | model_forward/N16 N/A/N/A | B_A8_matched | GLOBAL_ONLY_NATIVE_LOCAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | denoising_step/N16 N/A/N/A | D_native | NATIVE | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | denoising_step/N16 N/A/N/A | G75L30_nativeQ128 | ALL_NATIVE_LEGAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | denoising_step/N16 N/A/N/A | D_matched | GLOBAL_ONLY_NATIVE_LOCAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | denoising_step/N16 N/A/N/A | T_scope | GLOBAL_ONLY_NATIVE_LOCAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | denoising_step/N16 N/A/N/A | M1_R1_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | denoising_step/N16 N/A/N/A | M3_R2_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | denoising_step/N16 N/A/N/A | M3_R3_A8_current_output | GLOBAL_ONLY_NATIVE_LOCAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |
| selected_001 | mpk | ruler4k | 0/a212ef (requested 0,2) | denoising_step/N16 N/A/N/A | B_A8_matched | GLOBAL_ONLY_NATIVE_LOCAL | 0/0/0 | N/A | N/A | N/A/N/A | N/A | missing_sequence |

Counter twins matched accepted input/output digests when qualified and are untimed. Physical QK/PV pairs are native-legal eligible pairs; projection skipping is unmeasured. The table makes no answer-quality or request-speed claim.
