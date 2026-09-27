# Screen002 physical-work counters

Untimed counter twins replayed the accepted model-forward N4 inputs. Qualified twins match every accepted input/output digest. Pair fractions are over native-legal eligible pairs; they are physical QK/PV work, not latency or task quality. Projection skipping was not measured.

Cost table SHA-256: `a3d30baf6bae6ec0a1e54890e5e048bf67d4274bb7f98958f641e64c8c4af944`.

- dllm GPU `GPU-fc12ad5c-5334-5509-8fc6-465498fd3915`; profile `a908b1597ba347b762d692f28a80ce3cc33fcf7e32207aa2fe83139718c987f4`; source commits 699a9ec1c4d5b31d0a12cd046256ba59bfdfb203.
  - `v20_counter.py`: `7e563e690c4c25d2fc80e4568cbd9c4cd2bc5b3ae475e9653a66ef776f38f70d`
  - `replay_harness.py`: `60b8c09e98636b0cdd449dbdb4fb54f1cfd834b0e9648e068ac42b0c0881e879`
  - `v20_profile.py`: `4f5e4c9eba968bfd82fbfb5cd3cc196f9b09d513fcfc52c4969e42403767b7f2`
  - `v9_step_replay_profile.py`: `31266adc0cfee09e865cb4b347b2f5e15eaf3c1e4f1d12e0b125ee28a57afc13`
- mpk GPU `GPU-6139046a-b005-8fe5-a837-f8270472ab72`; profile `e781f09bc473f2724cd5f4af2374d2ef2e41cc33e16fbae9b1b69844f299d1ef`; source commits 699a9ec1c4d5b31d0a12cd046256ba59bfdfb203.
  - `v20_counter.py`: `7e563e690c4c25d2fc80e4568cbd9c4cd2bc5b3ae475e9653a66ef776f38f70d`
  - `replay_harness.py`: `60b8c09e98636b0cdd449dbdb4fb54f1cfd834b0e9648e068ac42b0c0881e879`
  - `v20_profile.py`: `4f5e4c9eba968bfd82fbfb5cd3cc196f9b09d513fcfc52c4969e42403767b7f2`
  - `v9_step_replay_profile.py`: `31266adc0cfee09e865cb4b347b2f5e15eaf3c1e4f1d12e0b125ee28a57afc13`

| Host | Dataset | Scope | Arm | Policy | Twin | A/D/H layer calls | QK skipped whole/static/current | PV skipped whole/static/current | Local QK/PV whole | Global QK/PV whole | Cost rows |
|---|---|---|---|---|---|---|---|---|---|---|---:|
| dllm | aime26 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_D_matched | P0 | qualified 2/2 | 240/0/0 | 0.0%/0.0%/0.0% | 0.0%/0.0%/0.0% | 0.0%/0.0% | 0.0%/0.0% | 2 |
| dllm | aime26 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_B_A8_matched | P0 | qualified 2/2 | 60/0/180 | 30.2%/32.2%/24.9% | 40.2%/42.9%/33.3% | 33.1%/44.1% | 21.5%/28.7% | 2 |
| dllm | aime26 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_M1_R1_A8_current_output | P0 | qualified 2/2 | 60/180/0 | 24.6%/25.8%/21.6% | 34.7%/36.6%/29.9% | 27.2%/38.2% | 17.0%/24.2% | 2 |
| dllm | aime26 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_M3_R2_A8_current_output | P0 | qualified 2/2 | 60/60/120 | 25.8%/27.2%/22.3% | 35.8%/37.9%/30.6% | 28.4%/39.5% | 17.9%/25.0% | 2 |
| dllm | aime26 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_M3_R3_A8_current_output | P0 | qualified 2/2 | 60/60/120 | 26.9%/28.4%/23.1% | 36.9%/39.1%/31.4% | 29.6%/40.6% | 18.9%/26.0% | 2 |
| dllm | aime26 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_triton_B_A8_matched | P0 | qualified 2/2 | 60/0/180 | 30.2%/32.2%/24.9% | 40.2%/42.9%/33.3% | 33.1%/44.1% | 21.5%/28.7% | 2 |
| dllm | aime26 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_triton_M3_R2_A8_current_output | P0 | qualified 2/2 | 60/60/120 | 25.8%/27.2%/22.3% | 35.8%/37.9%/30.6% | 28.4%/39.5% | 17.9%/25.0% | 2 |
| dllm | aime26 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P1_B_A8_matched | P1 | qualified 2/2 | 60/0/180 | 38.9%/40.9%/33.9% | 51.9%/54.5%/45.2% | 43.4%/57.8% | 25.7%/34.3% | 2 |
| dllm | aime26 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P1_M1_R1_A8_current_output | P1 | qualified 2/2 | 60/180/0 | 33.1%/34.3%/30.2% | 46.1%/47.9%/41.5% | 37.4%/51.8% | 20.5%/29.1% | 2 |
| dllm | aime26 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P1_M3_R2_A8_current_output | P1 | qualified 2/2 | 60/60/120 | 34.2%/35.5%/31.0% | 47.2%/49.1%/42.3% | 38.6%/53.0% | 21.4%/30.0% | 2 |
| dllm | aime26 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P1_M3_R3_A8_current_output | P1 | qualified 2/2 | 60/60/120 | 35.5%/36.9%/31.7% | 48.4%/50.6%/43.0% | 39.8%/54.2% | 22.7%/31.3% | 2 |
| dllm | aime26 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_T_scope | P0 | qualified 2/2 | 240/0/0 | 0.0%/0.0%/0.0% | 32.4%/35.0%/25.9% | 0.0%/36.4% | 0.0%/20.7% | 2 |
| dllm | aime26 | ALL_NATIVE_LEGAL | G75L30_nativeQ128 | P0 | qualified 2/2 | 120/0/120 | 12.5%/17.3%/0.0% | 18.7%/26.0%/0.0% | 7.8%/11.7% | 26.2%/39.3% | 2 |
| dllm | aime26 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_D_matched | P0 | qualified 2/2 | 40/0/0 | 0.0%/0.0%/0.0% | 0.0%/0.0%/0.0% | N/A/N/A | 0.0%/0.0% | 2 |
| dllm | aime26 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_B_A8_matched | P0 | qualified 2/2 | 10/0/30 | 21.8%/25.2%/7.1% | 29.1%/33.6%/9.5% | N/A/N/A | 21.8%/29.1% | 2 |
| dllm | aime26 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_M1_R1_A8_current_output | P0 | qualified 2/2 | 10/30/0 | 17.0%/19.7%/5.5% | 24.3%/28.0%/7.9% | N/A/N/A | 17.0%/24.3% | 2 |
| dllm | aime26 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_M3_R2_A8_current_output | P0 | qualified 2/2 | 10/10/20 | 17.8%/20.5%/5.7% | 25.0%/28.9%/8.1% | N/A/N/A | 17.8%/25.0% | 2 |
| dllm | aime26 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_M3_R3_A8_current_output | P0 | qualified 2/2 | 10/10/20 | 19.1%/22.0%/6.3% | 26.4%/30.4%/8.7% | N/A/N/A | 19.1%/26.4% | 2 |
| dllm | aime26 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_triton_B_A8_matched | P0 | qualified 2/2 | 10/0/30 | 21.8%/25.2%/7.1% | 29.1%/33.6%/9.5% | N/A/N/A | 21.8%/29.1% | 2 |
| dllm | aime26 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_triton_M3_R2_A8_current_output | P0 | qualified 2/2 | 10/10/20 | 17.8%/20.5%/5.7% | 25.0%/28.9%/8.1% | N/A/N/A | 17.8%/25.0% | 2 |
| dllm | aime26 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P1_B_A8_matched | P1 | qualified 2/2 | 10/0/30 | 27.7%/31.8%/9.8% | 37.0%/42.5%/13.0% | N/A/N/A | 27.7%/37.0% | 2 |
| dllm | aime26 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P1_M1_R1_A8_current_output | P1 | qualified 2/2 | 10/30/0 | 22.7%/26.1%/7.6% | 31.9%/36.7%/10.9% | N/A/N/A | 22.7%/31.9% | 2 |
| dllm | aime26 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P1_M3_R2_A8_current_output | P1 | qualified 2/2 | 10/10/20 | 23.5%/27.1%/8.0% | 32.8%/37.7%/11.2% | N/A/N/A | 23.5%/32.8% | 2 |
| dllm | aime26 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P1_M3_R3_A8_current_output | P1 | qualified 2/2 | 10/10/20 | 24.8%/28.5%/8.6% | 34.0%/39.1%/11.9% | N/A/N/A | 24.8%/34.0% | 2 |
| dllm | aime26 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_T_scope | P0 | qualified 2/2 | 40/0/0 | 0.0%/0.0%/0.0% | 21.1%/24.5%/6.2% | N/A/N/A | 0.0%/21.1% | 2 |
| dllm | aime26 | NATIVE | D_native | P0 | unavailable 0/2 | 0/0/0 | N/A/N/A/N/A | N/A/N/A/N/A | N/A/N/A | N/A/N/A | 2 |
| dllm | longbench_v2 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_D_matched | P0 | qualified 2/2 | 240/0/0 | 0.0%/0.0%/0.0% | 0.0%/0.0%/0.0% | 0.0%/0.0% | 0.0%/0.0% | 2 |
| dllm | longbench_v2 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_B_A8_matched | P0 | qualified 2/2 | 60/0/180 | 50.4%/51.8%/27.5% | 67.1%/69.0%/36.7% | 35.6%/47.5% | 55.1%/73.5% | 2 |
| dllm | longbench_v2 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_M1_R1_A8_current_output | P0 | qualified 2/2 | 60/180/0 | 39.9%/41.3%/19.1% | 56.7%/58.5%/28.2% | 22.6%/34.5% | 45.6%/63.9% | 2 |
| dllm | longbench_v2 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_M3_R2_A8_current_output | P0 | qualified 2/2 | 60/60/120 | 40.9%/42.2%/19.9% | 57.7%/59.5%/29.0% | 23.6%/35.5% | 46.5%/64.9% | 2 |
| dllm | longbench_v2 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_M3_R3_A8_current_output | P0 | qualified 2/2 | 60/60/120 | 44.7%/46.0%/23.2% | 61.5%/63.3%/32.4% | 28.7%/40.6% | 49.9%/68.3% | 2 |
| dllm | longbench_v2 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_triton_B_A8_matched | P0 | qualified 2/2 | 60/0/180 | 50.4%/51.8%/27.5% | 67.1%/69.0%/36.7% | 35.6%/47.5% | 55.1%/73.5% | 2 |
| dllm | longbench_v2 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_triton_M3_R2_A8_current_output | P0 | qualified 2/2 | 60/60/120 | 40.9%/42.2%/19.8% | 57.7%/59.5%/29.0% | 23.6%/35.5% | 46.5%/64.9% | 2 |
| dllm | longbench_v2 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P1_B_A8_matched | P1 | qualified 2/2 | 60/0/180 | 56.6%/58.0%/35.3% | 75.5%/77.3%/47.0% | 47.1%/62.8% | 59.7%/79.6% | 2 |
| dllm | longbench_v2 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P1_M1_R1_A8_current_output | P1 | qualified 2/2 | 60/180/0 | 47.0%/48.4%/25.3% | 65.9%/67.7%/37.1% | 33.1%/48.8% | 51.6%/71.5% | 2 |
| dllm | longbench_v2 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P1_M3_R2_A8_current_output | P1 | qualified 2/2 | 60/60/120 | 47.9%/49.2%/26.2% | 66.7%/68.5%/38.0% | 34.3%/50.0% | 52.3%/72.2% | 2 |
| dllm | longbench_v2 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P1_M3_R3_A8_current_output | P1 | qualified 2/2 | 60/60/120 | 51.4%/52.8%/30.1% | 70.3%/72.1%/41.8% | 39.6%/55.3% | 55.3%/75.2% | 2 |
| dllm | longbench_v2 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_T_scope | P0 | qualified 2/2 | 240/0/0 | 0.0%/0.0%/0.0% | 56.7%/58.5%/27.5% | 0.0%/34.2% | 0.0%/64.0% | 2 |
| dllm | longbench_v2 | ALL_NATIVE_LEGAL | G75L30_nativeQ128 | P0 | qualified 2/2 | 120/0/120 | 30.3%/32.1%/0.0% | 45.4%/48.2%/0.0% | 9.7%/14.6% | 36.9%/55.4% | 2 |
| dllm | longbench_v2 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_D_matched | P0 | qualified 2/2 | 40/0/0 | 0.0%/0.0%/0.0% | 0.0%/0.0%/0.0% | N/A/N/A | 0.0%/0.0% | 2 |
| dllm | longbench_v2 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_B_A8_matched | P0 | qualified 2/2 | 10/0/30 | 55.2%/55.8%/11.1% | 73.6%/74.4%/14.8% | N/A/N/A | 55.2%/73.6% | 2 |
| dllm | longbench_v2 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_M1_R1_A8_current_output | P0 | qualified 2/2 | 10/30/0 | 45.7%/46.2%/7.0% | 64.1%/64.8%/10.7% | N/A/N/A | 45.7%/64.1% | 2 |
| dllm | longbench_v2 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_M3_R2_A8_current_output | P0 | qualified 2/2 | 10/10/20 | 46.6%/47.1%/7.6% | 65.0%/65.7%/11.3% | N/A/N/A | 46.6%/65.0% | 2 |
| dllm | longbench_v2 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_M3_R3_A8_current_output | P0 | qualified 2/2 | 10/10/20 | 50.0%/50.5%/9.1% | 68.4%/69.1%/12.8% | N/A/N/A | 50.0%/68.4% | 2 |
| dllm | longbench_v2 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_triton_B_A8_matched | P0 | qualified 2/2 | 10/0/30 | 55.2%/55.8%/11.1% | 73.6%/74.4%/14.8% | N/A/N/A | 55.2%/73.6% | 2 |
| dllm | longbench_v2 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_triton_M3_R2_A8_current_output | P0 | qualified 2/2 | 10/10/20 | 46.6%/47.1%/7.7% | 65.0%/65.7%/11.4% | N/A/N/A | 46.6%/65.0% | 2 |
| dllm | longbench_v2 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P1_B_A8_matched | P1 | qualified 2/2 | 10/0/30 | 59.5%/60.1%/16.1% | 79.4%/80.2%/21.5% | N/A/N/A | 59.5%/79.4% | 2 |
| dllm | longbench_v2 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P1_M1_R1_A8_current_output | P1 | qualified 2/2 | 10/30/0 | 51.4%/51.9%/9.8% | 71.2%/72.0%/15.1% | N/A/N/A | 51.4%/71.2% | 2 |
| dllm | longbench_v2 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P1_M3_R2_A8_current_output | P1 | qualified 2/2 | 10/10/20 | 52.2%/52.7%/10.5% | 72.0%/72.8%/15.9% | N/A/N/A | 52.2%/72.0% | 2 |
| dllm | longbench_v2 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P1_M3_R3_A8_current_output | P1 | qualified 2/2 | 10/10/20 | 55.1%/55.6%/13.1% | 74.9%/75.7%/18.5% | N/A/N/A | 55.1%/74.9% | 2 |
| dllm | longbench_v2 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_T_scope | P0 | qualified 2/2 | 40/0/0 | 0.0%/0.0%/0.0% | 63.6%/64.3%/10.3% | N/A/N/A | 0.0%/63.6% | 2 |
| dllm | longbench_v2 | NATIVE | D_native | P0 | unavailable 0/2 | 0/0/0 | N/A/N/A/N/A | N/A/N/A/N/A | N/A/N/A | N/A/N/A | 2 |
| dllm | ruler4k | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_D_matched | P0 | qualified 2/2 | 240/0/0 | 0.0%/0.0%/0.0% | 0.0%/0.0%/0.0% | 0.0%/0.0% | 0.0%/0.0% | 1 |
| dllm | ruler4k | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_B_A8_matched | P0 | qualified 2/2 | 60/0/180 | 39.5%/42.3%/23.0% | 52.7%/56.4%/30.6% | 36.2%/48.3% | 44.4%/59.3% | 1 |
| dllm | ruler4k | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_M1_R1_A8_current_output | P0 | qualified 2/2 | 60/180/0 | 37.4%/40.0%/21.7% | 50.5%/54.1%/29.3% | 33.6%/45.7% | 43.0%/57.8% | 1 |
| dllm | ruler4k | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_M3_R2_A8_current_output | P0 | qualified 2/2 | 60/60/120 | 37.1%/39.6%/21.8% | 50.2%/53.7%/29.4% | 33.3%/45.3% | 42.7%/57.6% | 1 |
| dllm | ruler4k | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_M3_R3_A8_current_output | P0 | qualified 2/2 | 60/60/120 | 38.6%/41.3%/22.3% | 51.8%/55.4%/30.0% | 35.1%/47.1% | 43.9%/58.7% | 1 |
| dllm | ruler4k | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_triton_B_A8_matched | P0 | qualified 2/2 | 60/0/180 | 39.5%/42.3%/23.0% | 52.7%/56.4%/30.6% | 36.2%/48.3% | 44.4%/59.3% | 1 |
| dllm | ruler4k | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_triton_M3_R2_A8_current_output | P0 | qualified 2/2 | 60/60/120 | 37.1%/39.6%/21.7% | 50.2%/53.7%/29.3% | 33.3%/45.3% | 42.7%/57.5% | 1 |
| dllm | ruler4k | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P1_B_A8_matched | P1 | qualified 2/2 | 60/0/180 | 47.4%/50.0%/32.0% | 63.3%/66.7%/42.6% | 46.3%/61.8% | 49.1%/65.4% | 1 |
| dllm | ruler4k | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P1_M1_R1_A8_current_output | P1 | qualified 2/2 | 60/180/0 | 45.4%/47.9%/30.7% | 61.2%/64.6%/41.4% | 43.9%/59.3% | 47.7%/64.0% | 1 |
| dllm | ruler4k | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P1_M3_R2_A8_current_output | P1 | qualified 2/2 | 60/60/120 | 45.1%/47.5%/30.8% | 60.9%/64.2%/41.4% | 43.6%/59.0% | 47.4%/63.8% | 1 |
| dllm | ruler4k | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P1_M3_R3_A8_current_output | P1 | qualified 2/2 | 60/60/120 | 46.6%/49.1%/31.4% | 62.4%/65.8%/42.0% | 45.3%/60.8% | 48.5%/64.8% | 1 |
| dllm | ruler4k | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_T_scope | P0 | qualified 2/2 | 240/0/0 | 0.0%/0.0%/0.0% | 51.6%/54.3%/35.5% | 0.0%/47.1% | 0.0%/58.2% | 1 |
| dllm | ruler4k | ALL_NATIVE_LEGAL | G75L30_nativeQ128 | P0 | qualified 2/2 | 120/0/120 | 20.1%/23.4%/0.0% | 30.1%/35.2%/0.0% | 10.0%/15.0% | 35.1%/52.6% | 1 |
| dllm | ruler4k | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_D_matched | P0 | qualified 2/2 | 40/0/0 | 0.0%/0.0%/0.0% | 0.0%/0.0%/0.0% | N/A/N/A | 0.0%/0.0% | 1 |
| dllm | ruler4k | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_B_A8_matched | P0 | qualified 2/2 | 10/0/30 | 45.0%/47.5%/5.4% | 59.9%/63.3%/7.2% | N/A/N/A | 45.0%/59.9% | 1 |
| dllm | ruler4k | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_M1_R1_A8_current_output | P0 | qualified 2/2 | 10/30/0 | 43.3%/45.7%/5.1% | 58.3%/61.6%/6.9% | N/A/N/A | 43.3%/58.3% | 1 |
| dllm | ruler4k | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_M3_R2_A8_current_output | P0 | qualified 2/2 | 10/10/20 | 43.1%/45.5%/5.3% | 58.1%/61.3%/7.1% | N/A/N/A | 43.1%/58.1% | 1 |
| dllm | ruler4k | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_M3_R3_A8_current_output | P0 | qualified 2/2 | 10/10/20 | 44.2%/46.7%/5.1% | 59.2%/62.5%/6.9% | N/A/N/A | 44.2%/59.2% | 1 |
| dllm | ruler4k | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_triton_B_A8_matched | P0 | qualified 2/2 | 10/0/30 | 45.0%/47.5%/5.4% | 59.9%/63.3%/7.2% | N/A/N/A | 45.0%/59.9% | 1 |
| dllm | ruler4k | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_triton_M3_R2_A8_current_output | P0 | qualified 2/2 | 10/10/20 | 43.1%/45.5%/5.3% | 58.1%/61.3%/7.1% | N/A/N/A | 43.1%/58.1% | 1 |
| dllm | ruler4k | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P1_B_A8_matched | P1 | qualified 2/2 | 10/0/30 | 49.6%/52.2%/7.7% | 66.1%/69.6%/10.2% | N/A/N/A | 49.6%/66.1% | 1 |
| dllm | ruler4k | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P1_M1_R1_A8_current_output | P1 | qualified 2/2 | 10/30/0 | 48.1%/50.7%/7.5% | 64.7%/68.1%/10.0% | N/A/N/A | 48.1%/64.7% | 1 |
| dllm | ruler4k | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P1_M3_R2_A8_current_output | P1 | qualified 2/2 | 10/10/20 | 48.0%/50.5%/7.7% | 64.5%/67.9%/10.3% | N/A/N/A | 48.0%/64.5% | 1 |
| dllm | ruler4k | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P1_M3_R3_A8_current_output | P1 | qualified 2/2 | 10/10/20 | 48.9%/51.6%/7.4% | 65.5%/69.0%/9.9% | N/A/N/A | 48.9%/65.5% | 1 |
| dllm | ruler4k | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_T_scope | P0 | qualified 2/2 | 40/0/0 | 0.0%/0.0%/0.0% | 59.0%/62.2%/7.4% | N/A/N/A | 0.0%/59.0% | 1 |
| dllm | ruler4k | NATIVE | D_native | P0 | unavailable 0/2 | 0/0/0 | N/A/N/A/N/A | N/A/N/A/N/A | N/A/N/A | N/A/N/A | 1 |
| mpk | aime26 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_D_matched | P0 | qualified 2/2 | 240/0/0 | 0.0%/0.0%/0.0% | 0.0%/0.0%/0.0% | 0.0%/0.0% | 0.0%/0.0% | 2 |
| mpk | aime26 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_B_A8_matched | P0 | qualified 2/2 | 60/0/180 | 28.2%/29.2%/25.6% | 37.6%/38.9%/34.2% | 32.2%/43.0% | 16.4%/21.8% | 2 |
| mpk | aime26 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_M1_R1_A8_current_output | P0 | qualified 2/2 | 60/180/0 | 24.1%/25.3%/20.8% | 33.5%/35.1%/29.3% | 27.5%/38.3% | 13.8%/19.2% | 2 |
| mpk | aime26 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_M3_R2_A8_current_output | P0 | qualified 2/2 | 60/60/120 | 24.9%/26.1%/21.9% | 34.3%/35.8%/30.5% | 28.6%/39.3% | 14.2%/19.6% | 2 |
| mpk | aime26 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_M3_R3_A8_current_output | P0 | qualified 2/2 | 60/60/120 | 25.8%/26.9%/22.8% | 35.2%/36.6%/31.4% | 29.5%/40.2% | 14.9%/20.4% | 2 |
| mpk | aime26 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_triton_B_A8_matched | P0 | qualified 2/2 | 60/0/180 | 28.2%/29.2%/25.6% | 37.6%/38.9%/34.2% | 32.2%/43.0% | 16.4%/21.8% | 2 |
| mpk | aime26 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_triton_M3_R2_A8_current_output | P0 | qualified 2/2 | 60/60/120 | 24.9%/26.1%/22.0% | 34.3%/35.8%/30.5% | 28.6%/39.3% | 14.2%/19.6% | 2 |
| mpk | aime26 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P1_B_A8_matched | P1 | qualified 2/2 | 60/0/180 | 38.2%/39.4%/35.0% | 50.9%/52.5%/46.7% | 43.7%/58.2% | 21.9%/29.1% | 2 |
| mpk | aime26 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P1_M1_R1_A8_current_output | P1 | qualified 2/2 | 60/180/0 | 33.7%/35.1%/29.9% | 46.4%/48.2%/41.6% | 38.7%/53.2% | 18.9%/26.2% | 2 |
| mpk | aime26 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P1_M3_R2_A8_current_output | P1 | qualified 2/2 | 60/60/120 | 34.8%/36.0%/31.4% | 47.5%/49.2%/43.1% | 39.9%/54.5% | 19.5%/26.8% | 2 |
| mpk | aime26 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P1_M3_R3_A8_current_output | P1 | qualified 2/2 | 60/60/120 | 35.4%/36.8%/31.8% | 48.1%/49.9%/43.5% | 40.6%/55.2% | 20.2%/27.4% | 2 |
| mpk | aime26 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_T_scope | P0 | qualified 2/2 | 240/0/0 | 0.0%/0.0%/0.0% | 33.9%/35.8%/29.0% | 0.0%/39.0% | 0.0%/18.8% | 2 |
| mpk | aime26 | ALL_NATIVE_LEGAL | G75L30_nativeQ128 | P0 | qualified 2/2 | 120/0/120 | 12.2%/16.8%/0.0% | 18.3%/25.2%/0.0% | 7.6%/11.4% | 25.7%/38.5% | 2 |
| mpk | aime26 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_D_matched | P0 | qualified 2/2 | 40/0/0 | 0.0%/0.0%/0.0% | 0.0%/0.0%/0.0% | N/A/N/A | 0.0%/0.0% | 2 |
| mpk | aime26 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_B_A8_matched | P0 | qualified 2/2 | 10/0/30 | 16.8%/19.1%/6.1% | 22.4%/25.5%/8.1% | N/A/N/A | 16.8%/22.4% | 2 |
| mpk | aime26 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_M1_R1_A8_current_output | P0 | qualified 2/2 | 10/30/0 | 14.4%/16.6%/4.6% | 20.0%/23.0%/6.6% | N/A/N/A | 14.4%/20.0% | 2 |
| mpk | aime26 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_M3_R2_A8_current_output | P0 | qualified 2/2 | 10/10/20 | 14.7%/16.9%/4.9% | 20.3%/23.3%/6.9% | N/A/N/A | 14.7%/20.3% | 2 |
| mpk | aime26 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_M3_R3_A8_current_output | P0 | qualified 2/2 | 10/10/20 | 15.5%/17.7%/5.4% | 21.1%/24.1%/7.5% | N/A/N/A | 15.5%/21.1% | 2 |
| mpk | aime26 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_triton_B_A8_matched | P0 | qualified 2/2 | 10/0/30 | 16.8%/19.1%/6.1% | 22.4%/25.5%/8.1% | N/A/N/A | 16.8%/22.4% | 2 |
| mpk | aime26 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_triton_M3_R2_A8_current_output | P0 | qualified 2/2 | 10/10/20 | 14.7%/16.9%/4.9% | 20.3%/23.3%/6.9% | N/A/N/A | 14.7%/20.3% | 2 |
| mpk | aime26 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P1_B_A8_matched | P1 | qualified 2/2 | 10/0/30 | 22.7%/25.6%/9.5% | 30.3%/34.2%/12.7% | N/A/N/A | 22.7%/30.3% | 2 |
| mpk | aime26 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P1_M1_R1_A8_current_output | P1 | qualified 2/2 | 10/30/0 | 19.9%/22.8%/6.9% | 27.5%/31.3%/10.1% | N/A/N/A | 19.9%/27.5% | 2 |
| mpk | aime26 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P1_M3_R2_A8_current_output | P1 | qualified 2/2 | 10/10/20 | 20.5%/23.3%/7.7% | 28.1%/31.9%/10.9% | N/A/N/A | 20.5%/28.1% | 2 |
| mpk | aime26 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P1_M3_R3_A8_current_output | P1 | qualified 2/2 | 10/10/20 | 21.1%/23.9%/8.2% | 28.7%/32.5%/11.4% | N/A/N/A | 21.1%/28.7% | 2 |
| mpk | aime26 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_T_scope | P0 | qualified 2/2 | 40/0/0 | 0.0%/0.0%/0.0% | 19.2%/22.2%/5.9% | N/A/N/A | 0.0%/19.2% | 2 |
| mpk | aime26 | NATIVE | D_native | P0 | unavailable 0/2 | 0/0/0 | N/A/N/A/N/A | N/A/N/A/N/A | N/A/N/A | N/A/N/A | 2 |
| mpk | longbench_v2 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_D_matched | P0 | qualified 2/2 | 240/0/0 | 0.0%/0.0%/0.0% | 0.0%/0.0%/0.0% | 0.0%/0.0% | 0.0%/0.0% | 2 |
| mpk | longbench_v2 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_B_A8_matched | P0 | qualified 2/2 | 60/0/180 | 47.3%/48.6%/29.4% | 63.0%/64.8%/39.2% | 35.0%/46.7% | 52.4%/69.9% | 2 |
| mpk | longbench_v2 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_M1_R1_A8_current_output | P0 | qualified 2/2 | 60/180/0 | 43.6%/44.9%/26.8% | 59.4%/61.1%/36.6% | 30.4%/42.1% | 49.2%/66.7% | 2 |
| mpk | longbench_v2 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_M3_R2_A8_current_output | P0 | qualified 2/2 | 60/60/120 | 44.6%/45.9%/27.9% | 60.4%/62.1%/37.7% | 31.5%/43.1% | 50.2%/67.6% | 2 |
| mpk | longbench_v2 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_M3_R3_A8_current_output | P0 | qualified 2/2 | 60/60/120 | 45.0%/46.3%/27.8% | 60.7%/62.5%/37.6% | 32.2%/43.9% | 50.3%/67.8% | 2 |
| mpk | longbench_v2 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_triton_B_A8_matched | P0 | qualified 2/2 | 60/0/180 | 47.3%/48.6%/29.4% | 63.0%/64.8%/39.2% | 35.0%/46.7% | 52.4%/69.9% | 2 |
| mpk | longbench_v2 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_triton_M3_R2_A8_current_output | P0 | qualified 2/2 | 60/60/120 | 44.6%/45.9%/27.9% | 60.4%/62.1%/37.7% | 31.5%/43.1% | 50.2%/67.6% | 2 |
| mpk | longbench_v2 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P1_B_A8_matched | P1 | qualified 2/2 | 60/0/180 | 54.2%/55.4%/37.6% | 72.2%/73.9%/50.2% | 46.6%/62.1% | 57.4%/76.5% | 2 |
| mpk | longbench_v2 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P1_M1_R1_A8_current_output | P1 | qualified 2/2 | 60/180/0 | 50.8%/52.1%/35.0% | 68.9%/70.5%/47.6% | 41.7%/57.3% | 54.7%/73.8% | 2 |
| mpk | longbench_v2 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P1_M3_R2_A8_current_output | P1 | qualified 2/2 | 60/60/120 | 51.7%/52.9%/36.1% | 69.8%/71.4%/48.7% | 42.8%/58.4% | 55.5%/74.6% | 2 |
| mpk | longbench_v2 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P1_M3_R3_A8_current_output | P1 | qualified 2/2 | 60/60/120 | 52.1%/53.3%/35.9% | 70.1%/71.8%/48.5% | 43.6%/59.2% | 55.6%/74.7% | 2 |
| mpk | longbench_v2 | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_T_scope | P0 | qualified 2/2 | 240/0/0 | 0.0%/0.0%/0.0% | 58.6%/60.5%/34.6% | 0.0%/41.5% | 0.0%/65.8% | 2 |
| mpk | longbench_v2 | ALL_NATIVE_LEGAL | G75L30_nativeQ128 | P0 | qualified 2/2 | 120/0/120 | 28.7%/30.9%/0.0% | 43.0%/46.3%/0.0% | 9.8%/14.7% | 36.6%/54.9% | 2 |
| mpk | longbench_v2 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_D_matched | P0 | qualified 2/2 | 40/0/0 | 0.0%/0.0%/0.0% | 0.0%/0.0%/0.0% | N/A/N/A | 0.0%/0.0% | 2 |
| mpk | longbench_v2 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_B_A8_matched | P0 | qualified 2/2 | 10/0/30 | 52.7%/53.4%/10.6% | 70.2%/71.2%/14.2% | N/A/N/A | 52.7%/70.2% | 2 |
| mpk | longbench_v2 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_M1_R1_A8_current_output | P0 | qualified 2/2 | 10/30/0 | 49.4%/50.1%/9.1% | 67.0%/67.9%/12.6% | N/A/N/A | 49.4%/67.0% | 2 |
| mpk | longbench_v2 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_M3_R2_A8_current_output | P0 | qualified 2/2 | 10/10/20 | 50.4%/51.1%/9.5% | 67.9%/68.9%/13.0% | N/A/N/A | 50.4%/67.9% | 2 |
| mpk | longbench_v2 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_M3_R3_A8_current_output | P0 | qualified 2/2 | 10/10/20 | 50.6%/51.3%/9.9% | 68.1%/69.1%/13.4% | N/A/N/A | 50.6%/68.1% | 2 |
| mpk | longbench_v2 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_triton_B_A8_matched | P0 | qualified 2/2 | 10/0/30 | 52.7%/53.4%/10.6% | 70.2%/71.2%/14.2% | N/A/N/A | 52.7%/70.2% | 2 |
| mpk | longbench_v2 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_triton_M3_R2_A8_current_output | P0 | qualified 2/2 | 10/10/20 | 50.4%/51.1%/9.5% | 67.9%/68.9%/13.1% | N/A/N/A | 50.4%/67.9% | 2 |
| mpk | longbench_v2 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P1_B_A8_matched | P1 | qualified 2/2 | 10/0/30 | 57.3%/58.1%/14.6% | 76.5%/77.4%/19.5% | N/A/N/A | 57.3%/76.5% | 2 |
| mpk | longbench_v2 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P1_M1_R1_A8_current_output | P1 | qualified 2/2 | 10/30/0 | 54.5%/55.3%/12.1% | 73.7%/74.6%/17.0% | N/A/N/A | 54.5%/73.7% | 2 |
| mpk | longbench_v2 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P1_M3_R2_A8_current_output | P1 | qualified 2/2 | 10/10/20 | 55.4%/56.1%/13.0% | 74.5%/75.4%/17.9% | N/A/N/A | 55.4%/74.5% | 2 |
| mpk | longbench_v2 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P1_M3_R3_A8_current_output | P1 | qualified 2/2 | 10/10/20 | 55.5%/56.3%/13.4% | 74.7%/75.6%/18.2% | N/A/N/A | 55.5%/74.7% | 2 |
| mpk | longbench_v2 | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_T_scope | P0 | qualified 2/2 | 40/0/0 | 0.0%/0.0%/0.0% | 65.9%/66.8%/13.2% | N/A/N/A | 0.0%/65.9% | 2 |
| mpk | longbench_v2 | NATIVE | D_native | P0 | unavailable 0/2 | 0/0/0 | N/A/N/A/N/A | N/A/N/A/N/A | N/A/N/A | N/A/N/A | 2 |
| mpk | ruler4k | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_D_matched | P0 | qualified 2/2 | 240/0/0 | 0.0%/0.0%/0.0% | 0.0%/0.0%/0.0% | 0.0%/0.0% | 0.0%/0.0% | 1 |
| mpk | ruler4k | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_B_A8_matched | P0 | qualified 2/2 | 60/0/180 | 34.8%/36.4%/25.5% | 46.4%/48.5%/34.1% | 35.4%/47.2% | 33.9%/45.2% | 1 |
| mpk | ruler4k | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_M1_R1_A8_current_output | P0 | qualified 2/2 | 60/180/0 | 32.3%/33.7%/24.4% | 43.9%/45.8%/33.0% | 33.0%/44.9% | 31.2%/42.5% | 1 |
| mpk | ruler4k | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_M3_R2_A8_current_output | P0 | qualified 2/2 | 60/60/120 | 31.8%/33.1%/24.2% | 43.4%/45.2%/32.7% | 32.6%/44.4% | 30.7%/42.0% | 1 |
| mpk | ruler4k | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_M3_R3_A8_current_output | P0 | qualified 2/2 | 60/60/120 | 33.8%/35.3%/25.2% | 45.4%/47.4%/33.7% | 34.5%/46.3% | 32.9%/44.1% | 1 |
| mpk | ruler4k | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_triton_B_A8_matched | P0 | qualified 2/2 | 60/0/180 | 34.8%/36.4%/25.5% | 46.4%/48.5%/34.1% | 35.4%/47.2% | 33.9%/45.2% | 1 |
| mpk | ruler4k | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P0_triton_M3_R2_A8_current_output | P0 | qualified 2/2 | 60/60/120 | 31.8%/33.1%/24.2% | 43.4%/45.2%/32.8% | 32.6%/44.4% | 30.7%/42.0% | 1 |
| mpk | ruler4k | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P1_B_A8_matched | P1 | qualified 2/2 | 60/0/180 | 43.8%/45.7%/32.8% | 58.4%/60.9%/43.7% | 46.0%/61.3% | 40.4%/53.9% | 1 |
| mpk | ruler4k | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P1_M1_R1_A8_current_output | P1 | qualified 2/2 | 60/180/0 | 41.4%/42.8%/32.9% | 56.0%/58.0%/43.9% | 43.7%/59.0% | 37.9%/51.4% | 1 |
| mpk | ruler4k | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P1_M3_R2_A8_current_output | P1 | qualified 2/2 | 60/60/120 | 40.9%/42.3%/32.5% | 55.5%/57.5%/43.4% | 43.2%/58.5% | 37.3%/50.8% | 1 |
| mpk | ruler4k | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_P1_M3_R3_A8_current_output | P1 | qualified 2/2 | 60/60/120 | 42.8%/44.5%/32.9% | 57.4%/59.7%/43.8% | 45.1%/60.4% | 39.4%/52.9% | 1 |
| mpk | ruler4k | ALL_NATIVE_LEGAL | ALL_NATIVE_LEGAL_T_scope | P0 | qualified 2/2 | 240/0/0 | 0.0%/0.0%/0.0% | 47.9%/49.2%/40.4% | 0.0%/46.8% | 0.0%/49.6% | 1 |
| mpk | ruler4k | ALL_NATIVE_LEGAL | G75L30_nativeQ128 | P0 | qualified 2/2 | 120/0/120 | 19.6%/22.9%/0.0% | 29.4%/34.4%/0.0% | 10.0%/15.0% | 34.1%/51.2% | 1 |
| mpk | ruler4k | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_D_matched | P0 | qualified 2/2 | 40/0/0 | 0.0%/0.0%/0.0% | 0.0%/0.0%/0.0% | N/A/N/A | 0.0%/0.0% | 1 |
| mpk | ruler4k | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_B_A8_matched | P0 | qualified 2/2 | 10/0/30 | 34.6%/36.5%/4.3% | 46.1%/48.7%/5.7% | N/A/N/A | 34.6%/46.1% | 1 |
| mpk | ruler4k | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_M1_R1_A8_current_output | P0 | qualified 2/2 | 10/30/0 | 31.7%/33.5%/3.9% | 43.2%/45.7%/5.3% | N/A/N/A | 31.7%/43.2% | 1 |
| mpk | ruler4k | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_M3_R2_A8_current_output | P0 | qualified 2/2 | 10/10/20 | 31.1%/32.9%/3.8% | 42.7%/45.1%/5.2% | N/A/N/A | 31.1%/42.7% | 1 |
| mpk | ruler4k | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_M3_R3_A8_current_output | P0 | qualified 2/2 | 10/10/20 | 33.4%/35.3%/4.1% | 44.9%/47.5%/5.5% | N/A/N/A | 33.4%/44.9% | 1 |
| mpk | ruler4k | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_triton_B_A8_matched | P0 | qualified 2/2 | 10/0/30 | 34.6%/36.5%/4.3% | 46.1%/48.7%/5.7% | N/A/N/A | 34.6%/46.1% | 1 |
| mpk | ruler4k | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P0_triton_M3_R2_A8_current_output | P0 | qualified 2/2 | 10/10/20 | 31.1%/32.9%/3.8% | 42.7%/45.1%/5.2% | N/A/N/A | 31.1%/42.7% | 1 |
| mpk | ruler4k | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P1_B_A8_matched | P1 | qualified 2/2 | 10/0/30 | 41.7%/44.0%/6.4% | 55.7%/58.7%/8.5% | N/A/N/A | 41.7%/55.7% | 1 |
| mpk | ruler4k | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P1_M1_R1_A8_current_output | P1 | qualified 2/2 | 10/30/0 | 39.2%/41.3%/6.3% | 53.1%/56.0%/8.4% | N/A/N/A | 39.2%/53.1% | 1 |
| mpk | ruler4k | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P1_M3_R2_A8_current_output | P1 | qualified 2/2 | 10/10/20 | 38.7%/40.8%/5.8% | 52.6%/55.5%/8.0% | N/A/N/A | 38.7%/52.6% | 1 |
| mpk | ruler4k | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_P1_M3_R3_A8_current_output | P1 | qualified 2/2 | 10/10/20 | 40.7%/42.9%/6.3% | 54.6%/57.6%/8.4% | N/A/N/A | 40.7%/54.6% | 1 |
| mpk | ruler4k | GLOBAL_ONLY_NATIVE_LOCAL | GLOBAL_ONLY_NATIVE_LOCAL_T_scope | P0 | qualified 2/2 | 40/0/0 | 0.0%/0.0%/0.0% | 50.0%/52.8%/7.3% | N/A/N/A | 0.0%/50.0% | 1 |
| mpk | ruler4k | NATIVE | D_native | P0 | unavailable 0/2 | 0/0/0 | N/A/N/A/N/A | N/A/N/A/N/A | N/A/N/A | N/A/N/A | 1 |

The JSON preserves raw eligible/skipped/executed pair and multiply-accumulate counts by LOCAL/GLOBAL and whole/static-prefix/current-canvas. A/D/H counts are attention-layer calls, not decoder calls. N/A twins carry no physical-work claim. This report makes no policy selection or answer-quality claim.
