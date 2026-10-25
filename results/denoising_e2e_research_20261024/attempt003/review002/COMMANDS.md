# Recorded execution commands

The JSON catalog records exact worker/readout argument vectors and environment overrides from receipts.
Do not rerun commands into immutable existing result directories. A new run needs a new source/protocol binding and private inputs.

Receipt: attempt001/qualification001/launch_receipt.json

```bash
timeout --signal=TERM --kill-after=15s 1800 /home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt001/qualification001/code/worker.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt001
```

Receipt: attempt002/launches/short/allkept_layer29/launch_receipt.json

```bash
timeout --signal=TERM --kill-after=15s 14400 /home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt002/code/e2e_worker.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt002 --arm allkept_layer29 --suite short --stage panel
```

Receipt: attempt002/launches/short/dense_piecewise/launch_receipt.json

```bash
timeout --signal=TERM --kill-after=15s 14400 /home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt002/code/e2e_worker.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt002 --arm dense_piecewise --suite short --stage panel
```

Receipt: attempt003/launches/long/allkept_layer29/launch_receipt.json

```bash
/home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt006/code/long_worker.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003 --arm allkept_layer29 --suite long --stage panel
```

Receipt: attempt003/launches/long/current_error_fixed4/launch_receipt.json

```bash
/home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt006/code/long_worker.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003 --arm current_error_fixed4 --suite long --stage panel
```

Receipt: attempt003/launches/long/dense_full_fix51994/launch_receipt.json

```bash
/home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt006/code/long_worker.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003 --arm dense_full_fix51994 --suite long --stage panel
```

Receipt: attempt003/launches/long/dense_piecewise/launch_receipt.json

```bash
/home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt006/code/long_worker.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003 --arm dense_piecewise --suite long --stage panel
```

Receipt: attempt003/launches/long/margin_guard_weighted_bias/launch_receipt.json

```bash
/home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt006/code/long_worker.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003 --arm margin_guard_weighted_bias --suite long --stage panel
```

Receipt: attempt003/launches/long/response_guard_qdirection/launch_receipt.json

```bash
/home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt006/code/long_worker.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003 --arm response_guard_qdirection --suite long --stage panel
```

Receipt: attempt003/launches/long/sham_fixed4/launch_receipt.json

```bash
/home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt006/code/long_worker.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003 --arm sham_fixed4 --suite long --stage panel
```

Receipt: attempt003/launches/long/v31_optimized_separate_passes/audit/launch_receipt.json

```bash
/home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt005/code/v31_worker.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003 --arm v31_optimized --suite long --stage panel --pass-name audit
```

Receipt: attempt003/launches/long/v31_optimized_separate_passes/clean/launch_receipt.json

```bash
/home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt005/code/v31_worker.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003 --arm v31_optimized --suite long --stage panel --pass-name clean
```

Receipt: attempt003/launches/long/validity_reuse_mass_certificate/launch_receipt.json

```bash
/home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt006/code/long_worker.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003 --arm validity_reuse_mass_certificate --suite long --stage panel
```

Receipt: attempt003/launches/short/allkept_layer29/launch_receipt.json

```bash
timeout --signal=TERM --kill-after=15s 14400 /home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003/code/e2e_worker.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003 --arm allkept_layer29 --suite short --stage panel
```

Receipt: attempt003/launches/short/current_error_fixed4/launch_receipt.json

```bash
timeout --signal=TERM --kill-after=15s 14400 /home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003/code/e2e_worker.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003 --arm current_error_fixed4 --suite short --stage panel
```

Receipt: attempt003/launches/short/dense_full_fix51994/launch_receipt.json

```bash
timeout --signal=TERM --kill-after=15s 14400 /home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003/code/e2e_worker.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003 --arm dense_full_fix51994 --suite short --stage panel
```

Receipt: attempt003/launches/short/dense_piecewise/launch_receipt.json

```bash
timeout --signal=TERM --kill-after=15s 14400 /home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003/code/e2e_worker.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003 --arm dense_piecewise --suite short --stage panel
```

Receipt: attempt003/launches/short/margin_guard_weighted_bias/launch_receipt.json

```bash
timeout --signal=TERM --kill-after=15s 14400 /home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003/code/e2e_worker.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003 --arm margin_guard_weighted_bias --suite short --stage panel
```

Receipt: attempt003/launches/short/response_guard_qdirection/launch_receipt.json

```bash
timeout --signal=TERM --kill-after=15s 14400 /home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003/code/e2e_worker.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003 --arm response_guard_qdirection --suite short --stage panel
```

Receipt: attempt003/launches/short/sham_fixed4/launch_receipt.json

```bash
timeout --signal=TERM --kill-after=15s 14400 /home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003/code/e2e_worker.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003 --arm sham_fixed4 --suite short --stage panel
```

Receipt: attempt003/launches/short/v31_optimized/launch_receipt.json

```bash
timeout --signal=TERM --kill-after=15s 14400 /home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003/code/e2e_worker.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003 --arm v31_optimized --suite short --stage panel
```

Receipt: attempt003/launches/short/v31_optimized_separate_passes/audit/launch_receipt.json

```bash
/home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt005/code/v31_worker.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003 --arm v31_optimized --suite short --stage panel --pass-name audit
```

Receipt: attempt003/launches/short/v31_optimized_separate_passes/clean/launch_receipt.json

```bash
/home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt005/code/v31_worker.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003 --arm v31_optimized --suite short --stage panel --pass-name clean
```

Receipt: attempt003/launches/short/validity_reuse_mass_certificate/launch_receipt.json

```bash
timeout --signal=TERM --kill-after=15s 14400 /home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003/code/e2e_worker.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003 --arm validity_reuse_mass_certificate --suite short --stage panel
```

Receipt: attempt009/analysis_receipt.json

```bash
/home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003/analyze_v4.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003
```

Receipt: attempt009/native_bottlenecks_receipt.json

```bash
/home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003/trajectory_bottlenecks.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003
```

Receipt: attempt009/official_score_receipt.json

```bash
/home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003/score_v3.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003
```

Receipt: attempt009/output_diagnostics_receipt.json

```bash
/home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003/output_diagnostics.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003
```

Receipt: attempt009/paired_trajectory_receipt.json

```bash
/home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003/paired_trajectory.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003
```

Receipt: attempt009/plots_receipt.json

```bash
/home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003/plot_trajectories.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003
```

Receipt: attempt009/report_receipt.json

```bash
/home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003/report.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003
```

Receipt: attempt009/support_diversity_receipt.json

```bash
/home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003/support_diversity.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003
```

Receipt: attempt009/v31_long_receipt.json

```bash
/home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt005/launch_v31.py --suite long
```

Receipt: attempt009/verify_receipt.json

```bash
/home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003/verify.py --root /home/exouser/ljy/dlm/results/denoising_e2e_research_20261024/attempt003
```

Focused CPU follow-up (36 passed):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /home/exouser/dyh/v31_env_20261005/bin/python -m pytest -q results/denoising_e2e_research_20261024/attempt003/tests/test_online_selector.py results/denoising_e2e_research_20261024/attempt001/tests/test_pulsehold_inherited.py tests/test_trajectory_dynamics.py tests/test_dynamics_screen_protocol.py results/denoising_e2e_research_20261024/attempt005/tests/test_v31_pass_binding.py --junitxml=results/denoising_e2e_research_20261024/attempt003/tests/final_cpu001/results.xml
```

Supplemental execution-source verification (450 cells, two bindings, seven runtime files passed):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/exouser/dyh/v31_env_20261005/bin/python results/denoising_e2e_research_20261024/attempt003/verify_execution_bindings.py --root results/denoising_e2e_research_20261024/attempt003
```

Completion orchestration: attempt006/launch_long.py then attempt009/finish.py; the latter ran attempt005/launch_v31.py --suite long and the CPU jobs listed above.
The earlier test command with tests/test_online_selector.py failed before collection because that path does not exist; explicit correct paths passed22 tests, then the final suite passed36.
