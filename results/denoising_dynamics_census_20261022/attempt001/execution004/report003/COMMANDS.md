# Commands and execution receipts

Working directory: `/home/exouser/ljy/dlm`. Arguments below are exact worker argv from launch receipts; each receipt also records environment and timestamps. Shell wrappers checked live GPU clients before launch and preserved logs/exit status.

## census_long_launch.json

```bash
timeout --signal=TERM --kill-after=30s 1800 /home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_dynamics_census_20261022/attempt001/execution004/code/worker.py --root /home/exouser/ljy/dlm/results/denoising_dynamics_census_20261022/attempt001/execution004 --stage census --group long
```

- `PYTHONPATH=/home/exouser/ljy/dlm/src:/home/exouser/ljy/dlm`
- `HF_HUB_OFFLINE=1`
- `TORCHINDUCTOR_DETERMINISTIC=1`
- `VLLM_USE_V2_MODEL_RUNNER=1`
- `VLLM_CACHE_ROOT=/home/exouser/ljy/dlm/results/denoising_dynamics_census_20261022/attempt001/execution004/cache/vllm`
- `TRITON_CACHE_DIR=/home/exouser/ljy/dlm/results/denoising_dynamics_census_20261022/attempt001/execution004/cache/triton`
- `TORCHINDUCTOR_CACHE_DIR=/home/exouser/ljy/dlm/results/denoising_dynamics_census_20261022/attempt001/execution004/cache/inductor`
- `CUDA_CACHE_PATH=/home/exouser/ljy/dlm/results/denoising_dynamics_census_20261022/attempt001/execution004/cache/cuda`

## census_short_launch.json

```bash
timeout --signal=TERM --kill-after=30s 1800 /home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_dynamics_census_20261022/attempt001/execution004/code/worker.py --root /home/exouser/ljy/dlm/results/denoising_dynamics_census_20261022/attempt001/execution004 --stage census --group short
```

- `PYTHONPATH=/home/exouser/ljy/dlm/src:/home/exouser/ljy/dlm`
- `HF_HUB_OFFLINE=1`
- `TORCHINDUCTOR_DETERMINISTIC=1`
- `VLLM_USE_V2_MODEL_RUNNER=1`
- `VLLM_CACHE_ROOT=/home/exouser/ljy/dlm/results/denoising_dynamics_census_20261022/attempt001/execution004/cache/vllm`
- `TRITON_CACHE_DIR=/home/exouser/ljy/dlm/results/denoising_dynamics_census_20261022/attempt001/execution004/cache/triton`
- `TORCHINDUCTOR_CACHE_DIR=/home/exouser/ljy/dlm/results/denoising_dynamics_census_20261022/attempt001/execution004/cache/inductor`
- `CUDA_CACHE_PATH=/home/exouser/ljy/dlm/results/denoising_dynamics_census_20261022/attempt001/execution004/cache/cuda`

## smoke_launch.json

```bash
timeout --signal=TERM --kill-after=30s 1800 /home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_dynamics_census_20261022/attempt001/execution004/code/worker.py --root /home/exouser/ljy/dlm/results/denoising_dynamics_census_20261022/attempt001/execution004 --stage smoke
```

- `PYTHONPATH=/home/exouser/ljy/dlm/src:/home/exouser/ljy/dlm`
- `HF_HUB_OFFLINE=1`
- `TORCHINDUCTOR_DETERMINISTIC=1`
- `VLLM_USE_V2_MODEL_RUNNER=1`
- `VLLM_CACHE_ROOT=/home/exouser/ljy/dlm/results/denoising_dynamics_census_20261022/attempt001/execution004/cache/vllm`
- `TRITON_CACHE_DIR=/home/exouser/ljy/dlm/results/denoising_dynamics_census_20261022/attempt001/execution004/cache/triton`
- `TORCHINDUCTOR_CACHE_DIR=/home/exouser/ljy/dlm/results/denoising_dynamics_census_20261022/attempt001/execution004/cache/inductor`
- `CUDA_CACHE_PATH=/home/exouser/ljy/dlm/results/denoising_dynamics_census_20261022/attempt001/execution004/cache/cuda`

## operator_diagnostic002/launch_receipt.json

```bash
timeout --signal=TERM --kill-after=30s 900 /home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_dynamics_census_20261022/attempt001/execution004/operator_diagnostic002/code/diagnose.py --device cuda
```


## CPU and official scoring commands

```bash
PYTHONPATH=src:. /home/exouser/dyh/v31_env_20261005/bin/python -m pytest -q tests/test_trajectory_dynamics.py tests/test_dynamics_screen_protocol.py results/denoising_dynamics_census_20261022/attempt001/execution004/tests/test_production_observer.py results/denoising_dynamics_census_20261022/attempt001/execution004/operator_diagnostic002/tests/test_diagnose.py --junitxml=results/denoising_dynamics_census_20261022/attempt001/execution004/operator_diagnostic002/tests/cpu001.xml
PYTHONPATH=src:. /home/exouser/dyh/v31_env_20261005/bin/python results/denoising_dynamics_census_20261022/attempt001/execution004/operator_diagnostic002/code/diagnose.py --freeze
python scripts/v27_humaneval.py sandbox-check
PYTHONPATH=src:. python results/denoising_dynamics_census_20261022/attempt001/execution004/code/score_census.py
python results/denoising_dynamics_census_20261022/attempt001/execution004/code/candidate_readout.py
PYTHONPATH=src:. /home/exouser/dyh/v31_env_20261005/bin/python results/denoising_dynamics_census_20261022/attempt001/execution004/code/report_final.py
```

The scorer host sandbox preflight passes. The same preflight inside the restricted workspace sandbox fails, so code execution used the unchanged official bwrap sandbox on the host. Earlier exact launch argv and failures remain in execution001/002/003 launch/exit receipts. Read-only audits used git branch/log/status, nvidia-smi, source hashing, targeted rg/sed/cat and model-tokenizer checks.
