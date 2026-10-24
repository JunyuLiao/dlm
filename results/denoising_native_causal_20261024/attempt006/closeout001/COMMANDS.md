# Exact experiment and verification commands

Working directory: /home/exouser/ljy/dlm. Every GPU argv and environment override, including failed attempts, is preserved in command_receipts.json. Raw worker logs remain private.

## GPU workers

Receipt: results/denoising_native_causal_20261024/attempt001/launch_receipt.json

    timeout --signal=TERM --kill-after=30s 900 /home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_native_causal_20261024/attempt001/code/worker.py --root /home/exouser/ljy/dlm/results/denoising_native_causal_20261024/attempt001

Receipt: results/denoising_native_causal_20261024/attempt003/launch_receipt.json

    timeout --signal=TERM --kill-after=30s 900 /home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_native_causal_20261024/attempt003/code/worker.py --root /home/exouser/ljy/dlm/results/denoising_native_causal_20261024/attempt003

Receipt: results/denoising_native_causal_20261024/attempt004/launch_receipt.json

    timeout --signal=TERM --kill-after=30s 900 /home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_native_causal_20261024/attempt004/code/worker.py --root /home/exouser/ljy/dlm/results/denoising_native_causal_20261024/attempt004

Receipt: results/denoising_native_causal_20261024/attempt005/smoke_launch.json

    timeout --signal=TERM --kill-after=30s 1800 /home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_native_causal_20261024/attempt005/code/worker.py --root /home/exouser/ljy/dlm/results/denoising_native_causal_20261024/attempt005 --group smoke

Receipt: results/denoising_native_causal_20261024/attempt006/long_launch.json

    timeout --signal=TERM --kill-after=30s 1800 /home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_native_causal_20261024/attempt006/code/worker.py --root /home/exouser/ljy/dlm/results/denoising_native_causal_20261024/attempt006 --group long

Receipt: results/denoising_native_causal_20261024/attempt006/short_launch.json

    timeout --signal=TERM --kill-after=30s 1800 /home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_native_causal_20261024/attempt006/code/worker.py --root /home/exouser/ljy/dlm/results/denoising_native_causal_20261024/attempt006 --group short

Receipt: results/denoising_native_causal_20261024/attempt006/smoke_launch.json

    timeout --signal=TERM --kill-after=30s 1800 /home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_native_causal_20261024/attempt006/code/worker.py --root /home/exouser/ljy/dlm/results/denoising_native_causal_20261024/attempt006 --group smoke

## Corrected launches

    python results/denoising_native_causal_20261024/attempt006/launch.py --group smoke
    python results/denoising_native_causal_20261024/attempt006/launch.py --group short
    python results/denoising_native_causal_20261024/attempt006/launch.py --group long

## CPU and analysis

    PYTHONPATH=src:. /home/exouser/dyh/v31_env_20261005/bin/python -m pytest -q tests/test_trajectory_dynamics.py tests/test_dynamics_screen_protocol.py results/denoising_native_causal_20261024/attempt004/tests/test_state_restore.py results/denoising_native_causal_20261024/attempt006/tests/test_support_oracle.py --junitxml=results/denoising_native_causal_20261024/attempt006/tests/cpu001.xml

    PYTHONPATH=src:. /home/exouser/dyh/v31_env_20261005/bin/python -m pytest -q tests/test_trajectory_dynamics.py tests/test_dynamics_screen_protocol.py results/denoising_native_causal_20261024/attempt004/tests/test_state_restore.py results/denoising_native_causal_20261024/attempt006/tests/test_support_oracle.py --junitxml=results/denoising_native_causal_20261024/attempt006/tests/cpu002.xml

    PYTHONPATH=src:. /home/exouser/dyh/v31_env_20261005/bin/python results/denoising_native_causal_20261024/attempt006/analysis001/analyze.py
    python -m py_compile results/denoising_native_causal_20261024/attempt006/analysis001/analyze.py
    python results/denoising_native_causal_20261024/attempt006/closeout001/verify.py
    git diff --check

19 focused tests pass twice. Attempt004 earlier passed16; attempt005 passed2 bank tests but failed its GPU smoke. Attempt002 is CPU-only and launched no generation. Failed attempts remain preserved.

## Read-only audits

    git branch --show-current
    git log -6 --oneline
    git status --short --untracked-files=no
    nvidia-smi
    nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader

Authoritative GPU checks and generation used authorized host execution because the restricted sandbox cannot reliably initialize NVML. No other workload was terminated.

## Files and scope

changed_files.txt and artifact_inventory.json enumerate the exact campaign publication files and documentation. Private inputs, tensor captures, logs, caches and runtime overlays are excluded. Pre-existing AGENTS.md edits were not changed by this continuation. Existing HANDOFF.md and DECISIONS.md text is preserved. STATE.json remains unchanged. The ledger addition closes only the source-bound diagnostic leaves, not the acceleration study.
