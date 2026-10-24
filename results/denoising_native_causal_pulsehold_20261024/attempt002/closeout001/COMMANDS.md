# Execution and reproduction commands

Working directory for all commands: `/home/exouser/ljy/dlm`.
This records substantive test, worker, analysis and verification commands.
File reads, patches and intermediate read-only inspections are not a shell-session
transcript. Generation must not be rerun in the immutable attempt directory.
Any replay requires a new named attempt/source freeze and review of this failure.

Repository/hardware checks (read only):

```bash
git branch --show-current
git log -20 --format='%h %s'
git status --short --untracked-files=normal
git diff --numstat
nvidia-smi --query-gpu=name,uuid,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
```

Focused pre-generation test receipt `cpu004.xml`, 24 pass. The closeout rerun
below was executed verbatim and produced `cpu005.xml`, also 24 pass:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. /home/exouser/dyh/v31_env_20261005/bin/python -m pytest -q results/denoising_native_causal_pulsehold_20261024/attempt002/tests/test_pulsehold.py tests/test_trajectory_dynamics.py tests/test_dynamics_screen_protocol.py --junitxml=results/denoising_native_causal_pulsehold_20261024/attempt002/tests/cpu005.xml
```

The sole GPU launch used the frozen launcher:

```bash
/home/exouser/dyh/v31_env_20261005/bin/python results/denoising_native_causal_pulsehold_20261024/attempt002/launch.py
```

`launch_receipt.json` is authoritative for its exact worker argv, environment
overrides, timeout/deadline, source checkpoint and fresh idle-GPU checks.
The generated worker command was:

```bash
timeout --signal=TERM --kill-after=15s 7185 /home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_native_causal_pulsehold_20261024/attempt002/code/worker.py --root /home/exouser/ljy/dlm/results/denoising_native_causal_pulsehold_20261024/attempt002
```

It returned1 with `RuntimeError: full-request restore parity`; no retry.
Attempt001 was not GPU-launched. The launcher/worker do not run any other campaign.

First CPU partial reader (failed because it expected a disabled-control trace):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. /home/exouser/dyh/v31_env_20261005/bin/python results/denoising_native_causal_pulsehold_20261024/analyze_partial.py --root results/denoising_native_causal_pulsehold_20261024/attempt002
```

The corrected CPU reader was executed separately, retaining the failed001
directory. Current source explicitly checks that the disabled trace is empty.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. /home/exouser/dyh/v31_env_20261005/bin/python results/denoising_native_causal_pulsehold_20261024/analyze_partial.py --root results/denoising_native_causal_pulsehold_20261024/attempt002 --output-name analysis_partial002
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. /home/exouser/dyh/v31_env_20261005/bin/python results/denoising_native_causal_pulsehold_20261024/check_response_derivative.py --root results/denoising_native_causal_pulsehold_20261024/attempt002
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. /home/exouser/dyh/v31_env_20261005/bin/python results/denoising_native_causal_pulsehold_20261024/verify_partial.py --root results/denoising_native_causal_pulsehold_20261024/attempt002
```

The readers create outputs exclusively and will refuse to overwrite them. To
repeat CPU analysis, use a fresh output name and record a new reader receipt.
The complete-stage templates `analyze_development.py`/`verify_development.py`
were not executed; this failed stage has no complete-stage marker.

Source-preservation commits:

```bash
git commit -m 'Freeze bounded first-canvas pulse hold identification diagnostic'
git commit -m 'Qualify native pre-call phase capture for pulse hold diagnostic'
```

These produced17b8e7ba andc3dd9855 before GPU generation. No push was run. The
publication commit stages only this campaign and our HANDOFF/DECISIONS additions;
the preexisting review/document edits remain in the worktree. Exact changed-file
identity is recoverable with:

```bash
git diff --name-only 7febadb6 HEAD
git status --short --untracked-files=normal
```
