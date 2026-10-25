# Stage A execution and reproduction

Run from `/home/exouser/ljy/dlm`. The pinned interpreter and local model/scorer
paths are in the ignored `attempt002/private/runtime_paths.json`. Public launch
receipts use `<pinned_python>` to avoid publishing private host paths; resolving
that one token gives the exact executed worker argument vector. Source and
protocol hashes, process identities, start times and exit/observation receipts
are retained with each launch.

Resolve the interpreter for the commands below:

```bash
PY=$(python -c 'import json; print(json.load(open("results/denoising_mechanism_stage_a_20261025/attempt002/private/runtime_paths.json"))["python"])')
```

The pre-generation preparation entrypoints were:

```bash
"$PY" results/denoising_mechanism_stage_a_20261025/attempt001/code/prior_addendum.py
"$PY" results/denoising_mechanism_stage_a_20261025/attempt001/code/prepare.py
"$PY" results/denoising_mechanism_stage_a_20261025/attempt001/freeze.py
"$PY" results/denoising_mechanism_stage_a_20261025/attempt002/freeze.py
"$PY" results/denoising_mechanism_stage_a_20261025/phase_accounting_addendum.py
```

These writers are immutable-attempt writers. They must be copied and rebound to
a **new** attempt before reproduction, rather than run over existing artifacts.
Attempt001 generated no GPU data. Revision002 changed diagnostic GEMM accounting
before launch; its scientific choices and manifest stayed the same.

The focused prelaunch test invocation is reproducible as:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. "$PY" -m pytest -q \
  results/denoising_mechanism_stage_a_20261025/attempt002/tests/test_operators.py \
  tests/test_trajectory_dynamics.py tests/test_dynamics_screen_protocol.py
```

The retained `attempt002/tests/prelaunch001.xml` records all 19 passing tests.
Model verification read and hashed all 16 model/config/tokenizer files; runtime
verification checked seven installed source files. The verification receipts
record those exact identities without publishing private paths.

GPU execution began with:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" results/denoising_mechanism_stage_a_20261025/attempt002/launch.py
```

This launched the following frozen worker argument vectors sequentially:

```bash
"$PY" results/denoising_mechanism_stage_a_20261025/attempt002/code/worker.py --root /home/exouser/ljy/dlm/results/denoising_mechanism_stage_a_20261025/attempt002 --job smoke
"$PY" results/denoising_mechanism_stage_a_20261025/attempt002/code/worker.py --root /home/exouser/ljy/dlm/results/denoising_mechanism_stage_a_20261025/attempt002 --job full
"$PY" results/denoising_mechanism_stage_a_20261025/attempt002/code/worker.py --root /home/exouser/ljy/dlm/results/denoising_mechanism_stage_a_20261025/attempt002 --job a1
```

The interactive interruption ended the launcher while FULL was running. The
FULL worker survived. A detached recovery controller observed its original
PID/start time and completion marker, then launched only the unstarted A1 and
analysis jobs using the same frozen environment and sources:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" results/denoising_mechanism_stage_a_20261025/recover_controller.py
CUDA_VISIBLE_DEVICES='' "$PY" results/denoising_mechanism_stage_a_20261025/attempt002/code/analyze.py
```

No worker was killed or relaunched. The original launch receipts are unchanged;
`controller_recovery001/` discloses the unavailable FULL process wait status and
the measured worker duration plus the 15-second-polled allocation upper bound.
No campaign resource ceiling or hang timeout is applied.

Derived closeout (after the frozen analysis completes):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$PY" results/denoising_mechanism_stage_a_20261025/report_stage_a.py
```

The frozen launch/controller code is the exact source of environment variables,
cache binding, sampler/model configuration, warmups, job order and commands.
Private inputs are required to reproduce native requests. Public artifacts
contain IDs, hashes, numeric diagnostics and receipts; no prompts or generations.
