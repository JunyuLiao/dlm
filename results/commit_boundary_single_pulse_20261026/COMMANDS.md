# Commands and reproduction

Work from the repository root. `<pinned_python>` is the interpreter in the
ignored private/runtime_paths.json binding; do not publish host-specific paths.
All exact expanded worker vectors and source/protocol bindings are in launches/
receipts. Native inputs, gold, output text, tensors and unredacted logs remain
private/ignored. Scripts exclusively create outputs: never rerun in a frozen
completed directory; use a new named revision after a failure.

Initial read-only audit: git branch --show-current; git log --oneline -12;
git status --short; nvidia-smi GPU/compute queries; source SHA256 comparisons.
The sandbox could not expose the NVIDIA driver; the same read-only query outside
the sandbox confirmed an idle H10080GB, driver590.48.01,0MiB/no compute apps.
No process was killed. Existing AGENTS/HANDOFF/DECISIONS edits were snapshotted
privately and preserved. No paused worker resumed.

```sh
python results/commit_boundary_single_pulse_20261026/preparation001/setup.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 <pinned_python> results/commit_boundary_single_pulse_20261026/cpu_addendum001/analyze.py
python results/commit_boundary_single_pulse_20261026/preparation001/patch_boundary.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. <pinned_python> -m pytest -q results/commit_boundary_single_pulse_20261026/boundary/attempt001/tests --junitxml=results/commit_boundary_single_pulse_20261026/boundary/attempt001/tests/prelaunch001.xml
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 <pinned_python> results/commit_boundary_single_pulse_20261026/boundary/attempt001/freeze.py
```

Boundary generation source85d2f11d, protocol
9d10674b993bcb55120a6d49c2d95c4dae164ffc0b64e2a2048e6c0f654e4044.
After task-owned git add/commit, bind HEAD/protocol then start once:

```sh
<pinned_python> results/commit_boundary_single_pulse_20261026/boundary/attempt001/launch.py
```

One serial worker runs references then development; actual old supports and
aliases are retained. Allocation includes startup/warmups/cleanup. No timeouts
or forward/hour/cost ceilings. Worker errors preserve receipts and stop.

Panel preparation/CPU verification:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. <pinned_python> results/commit_boundary_single_pulse_20261026/panel/attempt001/code/roster.py
python results/commit_boundary_single_pulse_20261026/preparation001/panel_sources.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. <pinned_python> -m pytest -q results/commit_boundary_single_pulse_20261026/panel/attempt001/tests tests/test_trajectory_dynamics.py tests/test_dynamics_screen_protocol.py --junitxml=results/commit_boundary_single_pulse_20261026/panel/attempt001/tests/prelaunch002.xml
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 <pinned_python> results/commit_boundary_single_pulse_20261026/panel/attempt001/code/verify_inputs.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 <pinned_python> results/commit_boundary_single_pulse_20261026/panel/attempt001/freeze.py
```

Panel sourceb401e7e4, protocold7e0d864ff6acfcd5cacadd738f6e39b5b159ff590fe35be08ec95d08ddbe6e6.
Only after frozen-source binding and the boundary engineering completion:

```sh
<pinned_python> results/commit_boundary_single_pulse_20261026/panel/attempt001/launch.py
```

The controller runs fresh profilingFULL/A at seed42 across the24-question roster,
select.py freezes12questions with baseline-only ranking, then commits that manifest
before any sparse treatment. Seven-arm engineering qualification precedes the
seven-arm panel. All audit/clean comparisons and actual capture/JIT counters are
required. Invalid timing stays unavailable. Final analyze.py consumes unique,
complete source-bound audit/clean cells with all input hashes.
