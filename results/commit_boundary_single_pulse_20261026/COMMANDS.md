# Commands and reproduction

Canonical execution is **boundary/attempt001 and panel/attempt004**. The panel
attempt001 commands below are historical preparation, not the executed panel.
Study naming uses Asia/Hong_Kong (2026-10-26); launch receipts retain Unix UTC.

Canonical identities:

- Starting repository HEAD: `748e22b328e167cd64c64a46afda9bec1ff06d32`.
- Boundary source: `85d2f11d`; protocol:
  `9d10674b993bcb55120a6d49c2d95c4dae164ffc0b64e2a2048e6c0f654e4044`.
- Panel generation source: `723d4cb0a4498c61a363ddbd856de6c736419ed4`;
  protocol `4f6d95ed03ffc22dada08d113f7ef17acc0844b764cff71136344ddf4eac6f11`.
- Baseline-only cohort checkpoint: `b52335e84ebfd62812ce5919d877df21d831b665`;
  manifest `13b6ea729c1a4dacbdce283417539b0802ed8600978fc57d73ee5b88f37c02c3`.

The actual commands for the canonical panel qualification and serial execution
were (the ignored interpreter binding expands `<pinned_python>`):

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. <pinned_python> -m pytest -q results/commit_boundary_single_pulse_20261026/boundary/attempt001/tests results/commit_boundary_single_pulse_20261026/panel/attempt004/tests tests/test_trajectory_dynamics.py tests/test_dynamics_screen_protocol.py results/commit_boundary_single_pulse_20261026/reporting002/test_qualified_bank.py --junitxml=results/commit_boundary_single_pulse_20261026/panel/attempt004/tests/prelaunch002.xml
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 <pinned_python> results/commit_boundary_single_pulse_20261026/panel/attempt004/freeze.py
<pinned_python> results/commit_boundary_single_pulse_20261026/panel/attempt004/launch.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 <pinned_python> results/commit_boundary_single_pulse_20261026/reporting002/after.py
python results/commit_boundary_single_pulse_20261026/preparation001/brief_status004.py
```

The controller's exact serial subprocess vectors, start times, source/protocol
bindings, process start identities and exit statuses are in
`panel/attempt004/launches/*/{launch_receipt,process_identity,exit_receipt}.json`.
Its schedule is: two 24-question baseline profiles; baseline-only cohort selection
and manifest commit; seven arms on three qualification cells (21 audit/clean
pairs); seven arms on 24 panel cells (168 audit/clean pairs); analysis.
Every model request uses the original native output limits and denoising cap.

CPU addendum commands, without new model calls:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 <pinned_python> results/commit_boundary_single_pulse_20261026/cpu_addendum001/canonical_work_audit002.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 <pinned_python> results/commit_boundary_single_pulse_20261026/cpu_addendum001/rescore.py
```

CPU relay001 produced the completed boundary analysis before it stopped at the
failed panel003 marker. Relay002 reuses that completed analysis and runs the
independent online/legacy, fairness, position/content, resource and plot readers.
Exact reader vectors and hashes are in `reporting_execution00*/`.

Preserved failures/revisions: panel001 and002 launched no GPU workers; only
their positively identified waiting controllers were stopped. Panel003 launched
one worker which failed before model initialization because `code/select.py`
shadowed Python's standard-library `select`; zero model calls. Panel004 renames
that script `cohort_select.py` and adds a worker-directory import test. No native
worker or unrelated workload was killed. A CPU work-reader field alias and an
interpreter-wrapper verification failure are also retained; neither generated
model output. Never replay these obsolete attempts into their frozen directories.

The following section preserves the historical preparation commands.

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
