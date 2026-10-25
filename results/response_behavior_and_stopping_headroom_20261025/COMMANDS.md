# Commands and reproduction

Run from the repository root. `<pinned_python>` denotes the interpreter recorded
in the ignored private runtime binding; its package fingerprints are public.
Private model/input/scorer locations are intentionally omitted. The public
`launches/*/launch_receipt.json` files contain each exact worker argument vector,
protocol/source identity, start time and matched exit receipt. Process identities
and allocation counters are separate receipts. Private logs are not published.

CPU addendum:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 <pinned_python> results/response_behavior_and_stopping_headroom_20261025/cpu_reporting_addendum001/analyze.py
```

R preparation used the new attempt's `prepare.py`, `freeze.py` and
`verify_inputs.py` before the committed source checkpoint. CPU qualification:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. <pinned_python> -m pytest -q results/response_behavior_and_stopping_headroom_20261025/track_r/attempt001/tests tests/test_trajectory_dynamics.py tests/test_dynamics_screen_protocol.py --junitxml=results/response_behavior_and_stopping_headroom_20261025/track_r/attempt001/tests/prelaunch001.xml
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. <pinned_python> -m pytest -q results/denoising_e2e_research_20261024/attempt003/tests/test_online_selector.py --junitxml=results/response_behavior_and_stopping_headroom_20261025/track_r/attempt001/tests/inherited_selector001.xml
```

R executes one detached controller, started once after binding to source commit
`5ebd3e66b4b9ac54cbba24ded2cc4154ae5d84e3`:

```sh
<pinned_python> results/response_behavior_and_stopping_headroom_20261025/track_r/attempt001/launch.py
```

It runs six legacy arms, official scoring and the legacy equality gate before ten
panel workers. Each worker uses `code/worker.py --root <R_attempt> --phase
legacy|panel --benchmark all|aime26|humaneval --arm <frozen_arm>`. It scores each
completed shard with `code/scoring.py` using the same flags and runs
`code/analyze.py` only after all700 panel cell pairs complete. Exact expanded
vectors are in the launch receipts. The worker creates audit and clean complete
requests, with an independently counted warm request per initialized engine.

Monitoring was read-only: `git status`, source SHA256 checks, progress/completion
receipts, private controller heartbeats, identified PID/start-time checks, and
`nvidia-smi`. No process was preempted. No paused campaign was resumed.

After R's timing/readout completes, descriptive CPU closeout and the previously
sealed extraction checkpoint addendum use:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 <pinned_python> results/response_behavior_and_stopping_headroom_20261025/reporting001/close_track_r.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 <pinned_python> results/response_behavior_and_stopping_headroom_20261025/content_prefix_study001/analyze.py --scope original
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 <pinned_python> results/response_behavior_and_stopping_headroom_20261025/content_prefix_study001/analyze.py --scope panel
```

A detached `reporting001/after_r.py` relay waits for R's completion and runs those
commands sequentially. It verifies all900 original/new private generation
identity/output bindings without printing text, commits only this study's public
files, then prepares and qualifies H. One preflight log-directory error occurred
before this relay was created; it used zero model calls and no GPU worker.
The corrected launch was started once and does not duplicate R.

Never rerun these exclusive-create scripts in a completed directory. A revision
or relaunch needs a new named analysis/attempt; retain prior failures and markers.
Track H commands and its separate source/protocol binding are recorded after R
readout is frozen. Preparing H source is not an H model execution.
