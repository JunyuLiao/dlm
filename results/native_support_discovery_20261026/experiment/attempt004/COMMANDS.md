# Exact entry points

From the repository root. The ignored interpreter wrapper executes the pinned
environment, with its private path omitted from publication.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. results/native_support_discovery_20261026/private/python -m pytest -q results/native_support_discovery_20261026/experiment/attempt004/tests --junitxml=results/native_support_discovery_20261026/experiment/attempt004/tests/prelaunch001.xml
python results/native_support_discovery_20261026/experiment/attempt004/freeze.py
```

Commit the task-owned specification/source/protocol, create binding.json with
that checkpoint and protocol SHA256, then repeat focused tests as bound001.xml.
The controller is launched detached with the pinned Python. It waits for the
canonical profiling completion and committed cohort, verifies actual model,
runtime and official scorer files, and serializes these commands:

```bash
python results/native_support_discovery_20261026/experiment/attempt004/verify_environment.py
python results/native_support_discovery_20261026/experiment/attempt004/code/study_worker.py --job references
python results/native_support_discovery_20261026/experiment/attempt004/code/numerical_verify.py
python results/native_support_discovery_20261026/experiment/attempt004/code/bank_plan.py --stage early_maps
python results/native_support_discovery_20261026/experiment/attempt004/code/study_worker.py --job early
python results/native_support_discovery_20261026/experiment/attempt004/code/bank_plan.py --stage branches
python results/native_support_discovery_20261026/experiment/attempt004/code/study_worker.py --job treatments
python results/native_support_discovery_20261026/experiment/attempt004/code/analyze.py
```

Here `python` means the pinned interpreter selected by the controller. Exact
structured arguments, PID/start identities, exit status and allocated time are
under launches/. Required environment settings are frozen in launch.py. No job
has a campaign timeout. Completed/frozen directories cannot be rerun in place.
Before sparse requests, the controller commits only support banks, numerical
qualification, early-map choices and the unique branch manifest. A future
confirmation is launched only after the frozen development gates pass.
