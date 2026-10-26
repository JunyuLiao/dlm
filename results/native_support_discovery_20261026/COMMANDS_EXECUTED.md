# Executed commands and reproduction

All commands run from the repository root. The ignored `private/python` wrapper selects the pinned interpreter; actual private environment paths remain unpublished. Structured exact job arguments, source/protocol pins, launch/exit status and allocated times are in COMMANDS_EXECUTED.json and launches/ receipts.

The canonical attempt006 reuses complete source-compatible native references from004 and early controls from005 by reference_binding.json. It does NOT rerun the generic reference/early commands listed in the inherited frozen COMMANDS.md; actual launch.py and this executed-command ledger are authoritative. The original202-support manifest is byte-identical.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. results/native_support_discovery_20261026/private/python -m pytest -q results/native_support_discovery_20261026/experiment/attempt006/tests --junitxml=results/native_support_discovery_20261026/experiment/attempt006/tests/bound001.xml
python results/native_support_discovery_20261026/experiment/attempt006/freeze.py
results/native_support_discovery_20261026/private/python results/native_support_discovery_20261026/experiment/attempt006/launch.py
```

The controller launch was detached after source commit and binding. Its serial child commands were:

```text
<pinned_python> /home/exouser/ljy/dlm/results/native_support_discovery_20261026/experiment/attempt006/code/analyze.py
<pinned_python> /home/exouser/ljy/dlm/results/native_support_discovery_20261026/experiment/attempt006/verify_environment.py
<pinned_python> /home/exouser/ljy/dlm/results/native_support_discovery_20261026/experiment/attempt006/code/numerical_verify.py
<pinned_python> /home/exouser/ljy/dlm/results/native_support_discovery_20261026/experiment/attempt006/code/study_worker.py --job treatments
```

CPU-derived commands:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 results/native_support_discovery_20261026/private/python results/native_support_discovery_20261026/census_addendum/attempt001/recheck_and_plot.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 results/native_support_discovery_20261026/private/python results/native_support_discovery_20261026/profiling_report001/report.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 results/native_support_discovery_20261026/private/python results/native_support_discovery_20261026/failure_review001/review.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 results/native_support_discovery_20261026/private/python results/native_support_discovery_20261026/bank_identification001/analyze.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 results/native_support_discovery_20261026/private/python results/native_support_discovery_20261026/report001/report.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 results/native_support_discovery_20261026/private/python results/native_support_discovery_20261026/mechanism_analysis002/analyze.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 results/native_support_discovery_20261026/private/python results/native_support_discovery_20261026/closeout002/verify.py
```

Earlier failures/superseded sources and their tests/commands remain in their named directories. Two waiting CPU controllers were identified by PID, executable/script and process-start ticks before SIGTERM; their unlaunched.json receipts record zero model/GPU work. No GPU worker was killed.

Completed paths are immutable and exclusive-create guards reject in-place reruns. Reproduction needs a NEW named attempt, frozen source/protocol/binding and the ignored private runtime/manifest/tensor inputs. Public input identities and hashes are sufficient to audit provenance; they intentionally omit payloads. Source checkpoints and all task-owned Git commands are also recoverable from the branch log starting7067f209. No push was run.
