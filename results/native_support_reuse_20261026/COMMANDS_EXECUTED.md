# Executed experiment and test commands

Run from the repository root. The ignored private/python wrapper chooses the fingerprinted interpreter. GPU commands ran outside the restricted sandbox. Do not replay writers inside frozen attempts; create a new named directory and bind inputs. Exact child commands/source/protocol/time receipts are in COMMANDS_EXECUTED.json and launches.

```bash
nvidia-smi --query-gpu=name,uuid,driver_version,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python results/native_support_reuse_20261026/ranking_audit/attempt001/audit.py
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. results/native_support_reuse_20261026/private/python -m pytest -q results/native_support_reuse_20261026/attempt001/tests --junitxml=results/native_support_reuse_20261026/attempt001/tests/prelaunch001.xml
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. results/native_support_reuse_20261026/private/python -m pytest -q results/native_support_reuse_20261026/attempt001/tests --junitxml=results/native_support_reuse_20261026/attempt001/tests/prelaunch002.xml
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. results/native_support_reuse_20261026/private/python -m pytest -q results/native_support_reuse_20261026/attempt002/tests --junitxml=results/native_support_reuse_20261026/attempt002/tests/prelaunch001.xml
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. results/native_support_reuse_20261026/private/python -m pytest -q results/native_support_reuse_20261026/attempt003/tests --junitxml=results/native_support_reuse_20261026/attempt003/tests/prelaunch001.xml
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. results/native_support_reuse_20261026/private/python -m pytest -q results/native_support_reuse_20261026/attempt004/tests --junitxml=results/native_support_reuse_20261026/attempt004/tests/prelaunch001.xml
python results/native_support_reuse_20261026/attempt001/freeze.py
CUDA_VISIBLE_DEVICES='' results/native_support_reuse_20261026/private/python results/native_support_reuse_20261026/attempt001/verify_environment.py
python results/native_support_reuse_20261026/attempt002/freeze.py
CUDA_VISIBLE_DEVICES='' results/native_support_reuse_20261026/private/python results/native_support_reuse_20261026/attempt002/verify_environment.py
python results/native_support_reuse_20261026/attempt003/freeze.py
CUDA_VISIBLE_DEVICES='' results/native_support_reuse_20261026/private/python results/native_support_reuse_20261026/attempt003/verify_environment.py
python results/native_support_reuse_20261026/attempt004/freeze.py
CUDA_VISIBLE_DEVICES='' results/native_support_reuse_20261026/private/python results/native_support_reuse_20261026/attempt004/verify_environment.py
results/native_support_reuse_20261026/private/python results/native_support_reuse_20261026/attempt001/launch.py --stage controls
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 results/native_support_reuse_20261026/private/python results/native_support_reuse_20261026/attempt002/numerical_verify.py
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 results/native_support_reuse_20261026/private/python results/native_support_reuse_20261026/attempt002/numerical_verify002.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 results/native_support_reuse_20261026/private/python results/native_support_reuse_20261026/attempt002/code/plan_branches.py
results/native_support_reuse_20261026/private/python results/native_support_reuse_20261026/attempt002/launch.py --stage treatments
results/native_support_reuse_20261026/private/python results/native_support_reuse_20261026/supervise.py
results/native_support_reuse_20261026/private/python results/native_support_reuse_20261026/supervise002.py
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python results/native_support_reuse_20261026/ranking_audit/selection_pool002/analyze.py
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 results/native_support_reuse_20261026/private/python results/native_support_reuse_20261026/audit_off_bank_cpu001.py
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 results/native_support_reuse_20261026/private/python results/native_support_reuse_20261026/common_state_operators.py
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 results/native_support_reuse_20261026/private/python results/native_support_reuse_20261026/rescore.py
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 results/native_support_reuse_20261026/private/python results/native_support_reuse_20261026/build_report.py
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 results/native_support_reuse_20261026/private/python results/native_support_reuse_20261026/attempt002/code/analyze_reuse.py
```

The first numerical reader failed twice on legitimate empty banks; numerical_verify002 passed at unchanged tolerance. The original waiting CPU supervisor was stopped before any child/model; supervise002 used qualified004. Attempt003 launched no GPU worker. The stopped-controller identity and SIGTERM are separately recorded. No GPU worker was killed, paused campaign resumed or push run.
