# Closeout commands and file inventory

All commands ran from `/home/exouser/ljy/dlm`. Original production, operator and scoring commands are preserved in [report003/COMMANDS.md](../report003/COMMANDS.md); all earlier launch and exit receipts remain in their immutable execution directories.

## Native replay qualification

```bash
timeout --signal=TERM --kill-after=30s 900 /home/exouser/dyh/v31_env_20261005/bin/python /home/exouser/ljy/dlm/results/denoising_dynamics_census_20261022/attempt001/execution004/native_replay001/code/worker.py --root /home/exouser/ljy/dlm/results/denoising_dynamics_census_20261022/attempt001/execution004/native_replay001 --stage smoke
```

Environment overrides:

```text
PYTHONPATH=/home/exouser/ljy/dlm/src:/home/exouser/ljy/dlm
HF_HUB_OFFLINE=1
TORCHINDUCTOR_DETERMINISTIC=1
VLLM_USE_V2_MODEL_RUNNER=1
VLLM_CACHE_ROOT=/home/exouser/ljy/dlm/results/denoising_dynamics_census_20261022/attempt001/execution004/native_replay001/cache/vllm
TRITON_CACHE_DIR=/home/exouser/ljy/dlm/results/denoising_dynamics_census_20261022/attempt001/execution004/native_replay001/cache/triton
TORCHINDUCTOR_CACHE_DIR=/home/exouser/ljy/dlm/results/denoising_dynamics_census_20261022/attempt001/execution004/native_replay001/cache/inductor
CUDA_CACHE_PATH=/home/exouser/ljy/dlm/results/denoising_dynamics_census_20261022/attempt001/execution004/native_replay001/cache/cuda
```

## Final focused CPU suite

```bash
PYTHONPATH=src:. /home/exouser/dyh/v31_env_20261005/bin/python -m pytest -q tests/test_trajectory_dynamics.py tests/test_dynamics_screen_protocol.py results/denoising_dynamics_census_20261022/attempt001/execution004/tests/test_production_observer.py results/denoising_dynamics_census_20261022/attempt001/execution004/tests/test_candidate_readout.py results/denoising_dynamics_census_20261022/attempt001/execution004/operator_diagnostic002/tests/test_diagnose.py results/denoising_dynamics_census_20261022/attempt001/execution004/native_replay001/tests/test_replay.py --junitxml=results/denoising_dynamics_census_20261022/attempt001/execution004/closeout001/cpu_tests.xml
python results/denoising_dynamics_census_20261022/attempt001/execution004/closeout001/verify.py
git diff --check
```

GPU closeout used the exact nvidia-smi argv in [gpu_status.json](gpu_status.json). The restricted sandbox cannot access the NVIDIA driver; the authorized host check succeeded. The final suite passes 24 tests; the original report passes its 26 integrity checks and the later closeout passes 18 checks.

## Files changed and preserved

The exact 254-file publication set, sizes and hashes are in [artifact_inventory.json](artifact_inventory.json). [publication_files.json](publication_files.json) distinguishes inherited untracked scaffolding from new campaign code and results. `docs/DENOISING_DYNAMICS_CENSUS_20261023.md` records committed pre-generation specifications. This task also prepends its handoff and appends verified decisions to `HANDOFF.md` and `docs/DECISIONS.md`; pre-existing edits are retained separately. `AGENTS.md`, `STATE.json`, and `docs/RESULTS_LEDGER.md` are unchanged by this task.

Private prompts, gold, token IDs, raw completions, tensor captures, logs and runtime caches remain local and excluded. Public JSON field inspection passes; metadata fields named prompt_tokens contain scalar counts only. Downloaded research papers are excluded from publication; retrieval URLs and hashes are retained.

## Completed and incomplete scope

Completed: phase 0 audit, observer smoke, all 21 context-eligible dense requests and official scores, bounded inherited-state attention diagnostics, one-item logical native replay qualification. Nine original LongBench prompts remain blocked without truncation or replacement by explicit user instruction.

Partial: D1/D3/D4/D5 attention-level diagnostics only. No full-state in-place restoration, controlled sparse continuation, downstream MarginGuard oracle, matched native sparse panel, paired accuracy contrast, clean timing or seed replication. No selector or kernel optimization is promoted.

Next engineering gate: preserve and restore complete native scheduler/sampler/KV/layout/RNG state, then prove same-support continuation before downstream influence tests. Logical replay is useful evidence but physical page identities differ; it does not close this gate. Existing suffix oracles perform extra forwards and include cached-output variants, so they cannot be imported into this study.

The primary agent must address this missing native causal harness. The tested ResponseGuard candidate bank has mixed held-out attention evidence; the norm-based ValidityReuse certificate is too loose; MarginGuard remains untested. The broader causal acceleration question remains open.
