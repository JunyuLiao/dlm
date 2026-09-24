# Native numerical QK reuse — checkpoint 1

## Authority and identity
- Sole specification: user v5 (2026-09-24); transcript not consulted or committed.
- Branch: `research/numerical-qk-reuse-native-20260924`.
- Base: Junyu `053441c6c6741ada6728dfc97d8faa6ea2be72aa`.
- Fetched latest Junyu and Haowei refs once; both equal pinned refs.
- Haowei read-only reference: `b23f969a3c52417ba84a999ff7a31e3dd00bb697`.
- Old vLLM P0 remains independent and unresolved. This is native Torch.

## Current status
- WIP / GPU_UNQUALIFIED. No new model generation has run.
- Mathematical contract, cache lifecycle, cached-score executor, native runner exist.
- Runner/cache: 11 local CPU checks passed; combined discovery also reports missing
  local pytest for the separate Torch math suite. Remote CPU math subsequently passed: 4 math + 5 cache tests, 2 CUDA tests skipped.
- Native dense and fresh Junyu T have NOT yet reproduced in this round.
- M1 adapter integration and M3 natural generation remain pending.
- Historical failures/results are unchanged.

## Implementation
- `method_contract.md`: numerical_reuse_current_V, distinct score/decision clocks.
- `cache.py`: identity, invalidation, storage guard; pure CPU control tests.
- `cached_executor.py`: historical scores + current V; no Q/K input.
- `reference.py`: sequential retained-state mathematical reference.
- `runner.py`: attempt-0 receipts, native actual per-canvas decoder call counts.
- Score anchors: calls 0,8,16,...; M1 decision each call, M3 initially every 2.
- Geometry: Junyu per query head, Q128/KV64. Not old GQA8/Q2/KV32.
- Cache FP32 of Junyu BF16-transformed scores; current model/output BF16.
- Same-stream publication initially; CUDA Graph replay not yet qualified.
- Optional timing events are disabled until parity qualification. CPU generation
  boundary unavailable => N/A, never relabel device timeline as CPU wall.

## Host and execution
- Only authorized H100: `exouser@149.165.151.254`; idle at last check.
- Disk at last check: 61 GiB free. No GPU jobs launched by this checkpoint.
- Private root: `/home/exouser/dyh/numerical_qk_reuse_native_20260924`.
- Read-only Python: `/home/exouser/miniconda3/envs/ljy_dlm/bin/python`.
- Torch 2.6.0+cu124; Transformers 5.11.0; Triton 3.2.0.
- Full model snapshot: `/home/exouser/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b`.
- Archived Junyu binaries absent; privately rebuilding exact pinned source.
- Never deploy edits into an active worker. Private immutable source archives only.

## Immediate next actions
1. Remote CPU math tests with CUDA_VISIBLE_DEVICES empty; record actual source hash.
2. Commit/push this runnable WIP checkpoint and verify remote SHA.
3. Complete private Junyu binary rebuild, actual metadata inspection and M1 adapter.
4. GPU mathematical/dispatch gates, then 1–2 complete natural requests.
5. Frozen four AIME26 questions, seed42,8192/thinkingON/native adaptive; no240 matrix.
6. M3 R2 on same executor after M1; push each qualified checkpoint.

## Resume commands
From WSL: `cd /mnt/e/dlm/numerical_qk_reuse_native_20260924`.
CPU local: `python -m unittest discover -s tests -p test_numerical_reuse_cache.py -v`.
Runner contract: `python -m unittest discover -s tests -p test_numerical_qk_runner.py -v`.
Remote math: `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. OMP_NUM_THREADS=4 <python> -m pytest tests/test_numerical_reuse_math.py -q`.
No live model PID or pending GPU queue exists. Read STATE.json before launch.
