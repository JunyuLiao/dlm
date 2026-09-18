# RULER8K comparison, version13

One H100 worker. Final cohort:130 questions,10 from each of the13 pinned RULER tasks. Separate calibration26 and development13. Dataset manifests, source hashes and exact decoding settings are frozen in `setup.json` and `execution_contract.json`.

Conditions: dense; BLASST, existing mass-bound-only, full-dimensional centered and Gaussian2 centered, each at50% and75% target physical sparsity. Physical tiles are128 queries×64 KV tokens; prefix and canvas participate.

BLASST defaults to λ≤1 for both attention types. A full-calibration run at constant λlocal=λglobal=1 determines whether a target permits λ>1. Only a below-target whole-model result unlocks aggressive thresholds. Local infeasibility alone does not unlock them. Effective λ retains the established inverse-valid-KV-length rule; its per-call values are recorded.

Existing numerical kernels and previous experiment bundles are unchanged. No measured hardware speedup is claimed. The17 pre-launch CPU/CUDA tests passed; the worker additionally requires two-input actual-model dense/unpruned and sparse/reference parity before calibration or evaluation.

Run from `/home/exouser/ljy/dlm`, with:

```bash
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=src:. OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
/home/exouser/miniconda3/envs/ljy_dlm/bin/python -m experiments.diffusion_gemma_ruler8k_jl launch
```

`launch` resumes completed hash-checked shards and requires an idle GPU. Do not launch a second worker while `job.json`'s PID is alive. `phase.json`, `progress.json`, `completed.jsonl`, `run.log` and `supervisor_terminal.json` describe execution. The supervisor records30-minute health snapshots; the agent also checks status and reports progress on that interval.

After completion, raw-only regeneration and verification (no GPU inference):

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONPATH=src:. /home/exouser/miniconda3/envs/ljy_dlm/bin/python -m experiments.diffusion_gemma_ruler8k_jl report
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONPATH=src:. /home/exouser/miniconda3/envs/ljy_dlm/bin/python -m experiments.diffusion_gemma_ruler8k_jl verify
```

Final outputs include per-task official scores and equal-task macro, count-weighted whole/global/local tile sparsity, shared-state and trajectory-local operator diagnostics, dense retained mass, token agreement, paired uncertainty, effective thresholds, CSV/JSON exports and plots. A complete claim requires all1170 outputs and an independent matching report-regeneration proof.

Exposure caveat: selection is disjoint for this study but drawn from a previously generated/partly evaluated650-example pool, not a fresh unseen benchmark sample.
