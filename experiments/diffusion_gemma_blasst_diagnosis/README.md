# DiffusionGemma / BLASST diagnosis

## Historical versus corrected runs

The saved diagnosis bundle and its narrative report describe **legacy row-mask
execution**. They are preserved, not relabelled as results of the corrected
method. The same-state row-mask probes remain intentional counterfactuals;
their `closure_*` probes represent whole-tile execution.

The current endpoint/reverse commands use `physical_tile_v1` and reject old
output directories before modifying their manifests or policies. Pass a fresh
`--output-dir`, such as `results/diffusion_gemma_blasst_physical_tile_v1`.
The fixed historical narrative report rejects corrected endpoint bundles.
Use `python -m experiments.diffusion_gemma_blasst_diagnosis.history --output-dir
<new-root>` for version-labelled numerical summaries of new endpoint runs,
or the canonical nine-condition experiment/report for a full corrected sweep.

Read-only historical reanalysis plus separate, resumable reference-mask controls.
Run from the repository root in the `ljy_dlm` environment with `PYTHONPATH=src:.`
and `HF_HUB_OFFLINE=1`. CUDA stages require the H100; run them sequentially.

```bash
python -m experiments.diffusion_gemma_blasst_diagnosis.history --calibration
python -m experiments.diffusion_gemma_blasst_diagnosis endpoint-smoke
python -m experiments.diffusion_gemma_blasst_diagnosis endpoint
python -m experiments.diffusion_gemma_blasst_diagnosis diagnostics
python -m experiments.diffusion_gemma_blasst_diagnosis reverse-smoke
python -m experiments.diffusion_gemma_blasst_diagnosis reverse
python -m experiments.diffusion_gemma_blasst_diagnosis dense-eager
python -m experiments.diffusion_gemma_blasst_diagnosis.report
```

Outputs default to `results/diffusion_gemma_blasst_diagnosis/`. The historical
bundle is never modified. Endpoint/reverse/dense-eager use all 50 original final
prompts. Primary scores exclude VT. The diagnostic subset is fixed before running:
two prompts per non-VT task from each split, with no overlap or threshold tuning.

`same_state/shards/<sample>/snapshots.npz` stores block maxima, FP32 block maxima,
valid token counts, dense tile mass and pooled proxies for each sampled call.
`diagnostics.json` stores mask counters, FP32 output perturbations, region splits,
physical-mask reuse and fixed-row-mask regrouping probes. The native attention
output is not replaced by the observer. Final token parity against the original
native dense run is a required gate. The report separates sampled-state evidence
from full-rollout accuracy; it makes no kernel performance or novelty claims.

The reverse control reverses the *traversal* of globally aligned KV tiles,
including a padded partial tile, then restores decisions to their original
positions before softmax/AV and counter collection. It never permutes positional
embeddings or changes the original structural mask. Unit tests check partial
tiles, GQA, structural eligibility, ties and exact output against a literal oracle.

Prediction checkpoints and independent attention statistics are persisted by
the shared RULER runner. Observer snapshots are checkpointed before each prediction
commit, at sample transitions, and on orderly exit; report generation rejects
missing diagnostic shards. A hard kill during a snapshot write leaves that sample
uncommitted and eligible for rerun. No automatic deletion of user artifacts occurs.

During execution, inspect the authoritative process handle, progress log and GPU
every 15 minutes. A read timeout is not permission to launch a duplicate job.
