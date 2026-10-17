# 2026-10-17: integration, public analysis and private generation reproduction

The [method/setup catalogue](01_METHODS_AND_SETUP.md) defines all variants and
the [results record](02_RESULTS_AND_STATISTICS.md) distinguishes complete native
accuracy from operator/decision diagnostics. This page gives the source and
commands for reproducing those levels of evidence. None of the commands below
is an instruction to resume an older paused campaign. Clean timing stays paused.

## Integration map

| Source | Responsibility |
|---|---|
| [`v31_value_decision.py`](../../experiments/numerical_qk_reuse/v31_value_decision.py) | Full-V block moments, true renormalized outputs, fixed-K mass neighborhoods, post-O loss, disjoint swap proposals and joint admission, online oracle, C_gate, omitted-cache reference. |
| [`v31_value_online_kernel.py`](../../experiments/numerical_qk_reuse/v31_value_online_kernel.py) | Full-V preserve-mass proposal risks in physical order; Triton one-CTA Q128/head traversal, bounded working state; diagnostic hybrid only. |
| [`v31_value_reuse_guard.py`](../../experiments/numerical_qk_reuse/v31_value_reuse_guard.py) | Whole-layer directional/norm trust, exact adjoint cache, selected-only omitted reconstruction and identity/reset checks. |
| [`v31_value_reuse_merge_kernel.py`](../../experiments/numerical_qk_reuse/v31_value_reuse_merge_kernel.py) | Fused held-row dot/coefficient/update and selected-LSE merge. |
| [`v31_value_refinement_runtime.py`](../../experiments/numerical_qk_reuse/v31_value_refinement_runtime.py) | Opt-in layer29 adapter hook, exact budget checks, refresh/support identity, unchanged FA4 or named correction consumer, separate work counters. |
| [`v31_value_refinement_worker.py`](../../scripts/v31_value_refinement_worker.py) | Native worker; refuses a mode absent from the passed decision gate. Installs before model construction. |
| [`v31_value_interleaved_worker.py`](../../scripts/v31_value_interleaved_worker.py) | Per-request candidate/control dispatch within a common engine. |
| [`vllm_adapter.py`](../../experiments/numerical_qk_reuse/vllm_adapter.py) and [`v31_vllm_paired_bench.py`](../../scripts/v31_vllm_paired_bench.py) | Inherited V31 lifecycle, native sampling/consumer integration and dispatch/accounting receipts. |
| [`v31_same_state_suffix.py`](../../experiments/numerical_qk_reuse/v31_same_state_suffix.py) | Layer29 identical-state intervention, exact suffix/RNG/cache restoration, logits/sampler diagnostics; extra forwards kept separate from native N. |
| [`v31_value_decision_replay.py`](../../scripts/v31_value_decision_replay.py), [`v31_value_whole_layer_replay.py`](../../scripts/v31_value_whole_layer_replay.py), [`v31_value_reuse_guard_replay.py`](../../scripts/v31_value_reuse_guard_replay.py) | Independent saved-Q/K/V operator, whole-layer and trust screens. |
| [`v31_value_decision_batch.py`](../../scripts/v31_value_decision_batch.py), [`v31_value_coverage_batch.py`](../../scripts/v31_value_coverage_batch.py), [`v31_value_decision_analysis.py`](../../scripts/v31_value_decision_analysis.py) | Frozen initial/coverage schedules and paired request-cluster decision analysis. |
| [`v31_value_replicated_study.py`](../../scripts/v31_value_replicated_study.py) | Qualification → two-engine native shards → official score → report → unchanged promotion screen. |
| [`v31_verify_replicated_results.py`](../../scripts/v31_verify_replicated_results.py) | Independent full local coverage/hash/repetition/N-C-T/work audit; requires private bound generations and RULER row information. |
| [`v31_replicated_trajectory_diagnostics.py`](../../scripts/v31_replicated_trajectory_diagnostics.py) | Descriptive outcome/exposure/tail/product-decomposition analysis; not a causal estimate. |
| [`v31_value_publication_20261017.py`](../../scripts/v31_value_publication_20261017.py) | Public-only verification and all-statistic CSV/Markdown reproduction; no CUDA, model, private text or network. |
| [`v31_replicated_synthesis.py`](../../scripts/v31_replicated_synthesis.py) | Full local verification, consolidated tables and scientific PNG/SVG/PDF; `make_plot` can also render public summary JSON. |

The kernel changes reduce bounded-pair Python dispatch, keep proposal ranking
on device and avoid a held-call whole-layer O GEMM. These are implementation
properties, not measured latency wins. Cached correction and support selection
have different consumers and receipts.

The opt-in installation primitive is `install(mode)` from the runtime module,
called before constructing DiffusionGemma. Supported native evaluation modes
were `full_swap8`, `residual_direction_guard`, `residual`, and
`full_swap8_direction_guard`. Use the coordinator and worker rather than calling
the primitive in an unbound production run: the worker requires
`VALUE_REFINEMENT_GATE` and `VALUE_REFINEMENT`, and the coordinator binds source,
input schedule, `MAGE_K` token budget, qblock_max/fa4 selection and graph mode.
The runtime rejects conflicting selector/cached-refresh families. The receipt
must name layer `[29]`, the configured prefix budget, the actual consumer and
zero budget errors. No earlier GLOBAL layer is qualified by this suffix probe.

## Reproduce all public tables on a CPU

Clone this branch and run from the repository root with Python 3.10+:

```bash
PYTHONPATH=src:. python -m scripts.v31_value_publication_20261017 \
  --output results/reproduce_public_20261017/attempt001
```

Use a new output directory for each invocation. The exporter verifies the native
completion/protocol/report hashes, official scoring output hashes, all 15 score
labels and cohort keys per workload, unequal repeat counts, score arithmetic,
N/C/T totals and means, physical-work sums, zero measured graph captures, LOCAL
skips and reported budget errors. It exports 12 files plus a source/input/output
binding marker. Only public numeric records are read. It does not rerun the
official scorers or claim to verify private prompt, completion or model bytes.

The checked-in publication is
[`results/research_update_20261017/attempt001`](../../results/research_update_20261017/attempt001/native_statistics.md).
Its `final_complete.json` identifies every consumed public file by SHA256.

To reproduce the scientific figure from the public reports, use a writable
Matplotlib cache and a fresh figure directory:

```bash
MPLCONFIGDIR=/tmp/dlm_matplotlib PYTHONPATH=src:. python - <<'PY'
import json
from pathlib import Path
from scripts.v31_replicated_synthesis import make_plot, SUITES
study = Path('results/v31_value_decision_reuse_20261015/native_replicated_20261016_attempt001')
out = Path('results/reproduce_figures_20261017/attempt001')
out.mkdir(parents=True, exist_ok=False)
reports = {s: json.loads((study/'reports'/s/'summary.json').read_text()) for s in SUITES}
make_plot(reports, out)
PY
```

This needs Matplotlib 3.10+ for the recorded legend layout. PNG/SVG/PDF rendering
metadata can vary by version; score/interval arithmetic is fixed by the reports.

## Qualification tests

The new public report has CPU integrity tests:

```bash
PYTHONPATH=src:. python -m pytest -q tests/test_v31_value_publication_20261017.py
```

The source-bound H100 qualification used the following five modules:

```bash
PYTHONPATH=src:. "$DLM_PYTHON" -m pytest -q \
  tests/test_v31_value_decision.py \
  tests/test_v31_value_reuse_guard.py \
  tests/test_v31_same_state_suffix.py \
  tests/test_v31_value_refinement_runtime.py \
  tests/test_v31_value_native_study.py
```

`DLM_PYTHON` must be the pinned CUDA/vLLM interpreter on an idle H100.
The [recorded receipt](../../results/v31_value_decision_reuse_20261015/receipts/h100_qualification/attempt002/receipt.json)
has 71 passes, zero skips/failures and 12 source/test hashes at `79c994ca`.
CPU-only skips do not reproduce that qualification. Coordinator tests are
`test_v31_value_replicated_study.py`, `test_v31_value_process_matched_study.py`,
and `test_v31_value_warmup_factor.py`; the recorded 15-check receipt is separate.
Do not infer kernel performance from correctness tests.

## Reproduce operator or native generation with authorized private inputs

Git supplies public methods, numeric evidence, source/model fingerprints,
protocols and schedules' hashes. Prompts, gold, token IDs, raw completions,
Q/K/V snapshots, gated model weights, scorer-location files and runtime caches
stay private. Full generation/scoring requires authorized copies matching the
selected protocol; a public clone alone cannot recreate private inputs.

The private snapshot inventory is under the stable-refresh study, with 670
files/620 frames and a completed trajectory binding. Original selector replay
source is pinned by its `operator_protocol.json`; whole-layer and guard replay
have separate protocols and markers. On the authorized original installation,
the relevant commands are:

```bash
PYTHONPATH=src:. "$DLM_PYTHON" -m scripts.v31_value_decision_replay \
  --stage run --study "$DLM_OPERATOR_STUDY"
PYTHONPATH=src:. "$DLM_PYTHON" -m scripts.v31_value_whole_layer_replay \
  --stage run --study "$DLM_WHOLE_LAYER_STUDY"
PYTHONPATH=src:. "$DLM_PYTHON" -m scripts.v31_value_reuse_guard_replay \
  --stage run --study "$DLM_GUARD_STUDY"
```

Execute each stage's pinned source bytes, not today's changed implementation
against an old protocol. New preparations use `--stage prepare` with new dated
study paths. Never replace an old result, fake a gate or recalculate an old
source identity to accommodate a changed implementation.

For the replicated native study, the
[deployment recipe](../../results/v31_value_decision_reuse_20261015/native_replicated_20261016_attempt001/protocol.json)
is parent execution `28714b4a` plus the two coordinator/test overlay files from
`b485866c`. The frozen method code remains `79c994ca`. Restore the original
private source/input bundle and verify all fingerprints before execution.
The bundle includes pools, manifests, cell/shard/warmup schedules, passed
decision/H100 gates, model inventory and authorized scorer locations.

For a new exact-runtime repetition, prepare a fresh directory from that
qualified parent and completed four-job warmup-factor evidence:

```bash
PYTHONPATH="$DLM_FROZEN_EXEC/src:$DLM_FROZEN_EXEC" "$DLM_PYTHON" \
  "$DLM_FROZEN_EXEC/scripts/v31_value_replicated_study.py" \
  --stage prepare --study "$DLM_NEW_STUDY" \
  --parent "$DLM_QUALIFIED_PARENT" --factor "$DLM_COMPLETE_WARMUP_FACTOR"

PYTHONPATH="$DLM_NEW_STUDY/private/execution/src:$DLM_NEW_STUDY/private/execution" \
  "$DLM_PYTHON" "$DLM_NEW_STUDY/private/execution/scripts/v31_value_replicated_study.py" \
  --stage run --study "$DLM_NEW_STUDY"
```

Variables must point to authorized local paths, not placeholders copied into
metadata. Preparation refuses changed hashes; generation checks host/GPU/runtime
identity. Relocation or different runtime/model/input bytes requires a separately
named frozen protocol and fresh controls. This workflow is not a generic
portable launcher with automatic model/scorer downloads.

The coordinator runs four workload qualification/official-scoring smokes before
accuracy; preserves structurally failed/interrupted attempts; verifies completed
shards on resume; retains drift without parity retries; and checks about every
30 minutes. It does not launch clean timing or a full503 panel. Do not point
these commands at the earlier held independent-process or failed exact-parity
campaigns. Existing completed results should only be audited, not regenerated
into the same directory.

Full local independent verification, with private files present:

```bash
PYTHONPATH=src:. python -m scripts.v31_verify_replicated_results \
  --study "$DLM_COMPLETED_STUDY" --suite longbench_v2 \
  --output "$DLM_NEW_AUDIT/longbench_v2.json"
```

Repeat for `aime26`, `math500`, `ruler128k`. Original official scorers are driven
by the native coordinator's bound scoring stage: current MATH uses math-verify,
not the historical quick string scorer; RULER uses official task averaging.
Sanitized numeric records cannot be independently rescored for correctness
without the private completions/gold. Public arithmetic reproduction and full
official rescoring are separate claims.

## Historical artifacts and external dependencies

All public pending historical evidence is included with its existing status:
complete panels, failed attempts, paused sweeps, generation checkpoints and
diagnostic-only scores. The [October 15 publications](../research_week_20261015/01_VALUE_AWARE_METHODS.md)
remain the definitions/results entry point for those families. A partial file
or completed score does not change a campaign's held/diagnostic status.

Reference repositories are external local checkouts, not nested Git repositories
bundled into this branch. Their upstream URLs/commits and local-data/cache state
are recorded in the publication inventory. The unrelated local worktree is
also preserved externally. Build probes, runtime locks, private inputs and
caches are explicitly ignored. The inventory accounts for the pending entries
without mutating or publishing another checkout's private data.
