# V29 expanded confirmation protocol (2026-10-02)

This is a preparation protocol, not a result or a noninferiority decision.
Generation and scoring preserve every scheduled question/seed and every failed
attempt. V28 and its completion coordinator remain separate.

## Frozen inventory and comparison

| Suite | Questions | Distinct engine seeds | Timed repeats per seed | Arms | Timed requests |
|---|---:|---:|---:|---:|---:|
| LongBench-v2 | 59 (24/24/11 in 32K/64K/96K) | 8 | 1 | 4 | 1888 |
| AIME26 | 30 | 8 | 1 | 4 | 960 |
| HumanEval | 164 | 8 | 1 | 4 | 5248 |

All 8096 timed requests have matching default dense, native-hook, all-kept and
unchanged main cells. Every frozen item is also warmed once per fresh engine:
8096 warm requests, excluding separate qualification. AIME and HumanEval prompts
are short; thinking and generated output can be long. The existing manifests use
8192 output-token budgets. No budget, question or seed is shortened after seeing
an answer. LongBench uses the existing 59-question inventory; this is not a new
unseen holdout. Keep its historical selection provenance in the result notes.

The first confirmation fixes Q128, alias2, request_clear, legacy canvas buffers,
`kv_copy_backend=torch` and `merge_backend=torch`. The main effective method
remains R6/A64, dense-prefix risk, minus-ln2, carry_first, fused observation,
asynchronous routing and temporal sensitivity on the five GLOBAL layers.
This panel fixes the actual default `min_route_keys=0`, projected-V rank32 and
`observe_carried=False`; it is not the historical HF AIME +2K-gate variant.
Missing frozen-config Q128/False fields retain the adapter's original defaults;
effective runtime receipts still require the explicit resulting values.
Default dense retains its official graph default; the three adapter arms retain
PIECEWISE. Other settings and source/deployment are identical for a paired cell.
Copy/merge fusion and Q64 require separately named, committed protocols and new
qualification; they cannot silently enter this confirmation.

Each suite has its own protocol. Different suites may use different hosts. Within
a suite the frozen block-to-host assignment places all arms of each paired cell
on one host/GPU/software stack. Eight seeds are distinct load-time engine seeds;
there is no request RNG reset or identical stochastic-trajectory claim. Arm order
rotates deterministically across the eight blocks. Repeat numbers are cell labels.

## Qualification and accuracy contracts

Every assigned generation host must qualify every dataset and all four arms,
using that host's first frozen seed block and the longest frozen input for the
dataset: one warm and one timed request. Qualification checks full inventory,
closed workers, unchanged source, private/public joins, actual forward counts,
effective-path receipts, actual KV-layout probes and zero timed graph/JIT events.
The CPU scorer additionally runs public correct/wrong toy answers through the
real task scorers. Formal launch requires the scored qualification aggregate;
the proof binds generation host, GPU UUID, deployment, Torch/vLLM versions and
CPU threads. A different CPU scoring host does not change that generation identity.

Qualification does **not** require a correct, parsed or final-channel answer.
Wrong, unparsed, capped and missing-final responses remain scored failures and
must not trigger replacement of a question or seed. Missing gold/completion,
invalid joins, unavailable HumanEval sandbox, scorer exceptions and failed
public-toy self-tests are infrastructure failures and block formal launch.

LongBench uses the unchanged `v15_longbench_task` pinned NeMo MCQ contract;
strict correctness requires the correct final-channel option and EOS. AIME uses
the existing project `final_response` plus `numeric_score` contract, with strict
correctness requiring EOS; task correctness and caps are also reported. This is
the existing AIME numeric grader, not a newly introduced NeMo math grader.
HumanEval uses the existing complete-function final-channel extractor and the
official task tests through the existing bwrap sandbox; pass@1 does not require
EOS. A capped program that passes retains that historical pass@1 result.

Gold stays at its existing private CPU scoring location and its exact bytes are
pinned before generation. Generation manifests strip gold/test fields. No gold,
prompt, generated text, private identity, private path or private data hash is
published. Cross-host scoring uses an explicit `scorer_artifacts` map in each
private family entry, from every original `binding.files` path plus `DEPLOY_SHA`
to its scorer-local byte-identical copy. Require `binding_sha256` for the untouched
original generation binding. Missing, unknown, aliased or changed mappings reject
the family. Full nested config source pins, deployment SHA and source-set identity
remain checked; gold stays at its original CPU scorer location. The original
binding, config source paths and generation host are never rewritten.
Bindings explicitly preserve the model origin and freeze the generation host's
canonical `model_config_files` pair, plus `source_file_paths` and nested
`config_source_files`. A scorer looks up those canonical pins directly and never
tries to resolve an unavailable remote snapshot symlink. Generation validation
still checks that each live origin alias resolves to its originally pinned file.
Mapped workers also require private `worker_sha256` pins obtained from the
closed generation worker for `terminal.json`, `records.jsonl` and
`completions.private.jsonl`; scorer-local bytes must match all three before loading.

## Executable workflow

The scripts are CPU-importable and do not launch other workers. Use `python -B -m`
from the bound repository root. Replace the uppercase placeholders below with
private coordination paths; never commit populated private catalogs or families.

```text
python -B -m scripts.v29_expanded_panel spec --suite longbench --name lb_confirmation001 --engine-seeds 8101 8102 8103 8104 8105 8106 8107 8108 --host HOST_ALIAS --out NEW_SPEC
python -B -m scripts.v29_expanded_panel freeze --spec COMMITTED_SPEC --catalog PRIVATE_CATALOG --config REBOUND_MAIN_CONFIG --model EXISTING_MODEL --deploy COMMITTED_DEPLOY --host HOST_ALIAS --gpu-uuid GPU_UUID --out-dir NEW_PRIVATE_FREEZE
python -B -m scripts.v29_expanded_panel run --binding PRIVATE_BINDING --arm ARM --block FIRST_HOST_BLOCK --dataset DATASET --mode qualification --run-dir NEW_QUAL_WORKER
python -B -m scripts.v29_expanded_summary --family PRIVATE_QUAL_FAMILY --gold DATASET=EXISTING_CPU_GOLD --nemo-root PINNED_NEMO_ROOT --qualification --out-dir NEW_QUAL_SCORE
python -B -m scripts.v29_expanded_panel run --binding PRIVATE_BINDING --arm ARM --block BLOCK --mode benchmark --qualification-score QUAL_SCORE_SUMMARY --run-dir NEW_FORMAL_WORKER
python -B -m scripts.v29_expanded_summary --family PRIVATE_FORMAL_FAMILY --gold DATASET=EXISTING_CPU_GOLD --nemo-root PINNED_NEMO_ROOT --out-dir NEW_FORMAL_SCORE
```

Choose suite `aime` or `humaneval` for their separate specs. The seed list above
is a fixed example; commit the actual list before qualification. Repeat `--gold`
for every dataset in a suite. NeMo root is explicitly required for LongBench;
AIME/HumanEval do not use it. Set `OMP_NUM_THREADS=1` before worker launch.
The model and software environments must already exist; do not install scorer
dependencies into a running generation environment.

The private catalog maps each dataset to `manifest`, scorer-only `gold`, optional
preverified `gold_sha256` and optional frozen `selected_ids`; HumanEval also needs
its existing `task_contract` receipt. If gold is not local to generation, supply
the already verified byte pin without transferring it. All official task counts
remain mandatory. A family contains `workers:[{binding,run_dir},...]`; formal
families contain all 32 workers (8 blocks × 4 arms), and qualification families
contain each assigned host × dataset × arm. Files and output directories are
exclusive-new; open/failed/duplicate/missing workers are rejected, never filtered.

For a new generation host whose original source root is unavailable locally,
use `rebind-mirror --config ORIGINAL_CONFIG --old-root ORIGINAL_ROOT --mirror-root
VERIFIED_SOURCE_MIRROR --new-root NEW_DEPLOY --out NEW_CONFIG` on the destination
host. It checks original nested fingerprints, each source's pinned bytes in both
the mirror and destination, relative paths, collisions and symlink escapes.
Only source path keys and existing nested fingerprints change; original byte
hashes and every non-binding method field are preserved. External source paths
must still be readable and unchanged; they are never silently remapped. No old
root directory needs to be created and the original config remains untouched.

## Measurement and inference

W is request-boundary wall time, S the decode span and P prefill. N counts actual
denoising execution, including unused speculative work, and must equal scheduler
retired denoising plus unused denoising. Commit forwards are separate. S/N is
amortized decode cost, not isolated attention latency. Standard adapter D2H
metadata reads remain charged to all three adapter arms. KV numerical probes
occur only during warmup; zero timed CUDA graph, backend/inductor and monitored
Triton/CuTe JIT deltas are required. JIT receipts retain their event-coverage
limitations; a monitor counter is not a universal compiler audit.

Publish per-dataset absolute W/S/P/N statistics, correct counts, caps, paired
geometric W/S/SN/N/P ratios and question-cluster 95% CIs. Bootstrap resamples
questions and retains all their seed observations; more seeds do not make a
question independent. By-seed N distributions and ratios are descriptive and
do not select a winning seed. Cell McNemar remains exploratory. Do not pool
accuracy across incompatible task metrics. The noninferiority margin is unset:
report observed differences and uncertainty without declaring noninferiority
or treating nonsignificance as proof of no quality loss.

Calibrate runtime from qualified runs and the first complete formal block before
publishing an ETA. The full schedule may require hours to one or two days;
previous partial runs are not a basis for reducing it. Account for warmup,
engine loads, failed attempts and reserved GPU seconds separately from W. CPU
HumanEval grading time is separate. This preparation contains no speed result.

## CPU validation

```text
python -B -m unittest tests.test_v29_expanded_panel -q
```

Public toy tests cover complete inventories, source/settings/seed/count/JIT
drift, missing/duplicate/open workers, gold pins, qualification/formal separation,
constructor restoration, unchanged task delegation and EOS semantics, legitimate
unparsed failures, generation host/GPU/software proof binding, per-host coverage,
public scorer-toy delegation and private-field exclusion. They use stdlib fakes;
real pinned scorer self-tests and all-arm GPU qualification remain launch gates.
