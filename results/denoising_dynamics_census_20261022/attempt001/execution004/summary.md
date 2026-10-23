# Dense trajectory census and response diagnostics

The dense observer passes its bounded invariance checks. The eligible census
is complete, but the current evidence does not justify a sparse selector or a
native benchmark panel. No speedup, accuracy preservation or trajectory
acceleration is established.

| Workload | Completed / planned | Official correct | Denoising calls N | Emitted canvases C | Pooled N/C | Mean output tokens | Cap canvases |
|---|---:|---:|---:|---:|---:|---:|---:|
| AIME26 | 10 / 10 | 6 / 10 | 2,700 | 208 | 12.981 | 5,264.0 | 1 / 208 |
| HumanEval | 10 / 10 | 10 / 10 | 362 | 46 | 7.870 | 972.3 | 0 / 46 |
| LongBench-v2 | 1 / 10 | 0 / 1 | 2,780 | 64 | 43.438 | 16,384.0 | 21 / 64 |

Nine frozen LongBench prompts exceed the native 262,144-token context limit.
They remain blocked by explicit user instruction. The one runnable long item
has 185,546 prompt tokens and reaches its original output limit without a final
response. Its trace is confidence-limited on 2,631 of 2,780 calls. This is one
descriptive case, not a long-context population result.

Each request contains one native trailing uncommitted denoising call. The table
uses emitted C; the detailed report also gives completed-canvas N/C and
observed-interval counts. All native sampler settings and RNG order are
preserved. Clean S, S/N and W were not measured.

The observer smoke compares five measured requests: unobserved, minimal state
trace, rich trace, minimal repeat, and unobserved repeat. Output hashes, prompt
tokenization, active sampler-state hashes, final CPU/CUDA RNG states and
forward counts agree. There are 12,320 observer RNG checks across the census,
zero native stopping-predicate reconstruction mismatches, zero request-time
captures or JIT events, and zero LOCAL skips. This is not a full KV/page/layout
snapshot-and-restore certificate.

The corrected FP64 operator study covers 653 candidate records, or 635 distinct
physical supports after collapsing rounded budgets. All finite-difference gates
pass. It uses preserved inherited V31 trajectories, one layer, two heads, and
four sampled query rows. It does not intervene on the new dense trajectories.

| Held-out request family | Response correlation with future attention-bias motion | Current-error correlation | Response selection versus current-error selection: lower / equal / higher future bias motion |
|---|---:|---:|---:|
| AIME26 | 0.906 | 0.917 | 0 / 22 / 2 |
| LongBench-v2 | 0.869 | 0.919 | 5 / 15 / 4 |

These correlated groups provide mixed attention-level evidence. They do not
establish downstream logit, correction/break, stopping or task benefits. The
fixed-bank readout reproduces exactly. All 1,481 partial-support norm bounds
fall below 0.5 without violations; all-kept certificates are exactly one.

- **ResponseGuard:** no-go for a cheap proxy or native benchmark from this
  evidence. The candidate-bank oracle has not shown a consistent advantage
  over current-error selection on held-out attention states.
- **MarginGuard:** not evaluated. An exact downstream influence oracle is
  required before a proxy.
- **ValidityReuse:** the tested norm bound is too conservative to establish a
  useful refresh policy. This does not reject the broader validity approach.

The next engineering gate is exact native full-state restoration and
same-support continuation, followed by downstream influence diagnostics on
dense states. Those tasks remain unvalidated; they are not scientific negative
results. No matched PIECEWISE/all-kept/V31 native panel ran in this stage.

All 22 focused CPU tests and 26 integrity checks pass. Every pinned checkpoint
file matches its size and SHA256. The H100 is idle with no compute clients.
Conservative launched-worker wall time is 1,284.865 seconds, including startup,
warm-up, failed attempts and shutdown; it is not active GPU time or clean
request latency.

See the [detailed report](report003/summary.md),
[method decisions](report003/method_decisions.json),
[commands](report003/COMMANDS.md),
[integrity receipt](report003/integrity_receipt.json), and
[GPU closeout](gpu_closeout.json).

The later [dense native request replay qualification](native_replay001/summary.md)
passes logical KV/input/logit/sampler/RNG equality across all 30 layers on one
item. Physical page assignments differ; in-place restoration remains unqualified.
Its additional worker reservation is 179.500 seconds. The combined conservative
reservation is 1,464.365 seconds. Report003 remains unchanged as the earlier
diagnostic report; this addendum does not promote a selector.
