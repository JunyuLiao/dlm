# Independent external-profile verification — 2026-10-18

The single external Nsight trace completed successfully and reproduces the
original clean worker. Both explicit warmups and all eight measured requests
match the original engine 2 records on output hash, N/C/T, finish reason, native
call counts, frozen input/runtime/method fields, and the entire adapter receipt.
Public/private coverage is 10/10; binding validation has no errors. The original
cache is unchanged. No retry or additional GPU work was performed by this review.

The native source is `6aab86325990edf6e5ab6bb72bcd909781234e97`; the standalone
runner is `2f5687d4c221e9b38cb5897e6e4443374aa81ae5`. The original completion,
frozen source/input files, model inventory file, runner dependencies, copied
schedule, and Nsight executable hashes validate. Exact parity and the full
SQLite activity summary were independently recomputed and agree with the saved
artifacts. `verification.json` binds the protocol, receipt, parity, activity
summary, raw public records, private logs/generation records, and trace files.

## Request attribution

There is exactly one matching sequence of 33 consecutive, same-thread
`cudaDeviceSynchronize` calls, at sync offset 3276. Its 11 request windows cover
the unrecorded implicit warmup, two recorded explicit warmups, and eight measured
requests. Frozen source order and all ten recorded request/prefill/decode
durations agree with that alignment. No GPU activity crosses a proposed request
or prefill/decode boundary. Request-window attribution is therefore qualified,
separately from the successful output-parity check.

All six monitored inference-JIT warnings occur at worker-log lines 204–211,
timestamped 08:39:29–08:39:37 UTC. The first explicit warmup completion is
line 212, the second is line 213, and measured completions begin at line 214.
Thus these observed warnings precede measurement by frozen source and log order.
There are no later monitored JIT warnings, and recorded requests have zero new
CUDA graph captures. This does not prove that every possible compilation event
is monitored.

## Descriptive dynamic-arm trace measurements

These are means over two traced repetitions per input, after excluding all
warmups. Prefill fraction is summed prefill duration divided by summed request
duration. The two inputs have different native trajectories.

| Input tokens | Traced request W | Traced prefill | Prefill fraction of W | Attention-named decode activity sum |
|---:|---:|---:|---:|---:|
| 41,189 | 8.91354 s | 1.09216 s | 12.25% | 1.04332 s |
| 111,616 | 11.09166 s | 4.90992 s | 44.27% | 1.09904 s |

Kernel names provide a heuristic category, not complete attention-module
accounting. The category can contain multiple attention paths and omit other
attention-related work. Activity sums are not necessarily serialized elapsed
time. Neither this category nor qualified request boundaries distinguish dense
refresh, mask construction, held sparse use, commit, or the request critical
path. The 1.04–1.10 seconds must not be presented as removable refresh cost or
predicted speedup. Trace durations are not clean latency; matching output and
work counters does not remove profiler overhead. Two examined inputs do not
establish general context-length scaling.

## Budget and completion

The profile process, CPU validation, SQLite export, and CPU summary all exited
with code 0 without timeout. The common reservation used 326.282699 seconds of its
600-second limit, including those stages. The shared campaign ledger charges
5423.862923 seconds total, approximately 90.40 minutes, below the 7200-second cap.
The ledger still includes the original failed profiling attempt. All reservations
are finalized.

This trace resolves clean-output reproduction and request-window attribution.
A refresh-specific critical-path bound and an incremental clean-latency benefit
remain unresolved.
