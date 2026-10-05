# User-requested pause, after both short suites completed

Stopped 2026-10-03 04:34; rechecked 04:40 (US Central, UTC-5). Three GPUs have
no compute processes. Six local follow-up/scoring coordinators are stopped.
No automatic restart. Original frozen source, queue scripts, statuses and run
directories are preserved; their old `armed`/`active` fields are historical,
not current liveness. This pause receipt takes precedence.

| Suite | Completed workers | Timed records | State |
|---|---:|---:|---|
| AIME | 32/32 | 960 | Generation complete; unscored |
| HumanEval | 32/32 | 5248 | Generation complete; unscored |
| LongBench | 11/32 | 649 | Partial; b2_native interrupted during warm-up, 0 timed rows |

All 75 closed workers exited successfully. The separately interrupted worker
is not a completed sample or a zero-cost job. Closed reservation time totals
46,574.813729 GPU seconds; its additional process age at stop was 643.62 seconds
(approximate reservation, not measured CUDA compute). All 6,857 exported timed
records report zero new CUDA captures. No accuracy/noninferiority or paired
speed claim is made from this unscored snapshot.

Each suite provides `requests.jsonl.gz` (individual W, prefill, S, S/N, N,
commit/scheduler counts, output length, numeric adapter/method counters and
effective_method), `workers.csv` (accounting sums) and `inventory.json`.
Anonymous sequential question labels are consistent within a suite; they are
not prompt/token hashes. Prompt contents, original item indices, generated
text, gold, private paths and raw fingerprints are omitted. Request-only
numbers cannot reconstruct private grading; raw records/completions and worker
files are backed up privately on the coordinator and retained on each host.

Exporter: `scripts/v30_export_paused_records.py`, three privacy/format tests.
Raw private archives MUST NOT be uploaded. The public gzip files are under
1 MB in total with their inventories.

## Resume

Only resume on user instruction. Reuse the 11 validated complete LongBench
workers; rerun b2_native and the 20 unstarted workers in new run directories,
with original frozen source, binding, arm order, full warm-up and seeds. Merge
these into a new family and strictly validate before scoring. Do not restart
the original supervisor into existing directories or change its status to
complete. AIME/HumanEval need scoring, not regeneration. Frozen follow-up
queues depend on old barrier paths and require a new reviewed continuation.
A private resume manifest records exact original paths and artifact hashes.
