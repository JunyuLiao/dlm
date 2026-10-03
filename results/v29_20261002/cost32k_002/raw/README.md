# Saved diagnostic evidence: 002

These are the full per-arm measurement files actually emitted by the frozen
worker, plus sanitized single-request records and terminal receipts. They are
not a newly reconstructed trace. All four arms are included; numerical scope
measurements, call counts, allocator counters, effective-method receipts and
available KV layout/shape/stride checks are retained. Per-request profiled
wall/prefill/decode durations remain only diagnostic, with explicit flags;
these records cannot enter a formal speed/quality summary.

The run inventory was 214,318 bytes including private files. No Chrome trace
or persisted per-event timeline was found. The frozen profiler used
`record_shapes=False`, `with_stack=False`, `profile_memory=False`; it summarized
its in-memory events before process exit. Individual event timestamps and CUDA
event durations were not saved. No tensor values or prompt/token content was
saved in these profiler files. Operator input shapes and raw trace args were
not persisted; KV-layout receipt shapes are separate available evidence.
A complete Chrome trace cannot be recovered from these files. No GPU rerun,
synthetic timestamps, inferred event ordering or reconstructed trace was used.

The twelve sanitized evidence files total 36,791 uncompressed bytes;
their deterministic gzip copies total 13,428 bytes. Each compressed file
is well below25MiB. Read plain `.json`/`.jsonl` directly or decompress the
corresponding `.gz`. `manifest.json` lists each original file size, public size,
removed field names and SHA256 of the **public compressed artifact only**.
It does not contain hashes of private originals. Gzip roundtrip verification
passed for every file; private source files were read only and left unchanged.

Redaction rules:

- Remove host/GPU UUID, run/request/item identity, private index, adapter hashes,
  method fingerprints, private binding/source-hash metadata and credentials.
- Remove any private absolute path or private64-hex digest in retained strings.
  Public deploy commit and protocol/software settings remain for provenance.
- Keep numeric durations/counts, tensor layout shapes/strides and token **length
  counts**; there are no token IDs, decoded tokens, prompt, gold or generated text.
- Do not ingest or publish completions, launch/binding plans or worker logs.
  Logs may contain private paths; completions contain generation text.
- Add explicit diagnostic/overhead flags to records/terminal receipts. Preserve
  the original profiler field names and measurements without correcting old
  instrument readings or fabricating missing measurements.

Known interpretation limits still apply:002 misses actual shared-backbone
GLOBAL/LOCAL classification; its `host_copy_enqueue` name means CUDA copy API
CPU time, which can include implicit waits, not pure enqueue.003 events are
inclusive entry-stream spans, include host dispatch gaps and instrumentation,
and nested/side-stream scopes overlap. Fused observation includes normal dense
output; its inclusive share is not its incremental overhead. Missing scopes
mean unknown, not zero. Neither dataset is an accuracy evaluation.


Timing boundaries: each terminal receipt's `gpu_reserved_seconds` is the
worker wrapper's internal perf-counter span after its initial imports and frozen
validation, through panel execution/receipt checks. The aggregate's per-arm
reservation is measured externally around subprocess launch and completion,
so it also includes process startup, profiler setup/imports and final teardown.
The campaign reservation includes inter-engine supervisor/cache/idle-check work.
These nested spans are different and must not be added or substituted. The
resource ledger uses the more complete external supervisor reservation:
1081.689751653932 seconds.
