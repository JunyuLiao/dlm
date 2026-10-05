# AIME four-arm generation/scorer qualification

Source `1931db5a6540409946641ebb0b04c06507d48402`. The strict qualification proof passed for all four arms on dlm2, with the unchanged frozen generation binding and an independent CPU scorer on mpk. Formal generation has not started.

Each arm used one fresh engine, seed 29001, the frozen longest prompt, one warm request and one timed request, with the complete 8192-token budget. This is a pipeline qualification, not accuracy or speed evidence. All four timed outputs were capped and unparsed and were scored incorrect; valid unanswered/truncated outputs are retained, not used to reject or filter the task.

The successful dense worker from attempt 001 was reused by exact original binding and closed-artifact byte pins. Attempt 002 ran only the other three arms. The earlier supervisor cleanup error and its immutable artifacts remain retained. The bounded idle gate rejected active compute PIDs and required two stable readings before each new worker.

| Arm | Reserved GPU seconds | Timed request wall seconds | Denoise forwards | Correct / capped / unparsed |
|---|---:|---:|---:|---|
| dense | 239.994347 | 9.668869 | 512 | 0 / 1 / 1 |
| method | 153.416628 | 9.339806 | 427 | 0 / 1 / 1 |
| native | 237.570613 | 10.126167 | 480 | 0 / 1 / 1 |
| allkept | 149.425910 | 8.590140 | 403 | 0 / 1 / 1 |

Total supervisor reserved GPU time, including the retained original dense worker, was 780.407499 seconds. New attempt 002 reserved 540.413152 seconds. This process span includes engine setup, warmup, the timed request and shutdown; request wall/decode spans are narrower and must not be confused with resource use. Forward counts are the runner-observed model call counts, including one unused speculative denoising call per timed request, and are not accepted-token counts.

The strict scorer mirrored only required frozen source/config/provenance and closed generated artifacts into a new private directory. Every original path was explicitly mapped and byte-verified; the original generation binding was preserved. Gold stayed at its existing scorer-only location on mpk. CPU scoring used the registered scorer interpreter with CUDA hidden, zero GPU launches, and a passing official-scorer public toy self-test. Raw inputs, completions, gold, artifact hashes and device identity remain private.

The initial CPU export attempt stopped before transfer because its source-file allowlist omitted three frozen C++/CUDA/header files. The allowlist was corrected to the verified required source types; no generation source, tolerance, sampler, or output was changed.

Additional numeric evidence is preserved in `requests.csv` and `request_receipts.jsonl` (four timed request receipts), `workers.json` (four closed workers), `arms.csv` (arm-level scored aggregates) and `summary.csv` (scalar qualification metadata). These are explicit field allowlists derived from retained original closed artifacts, not raw private proof exports. Anonymous receipt labels identify an arm and its timed request; original task indices, run identifiers, device identifiers, private paths, hashes, prompts, token arrays, completions and gold are omitted. Token counts are scalar lengths only.

Per-request score fields are taken from the strict scored arm aggregate because each qualification arm has exactly one timed request; no multi-request score assignment is inferred. Original optional nested `receipts` were unavailable (`null`), and this is recorded rather than manufacturing a detailed execution trace. Compilation/capture counters and denoising/commit counts come directly from the original request records. `workers.json` separately preserves the worker-reported and supervisor-reserved GPU spans. Warm-request records were not present in the retained timed `records.jsonl`, so these exports do not fabricate warm-request detail. Original private files are retained unchanged.
