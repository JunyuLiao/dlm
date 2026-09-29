# Generation records, v27 long-context RULER panel (1,144 requests)

Redacted per-request receipts of every execution of the v27 long-context RULER panel (32K and 64K),
exported with the same redaction as `../generation_records_v27/`. Protocol `v27_long_ruler_42883cd36fcb51c5`
(`../specs/v27_long_ruler.json`). Results and scoring: `../long_ruler_panel/`.

Each JSON is one request. `attempt00.json` is the first output, which is the one scored for quality.
`warm1.json` is the timing repeat of the same cell. Each record contains:
- the generated `completion_tokens`, `raw_completion`, and the extracted `prediction`;
- the `per_canvas` native trajectory (calls, stopping flags), `total_decoder_calls`, `output_tokens` and `termination_reason`;
- `request_wall_seconds`, the GPU timeline, the decoding `metadata`, and the router `counters`.

**Removed:** original prompts (the 32K/64K contexts), prompt tokens, `prompt_hash`, `prompt_token_hash`,
gold/reference answers, and absolute server paths. The ledgers, which also carry prompt token hashes, are not exported.

**Kept:** question IDs, seeds, cell IDs and configuration fingerprints.

**Leakage caveat.** RULER answers are strings that occur in the context (for example needle values),
and a correct `prediction` or `raw_completion` reproduces them. These records therefore reveal part of
the answer for every cell the model got right, and partial context content in general. Zero leakage
of benchmark content is not claimed. Only the prompts and gold fields themselves are removed.

`index.csv` has one row per file: run (= dataset), host, cell_id, file, role, dataset, id, seed, arm,
condition, decoder_calls, canvases, output_tokens, termination, request_wall_seconds, and for the
first output `first_score`, `strict_correct` and `parsed`, joined on cell_id from
`../long_ruler_panel/cells.csv`. `termination` is the record's own `termination_reason`. For every
first output it equals the `termination` column of `cells.csv`, and decoder calls, output tokens and
canvases also agree. `source_run` names the run directory that produced the record. `manifest.json`
has the sha256 of every record file.

## Where each record comes from

Only executions whose ledger `run` event has `ok: true` are exported. All scheduled executions qualified.

| dataset | host | run directory | cells | records |
|---|---|---|---:|---:|
| `ruler32k` | dllm | `v27_long_ruler_006` | 143 | 286 |
| `ruler32k` | mpk | `v27_long_ruler_008` | 143 | 286 |
| `ruler64k` | dllm | `v27_long_ruler_008` | 143 | 286 |
| `ruler64k` | mpk | `v27_long_ruler_008` | 143 | 286 |

- The dllm half of `ruler32k` comes from run 006, whose ledger was read only through its first
  `worker_end`. The aborted 64K part of run 006 is not exported. The dllm half of `ruler64k` comes from run 008.
- Between runs 006 and 008 only memory changed: the O(n^2) prefill mask mapping is elided in 008.
  See `../long_ruler_panel/README.md`.

## Arms

Arm definitions are in `../specs/v27_long_ruler.json`. `D_c64` is the strongest dense baseline.

| arm | first | warm |
|---|---:|---:|
| `D_native` | 52 | 52 |
| `D_c64` | 52 | 52 |
| `T_scope` | 52 | 52 |
| `M1_R1_A8_c64` | 52 | 52 |
| `M2c_R1_A8_c64` | 52 | 52 |
| `M3_R3_A8_c64` | 52 | 52 |
| `M1_R1_A64_one_fused` | 52 | 52 |
| `M2c_R1_A64_one_fused` | 52 | 52 |
| `M3_R6_A64_one_fused` | 52 | 52 |
| `M3_R3_A64_one_fused` | 52 | 52 |
| `B_A64_fused` | 52 | 52 |

## Known limits

- Host is assigned as (question index + seed index) mod 2 within each dataset, so host and seed are not independent.
- Warm repeats are timing only, not independent quality samples.
- `request_wall_seconds / decoder_calls` is an amortized cost, not a direct forward time.
