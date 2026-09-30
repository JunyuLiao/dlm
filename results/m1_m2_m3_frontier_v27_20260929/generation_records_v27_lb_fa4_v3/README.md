# Generation records, v27 LongBench-v2 long FA4 panel v3 (816 requests)

Redacted per-request receipts of every execution of the v27 LB-long panel on FA4 + piecewise_v1 (v3), 32K and
64K bins, exported with the same redaction as `../generation_records_v27/`. Protocol `v27_long_lb_fa4pw_v3_b5399ab5c46cd59e`
(`../specs/v27_long_lb_fa4pw_v3.json`). Results and scoring: `../fa4_panel_v3/`.

Each JSON is one request, `attempt00.json`, the first output (the one scored). Each record contains:
- the generated `completion_tokens`, `raw_completion`, and the extracted `prediction`;
- the `per_canvas` native trajectory (calls, stopping flags), `total_decoder_calls`, `output_tokens` and `termination_reason`;
- `request_wall_seconds`, the GPU timeline, the decoding `metadata`, and the router `counters`.

**Removed:** original prompts (the 32K/64K contexts), prompt tokens, `prompt_hash`, `prompt_token_hash`,
gold/reference answers, and absolute server paths. The ledgers, which also carry prompt token hashes, are not exported.

**Kept:** question IDs, seeds, cell IDs and configuration fingerprints.

**Leakage caveat.** LongBench-v2 is multiple choice; a `prediction` is a choice letter, and `raw_completion`
may quote or paraphrase the context. Zero leakage of benchmark content is not claimed. Only the prompts and
gold fields themselves are removed.

`index.csv` has one row per file: run (= dataset), host, cell_id, file, role, dataset, id, seed, arm,
condition, decoder_calls, canvases, output_tokens, termination, request_wall_seconds, and `first_score`,
`strict_correct` and `parsed` joined on cell_id from the panel's scored table. For every record the
termination, decoder calls, output tokens and canvases agree with the scored table. `manifest.json` has the
sha256 of every record file.

## Where each record comes from

Only executions whose ledger `run` event has `ok: true` are exported. All scheduled executions qualified.

| dataset | host | run directory | cells | records |
|---|---|---|---:|---:|
| `longbench_v2_32k` | dllm | `v27_long_lbfa4v3_001` | 204 | 408 |
| `longbench_v2_64k` | dllm | `v27_long_lbfa4v3_001` | 204 | 408 |

## Arms

Arm definitions are in `../specs/v27_long_lb_fa4pw_v3.json`. `D_fa4_allkept` is the primary dense reference.

| arm | first | warm |
|---|---:|---:|
| `D_native` | 48 | 0 |
| `D_fa4` | 48 | 0 |
| `D_fa4_allkept` | 48 | 0 |
| `M1_R1_A8_fa4` | 48 | 0 |
| `M2c_R1_A8_fa4` | 48 | 0 |
| `M3_R3_A8_fa4` | 48 | 0 |
| `B_A64_fused_fa4` | 48 | 0 |
| `M1_R1_A64_fused_rp_fa4` | 48 | 0 |
| `M1_R1_A64_fused_dp_fa4` | 48 | 0 |
| `M2c_R1_A64_fused_rp_fa4` | 48 | 0 |
| `M3_R3_A64_fused_rp_fa4` | 48 | 0 |
| `M3_R6_A64_fused_rp_fa4` | 48 | 0 |
| `M3_R6_A64_pairs_fused_rp_fa4` | 48 | 0 |
| `M3_R6_A64_nolast_fused_rp_fa4` | 48 | 0 |
| `G75L0_fa4` | 48 | 0 |
| `G75L15_fa4` | 48 | 0 |
| `G75L30_fa4` | 48 | 0 |

## Known limits

- One host (dllm), first outputs only; seeds 101/202.
- `request_wall_seconds / decoder_calls` is an amortized cost, not a direct forward time.
