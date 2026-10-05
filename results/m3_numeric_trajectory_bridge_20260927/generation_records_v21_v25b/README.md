# Generation records v21–v25b (448 requests, published at the user's request 2026-09-28)

Redacted per-request receipts of every scored or diagnostic generation run since v21, in the same spirit as the v20 export (`../../fan_m1_m3_multidataset_20260927/generation_records/`). Each JSON is one request (`attempt00.json` = first output, used for quality; `warm1.json` = timing repeat of the same cell) and contains:
- the generated `completion_tokens`, `raw_completion`, and the extracted `prediction`;
- `per_canvas` native trajectory (calls, stopping flags) and `total_decoder_calls`, `output_tokens`, `termination_reason`;
- `request_wall_seconds`, GPU timeline, decoding `metadata`, and router `counters` (B0/BO/A/D/H layer-calls, route storage, copy/pad bytes, memory).

**Removed:** original prompts, prompt tokens, prompt hashes, gold/reference answers, credentials, and absolute server paths.

**Kept:** question IDs, seeds, host/GPU IDs, and configuration fingerprints.

`index.csv` has one row per file, with: run, host, cell_id, role, dataset, id, seed, arm, condition, calls, canvases, tokens, termination, request wall, and first-output score (`first_score`, `strict_correct`, `parsed`). `manifest.json` has the sha256 of every file.

| Run | Records | Frozen protocol | What it is |
|---|---:|---|---|
| `v21_natural_zero_pruning` | 12 | `v21_zero_pruning_1ee6332c82377856` | 2 LB × 2 seeds × native / legacy all-kept / new all-kept, first only (not scored) |
| `v23_bootstrap6_preview_aime` + `v23_bootstrap6_lb_rest` | 240 | `v21_bootstrap6_b8b2c59449fd0ebf` | LB 6 × 2 seeds + AIME 4 × 2 seeds × 6 arms × first/warm; LB blocks 4–11 ran under a second binding (method sources byte-identical) |
| `v24_bootstrap6_ruler` | 90 | `v21_bootstrap6_ruler_4dbecd0c10841a11` | RULER 13 tasks × seed 101 × 6 arms first, plus warm on 2 tasks |
| `v25_aligned6_pilot` | 90 | `v25_aligned6_pilot_56d3b98fe1261756` | LB 2 × 2, AIME 1 × 2, RULER 2 × 1 × 6 arms (aligned16 variants) |
| `v25b_aligned_bridge` | 16 | `v25b_aligned_bridge_2dafaf7cd531bff3` | odd-K LB and KDIV-8 control × 2 seeds × logical/aligned16 M3 × first/warm |

Arms:
- `D_native`: unchanged native attention.
- `T_scope`: fresh Junyu value-aware T.
- `M3_R3_A8_incumbent`: M3, R3/A8, score origin 0.
- `M1/M3/B_native_bootstrap2_observe1`: calls 0–1 native (call 1 also observes); after that R1/R3/held with A at 9, 17, ….
- `*_aligned16`: same method with 16-aligned route storage; tokens were identical to logical storage on every tested cell.

Known limits:
- Host and seed are confounded in v23.
- Warm repeats are timing only, not independent quality samples.
- `request_wall_seconds / calls` is amortized cost, not direct forward time.
- AIME `aime26/14` hit the 8,192-token cap in every arm in v25.

Reports: `../v23_bootstrap6/` (`bootstrap6_report.md`, `v24_errata_and_audit.md`, `v24_direct_cost.md`, `v25_route_storage_report.md`, `v25b_memory_and_gate_audit.md`).
