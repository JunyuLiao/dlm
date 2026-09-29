# v27 (M1/M2/M3 frontier) — Tiers 1–3 done; strong-dense correction; next = kernel or long context

Authority: user v27 doc + A64 doc + chat. GPUs are the user's own: no time window and no budget stop. Record GPU seconds only.
Results: `results/m1_m2_m3_frontier_v27_20260929/`.

## Done
- **CP0:** effective A/R identity, hold-only B, R6/R12, A64, threshold shifts, length gate, layer subsets and cross-layer shared support (GLOBAL pairs/one, LOCAL blocks), compact M2, D_fast.
- **Tier 1 direct cost** (`direct_cost_report.md`, with erratum), component probe, 45-point composed screen.
- **Tier 2 dev** (288 executions) and **Tier 3 frozen comparison** (888/888), in `tier3_report.md`:
  - The per-call ranking reproduces.
  - On new LB questions most sparse arms took more calls (×1.25–1.45), so the primary M3 R6/A64 is slower per request (1.23).
  - M3 R3/A64 is request-neutral.
  - AIME keeps quality for the primary (7/16 = native).
- **Strong dense:** `D_fast` (GLOBAL repeated-KV SDPA) is 0.917–0.926 of native on LB, which equals the best sparse variant. The v27 per-forward gains "vs native" are not sparsity gains.
- **LOCAL sparsity is not viable** (the native LOCAL call is 0.04–0.07 ms; our consumer needs ≥ 0.13 ms).
- **Cross-layer sharing** removes most selection overhead.
- 1,176 redacted generation records are in `generation_records_v27/`.

## Running
- mpk: `v27layers` rerun under deploy `v27_lay2_1cdae21` (consistency copy of the dllm run; not needed for conclusions).

## Next (needs the user's direction)
Two options:
- (a) A block-sparse kernel matching SDPA per-tile cost (BF16 scores, tile sizes, or a FlashAttention-style block-sparse kernel), benchmarked against the repeated-KV SDPA.
- (b) Long-context (RULER 32K/64K) direct cost against D_fast. The score cache needs more than the 4 GiB cap at 64K.
