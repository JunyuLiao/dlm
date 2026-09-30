# v27 exploration plan: maximising total gain on the FA4 + piecewise substrate

**Status.** Written 2026-09-30 while the LB-long FA4/piecewise panel (v3, 17 arms, dllm) runs. Each item says:
- which axis it moves: per-forward cost P, or the number of forward calls N;
- its expected size, from the measurements in `substrate/`;
- its novelty or collision status.

The paper's identifiable contribution stays Fan's M1/M2/M3 selector. Items marked as collisions are engineering stacked on top and are reported as such.

## Where the time goes (64K, piecewise, from `substrate/piecewise_64k.jsonl` and `official_baseline/`)

- **A dense decoder forward is about 40 ms.** GLOBAL attention (FA4) is about 12.5 ms of it (31%). The rest is MoE, projections, LOCAL attention and the LM head, which our method does not touch.
- **Per-forward ceiling from GLOBAL key sparsity alone is about 0.72×.** A held FA4 block-sparse call costs about 29 ms.
- **Per-canvas overhead of B/M3.** Each canvas pays one bootstrap-dense call and one observation call. The observation call measured about 78 ms for B (route not pipelined) and about 66 ms for M3 R6 (pipelined).
- **The fused observation kernel is not the problem.** In `observe_bench.json` it costs +1.3 ms per layer over FA4 dense at 60K, and more split-KV does not help. The remaining roughly 6 ms per layer is outside the kernel; the in-model breakdown is in `v27_observe_profile`.

## P: per-forward items

| item | expected effect | status |
|---|---|---|
| **B with the bit-identical pipelined route** (`B_A64_fused_rp_fa4`) | the B observation call from about 78 to about 66 ms: roughly −12 ms per canvas | free and exact; missing from v3; goes into a supplementary panel |
| **Observation-call breakdown** (sketch projection, summary allocation, route, publication) | target the ~6 ms per layer outside the kernel | profiling on mpk; engineering of our own method |
| **Observe at call 0** (merge bootstrap-dense into the observation call) | −1 dense call per canvas, about −8 to −11 ms per canvas | close to MAGE's step-1 observation. The DiffusionGemma initial canvas is random tokens, not masks, so quality is uncertain. Low priority. |
| **Carry held maps across canvases** (refresh every K canvases or on drift) | removes most per-canvas bootstrap and observation cost: about −57 ms per canvas at 64K (about 14%) | **collision.** Prefilling-dLLM (2606.10537) selects once after prefill for the whole generation; PulseCol (2605.20813) refreshes periodically. Engineering only. |
| **Query-side reuse for stable canvas tokens** (reuse prefix-attention output and LSE) | could cut GLOBAL prefix work for accepted tokens | **collision** with LoSA (2604.12056) and FlashBlock (2602.05305). Not pursued as a contribution. |
| **Compile the post-prefill encoder** (as the official path does) | −about 20–30 ms per canvas for every arm, dense included | baseline hygiene: shrinks the denominator equally and never counts as a contribution. Needs dynamic-KV handling. |

## N: forward-count items

| item | expected effect | status |
|---|---|---|
| **Sparse-induced step inflation** | calls per canvas of each sparse arm vs D_fa4_allkept; read from the v3 panel's paired cells | measurement |
| **Density gate** (`ent0.05`, `ent0.02`, `stall2`, `ent0.05_stall2`): the rest of the canvas runs dense once the stop statistic is near its threshold or acceptance stalls | removes sparse-induced extra steps near convergence | implemented. It needs the sampler-state mirror on the substrate (fixed in 2e7a0f7) and calibration from entropy trajectories (`v27_entropy_probe`, mpk). |
| **Official stopping knobs** (`confidence_threshold`, `stability_threshold`, `entropy_bound`, `max_denoising_steps`) | fewer steps for every arm | not a method contribution: any knob change applies to dense too. Measured as a dense-only frontier for context. |

## Order of work

1. mpk, in idle windows while chw is not using it:
   - observation-call breakdown;
   - entropy trajectories for D_fa4_allkept, B_rp and M3 R6, with and without gates;
   - pick gate thresholds on development items only. These items are never the scored ones.
2. After v3 finishes on dllm, run a supplementary panel with the same items and seeds:
   - D_fa4_allkept and D_fa4 as references;
   - B_rp;
   - the calibrated gate variants of B_rp and M3 R6;
   - any observation-call fix that is exact.
3. If a cross-canvas carry is built, label it as a Prefilling-dLLM/PulseCol-style engineering stack and never as the contribution.
