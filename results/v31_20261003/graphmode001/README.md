# v31 graph-mode check: does vLLM's cudagraph mode change DiffusionGemma's denoising? (2026-10-03)

Script: `scripts/v31_vllm_graphmode_check.py`. Logit comparison: `scripts/v31_compare_graphmode.py`.
- Host: dllm, vLLM 0.30.0 / torch 2.13.0+cu130, dense only (no adapter), `--block-size 32`, memory 0.85.
- Data: 3 LongBench-v2 32K items × 2 panel seeds (the 32K cells of `vllm_check_1002/cells.json`).
- Before every request the default torch CPU and CUDA generators are reseeded with a per-cell seed. The official
  sampler draws the initial canvas and all Gumbel noise from them. Each cell runs twice with the same seed.

## Findings

1. **Seeded vLLM runs are exactly reproducible.** In every mode, the two same-seed runs of a cell give identical
   sampler-call counts and identical output-token hashes (`graphmode_*.jsonl`). vLLM's refusal of per-request
   `seed` therefore does not block paired panels: reseeding at the request boundary works for batch 1.
2. **FULL (vLLM default) and PIECEWISE differ from the second denoising call on.**
   - The first call's temperature-scaled logits are bitwise identical (max abs 0.0).
   - At call 2, on item 2 / seed 1111: max abs 9.8, mean abs 0.66, 47 of 256 argmax positions differ,
     mean KL 0.051 (`logits_compare_item2_seed1111.json`).
3. **Eager mode** (`enforce_eager`, no CUDA graphs) differs from both compiled modes already at call 1: mean abs
   0.78, 34 argmax positions, KL 0.045. Kernel and fusion differences in this bf16 MoE model move logits by this
   much, so a logit diff alone does not identify a wrong mode.
4. **Behaviour over whole requests** (6 cells, repeat 0):

   | mode | sampler calls | output tokens | canvases | denoising calls per canvas |
   |---|---:|---:|---:|---:|
   | eager (no graphs) | 1230 | 26,928 | 107 | 10.50 |
   | PIECEWISE | 1053 | 23,449 | 93 | 10.32 |
   | default (FULL decode graphs) | 2000 | 30,223 | 120 | **15.67** |

   - Eager and PIECEWISE agree.
   - FULL needs about 50% more denoising calls per canvas on these long prompts.
5. **The same direction appears in the large V30 short-task panel** (unpaired engine seeds, 240 AIME and 1,312
   HumanEval requests per arm):
   - default dense: N 351 / 55.5 and output 6,593 / 1,344 tokens;
   - the three PIECEWISE arms: N 280–288 / 48.5–48.7 and output 5,346–5,446 / 1,213–1,216 tokens.

   On AIME the steps per canvas are equal (13.6 vs 13.5, as in the HF panels' 13.4); FULL emits more canvases.

## Reading and consequence

- FULL-graph decoding in vLLM 0.30.0's DiffusionGemma deviates systematically from both eager and PIECEWISE
  execution once denoising has state from a previous step. The root cause is not located. The self-conditioning
  MLP runs outside the graph in both modes, so it is not that path.
- **Primary dense reference from now on: vLLM dense in PIECEWISE mode, no hooks, seeded per request.**
  - It agrees with eager.
  - It is the method's own execution mode, so dense and method differ only in the skipped tiles.
  - At long context its per-step time equals FULL's (64K: 40.2 vs 40.2–40.8 ms).
- vLLM's default FULL dense is still reported as a secondary reference, with this deviation stated. This is a
  candidate upstream report.
- Speed is reported as per-forward cost (S/N, plus direct timing on common states) and forward count
  (canvases C × denoising calls per canvas), both on seed-paired requests.
- Small sample (6 cells). The seed-paired panel `v31_paired` (LongBench 32K / 64K, all E14 items × 2 seeds,
  three hosts) measures this at scale.
