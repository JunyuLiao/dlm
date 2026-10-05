# v23 corrections to v22 wording (existing measurements unchanged)

These corrections narrow claims in `q16_calibration.md`, `morning_brief_zh.md` and `fan_update_zh.md` (v22, `6b795ca6`). Those files are preserved as published. No new GPU stage was run for these caveats.

1. **The zero-attention oracle is not a rigorous speed ceiling.**
   - Returning zeros from the five GLOBAL layers changes the hidden states downstream, and potentially MoE routing and expert load. That applies to GLOBAL as well as LOCAL.
   - The spy also changes event, launch and allocation work between paths.
   - `no_global_attention` (LB median 0.84×) is therefore a non-equivalent intervention, not an upper bound on what a GLOBAL support policy can save.
   - The in-forward CUDA event sums (LB GLOBAL ≈ 21–26 ms of 140–169 ms) are instrumented cost evidence on those states. They do not prove a universal Amdahl limit.
2. **The AIME share has a narrow scope.** The AIME attention-share samples had only ~0.4–0.6K keys (early canvases). The 1.3% GLOBAL share describes those states, not every AIME step, and not later canvases with longer prefixes.
3. **The Q16 work saving is interpolated.** The ~4–5% equal-error Q16 work saving and the "<0.5% of a forward" figure are interpolations and arithmetic. They are not an executed production policy or a theorem. Stopping Q16 promotion remains justified by the frozen calibration rule alone: the selected point retains 1.158× coarse work.
4. **The all-kept arm does not settle attribution.** Its extra calls (v20 3009 vs native 2360) show that a no-pruning numerical path also perturbs trajectories. That does NOT identify what fraction of sparse M3's extra work comes from numerics versus pruning/history. The zero-pruning changes went in both directions. A CI crossing 1 is neither evidence of a hidden positive effect nor evidence of equivalence.
5. **No sample-size guarantee.** The "~50 questions for ±5%" figure came from a back-of-envelope cluster SD taken from one 6-cluster interval. It is not a paired-variance power calculation, and it guarantees nothing. Warm repeats are timing data, not independent quality samples.
6. **The residual is not all MoE.** The part of the forward not measured as GLOBAL attention is not by definition MoE/projection. It includes LOCAL attention, norms, embeddings/LM head, launch gaps and everything else in the decoder.

Unchanged: the v20 same-input LB full-forward saving of coarse M3 (≈4–9% on selected states) stands. The v22 negative concerns only the incremental Q16 geometry.
