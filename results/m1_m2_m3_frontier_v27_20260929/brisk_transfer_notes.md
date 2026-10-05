# What we take from BRISK-DLM, and what we do not

**Source and scope.** The manuscript Fan shared on Slack on 2026-09-27: "From Position Risks to Block Survival: Faster Generation for Diffusion Language Models" (Chen, Wan, Yu, Lai).
- Fan added that "its prefix-condition paradigm does not generalize to all DLMs".
- I did not read the PDF in this session. These notes rely on the section summary in the user's v27 task file (§§2.2–3, §3.3, Appendix F / Table 15 / Figure 9).
- Nothing here claims acceptance at a venue.
- The manuscript is not committed or redistributed.

## Transferable ideas

1. **Optimize progress per unit of real execution cost, under a quality constraint.**
   - BRISK optimizes verified progress per model evaluation.
   - Our analogue is a quality-constrained complete-request time.
   - Maximum sparsity, minimum local L2 and fewest calls are each only one column of that target.
2. **Separate "does the algorithm help" from "how cheaply it executes" (Appendix F).**
   - Their reference implementation raised tokens per forward (TPF) 3.37 → 3.74, yet lowered tokens per second (TPS) 90.0 → 60.4.
   - Only the optimized execution (TPS 97.4) turned progress into a net gain.
   - For us, this is why v27 measures the same selection policy at several execution costs:
     - the pure sparse forward with the bitmap given for free (`prepared_support_floor`);
     - the held call H;
     - the redecision call D;
     - the anchor call A.
   - The compact M2 is an execution change, not a new selector.
3. **Their component timing is not end-to-end speedup.** Figure 9's 17.3 → 0.74 ms is head runtime. Likewise, our route-kernel or attention-core numbers are not request speedups.

## Not transferable

- **The prefix-acceptance product.** DiffusionGemma has no "first rejected prefix position invalidates everything after it". It accepts via its native stable/confidence and entropy-budget rules. We do not copy position-risk weights, freeze accepted tokens, lower stop thresholds or train a corrector.
- **Lossless guarantees.** BRISK preserves its target distribution through its verifier/rejection correction. Sparse attention in DiffusionGemma has no such guarantee, so our quality must be measured on complete answers.
- **Their numbers.**
  - Their performance windows use a fixed 2,048 client-visible tokens on A100 with MATH-500.
  - Our evaluation uses natural EOS and native adaptive stopping.
  - Neither the fixed-2048 setup nor their "up to 37.4%" figure transfers.

## Diagnostics planned under this lens (no method change)

- Committed output tokens per denoise call is reported only together with complete-answer quality and W, because long repetitive thinking also inflates it.
- For each canvas we record native-stop vs candidate-stop disagreements where the step-level state exists. Where only final tokens exist, the entry is marked NOT_RECORDED rather than reconstructed.
