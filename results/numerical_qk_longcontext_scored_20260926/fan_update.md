# DRAFT for Fan (not sent): v15 scored long-context diagnostic

**Why.** On AIME (v14), the tested frozen GLOBAL-only configurations showed no execution gain; GLOBAL attention is a small share of a step at those prefixes. v15 checks, with real scored answers, whether the arithmetic saving appears where long prefixes give it room. The main paper scope is still for us to discuss.

**Setup.**
- 12 untruncated LongBench-v2 questions, 10–20K input tokens, 4 domains, selected score-blind from the existing development pool. Three were seen in earlier project panels.
- Seeds 17/29; native adaptive denoising; thinking ON; 8192-token output budget; pinned NeMo prompt and MCQ scorer on the final answer channel.
- Five frozen v14 arms, GLOBAL-only with native LOCAL layers: D_native, T_G_original, T_P, B8_P, CVM_T.
- 240/240 executions, all warm repeats accepted, no length caps.

**Results** (ratio <1 = faster; summed-time ratio = pooled calls × pooled time/call):

| | correct / 24 | vs D_native summed | calls × per-call |
|---|---:|---:|---|
| D_native | 12 | 1 | |
| T_G_original | 14 | 1.002 | 1.015 × 0.988 |
| T_P | 14 | 0.933 | 0.937 × 0.996 |
| B8_P | 14 | 0.893 | 0.967 × 0.923 |
| CVM_T | 14 | 0.876 | 0.930 × 0.942 |

- **CVM_T vs B8_P** (the incremental question):
  - quality 14 vs 14 (paired 2 vs 2);
  - summed time 0.981 [0.82, 1.18], geometric 0.953 [0.82, 1.12], with the sign flipping between seeds;
  - per call, CVM_T is ~2% costlier than B8_P.
- Its add-only restorations keep about 0.48–0.52 of the prunable prefix, vs B8_P 0.33–0.36 and fresh T ~0.40 (two diagnostic states).

**Reading (Decision A).**
- At long context, simple anchor-bitmap reuse gives a real per-call saving: ~7.7% of complete-request time per decoder call. Accuracy is not worse on this small panel, though that is not a noninferiority result.
- Live temporal protection (CVM-T) adds no demonstrated quality or latency benefit over that bitmap. Its independent contribution remains unestablished.
- Bitmap reuse itself is close to prior art (one-observation reuse, e.g. MAGE).
- The AIME negatives stand.

**Proposed decision.** Stop CVM-T as the contribution. Keep the long-context bitmap-reuse speedup as an engineering observation, and discuss whether any mechanism has a plausible increment over B8_P before spending another round.
