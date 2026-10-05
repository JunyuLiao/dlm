# v25b: aligned16 exposure audit, code guards and odd-K bridge

Append-only. This corrects the scope of the v25 conclusions; v25 data are unchanged.

## CP0: which pilot runs actually exercised the optimized path

From raw pilot receipts (`v25b_alignment_exposure.{csv,json}`, `scripts/v25b_alignment_exposure.py`, 21 tests):
- **Phase conservation:** B0/BO/A/D/H conserve exactly in all 90 pilot runs.
- **Aligned counters:** copies, allocated buffer bytes and sketch pads match the per-canvas prediction exactly (zero for logical arms).
- **Real K per canvas:** K_c = prompt tokens + 256·(c+1), because every non-final canvas commits 256 tokens. So K mod 16 is constant within a request; this was verified against the counters.

| Pilot input | Prompt tokens | K mod 16 | KDIV | Odd-K path | aligned M3 affected BO+A / D calls |
|---|---:|---:|---:|---|---|
| LB …aacb1f149 | 19,292 | 12 | 4 | **no** | 9 / 13 |
| LB …067c4480 | 13,720 | 8 | 8 | **no** | 16 / 18 |
| AIME /14 | 131 | 3 | 1 | yes (tiny K) | 76 / 120 |
| RULER multikey_1 | 3,315 | 3 | 1 | yes | 1 / 1 |
| RULER single_1 | 3,949 | 13 | 1 | yes | 1 / 0 |

**Neither pilot LB input exercised the KDIV = 1 route path.** The v25 LB request ratio of 1.001 therefore measured only copy cost on already-fast classes. It says nothing about odd-K long-context requests.

On odd-K AIME and RULER, the route per layer is small (≈0.2–1.0 ms), so the expected effect is below single-warm noise. Pilot LB bootstrap M3 had 41 canvases, so B0 = BO = 41; A/D/H = 24/87/233 model calls, totalling 426.

## CP1: code guards (commit `cac0cbd4`)

**A. Physical memory accounting.**
- `ScoreCache.physical_bytes` counts resident storage: a pitched buffer counts in full, and a shared storage counts once.
- `reserve_physical` checks the budget before allocation.
- The transient peak includes the old entry + the logical producer output + the new pitched buffer.
- `aligned_pad_bytes` was always the total **allocated** pitched-buffer bytes. It is kept as-is and aliased to `aligned_buffer_bytes_allocated`, and `aligned_extra_pad_bytes` and `aligned_copy_bytes` are added. None of these is measured DRAM traffic.
- The peak updates only at publication, so there is no per-call host work.

**B. Qualification gates.**
- `_compare_call` / `_sequence_verdict` require identical layer sets equal to the expected GLOBAL set. B0's empty set is the only legal empty case.
- They also require identical call indices, phases, clocks and summary-layer sets.
- Execution, level 1, level 2 and pilot eligibility are reported separately. A failed gate records an error, so the stage exits non-zero while keeping the report.
- Negative tests cover a missing layer, a flipped skip bit, a changed logit, a phase mismatch, a missing call and a missing summary layer.
- The published v25 comparisons were re-audited: all 186 call records covered complete layer sets, so the reported zero mismatch was not vacuous.

## CP2: bridge (frozen `v25b_aligned_bridge_2dafaf7cd531bff3`, 16/16, 0 failures, separate STATE allocation)

Design:
- **Inputs:** the predeclared odd-K profile input `…067c5456` (K mod 16 = 7). The frozen LB six contain no K mod 16 = 0 input, so the control is the highest-KDIV one, `…067c4480` (KDIV 8).
- **Runs:** logical vs aligned16 bootstrapped M3, seeds 101/202, first + warm. Host is counterbalanced, and both arms of a cell run on the same GPU.

| Input | Host / seed | Tokens (first & warm) | Calls | aligned copies / sketch pads | Warm request aligned / logical | Correct |
|---|---|---|---:|---|---:|---|
| …067c5456 (odd K) | mpk / 101 | identical | 230 = 230 | 165 / 430 | **0.9742** | both no |
| …067c5456 (odd K) | dllm / 202 | identical | 324 = 324 | 220 / 590 | **0.9739** | both no |
| …067c4480 (KDIV 8) | dllm / 101 | identical | 100 = 100 | 80 / 170 | 1.0011 | both yes |
| …067c4480 (KDIV 8) | mpk / 202 | identical | 89 = 89 | 70 / 160 | 1.0017 | both yes |

Tokens and calls are identical, so call count does not explain the time difference.
- On the odd-K input the request saving is **≈2.6% on both hosts and both seeds**, consistent with the direct same-state sequence ratio 0.977.
- On the KDIV-8 control the copy cost is **+0.1–0.2%**.

Caveats: one warm repeat per cell and one odd-K and one control question. This is a mechanism check, not a population benchmark; the KDIV-2 class was not measured.

## Corrections to v25 wording

1. "No measurable request gain": this holds only for inputs whose K is already ÷4/÷8. On the tested odd-K LB request the gain is ≈2.6%. The v25 pilot never exercised that path on LB.
2. Direct profile figures: the odd-K state's model-forward sums were native 2268.8 ms, logical M3 2243.7 ms and aligned M3 2191.4 ms. That is a ≈2.3% saving of the whole forward sequence, not 8–9%. The 8–9% refers only to BO/A calls, and must not be multiplied again by the BO/A share.
3. "T and B faster than M3 at similar quality in v23": wrong for B. v23 matched B was 4/12 vs M3 6/12. Only the 2-question v25 pilot had equal quality.
4. "2–4% ceiling" for a parallel summary builder: this is a local estimate under stated assumptions, not a universal bound. The builder is neither implemented nor measured, so no claim is made about whether it would change any ordering.
5. The 18 qualification sequences are teacher-forced replays of up to 16 calls, not 18 natural generations, and not every sequence reached call 9 (A).

## Status

`aligned16` stays a selectable, qualified implementation. Its measured effects are logits, decisions and tokens identical on every tested state and run; about 2.6% faster on the odd-K LB request; and +0.1–0.2% on the aligned-class control. It is an execution change, not a method contribution. The M3-versus-native/T/B method comparison is a different contrast and is not changed by this result.
