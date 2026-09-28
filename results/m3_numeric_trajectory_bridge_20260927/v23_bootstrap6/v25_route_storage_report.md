# v25: aligned route storage — qualified, exact on tested runs, no measurable request gain

**Candidate.** `route_storage=aligned16`:
- The grouped-Q producer is unchanged at the real K.
- Its FP32 bits are copied into a 16-key-pitch score buffer with a −inf tail.
- The projected-V sketch is padded per route; tiles, prefix, reference, legal pairs and the output consumer keep the real K.

The contract ([v25_qualification_contract.md](v25_qualification_contract.md)) was frozen before any measurement. Corrections to v24 wording: [v25_corrections.md](v25_corrections.md). Evidence: [v25_aligned16_evidence.json](v25_aligned16_evidence.json) and `v25_pilot_scored.*`.

## Qualification (deploy `3ee98574`, both H100s)

Parent vs aligned16 bootstrapped M1, M3 and B were replayed on six real states (AIME, LB, RULER per host) from canvas start. That is 18 sequences of full B0/BO/A/D/H cycles with live causal T and current V.

- **Level 2 (behaviour):** 18/18 sequences have every call's logits bitwise equal, and every per-layer decision identical.
- **Level 1 (intermediates):** stored scores are bit-identical by construction.
  - Stored prefix summaries are bitwise equal wherever K was already ÷4/÷8.
  - On odd-K states (K = 365, 13,703), 8–35% of summary elements differ: z max abs 9.5e-7, mu max abs 5.7e-6, no non-finite change.
  - Relative/ULP maxima are large only on near-zero elements (e.g. mu max rel 0.53), so "last-ulp" is not an accurate description.
- **Level 3** (numerical-variant gate) was not triggered, because no logits or decisions changed.

## Direct cost (same states, native brackets, N16 from canvas start)

| State | Keys | aligned16 / logical parent, per call | Sequence ratio M3 / B | vs native, M3 → aligned; B → aligned |
|---|---|---|---|---|
| mpk LB canvas0 | 13,703 (odd) | BO 0.917, A 0.909, D 0.955–0.960, H ~1.00 | 0.977 / 0.980 | 0.989 → 0.966; 0.966 → 0.947 |
| dllm LB canvas1 | 17,540 (÷4) | copy cost BO/A +0.7% | 1.001 / 1.001 | unchanged |
| AIME, RULER | 365–4,216 | ±0.5% | 0.995–1.005 | unchanged |

## Pilot (frozen `v25_aligned6_pilot_56d3b98fe1261756`, 90/90 executions, 0 failures)

Inputs were chosen by sha256(id) before any output, with host counterbalanced by question and seed. The pilot covers LB 2 questions × 2 seeds, AIME 1 × 2, and RULER 2 tasks × 1 seed (warm repeats on one task).

- **aligned16 M3 vs logical M3:** complete tokens identical on **all 8 first-output cells**, with identical calls and canvases. Paired warm request time is LB 1.001 (0.998–1.003), AIME 0.992, RULER 0.998. The component gain does not reach the request level: BO + A are ~14% of calls, and only odd-K canvases benefit.
- **Quality**, identical across all six arms: LB 2/4, AIME 0/2 (every arm hit the 8,192-token cap on `aime26/14`), RULER 2/2. These are 2-question samples, not noninferiority evidence.
- **LB paired warm time vs native:** fresh T 0.869, aligned M1 0.877, aligned B 0.911, M3 ~0.99 under either storage. Descriptive only.

## Decision

- `aligned16` is a qualified implementation: same logits, decisions and tokens on every tested state and run. It is kept selectable, but **not claimed as a speedup**: its measured request effect is ~0.
- The parallel-summary builder probe (v25 §5.2) was **not run**. Even removing all route time at BO/A bounds the request gain at ~2–4% on LB. That could not change the observed ordering (T and B faster than M3 at similar quality in both v23 and this pilot), and only 2 of 512 campaign executions remain.
- Overall: unchanged M1/M3 semantics can run the observation path cheaper without changing outputs, but the dominant request-level differences come from call counts. This v25 evidence does not establish an incremental M3 advantage over fresh T or matched B.
