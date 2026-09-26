# v15 morning brief — scored LongBench-v2 long-context development panel (native adaptive)

**Scope.** This is a frozen, scored diagnostic on the existing LongBench-v2 development pool. It is NOT a replacement for AIME, and the AIME negatives (v14) stand.
- Panel: 12 untruncated questions, 10–20K input tokens, 4 domains, two length bins × seeds 17/29 × five frozen v14 arms. Attempt 0 plus one warm repeat.
- Execution: **240/240, 0 failures, 120/120 warm repeats accepted** (same tokens, calls and termination; no compile or new library load).
- Setup: 3 executions on an excluded item.
- Budget: GPU 2.87 of 6 h; wall about 3.2 of 8 h; 243 of 250 executions.
- Every first output ended at EOS (no length caps), with about 3.2K output tokens each.
- **Verdict: Decision A.** Simple reuse and CVM_T both come in under dense on this panel, but CVM_T's incremental value over B8_P is not established.

## Q1. Did complete requests, including prefill, become faster?
Primary time is the accepted warm request wall around `generate`, which includes the real prefill. Ratios are candidate/reference (<1 = faster). Intervals are question-cluster bootstrap 95%, descriptive only.

| pair | summed-time ratio | = pooled calls × pooled time/call | paired geometric |
|---|---|---|---|
| **CVM_T / D_native** | **0.876** [0.767, 0.998] | 0.930 × 0.942 | 0.913 [0.826, 1.002] |
| **CVM_T / B8_P** | **0.981** [0.815, 1.180] | 0.962 × 1.021 | 0.953 [0.817, 1.124] |
| B8_P / D_native | 0.893 [0.775, 1.038] | 0.967 × 0.923 | 0.958 [0.826, 1.097] |
| CVM_T / T_P | 0.939 [0.797, 1.134] | 0.993 × 0.946 | 0.941 [0.809, 1.117] |
| CVM_T / T_G_original | 0.874 | 0.917 × 0.953 | 0.907 [0.751, 1.094] |
| T_P / T_G_original | 0.931 | 0.924 × 1.008 | 0.963 [0.844, 1.122] |

- Leave-one-question-out, summed CVM_T/D: 0.848–0.908.
- The CVM_T/B8_P ordering is unstable:
  - per seed, geometric 1.05 (seed 17) vs 0.86 (seed 29);
  - per bin, summed 0.965 (10–15K) vs 0.995 (15–20K);
  - leave-one-out, summed 0.915–1.039.
- True generation latency after prefill and TBT are N/A (no qualified boundary or committed-output events).

## Q2. Did complete answers retain quality, and how many different questions support that?
Task score (NeMo MCQ on the final answer channel) and strict score (task-correct AND EOS) are identical here because there were no caps.

| arm | correct / 24 | seed 17 / 29 | questions with any correct | unparsed |
|---|---:|---|---:|---:|
| D_native | 12 | 7 / 5 | 7 | 0 |
| T_G_original | 14 | 7 / 7 | 7 | 0 |
| T_P | 14 | 6 / 8 | 8 | 0 |
| B8_P | 14 | 8 / 6 | 8 | 1 |
| CVM_T | 14 | 7 / 7 | 8 | 0 |

- **CVM_T vs B8_P:** paired 2 vs 2. Every disagreement comes from 4 of the 12 questions.
- **CVM_T vs D_native:** paired 4 vs 2, from 4 questions.
- 4 questions are wrong in every arm and 3 are right in every arm. This is 12 distinct questions (24 first outputs per arm), not 240 samples.
- No noninferiority or equivalence claim is made.

## Q3. How much comes from cost per call vs work count?
Exact pooled log decomposition of the summed-time ratio:
- **CVM_T/D:** calls 55%, per call 45%.
- **B8_P/D:** calls 30%, per call 70%.
- **CVM_T/B8_P:** in log points, calls −0.039 + per call +0.020 = net −0.019.

The per-call factor is amortized request wall per decoder call (prefill, anchors, commits included), not pure GPU latency. Call-count differences are trajectory effects; their sign is not stable across seeds.

**Counterfactual estimates** (labeled; not measured request time) use the v14 teacher-forced LongBench step costs priced on this panel's own native canvas histories (`phase_accounting.json`):
- per decoder step: CVM_T 0.929–0.942, B8_P 0.900–0.917, T_P 0.993–0.997, T_G 0.989–0.993;
- these agree with the measured pooled per-call factors (CVM_T 0.942, B8_P 0.923, T_P 0.996, T_G 0.988).

**Measured phases:** anchors are 15.1% of routed GLOBAL calls for B8_P and CVM_T. There was no native fallback. Peak allocated memory is 58.4 GiB in every arm (model + KV). CVM_T's live metadata at request end is ≤30.9 MB (B8_P 29.0 MB); that is not a peak measurement.

## Q4. Does CVM add anything beyond simple bitmap reuse?
**Not demonstrated.** Quality is the same (14 vs 14). The time difference sits inside noise with an unstable sign.
- Per call, CVM_T is ~2% costlier than B8_P (1.021).
- The predetermined attribution twins explain why (`attribution_twins.json`; a separate diagnostic from the D_native-captured canvas-1 states, where the D_native twin reproduced the formal call counts 13/13 and 16/16):
  - Mean executed fraction of the prunable prefix: CVM_T 0.48–0.52, fresh T 0.40–0.41, T_P 0.39–0.41, B8_P 0.33–0.36.
  - CVM_T made 16–19K tile restorations per canvas.
  - Add-only restoration within an A8 epoch accumulates support beyond what fresh T keeps at any single step. CVM_T therefore does more attention work than B8_P, and even more than fresh T.

## Q5. Single recommended next action, and what is still unproved
**Next action:** take Decision A to Fan.
- Stop developing CVM-T as the paper's contribution.
- Record B8_P-style held-bitmap reuse as an engineering long-context speed result. Its per-call saving is ~7.7% vs native on this panel, and the idea is prior art (one-observation reuse, e.g. MAGE).
- Any further round needs a mechanism with a plausible increment over B8_P, not more CVM runs.

**Still unproved:**
- any CVM_T increment over B8_P, in quality or latency;
- quality equivalence of any arm to native (12 questions; intervals cross zero);
- generality beyond 4 domains and 10–20K tokens (Code-repo and dialogue domains have no items in this range);
- whether the B8_P / CVM_T savings hold on a larger frozen panel;
- AIME, where the tested configurations showed no gain.

Files (this dir): `panel_selection.json`, `frozen_protocol.json`, `smoke_protocol.json`, `setup_receipts.json`, `complete_request_results.{csv,json}`, `per_question.csv`, `phase_accounting.json`, `attribution_plan.md`, `attribution_twins.json`, `test_receipts.txt`, `private_receipt_index.json`, `fan_update.md`. v14 corrections: `../numerical_qk_cvm_t_20260926/corrections_v15.md`.
