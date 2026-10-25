# Completed short end-to-end leaf — long-context phase pending

10AIME26 and10HumanEval problems, each seeds42/43,9arms:360 scored cells. This is the complete short leaf of the user-requested25-question screen;5LongBench problems x2seeds per arm remain in progress. No whole-panel completion or production promotion.

| Arm | AIME correct/20 | AIME N/C | AIME mean W(s) | HumanEval correct/20 | HE N/C | HE mean W(s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense_piecewise | 12 | 12.767 | 5.0533 | 20 | 8.326 | 0.7502 |
| allkept_layer29 | 11 | 12.998 | 5.6511 | 19 | 7.934 | 0.8208 |
| current_error_fixed4 | 9 | 13.119 | 20.5766 | 19 | 7.901 | 2.1283 |
| response_guard_qdirection | 15 | 12.410 | 16.6819 | 19 | 8.000 | 2.0702 |
| margin_guard_weighted_bias | 10 | 13.074 | 19.7439 | 18 | 7.978 | 2.2480 |
| validity_reuse_mass_certificate | 12 | 12.570 | 55.0437 | 19 | 8.196 | 5.5481 |
| sham_fixed4 | 10 | 12.688 | 17.4589 | 19 | 8.329 | 2.0834 |
| v31_optimized | 6 | 14.291 | 7.2421 | 19 | 9.253 | 0.9486 |
| dense_full_fix51994 | 12 | 12.767 | 4.9912 | 20 | 8.326 | 0.7388 |

ResponseGuard AIME15/20 versus corrected FULL12/20, inherited optimized V316/20, current-error9/20 and sham10/20. No dense-correct→response-wrong AIME cells;3 reverse transitions. Accuracy difference versus dense+15pp, descriptive paired question-cluster95% interval[0,+35]pp. Against sham+25pp[+10,+40], but native N/C ratio versus sham0.998 with interval[0.946,1.040]. This does not establish response-specific trajectory acceleration.

On HumanEval ResponseGuard19/20 versus dense20/20: one observed task regression. General accuracy preservation is not established. Its AIME paired geometric N ratio versus dense0.857[0.764,0.948], while N/C0.970[0.923,1.009] and output-length ratio0.874[0.783,0.956]. Much of the fewer-call effect accompanies shorter output. W ratio versus dense3.152[2.779,3.537]; no speedup. HumanEval W ratio2.379[1.893,2.890].

All methods use native proposal/acceptance/re-noising/self-conditioning/temperature/canvases/stopping/caps/EOS unchanged. New selectors affect only GLOBAL layer29; V31 retains5GLOBAL layers andK128, so physical work differs. LOCAL remains dense. Actual FULL decode versus PIECEWISE dispatch is audited; audit/clean full output hashes,N,C,T agree exactly for all360cells. All clean timing records have zero new capture/JIT events.

Certified reuse refreshes5731times, holds5times and records4088failed certificate checks across40shortrequests. The conservative bound is valid under its assumptions but fails practical reuse here. Discovery overhead is separately counted; skipped sparse-consumer tiles do not imply less total attention work.

Per-canvas calls derive from actual pre-call native commit flags. The native pipeline includes an uncommitted tail after last emitted canvas; totalN is retained and committed-only counts/tail are separately reported. CPU emission intervals may lag and are not used as exact native-canvas counts.

V31 mixed-mode accounting failure is preserved. Its accepted data come from separately bound constant-mode audit and clean processes; same inherited algorithm/budgets/runtime. Failed CPU scorer/path reader preparations are preserved; official task rules and generation records are unchanged.

The questions are previously exposed; bootstrap intervals retain two seeds within each of10question clusters per task. Nonsignificance is not noninferiority. No response-contraction mechanism is identified by repeated full-request intervention; no exact downstream MarginGuard oracle is claimed. Nine original over-limit LongBench items remain blocked.

Short successful-family allocation: 9306.713s; counted model forwards including startup/profile: 146600. Earlier failed qualification/smaller preflight/V31 mixed-mode failure are separate and preserved. This is allocated-worker time, not active GPU or clean request latency.

Inspect scoring_short002/official_scores.json, analysis_short001/summary.json, paired_trajectory_short001/summary.json, closeout_short002/verification.json and short_final_complete.json. Master protocol88e46d4088a094968ebbeaed05b0ce4b9cf6e824a643675eba257ac9920d35dd; parent source4ac2b53f, V31 repair binding1429aaf3. Long phase remains running, with no changes based on these short outcomes.
