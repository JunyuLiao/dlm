# v20 evidence addendum for v21 CP0

The published generation-record manifest and every record SHA-256 were checked before this CPU audit. There are 400 first and 400 accepted warm receipts across the three frozen datasets; the separately timed historical arm contributes 100 executions.

Qualified task correctness and strict EOS come from unchanged v20 offline `score_firsts` in the pinned CP5 environment; the allowlisted 400-cell output is copied as `qualified_first_scores.json`. No text-derived score substitutes for it.

Source score-export SHA-256: `bb4ccbec83ea50172566a5afb35e6aa9581a995ec114c6f737335d1de1d8d97b`. Export manifest protocol SHA is the exact CRLF expansion of the deployed/Git LF protocol bytes; this is a byte-line-ending difference, not a changed parsed protocol.

Request records preserve completion tokens, per-canvas native stopping/cap flags, schedule-step arrays, calls, A/D/H totals when available, and request/device timing. They do not preserve per-step attention tensors, decision-age distributions, or a synchronized prefill-excluded generation wall time. Historical G75L30_nativeQ128 is a native-Q128 transplant rather than the exact older vLLM grouping.

## Interpretation of the verified LongBench work

Across the same 12 question-seed first outputs, native uses 2360 calls / 140 canvases; R2 uses 2842 / 159; R3 uses 2514 / 147; fresh T uses 2223 / 128. These pooled work counts are not pooled cross-host absolute timing comparisons. The exact work identities are:

- R2/native: 1.2042 calls = 1.1357 canvases × 1.0603 calls/canvas.
- R3/native: 1.0653 calls = 1.0500 canvases × 1.0145 calls/canvas.
- R3/T: 1.1309 calls = 1.1484 canvases × 0.9847 calls/canvas.

Thus extra canvases account for much of the extra total work; R3 has fewer calls per canvas than T in this pilot. This does not identify why outputs became longer or prove that length caused quality changes. Once committed tokens diverge, matching canvas indices do not supply matching model inputs.

D_matched has 3009 calls / 148 canvases, 27.5% more calls than native despite retaining all legal GLOBAL support. Its 12 cap-flagged canvases split into 11 cap-only and one both-cap-and-native-stop canvas. This is a reason to isolate output arithmetic and execution effects, not proof of a bug or a complete explanation of M3. Native/M1/R2 score 6/12; R3 5/12; B and D_matched 4/12. B therefore does not dominate quality. Six unique questions give large uncertainty.

The existing same-state full-forward advantage remains real in measured LongBench states: R2/R3 are about 0.958/0.950 of native on mpk and 0.918/0.912 on dllm. Short measured AIME/RULER states cost about 2–3% more. These common-input replay measurements are not natural fixed-step generation or a full-panel forward average. The per-host ratio-of-totals and paired geometric request ratios are both retained in trajectory_audit; neither is substituted for the other.

The frozen six-cell text inspection describes common prefixes, repetitive 16-token windows and final-response behavior only. No hidden-state, per-step confidence or causal numerical evidence can be recovered from final text. CP1 must collect new separately labeled same-state evidence, preserving native stopping and the old selector-cache arithmetic.
