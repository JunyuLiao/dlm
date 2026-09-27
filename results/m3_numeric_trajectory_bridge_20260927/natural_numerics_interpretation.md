# Natural zero-pruning diagnostic: effects are not uniform

Executed source `525880efbfc9781976a3b45d131bd56eb96a6346`, frozen protocol `v21_zero_pruning_1ee6332c82377856`. Twelve first-only executions completed without failed or missing cells. Two exposed LB inputs × seeds101/202 × native/legacy-all-kept/new-all-kept; the complete comparison block stayed on its assigned GPU. No warm timing, no task-quality promotion and no geometry selection from these answers.

| Input suffix | Seed | Native calls / canvases | Legacy all-kept | New all-kept |
|---|---:|---:|---:|---:|
| 067c5456 (mpk) | 101 | 246 / 13 | 407 / 19 | 278 / 14 |
| 067c5456 (mpk) | 202 | 318 / 17 | 335 / 16 | 240 / 12 |
| acb2a9de (dllm) | 101 | 110 / 7 | 86 / 6 | 81 / 6 |
| acb2a9de (dllm) | 202 | 86 / 7 | 89 / 6 | 167 / 11 |

All requests ended with the recorded EOS termination. Disjoint per-canvas flags record native stop and no iteration-cap-only canvas in this diagnostic. All-kept variants still use five GLOBAL layers and native LOCAL. The router reports each all-kept output as a materialized-current-output call; its A count must not be mistaken for sparse M3's periodic A8 mix.

Changing only current-output score arithmetic changes the no-pruning trajectory. The direction varies by input and seed. In the second input's seed202, the new output has both more canvases and more refinement per canvas. In the first input, the new mode reduces both relative to legacy. Neither a universal numerical bug nor a complete explanation of sparse M3's work inflation follows. The small operator-error improvement remains a qualified numerical property, not proof of better task answers or stable request length.

Per-cell work, first cold wall time and absent warm fields are retained in `natural_zero_pruning_001.{csv,json}`. Cold diagnostic wall time is not a clean scored timing comparison. No failed requests were replaced and no extra seeds or output caps were used.
