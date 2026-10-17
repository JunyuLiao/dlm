# Target audit plan

The development gate promotes only `value_mass_gated_online_swap8` and the matched sensitivity arm `value_mass_gated_online_swap16`. Each uses the exact V31 native-mass prefix budget per `(head,128-row query block)` and swaps at most 8 or 16 tiles from a two-budget candidate pool using the online V2 projected residual. This is a paired, accuracy-only audit against the completed V31 control records. Four resumable shards preserve the 503-cell frozen order and seed.

The full-information, C_gate, phase, and V31∩V2 arms are closed as development diagnostics: none showed a better accuracy/call tradeoff than swap8. Timing remains paused. No target answer or scorer result is used to change the selector or threshold.
