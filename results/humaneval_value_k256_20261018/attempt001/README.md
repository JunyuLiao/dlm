# HumanEval164 value methods at K256

Completed seed42 study, thinking enabled and disabled, output limit8192,
native DiffusionGemma sampler, one H100. Campaign started2026-10-18;
report verified2026-10-19 Hong Kong time. Native source20e6c0be, protocol
SHA2567b81c845fdce07743d3ed5dd91c4b6b4a54640e071b594ef7f690d6f400bd4d5.

Read [the detailed report](analysis/attempt001/summary.md) and
[numeric results](analysis/attempt001/summary.json). They keep clean accuracy
and latency separate from audit accuracy and physical work. Audit/clean
trajectory differences are substantial even for dense controls; the cause is
unlocalized. Calibration matching is established only for the audit population.

- [Protocol](protocol.json), [committed plan](spec.md).
- [Sanitized per-item metrics](analysis/attempt001/per_item_metrics.jsonl):
  5,904 rows, no prompts, token IDs, gold code or generated text.
- [Independent binding/metric verification](independent_verification.json).
- [Independent official rescoring](independent_rescoring.json):5,904/5,904 agree.
- [Canonical scorer gate](qualification/scorer_gate.json):328/328 pass.
- [Engine inventory](receipts/attempt_inventory.json):58 complete, zero failures.
- [Verification seal](verified_complete.json); [native completion](final_complete.json).
- [Thinking work calibration](calibration/thinking_on/full164_threshold.json)
  and [non-thinking calibration](calibration/thinking_off/full164_threshold.json).

K256 means four selectable prefix KV64 blocks per query-head/Q128 tile;
canvas/boundary support stays dense and inherited carry is retained. V2 uses
a variable-budget threshold. V2/V3a use existing Gaussian32 on all GLOBAL
layers; full-value swaps and direction-guarded residual modify GLOBAL layer29
only. LOCAL, prefill and commits remain dense. The V31 reuse baseline is
inherited from Yuhan. No production method or native sampler was rewritten.

V3a is the best clean point candidate but does not establish superiority:
thinking-on159→160 correct and1.7% lower W versus V31; thinking-off155→154
correct and5.7% lower W. Both latency intervals include no gain. All requested
sparse methods are slower than dense FULL. Full swap observation remains costly.
First-canvas matching does not equalize total-trajectory work.

The original coordinator summaries remain unchanged at summary.md/json.
The detailed report adds complete per-pass trajectories, uncertainty, accounting
and limitations. Raw generations, manifests, gold, caches and execution snapshots
remain private. GPU engine reservation15,032.143 seconds. No worker is queued;
older LongBench campaigns remain paused.
