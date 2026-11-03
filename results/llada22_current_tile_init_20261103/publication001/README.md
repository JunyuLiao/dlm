# Current-tile initialization: completed LLaDA2.2-mini HumanEval evaluation

Current-tile-initialized V2 scores102/164 (62.20%), versus original V2's10/164
(6.10%), V31's83/164 (50.61%), V3a's103/164 (62.80%), native dense141/164
(85.98%) and all-kept FA4138/164 (84.15%). It preserves every original V2
pass and adds92. Output mean62.9->214.2, substantive function bodies28->144,
syntax errors98->10. Native EOS/acceptance/editing/stopping rules are unchanged;
all164 new outputs terminate with EOS, none reaches the1024-token cap.

The selector initializes running log mass and normalized Gaussian32 output from
its mandatory last KV64 tile, then scans tile0 through J-2. Each tile contributes
once. Chronological tile0 is no longer forced. V2's preserve-mass skip semantics,
projection, optimized chunk kernel, full-dimensional FA4 consumer, layer scope
and four-call reuse are unchanged. Original source-pinned runtime and native
sampler are used. No mass ranking, top-k repair, EOS mask or extra model forward
is added.

Six original handwritten nonbenchmark code development prompts, seed142, four
count-only threshold candidates, and a six-prompt repeat select0.29084308668979075
with50.9271% sparsity. Final164 problems, seed42, run once at that frozen
threshold. Achieved benchmark sparsity61.3512% misses48–52%; disclose it and
perform no further recalibration, per the user's standing instruction. This is
not a matched-50% benchmark comparison. The end-to-end result combines the
initialization change with its separately recalibrated threshold; it is not a
fixed-threshold ablation of initialization alone.

The large recovery supports harmful chronological initialization as an important
failure mechanism. It does not establish complete recovery to dense accuracy,
held-out generalization, or equivalent compute budgets. New-minus-original V2
accuracy is+56.10pp,95% problem-bootstrap CI[48.17,64.02]; new-minus-V31 is
+11.59pp,[1.83,21.95], interpreted at their different achieved sparsities. The
new variant is one pass below V3a, with uncertain paired difference[-10.38,9.15]pp.

Steps per canvas decrease4.634->3.627, while canvases/request recover3.07->7.62.
Total forwards/request increase14.21->27.65 because outputs grow. These are
native denoising statistics, not clean latency or per-forward speedup evidence.

Full comparison table and source-bound audit:
[attempt001/analysis001/README.md](../attempt001/analysis001/README.md).
Nine CPU tests,30 CUDA recurrence/consumer checks and12 exact historical-control
replays pass. All164 source-bound records, four41-cell shards, private token
hashes, prompt/seed/cap identities and official scorer fingerprint reconcile.
Maximum sparse-consumer versus mathematical masked-attention relative L2 in
qualification is0.002184. GPU checks pass before/after; H100 released0MiB/0%.
Generation finishes before the second scheduled900-second health sample.

This repeats a previously examined HumanEval cohort and is exploratory. Only
sanitized records/aggregates are published; raw prompts, gold, outputs, token
IDs and unsanitized process paths remain private.
