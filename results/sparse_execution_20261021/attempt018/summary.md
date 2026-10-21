# Best-kernel synthetic result, 2026-10-21

The requested overall task remains incomplete: V2 refresh is faster than the prior449us implementation, but does not meet dense refresh parity. Native combined qualification, matched native controls, workload calibration and the full scored panel have not been relaunched. The old campaign remains preserved as partial.

Q256/H16/Hkv2/D512 BF16, all required12cells pass finite/output/mask checks; actual timed captures0. Separate warmup/graph capture and clean kernel/event plus host/module timings. The table below is synthetic graph/event complete pipeline timing, not native model/request latency. Values in microseconds.

| Keys | Actual sparsity | V2 complete refresh | Plain dense | Native paged dense | V31 direct paged refresh | Held resident |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 50.000% | 80.58 | 46.95 | 137.16 | 148.81 | 39.99 |
| 4096 | 49.951% | 222.77 | 180.42 | 199.65 | 219.17 | 73.20 |
| 8192 | 50.049% | 427.90 | 363.58 | 287.47 | 301.85 | 118.60 |
| 16384 | 49.939% | 860.36 | 717.77 | 471.89 | 492.26 | 210.20 |

At8K,427.90us improves the prior449.02us by4.70%, but is17.69% slower than plain dense and48.85% slower than native paged dense. Cached-prefix projection413.85us is a separately qualified within-canvas optimization; it is not the full-current-V number. V31 direct paged refresh301.85us is5.00% above native dense287.47us. Short contexts retain the contiguous V31 dispatch; direct paged is shown for transparency.

The complete V2 pipeline charges current full-V Gaussian32 projection/RMS, sketch packing, true-online QK/softmax/PZ/admission/conditional full PV, and compact lists. Attention-module measurements separately charge K/V copying and resident mask construction. Held resident is the direct paged synthetic consumer, not a native request speedup; prior bucket host/module results remain separately documented.

Implementation: skip per-score bounds predicates on complete interior tiles and use a masked last-key tile, with generic irregular-query fallback. The numerical sampler and ordered threshold rule are unchanged. Four output-channel groups still repeat QK/routing. Unique QK producer queues, interleaved QK, split-key, cross-WG lookahead, wide-QK bundles, early K/V handshakes and descriptor variants did not beat this design. They are retained in attempt002-017 with source/build, numerical and timing receipts. Static literal-address attempt012 fails launch qualification and is abandoned.

Source/runtime hashes and exact compiled source are in qualified_source/source_manifest.json and matrix001/receipt.json. Compilation and captures are outside all measured regions. These synthetic thresholds are not per-workload calibration. Full benchmark short V31 budgets remain K128 and LongBench-v2 K8192. No accuracy, paired uncertainty, N/C/T/S/W or native trajectory claims are synthesized from these diagnostics.

The chosen interior/tail library passes all43combined H100tests with0failures/0skips in41.45s (regression001/junit.xml). Coverage includes resident current-Q/K/V/page metadata, held mask graph changes, projection and compact-list equivalence, V31 exact budgets and online refresh current-V/irregular-tail graph tests. This operator qualification is not combined native trajectory/timing qualification.
