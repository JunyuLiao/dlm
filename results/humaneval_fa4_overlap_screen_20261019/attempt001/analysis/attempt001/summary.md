# HumanEval same-input FA4 reproducibility diagnosis

Native evaluation remains gated pending the no-overlap qualification. This completed operator study contains no generation, accuracy, or clean timing result.

Each cell uses two independent processes and 256 calls per process, compared with that process's first output. Thus each cell has 512 calls and 510 nontrivial repeat comparisons. Factor order reverses in process 2. The two complete captured inputs are previously examined q129/seed42 anchors; they are diagnostics, not held-out confirmation.

| FA4 path | Actual splits | Overlap | Changes / comparisons, KV2252 | Changes / comparisons, KV3532 |
|---|---:|---|---:|---:|
| Mass observing | 16 | True | 0/510 | 4/510 |
| Mass observing | 16 | False | 0/510 | 0/510 |
| Plain | 16 | True | 0/510 | 0/510 |
| Plain | 16 | False | 0/510 | 0/510 |
| Mass observing | 1 | True | 0/510 | 4/510 |
| Mass observing | 1 | False | 0/510 | 0/510 |
| Plain | 1 | True | 0/510 | 0/510 |
| Plain | 1 | False | 0/510 | 0/510 |

All 2,048 mass-observing calls with overlap disabled match their per-process baseline outputs (2,040 nontrivial comparisons). All 4,096 plain calls are stable in this screen. All 4,096 observing calls reproduce the captured native attention-mass auxiliary bytes; Q/K/V match their saved input bytes after each case. There are zero CUDA captures and zero model/sampler calls.

The first automatic-split change affects 2,322 BF16 elements in head5, query tile1 (64-query tiles), with maximum absolute change0.0625 and relative L2 0.000187935. The first single-split change affects 5,318 elements in head7, query tile0, maximum0.125. Private baseline/changed tensors are retained under the corresponding engine.

This identifies a same-input failure in the customized mass-observing SM90 FA4 overlap path. Attention-mass statistics remain unchanged, narrowing the failure beyond the observed QK statistics to the output computation. The exact failing instruction/register/shared-memory mechanism has not been proved. Disabling overlap is a supported correction candidate, and must pass native full-observer/pure and independent-process qualification before use in headline results. The altered floating-point evaluation and any latency cost require a new runtime identity.

These conditional repeat counts are not iid failure-probability estimates. Stable operator trials do not establish universal determinism or method noninferiority. Prior unqualified HumanEval accuracy rankings remain unusable.

Protocol: `906d41b7bbb6f93895aed80452e256ab8f8ef298b95fd236938aee783619fbbe`; source `c7e7c67e29be2afbb5f6a0de594f571297c51bb7`. GPU reservation seconds: engine001 55.959, engine002 50.147.
