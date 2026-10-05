# Triton permutation microbenchmark

The frozen Triton gather/scatter implementation passed exact GPU equality checks against Torch, but was slower in all three tested nominal length bins. Triton/Torch median-time ratios have a geometric mean of **1.626493**.

| Nominal bin | Torch gather + scatter, ms | Triton gather + scatter, ms | Triton / Torch |
|---|---:|---:|---:|
| 32K | 0.051568 | 0.084768 | 1.643810 |
| 64K | 0.048752 | 0.079024 | 1.620939 |
| 96K | 0.047984 | 0.077488 | 1.614872 |

This is one same-host worker in the existing qualified environment: Torch 2.13.0+cu130 and Triton 3.7.1. Both implementations use identical held orders, synthetic native-stride BF16 query tensors and token-major output tensors; both allocate gather and scatter outputs on every call. Each bin uses 8 warm-ups followed by 100 rotated CUDA-event timing repeats. Exact gather, scatter and inverse checks passed before timing. No CUDA graphs were requested.

CUDA events surround Python component calls. Host dispatch can leave GPU waiting gaps inside that interval, so these numbers measure the current Torch/Triton component-call cost rather than isolated device-kernel bandwidth or execution duration. The exact oracle and device-to-host validation do not repeat inside timed calls. Triton has additional Python checks and launcher work; their contribution was not measured, so the full difference cannot be attributed to them.

Source and public spec were frozen at commit `e69b5903365bdd655b744c1682d27e4232ff80ac`. Only three source modules, the CPU test and public spec were deployed. All CPU import, unit and snapshot-validation checks returned zero before launch. The 144 private snapshots and committed bytes were bound by a private manifest; the same frozen gate selected the first CPU-accepted historical state in each nominal bin. Private snapshot names, orders and need bits are omitted. Report and GPU identity hashes remain private; the report hash was verified during collection.

The worker exited zero and reserved **40.294034620 GPU seconds**, including CPU selection, first construction/JIT, qualification, timing and teardown. Subsequent checks found no GPU compute process, zero used memory and zero utilization. All caches were newly created in the user's isolated directory; no packages or other-host jobs were changed.

The source's internal body span is **1.687286930 seconds**. Its timer starts after CPU `select_accepted` and Torch import, before CUDA tensor creation; it includes construction/JIT, GPU qualification and warmed timing, but excludes selection/import and process teardown. CPU selection runs while the GPU is reserved, before this benchmark creates CUDA tensors. This body span, supervisor reservation and CUDA-event medians are separate measurements.

This negative measures standalone permutation cost. Offline search and first construction are excluded from the CUDA-event medians, although their wall time is included in reservation accounting. No selector, attention consumer, model, request lifecycle or quality evaluation ran. Do not add these times to consumer measurements from another host. The previous Torch held-regroup result remains unchanged, and the frozen source was not tuned or rerun after observing this result.

The held attention GPU follow-up was paused after this three-bin negative; any prepared deployment remains retained and unrun.
