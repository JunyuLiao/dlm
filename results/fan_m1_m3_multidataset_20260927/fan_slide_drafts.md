# Local English slide text and Chinese notes — full700, G75 pending

Draft only; no shared slides/messages edited. First84 archived unchanged.

## Slide 1 — Periodic M1, with distinct numeric and decision clocks

- M1 uses observed historical QK and current projected V to redecide support.
- M3 invokes M1 at R2/R3; held calls skip selection. A8 numeric anchors reset
  decision age (R3: A H H D H H D H A H H D).
- Current kept-tile QK/PV output; Q/K/V linear projections remain. Common fast-T.
- Primary GLOBAL5/nativeLOCAL25, frozen P0/Triton; ALL measured and cost-rejected.
- Native B256/adaptive48max/thresholds/outputcaps/seeds unchanged. Source00a2c4d.

讲稿：这是真正周期执行M1，不是每2/3步重算全QK，也不使用旧的最终attention权重。
A8和GLOBAL-only是工程控制点，不声称是Fan明确指定或已知最优。

## Slide 2 — Full-forward savings exist in long context

| Selected GLOBAL/P0/Triton, method/native | mpk | dllm |
|---|---:|---:|
| LongBench R2 model-forward | .958 | .918 |
| LongBench R3 model-forward | .950 | .912 |
| LongBench B_A8 model-forward | .932 | .886 |
| LongBench R2 whole denoising step | .957 | .924 |
| LongBench R3 whole denoising step | .948 | .916 |

- Same-input state-correct teacher-forced replay, native-reached N<=16, brackets,
  no accepted JIT. Actual QKV and separate untimed counter twins qualified.
- AIME/RULER no forward headroom. RULER natural4 calls, N16 missing retained.
- Physical sparsity denominators refer to routed layers, not the whole model.

讲稿：单次成本是真实完整调用计时，不是总时间除调用数。LongBench有4–9%收益，
但简单B仍更便宜；不能从这种重放直接推导完整答案与请求收益。

## Slide 3 — Full700 does not support an incremental M3 gain

700 executions,50 valid question-seed blocks,350 accepted warm pairs.
RULER13q×2seeds; AIME/LB6q×2seeds. Exposed development inputs, not noninferiority.

| Dataset | native/T/R2/R3 correct | R2/native E2E | R3/native E2E |
|---|---|---:|---:|
| RULER | all24/26, macro92.31% | 1.075 | 1.048 |
| AIME | 5/8/6/5 of12 (strictEOS) | 1.096 | 1.135 |
| LongBench | 6/6/6/5 of12 | 1.111 | 1.031 |

- LB calls native2360,T2223,R2 2842,R3 2514; extra adaptive work consumes savings.
- LB R2/R3 E2E/freshT1.224/1.136, versusB1.111/1.031. No added M3 advantage.
- AIME capped requests remain counted; R3 has one additional task-correct at cap,
  kept separate from strictEOS. No output-budget or stopping rescue.
- First84 R3 .636/native did not persist: full-panel ratio1.031. Do not select R
  from early outcomes or turn non-significance into quality equivalence.
- Separate G75L30_nativeQ128100 is running; new port, not old grouping equivalence.

讲稿：完整面板推翻了早期两题的乐观点估计。R2多出来的调用吃掉单次节省；
R3没有超过fresh-T，且LB正确数更低。今天交付实现和可复核的负结果，
不把方向判死刑，也不靠继续扫参数硬找正结果。G75参考完成后补最终表。
