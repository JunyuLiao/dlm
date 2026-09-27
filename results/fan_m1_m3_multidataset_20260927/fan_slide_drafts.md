# Local English slide text and Chinese speaker notes — preliminary

Draft only. No shared slides or messages were edited/sent. Updated after direct
numerical qualification, after the first84 scored outputs. Replace PENDING with
actual scored evidence when available; retain failures and missing cells.

## Slide 1 — Periodic M1 is executed, with two separate clocks

- M1 redecides support from observed historical QK and current projected V.
- M3 invokes that same M1 at decision interval R2/R3; held calls bypass selection.
- A8 refreshes actual numerical QK, invalidates/rebuilds the bitmap, and resets
  decision age. A8 is our controlled engineering choice, not a meeting mandate.
- Current output uses current QK/PV on retained tiles. Q/K/V projections remain.
- Common fast-T and unchanged native adaptive sampler/stopping across T arms.
- Selected answer point: GLOBAL-only (5 layers), native LOCAL (25 layers), P0,
  generic Triton; all-layer execution was measured and slower at this point.

R3 first12 phases: A H H D H H D H A H H D.
A = numeric anchor + decision; D = M1 decision; H = held support.

中文讲稿：我们实现的 M3 在刷新时调用真正的 M1，而不是每两三步重新算全部QK。
数值锚点A和决策间隔R分别控制，当前输出不用旧的最终attention权重。五层GLOBAL
是明确的受限实验；公共fast-T不算方法创新。原生停止阈值和输出预算没有变。

## Slide 2 — Long-context forward savings survive full-step timing

Qualified actual-QKV/support references; separate untimed physical counter twins.
Direct full-network decoder costs, not attention-kernel latency or request/calls.
Same-input teacher-forced sequences, native-reached lengths only; three bracketed
blocks, no accepted new JIT. Ratios method/native; lower than1 is faster.

| LongBench, selected GLOBAL/P0/Triton | mpk | dllm |
|---|---:|---:|
| R2 full model-forward ratio | 0.958 | 0.918 |
| R3 full model-forward ratio | 0.950 | 0.912 |
| B_A8 full model-forward ratio | 0.932 | 0.886 |
| R2 whole denoising-step ratio | 0.957 | 0.924 |
| R3 whole denoising-step ratio | 0.948 | 0.916 |

AIME model-forward R2/R3 remain ~2–3% slower than native. RULER naturally stops
at4 calls, so qualified N4 is retained and an unavailable N16 is not fabricated.
Each host has its own questions/states and absolute ms table. Never splice
native from one GPU with candidate latency from the other.

中文讲稿：长上下文上已经直接测到完整forward和完整step变便宜，但简单B仍更便宜。
AIME还没有forward收益，RULER不能强迫跑16步。这里是共同输入的重放计时，不能
推出自然生成更快或答案正确。Triton选择差距接近零，不宣传为consumer突破。

## Slide 3 — Adaptive answers, work and E2E are the remaining gate

- Frozen first84: two questions/family, seed101, all7 arms and first/warm repeats.
- Full plan: 13 RULER +6 AIME +6 LongBench questions, seeds101/202,700 executions.
- All comparator arms for one question-seed stay on one GPU; exposed development
  inputs, not independent held-out validation.
- Report first-output quality separately by task, EOS/cap/parse/failure outcomes,
  total decoder calls, calls/canvas, output lengths and accepted warm E2E.
- Report both native and fresh-T/B references. B is a substantive competing
  baseline, not evidence that periodic redecisions add value by themselves.
- Observer ON/OFF token/call/phase parity is qualified on the bounded bridge.
  Optional device span starts after initial encoder prefill; includes host gaps
  and later commits. It is not decode-only wall time or GPU-active time.
- First84: all84 recorded, six valid blocks, all42 warm pairs accepted. Two
  questions/family, seed101; not a noninferiority trial.
- LongBench: native/T/M1/R2/R3 each2/2; matched dense/B1/2. R2 calls836 vs
  native707 erase its cheaper forward: E2E/native1.106. R3 calls483, E2E/native
  .636, but E2E/freshT1.030: no demonstrated incremental advantage over freshT.
- AIME all1/2 with one capped/unparsed answer each; R2/R3 E2E/B1.170/1.248.
- RULER two-task subset score mean.5 for all, R2/R3 E2E/native1.013.
- Keep all frozen arms for the remainder; do not select a winning R on these data.

中文讲稿：接下来最关键的是原生adaptive是否增加调用数、是否保持完整答案质量，
以及能否超过freshT和B的整请求时间。先发布84次完整配对，再按预算完成剩余块。
旧G75/L30已有小样本adaptive证据，但缺完整直接计时，新的native-Q128移植要独立
标注，不能混成旧vLLM实现。正负结果和未完成项都会留在表里。
