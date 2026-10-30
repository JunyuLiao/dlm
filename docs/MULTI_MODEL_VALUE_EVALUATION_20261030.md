# Multi-model value-direction evaluation — 2026-10-30

Status: authorized implementation and execution; no GPU-hour ceiling. Parent
commit: `6149df10d5c2a1f270755129c7d97318ffe5f3e7`; branch `multi-models`.
Canonical campaign: `results/multi_model_value_20261030/attempt001/`.

## Question and lineage

Test whether the existing V2 online Gaussian32 and V3a full-support singleton
selectors improve accuracy or native denoising behavior over mass ranking on
models with separately parameterized K and V. This is a portability study, not
a causal intervention on Gemma's K/V tying. Gaussian32 is Junyu's prior method;
cross-step reuse machinery is inherited from Yuhan. Every cross-model port is
a new named variant. Historical DiffusionGemma evidence is unchanged.

## Frozen design

- Models: inclusionAI/LLaDA2.2-mini; relaxe-system-lab/UltraLLaDA;
  Gen-Verse/TraDo-8B-Instruct. Resolve and freeze model revisions before loading.
- Arms: `dense_native`, `allkept_triton`, `mass_topk_reuse_r4`,
  `value_v2_gaussian32_reuse_r4`, `value_v3a_singleton_reuse_r4`.
- Native masks, proposals, edits, random draws, sampler schedules, stopping,
  EOS behavior, and output caps are shared within a model/workload cell.
- All applicable denoising layers are eligible. Genuine prompt prefill and
  block commits stay dense. Full-bidirectional models recompute prefix states.
- Selection is per query head and Q128 group, physical KV64 tiles; padded
  query rows do not vote. Current generation-block/boundary KV tiles are kept.
- Initial selection and every fourth call use current Q/K/V; maps are held
  between selections. Reset at block boundaries or structural index changes.
  No cross-block carry, sticky scores, compensation, or hidden model forwards.
- Discovery is attention-only and returns a sparse output on the same call.
  Full-dimensional current V is always used for retained output attention.
- Projection: fixed Gaussian32, seed 1729 plus layer identity, independent
  per KV head, entries N(0, 1/32); reproject current V at every selection.
- V31 port ranks worst-row normalized tile mass; V3a ranks singleton deletion
  risk against full-support projected output. Preserve V2 ordered online
  preserve-mass semantics; no top-k repair of V2's threshold map.
- Target consumer sparsity 48–52%, including mandatory tiles. For top-k keep
  round(J/2) legal tiles inclusive of mandatory tiles, clipped to feasibility.
  Publish infeasible rounding cases and total discovery/consumer work.
- Calibrate V2 by computation only on separate development tasks: six math,
  six code, thirteen synthetic retrieval tasks; at most seven threshold
  candidates per model/workload, followed by native confirmation. No test-score
  tuning. Freeze selected thresholds and disclose main-panel drift.
- AIME26: 30 fixed problems x seeds 42/43/44. HumanEval: 164 x seed42.
  RULER128K: 130 fixed problem identities x seed42 on LLaDA2.2 and UltraLLaDA.
  TraDo128K is unsupported, not truncated or extrapolated. No silent item loss.
- Proposed output caps: LLaDA2.2 AIME32768/HE1024/RULER128; UltraLLaDA
  AIME4096/HE1024/RULER128; TraDo AIME2000/HE2000. Native block lengths:
  LLaDA2.2=32, UltraLLaDA=64, TraDo=4. UltraLLaDA uses one scheduled
  unmasking step per generated token, temperature0. LLaDA2.2 uses released
  defaults threshold0.5, editing_threshold0, max_post_steps16, steps32,
  temperature0 and native EOS stopping. TraDo uses dynamic threshold0.9,
  temperature1, top_p1, top_k0, native EOS stopping.
- Official benchmark scoring; paired problem-clustered uncertainty. Report
  correct count/accuracy, N, C, request-level N/C, T, tokens/N, edits/refinement,
  finish/cap status, physical tiles, discovery work, reuse age, GPU seconds.
  Deterministic seed duplicates are not independent evidence.
- Latency is secondary accounting only, with instrumented times labeled.
  No end-to-end speedup claim. Kernel optimizations require parity against an
  independent reference and all-kept/native controls before scored generation.

## Execution and preservation

Spec commit -> frozen config -> source/model/runtime/scorer pins -> CPU/GPU
qualification -> receipts -> disjoint calibration -> complete resumable shards
-> official scoring -> fairness/coverage audit -> summary and completion marker.
Failure/relaunch uses a new immutable attempt leaf. Source changes are rebound
before new generation; no frozen protocol or evidence is overwritten.

Use one GPU worker. Check live driver/process identity before launch. A separate
CPU monitor records nvidia-smi, memory/utilization/temperature, worker liveness
and record progress every 900 seconds throughout the campaign. The agent also
inspects those receipts and failures until completion. No GPU-hour or wall-time
ceiling is imposed by this protocol. Only correctness, unavailable resources,
or user steering can stop a panel; retain incomplete evidence honestly.

Private inputs, raw outputs, model weights, downloaded source, environments,
logs and tensor state stay in ignored `private/` directories. Public results
contain sanitized identities, measurements, receipts and aggregates only.
The initial engineering qualification uses non-benchmark synthetic prompts.
