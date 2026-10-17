# V2 online refinement panel (2026-10-10)

Accuracy-first development panel for the online value-aware V2 selector. The
native adaptive sampler and GLOBAL-only sparse consumer are unchanged. Timing
is intentionally paused. The panel contains 15 seed-1 LongBench-v2 development
items, official scoring, selector receipts, a six-snapshot mass-anchor
diagnostic, and a streaming full-dimensional-V diagnostic.

Complete arms: `value_v2_online_preserve_mass`,
`value_v2_online_mass_anchor32`, `value_v2_online_jcgate`, and
`value_v2_online_jcgate_anchor16`. The last combined arm is retained as a
negative development result. Anchor64 smoke attempts are retained as
resumable interrupted OOM attempts; no clean timing worker was launched.

The source identity is recorded in each attempt `config.json`; the analysis and
scorer completion markers are under `reports/refinement_analysis_20261010/`.


The complete 503-item official audit of value_v2_online_jcgate is scored in
scoring/audit/official_with_control.summary.json: 256/503 (50.9%) versus
268/503 (53.3%) for the V31 control, with 78.07% versus 83.16% GLOBAL
sparsity. The candidate uses 37.12 fewer denoising forwards per item on
average, but its denser selector and lower accuracy reject it before timing.
Timing remains paused and resumable. The next justified accuracy direction is
a fixed-budget mass-gated value selector.
