# Math500 prefix-proxy distribution

## Setup

Collect post-normalization/post-RoPE prefix-only block proxies with native masks
and GQA for DiffusionGemma and fast-dLLM-v2, then generate distribution plots
and summaries.

## Purpose

Measure whether the prefix proxy has a stable distribution suitable for a
shared routing threshold.

## Findings

Global/local availability and distribution shape differ across models; the
output is diagnostic evidence, not a validated shared threshold. Canonical
artifacts are under `results/proxy_diagnostics/math500_prefix_distribution/`.
