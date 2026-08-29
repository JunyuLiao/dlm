# RULER16K stepwise prefix-proxy diagnostic

## Setup

Collect a deterministic per-row proxy reservoir for one 16K RULER prompt while
tracking the masked or unresolved denoising state at each call.

## Purpose

Test whether standardized proxy tails remain invariant as denoising progresses.

## Findings

The state changes dramatically and upper-tail statistics drift across steps and
layer/head strata. This motivates a multi-prompt state-conditioned threshold
study but does not itself validate such a policy. Results are under
`results/proxy_diagnostics/ruler16k_stepwise/`.
