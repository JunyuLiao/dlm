# Literature boundary for the failure-map interpretation

Checked primary papers on2026-10-19 while the frozen GPU screen was running.
These notes do not change the protocol, candidates or promotion gates.

[MAGE](https://arxiv.org/html/2602.14209v1) already combines first-step exact
attention, reused sparse support, layer-adaptive allocation and sparse-aware
fine-tuning. Its experimental setup explicitly uses static unmasking to control
the number of steps, and reports configurations with1/2/4 tokens per step.
Therefore neither attention-mass reuse nor mass-based budget redistribution is
our contribution. Its static-step evaluation does not directly answer this
study's native adaptive stopping/convergence question. This is a distinction
between evaluation settings, not evidence that MAGE fails on our model.

[SparseD](https://arxiv.org/abs/2509.24014) already identifies sensitivity in early
denoising and combines early dense attention with later reused sparse patterns.
Thus a dense-output oracle helping an early call, by itself, would not establish
algorithmic novelty. A prospective method must demonstrate a mechanism and
advantage beyond these strong static controls and our inherited V31 baseline.

The current diagnostic can establish reproducible selected-case failures and
conditional repair evidence. It cannot establish paper novelty, global
noninferiority, or a deployable speedup. The original frozen plan still gates
any algorithm development on the experimental results.
