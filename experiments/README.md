# Experiment packages

Reusable Python experiment packages live here; one-off RULER launch/report
scripts live under [`scripts/ruler/`](../scripts/ruler/).

| Package | Setup and purpose | Main finding | Results |
|---|---|---|---|
| `diffusion_attention_analysis` | Dense DiffusionGemma observation plus gated replay | Temporal structure exists, but 62% sequence agreement made replay a no-go | `results/attention/analysis/` |
| `diffusion_attention_threshold_modeling` | Distribution fitting, Math500 probes, and fresh RULER16K routing | No compact portable threshold table passed all held-out density gates | `results/attention/threshold_modeling/` and `results/attention/routing/` |
| `math500_prefix_proxy_distribution` | Prefix-only proxy collection for Math500/RULER inputs | Proxy geometry differs by model and attention type | `results/proxy_diagnostics/math500_prefix_distribution/` |
| `ruler16k_stepwise_proxy_diagnostic` | Single-prompt, per-step state/tail diagnostic | Upper-tail drift motivates state-conditioned follow-up work | `results/proxy_diagnostics/ruler16k_stepwise/` |
| `diffusion_gemma_solattn_vs_blasst_ruler16k` | Isolated 64x64 Sol-Attn Gaussian versus calibrated BLASST reference-mask comparison | Disjoint 10/50 RULER16K manifests, offline local/global calibration, and paired final reporting | `results/diffusion_gemma_solattn_vs_blasst_ruler16k/` |

Each results leaf contains `SUMMARY.md`; generated reports and raw artifacts
remain adjacent to it.
