# Experiment results

Results are grouped by research question rather than by the date or model name
used to produce them. Each major run keeps its raw artifacts, generated reports,
and a short `SUMMARY.md` covering setup, purpose, findings, and status.

| Group | Contents |
|---|---|
| [`attention/`](attention/) | Dense attention structure, routing, and threshold-modeling studies |
| [`blasst/`](blasst/) | BLASST sparsity/accuracy evaluations across model families |
| [`kv_pruning/`](kv_pruning/) | Oracle and block-maximum KV-tile pruning studies |
| [`math500/`](math500/) | DiffusionGemma reasoning baselines and pruning experiments |
| [`proxy_diagnostics/`](proxy_diagnostics/) | Prefix-proxy distributions, routing manifests, and stepwise diagnostics |
| [`systems/`](systems/) | SGLang/FDFO runtime and final-commit profiling |
| [`research/`](research/) | Literature review and future-work notes |

Sparsity values in these reports are generally dense-first/reference-mask
measurements. They should not be read as wall-clock speedups unless a report
explicitly labels a physical timing result.
