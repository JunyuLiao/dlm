# Intake CPU audit, 2026-10-01

See `docs/INTAKE_AUDIT_20261001.md`. `summary.json` contains aggregate checks only.

Inputs: existing private closed ledgers and scored CSVs for E4/E5/E6/E6b/E7/E8/E9/E10/E11, and the existing full-dataset LongBench rendering log. No new model generations.

The hardened `scripts.v27_fa4_panel_summary.read_cells` checks execution provenance. Existing paired point metrics and correct counts were recomputed and matched; historical CIs, CSVs and ledgers were not rewritten. The 17 CPU regression cases run on the registered dlm2 interpreter in a fresh user-owned directory. GPU seconds: 0.
