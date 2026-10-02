# Historical accuracy and variant audit

Source-code audit base: `7d9195047`. Frozen V18b inference remains `8704cd072`.
Full interpretation: `docs/VARIANT_FAIRNESS_AUDIT_20261002.md`.

`historical_accuracy.json` recomputes aggregates from the existing qualified E13,
E14 and E15 scores. LongBench pools 18 seeds per question; AIME appears only in
E15 and has six. Correct counts are unchanged. No generation was rerun for this
audit. Inputs remain private; the output contains no question identifiers, text,
answers, token sequences or private paths.

Reproduction, with private score locations supplied by the coordinator:

```text
python -m scripts.v27_historical_accuracy_audit \
  --scored <E13-scored.csv> --scored <E14-scored.csv> --scored <E15-scored.csv> \
  --arm M3_R6_A64_fused_dp_async_m1ln2_c0_fa4 --base D_fa4_allkept \
  --bootstrap-reps 20000 --bootstrap-seed 7 --exact-max-items 16 --out <new-aggregate.json>
```

The checked-in aggregate additionally labels the source panels and comparison.
The script validates qualified pair identity, but is not a replacement for the
original full worker-provenance audit. Item-cluster bootstrap and exact item
sign-flip answer different statistical questions; cell McNemar remains exploratory.
These results do not establish noninferiority.

Validation: 188 CPU tests plus 20 subtests pass, including 18 new accuracy-audit
tests and six new adapter constructor tests. Production frozen-config CPU
rebind/validation passes with unchanged mathematical settings. No GPU was used.

Separate read-only CHW check at `6f7279c1625f7fa53bacb233b96fb69a385df8c1`:
`gate_reorder_value.py::skipped_decisions` with synthetic all-skippable rows and
one key block returns 64/128/128/256 row-block units for 64/65/128/256 valid rows.
Only the 65-row case overcounts. This is a partial-tile diagnostic defect, not
evidence that the peer's 256-row results are wrong. No peer implementation is
copied here or changed.
