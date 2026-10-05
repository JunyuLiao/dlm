# CP0 v15 evidence audit (2026-09-26)

Status: **blocked for strict warm timing acceptance**. The existing attempt-0 outputs remain inspectable; this audit did not generate or rescore answers.

- Frozen manifest SHA-256 `80c6b523110db96530d06ab252fc2e27b9eefceee19a34d60b526c67df06ec72`, private gold SHA-256 `76cd1cc9e5770a59405e3ac601677509415b4a883925e198a5b3190f674df481`, and panel ledger SHA-256 `f6ec32afbc67061c684f9fdcfbc4bc3ae315efd98be3bd393fe39d2140c35ecf` match their frozen/index references.
- All 12 IDs, 240 scheduled executions, and 120 successful attempt-0 private receipts were accounted for. The audit checked prompt text and token hashes, receipt/ledger seed and identity, completion token hashes, per-canvas calls, termination, scorer/extractor/dataset source SHA, and the NeMo scorer source SHA. No mismatch was found in these checks.
- All 120 warm comparisons are rejected by the v18 strict rule because `phases` is absent on both attempt-0 and warm ledger rows. The stored v15 acceptance marks these rows accepted. Timing conclusions depending on accepted warm repeats need independent repair or fresh qualified timing; do not relabel these historical rows as strictly accepted.
- The v15 scorer source uses `zip(texts, preds, golds, terminations)` without an equal-length input guard. The new audit includes an explicit length-check helper and adversarial test, but the historical scorer file was not modified in this work package. Future scoring must call a guarded path.
- The 253 examples skipped for `>400,000` characters were never token counted. Label the eligible pool **heuristic-prefiltered**, while retaining the already selected 12 questions.

Evidence: [audit JSON](cp0_evidence_audit.json), [audit script](../../scripts/v18_evidence_audit.py), [tests](../../tests/test_v18_evidence_audit.py). Private inputs remain on `exouser@149.165.151.254` under `/media/volume/dllm-1/dyh/numerical_qk_longcontext_scored_20260926`; isolated audit copy/output is in its `v18_audit_stage` directory. Runtime: `/home/exouser/miniconda3/envs/ljy_dlm/bin/python` with user site enabled.
