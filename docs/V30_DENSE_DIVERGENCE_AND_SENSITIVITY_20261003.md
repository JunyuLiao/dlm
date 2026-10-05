## V30 follow-up queue armed — 2026-10-03 03:38 (UTC-5)

The new continuation is actually armed on94134a91b, phase waiting_prior_queues,
0newGPU seconds. All three own-host campaign003 CPU preparations passed, with
nine full-inventory sensitivity bindings and eight dense diagnostic bindings;
CUDA uninitialized. Source CPU tests33passed; private queue/payload guards6;
remote sensitivity/config/adapter regressions26. Earlier source freezes and
running worker deployments are unchanged. Receipt:
results/v30_20261003/campaign_queue001/status.json.

After original formal strict scores, original component/dense/engineering work,
and peer29-case qualification finish, run32fresh-engine dense diagnostics,
then60sensitivity pipeline qualification workers and nine complete strict CPU
scores. No automatic formal launch or quality/speed claim. Full eight-seed
specs exist for59LongBench,30AIME,164HumanEval questions per mode. Raw first
logits/canvas/RNG traces and hashes stay private; sanitized phase/count/graph
traces and boolean comparisons are retained. The first sampled state only
locates an initial difference; it cannot rule out a later divergence.

Original formal snapshot: longbench 8/32, aime 25/32, humaneval 25/32;
zero closed-worker failures. GPU resource seconds per suite remain in the
receipt; active worker spans are not yet included. Diagnosis is pending.

Operational continuation: private campaign_queue001.py/PID file, inputs in
prepare_*_003, scorer campaign_score001.py. Do not edit these now-pinned helper
files; a correction requires a new queue/run. Inspect status, preserve failures,
then review first-divergence comparisons and all nine strict qualifications
before choosing the matched engineering baseline for full confirmation.

## Dense divergence and M3 query-weight ports — 2026-10-03 03:16 (UTC-5)

Root reviewed installed vLLM sampling and adapter code. Native-hook prepare
performs a GPU-to-CPU metadata stack.tolist() each forward; no-hook dense does
not. The official compiled sampler draws from CUDA RNG without a request-local
generator argument. These facts do NOT prove the cause of divergent steps.
The new independent sync-only arm isolates the metadata read. Separate invasive
first-sample protocols retain initial RNG/canvas and first-sample input/output
privately; traces change synchronization and cannot establish timing gains.
Four arms, two engine seeds and two fresh-engine repeats per protocol (32
engines across sync-only/trace protocols); original V29 queues remain pinned.
New diagnostics have CPU tests, but are NOT deployed/armed in this commit.

Two named query-weight variants now have vLLM code: unit_v30 and confidence_v30
(previous completed call's processed p_top, 1+3*sqrt(1-p_top); first call fully
protected). This ports the confidence prior, NOT Junyu's C_gate or fresh
value-aware router. Preserve M3 DP/R6/A64/Q128/carry0 and torch copy/merge.
C_gate remains rejected in vLLM because the actual accepted mask is unavailable.
All three modes have new full-inventory eight-seed specs for LongBench (59),
AIME (30), HumanEval (164). Qualification/freeze/deploy/launch remains pending;
no variant GPU or E2E performance result exists. Short-task accuracy is required.

Remote CPU regression: 25 passed, no skips, CUDA uninitialized, 0 GPU seconds.
Per-case evidence: results/v30_20261003/sensitivity_cpu001. Local diagnostic,
privacy-comparator and spec/receipt tests: 12 passed. Existing adapter guard,
lifecycle and merge tests also passed. Last direct GPU check found one active
original worker on each of dllm/dlm2/mpk; no new GPU launch. Verified parent:
ed711c8b4afd5bf93a457f03bfc5714a5b36dcb9. Do not mislabel the new source/config freeze as a byte-identical rebind.

Code entry points: scripts/v30_dense_first_divergence.py, scripts/v30_compare_first_sample.py, scripts/v30_sensitivity_panel.py, scripts/v30_new_config.py. Formal speed/accuracy comparisons remain pending.
