# Independent Junyu adapter CPU qualification

2026-10-03 04:13 (UTC-5). Ten tests passed, no skips, failures or CUDA
initialization attempts; 0 GPU seconds. These test CPU tensors and a fake peer
router, not the real CUDA kernel or model accuracy/performance. The actual
standalone E2E runner is not implemented or armed in this checkpoint.

- `receipt.json`: environment, exact repository source hashes and limitations.
- `cases.jsonl`: all ten cases and their setup/call/teardown results.
- `status.json`: read-only 04:09 snapshot of earlier formal/qualification queues;
  active workers' unfinished reservation spans are not included in GPU seconds.

Coverage includes prior-call weights, new-canvas reset, padded-query handling,
five-GLOBAL eligibility, output orientation, exact scale forwarding, no M3
installation, dense-kernel dispatch without projection, all-kept enforcement,
failure cleanup, policy provenance and rejection of wrong/incomplete receipts.
Zero skipped tiles is allowed for finite-policy runs: it is a valid possible
negative result, not by itself evidence of an execution failure.
