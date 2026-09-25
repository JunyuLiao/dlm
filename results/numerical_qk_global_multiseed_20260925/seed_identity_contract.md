# v13 seed and identity contract

- **Generation seed** (17 or 29) is part of every cell: `cell_id = sha256(protocol_id, model_revision, arm_config_hash,
  question_id, seed)[:24]`. It is threaded into `runner._one`, which sets `GenerationRequest.seed`, and
  `adapter.generate` calls `seed_everything(seed)`; that global torch RNG drives the native sampler. The driver asserts
  the receipt's seed equals the scheduled seed. The seed appears in the ledger, the private receipt path
  (`cells/<cell_id>/attempt00.json`) and every public row.
- `arm_config_hash` is the effective arm config without volatile fields (ids, seeds, phase, manifest, fingerprints).
  The driver recomputes it at start and refuses to run if it differs from `frozen_protocol.json`.
- **Execution key** = `(cell_id, role, repeat)`. A warm repeat is a timing repeat of its OWN cell only. Another seed
  is a different cell, never a repeat.
- **Resume** requires the same protocol_id, arm hashes, model revision, source hashes and private root. Any ledger
  execution outside the frozen schedule (for example old seed-42 rows) is refused, and so are duplicates.
- Every scheduled execution runs exactly once. Failures are recorded and count as incorrect or missing. There are
  no silent retries. A device error stops the worker, and a clean worker resumes the remaining schedule.
- **Warm acceptance:** the warm run is OK, has 0 new Triton compilations, and matches its attempt 0 exactly on
  token hash, per-canvas calls and termination. Rejected rows stay in the raw ledger and appear as unaccepted
  timings.
- The v10 driver's hard-coded `seeds=[42]` in `arm_config` is replaced by the caller's seeds (default 42 kept for old
  callers). `_one` was never seed-hard-coded; the v10 driver passed 42 explicitly.
- Tests: `tests/test_v13_seed_driver.py`, 7 passed (CPU mocks).
