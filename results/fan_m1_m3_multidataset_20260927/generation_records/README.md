# v20 generation records (800 requests)

These are generated outputs and request-level telemetry for the frozen v20 Fan panel: 700 core executions plus 100 separately named G75L30_nativeQ128 executions. Each JSON record contains generated completion tokens, raw generated completion, prediction, per-canvas native stopping flags and schedule steps, decoder calls, timing, and available router counters. There are 400 first and 400 accepted warm records. The core first84 and remainder retain their original stage labels.

The export excludes prompts, prompt hashes, gold/answer-source data, credentials, private environment details and absolute private receipt paths. Question IDs, seeds, arm names, host/GPU IDs and source SHA-256 identities remain for reproducibility. Source tarballs were read without extraction or modification. The manifest binds every exported byte to its source archive and original receipt hash.

Per-canvas schedule_steps are the actual native adaptive decoder-call trajectory. Full per-step QKV/attention tensors, individual attention masks and intermediate activations were not captured in these request receipts and are not reconstructible from this export. The GPU timeline starts after the first actual encoder forward and includes host launch gaps and later encoder/commit work; it is not synchronized prefill-excluded generation wall. Request wall/call is amortized, not direct forward timing.

The historical G75L30_nativeQ128 arm is a native-Q128/K64 transplant, not identical to the older vLLM physical grouping. Its runs were later than the core panel; timing contrasts may include temporal drift. No gold or scoring input is needed to inspect these records. See manifest.json for exact counts and hashes.
