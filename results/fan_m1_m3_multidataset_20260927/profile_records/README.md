# Direct profile records

The six `.json.gz` files are byte-preserving gzip copies of the frozen direct-profile JSON. They include accepted timing repetitions, native brackets, first/cold observations, physical work counters, operator-probe numerical measurements, phase evidence, source hashes, and explicit missing target/sequence records.

The source audit found no raw prompts, gold answers, completion tokens, input IDs, hidden states, or tensor payloads. No fields were removed. Input/output and QKV values appear only as digests and scalar statistics. Absolute host and model paths are provenance metadata.

The profile replay uses captured native states. `model_forward` times complete `model.forward` calls with `state.begin` outside the timer; `denoising_step` times the full native step. CUDA events include stream launch gaps, and direct sequences are not natural generated requests. Missing native calls or N16 sequences were not synthesized. See `manifest.json` for SHA-256 checksums.
