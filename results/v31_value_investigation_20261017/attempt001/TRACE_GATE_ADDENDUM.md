# Conditional trace localization, 2026-10-17 10:16 UTC

The first completed crossed engine found a reference/device directional-residual
mismatch on question0, seed1: N321 versus310, both incorrect. Both V31 bookends
were identical (N218). Swaps matched exactly on all four cells in that engine.
This is an observed implementation-associated discrepancy, not yet a diagnosed
mathematical error or a reproducible population effect.

If the remaining crossed engines do not resolve this discrepancy, use the one
already-budgeted trace engine to localize this case. Change the proposed trace
seed from42 to1 and replace the two candidate slots with reference and device
directional residual. Preserve V31 before/after. This is explicitly a post-hoc
mechanism diagnostic selected because a mismatch occurred; it cannot support
accuracy selection or benchmark inference. Do not add a second trace engine.

Match the crossed study's four explicit warmups (question0, seed42, both
implementations of both methods) and implicit V31 warmup. The measured group
has four requests, question0 seed1. Fingerprint full logits for only the first
six denoising calls. For up to512 denoising calls per request, retain compact
native canvas/argmax/step/RNG/entropy summaries, layer29 Q/output/support hashes,
and correction-cache tensors at observations privately. Later full-logit
equality remains unknown. Generation itself retains the native output budget
and stopping rule; the observer does not terminate a request at512 calls.

The longer lightweight trace can locate a late cache/output discrepancy without
copying vocabulary-sized logits at every call. Reconstructed acceptance uses
the existing sampler formula and is labeled separately from captured native
state. No model forward, random draw, stopping override, or selector change is
introduced. Compare complete traced outputs against untraced crossed repeats
and control bookends; do not discard valid mismatches.

Launch only after the crossed run finishes, with cumulative diagnostic
reservation below50 minutes; retain the overall60-minute target. Full503 stays
paused. If the trace cannot resolve the discrepancy, report that limitation and
stop rather than extending the diagnostic automatically.
