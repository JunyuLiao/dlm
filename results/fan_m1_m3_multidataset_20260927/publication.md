# v20 code and data publication

The user explicitly authorized publishing code, data and all shareable artifacts to
`coconight01/dlm_test`, branch `research/fan-m1-m3-cost-refresh-20260927`.
This supersedes the earlier pending destination confirmation. Research remains
complete; publication does not authorize or launch another GPU experiment.

## Contents

- Tested implementation, worker/scorer/export scripts and tests; immutable production
  identity remains `00a2c4d3c93a7f685a4f75fba8ea47991fa5c383`.
- Complete main700 and separate G75 reference100 score/work/timing summaries,
  original first84 checkpoint, physical counter data, failures/missing states,
  qualification receipts, resource accounting, Chinese final report and local slides.
- [Generation records](generation_records/README.md): all800 recorded first/warm
  request outputs, generated text, per-canvas calls/stopping metadata and timing,
  with archive/record hashes and the actual available schema documented.
- [Profile records](profile_records/README.md): shareable raw replay measurement
  repetitions and counter/operator evidence, with source hashes and export checks.

Source prompts, benchmark gold, credentials, model weights and captured input
tensors are excluded. Original private archives/deployments remain unchanged.
The records do not contain unrecorded per-denoising-step hidden states or full QK
tensors; no missing trace is reconstructed or implied. Generated text is model
output, not benchmark gold or a correctness guarantee.

## Remote verification

Prepared for a normal fast-forward push to the existing authorized branch.
The follow-up publication receipt records the independently checked remote SHA.
