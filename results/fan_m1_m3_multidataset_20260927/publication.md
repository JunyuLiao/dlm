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

Normal fast-forward push succeeded from 3f2b4db to
45046b297e6f1e6a13fa3bcb68138bc444d2095. Independent git ls-remote
returned that exact branch SHA at 2026-09-27T21:36:42Z.

All800 generation-record SHA checks passed; exported token counts and per-canvas
call totals were conserved. All24 dataset/method first-output work groups match
the published core/historical summaries. Six profile gzip roundtrips and source
SHA checks passed. Raw profile JSON143,127,943bytes compresses to3,978,833bytes;
exported generation records total24,026,799bytes. This follow-up commit records
publication verification only; the cited data commit is immutable.
