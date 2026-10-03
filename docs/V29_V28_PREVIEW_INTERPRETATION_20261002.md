# Completed V28 preview, read-only interpretation (2026-10-02)

Source: branch `research/vllm-variants-20261002`, publication `91db65396`;
`results/v28_20261002/seed4_preview001/summary.json` and receipts. Generation
remains `dcfdb8730`; repaired CPU scorer is `7c8c608ed`. The repair canonicalizes
model snapshot paths; all original file hashes matched, and no generation
binding, output or task grader changed.18 scorer unit tests and real public
correct/wrong NeMo examples pass.24 workers closed,288 timed requests,48 per
variant; six distinct questions, four engine seeds, two sequential repeats.
Reserved7914.059110 GPU seconds. This remains a preview.

Ratios below are candidate/reference (lower is less time); intervals cluster by
question, with only two independent questions in each length stratum. Such small
cluster counts give fragile intervals and do not support a general headline.

| Comparison | W ratio [95% CI] | S/N ratio | N ratio | Correct /48 |
|---|---|---:|---:|---|
| Q64 release / Q128 release | .94245 [.89175,.99603] |1.00516|.92029|41/38|
| Q128 release / Q128 legacy |1.01228 [.99035,1.03470]|.99703|1.01815|38/40|
| Q128 legacy / default dense |.81008 [.73433,.89363]|.93823|.82296|40/29|
| Q128 release / all-kept release |.99576 [.91015,1.08942]|.90967|1.09672|38/36|
| Matched native / default dense |.78977 [.72001,.86628]|1.02514|.73670|43/29|

Q64 is a follow-up candidate, not a demonstrated per-step implementation gain.
Its average S/N is slightly higher than matched Q128; its W point improvement
comes with lower N. Its W ratios by32/64/96K are1.02953/.83588/.97273; by engine
seed they are.95368/1.10980/.87947/.84754. The sign changes with seed and length.
The prior component-only Q64 attention improvement cannot substitute for these
request observations. Keep all seeds and expand question coverage.

Canvas-release hygiene has no supported W gain here. Main versus matched
all-kept has a lower amortized S/N, but additional denoising steps offset that
benefit: W interval crosses1. Native itself beats default dense in this preview
despite higher S/N, also through lower N. Thus the method/default-dense comparison
does not isolate the sparse contribution.

Supplementary point ratios calculated from the same48 public geometric means:
legacy main/native1.025715, release main/native1.038311, Q64/native.978556. These
pairs were not among the frozen family comparisons; no CI is inferred by dividing
other intervals. Matched native has43correct, versus40/38/41 for those candidates.
There is no demonstrated general speed advantage with preserved accuracy over
the matched native condition.

Decision: retain unchanged main as the four-arm full-coverage confirmation,
with59 LongBench questions and8 engine seeds. Q64 merits a separately frozen
confirmation; it must retain Q128 and strong dense controls and include all
lengths, rather than selecting only the favorable64K preview. Short-prompt
AIME/HumanEval use separate full task inventories. Regroup's equally fused
component comparison remains approximately tied and is not promoted on these
unrelated Q64 results. No noninferiority claim is made.
