# Supplemental retrospective extraction checkpoints

This is a descriptive addition to the original frozen text rubric. Its source
and definition are sealed before any NEW confirmation text is inspected. Panel
generation had begun, but no new raw completion has been opened for designing
this metric. It changes no selector, native sampler, primary endpoint, cohort,
multiple-comparison rule or R promotion decision. It is not a new confirmatory
primary analysis and does not replace the pre-generation text_metrics.py rubric.

Use the same official final-response and task extractors, first on every original
core-arm short output, then on every new panel output. Check newline, closing
brace, code-fence and full-output character checkpoints. For AIME, ask only if
the official numeric parser extracts a candidate, never whether it matches gold.
For HumanEval, use the task entry point and official code extractor; execute no
prefix tests. Report the first successful checkpoint and character fraction,
including every missing result. Do not scan for a gold-matching answer.

An extractable candidate may be wrong or later revised. Character positions are
not denoising calls, token times or canvas boundaries. These retrospective
checkpoints neither define nor test an early-stop policy. Correct-in-both
descriptions remain post-treatment subsets; the full official quality panel is
primary. Metrics/IDs/hashes only are public; raw text stays private. No model
forward or GPU work is added, and CPU analysis runs after the timed R panel.
