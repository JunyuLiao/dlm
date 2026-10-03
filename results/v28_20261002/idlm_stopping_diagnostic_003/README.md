# I-DLM dense stopping diagnostic 003

Three sequential toy requests in one official dense engine, seeded once with seed 0, all naturally stopped at native EOS. Completion lengths were 868, 549 and 1200 tokens. Each response contained one opening and one closing thinking tag; each contained the expected answer substring after the closing tag. That substring check does not establish exact final-answer correctness.

The previous 512-token diagnostic therefore had a budget shorter than every completed response observed here. This supports budget-limited thinking for this toy rather than an obvious sustained repetition or EOS-list failure. It is not a paired causal comparison: the longer first request consumes additional RNG before subsequent requests. These are three calls under one engine seed, not three independent seeds or a quality benchmark.

The new run uses the pinned official model chat template with default thinking, the original IDLMBlockN4 configuration, unchanged temperature/top-k/top-p, native EOS policy, and a 4096-token budget permitted by the [official project example](https://github.com/Introspective-Diffusion/I-DLM/blob/a23c1a12ef997c7f3ad616b25bcfb62db39ded68/README.md#L105). No diagnostic early stopping or algorithm changes were applied. The existing one-line Python 3.10 annotation compatibility patch is recorded in summary.json.

| Request | Completion tokens / EOS position | Think closes | Repeated 8grams | Tail 256 repeated 8grams |
|---|---:|---:|---:|---:|
| 1 | 868 | 1 | 0.000000 | 0.000000 |
| 2 | 549 | 1 | 0.007380 | 0.000000 |
| 3 | 1200 | 1 | 0.010059 | 0.004016 |

EOS positions are one-based in actual output token IDs. The pinned [API returns output IDs](https://github.com/Introspective-Diffusion/I-DLM/blob/a23c1a12ef997c7f3ad616b25bcfb62db39ded68/inference/sglang/sglang/srt/managers/tokenizer_manager.py#L1537). Repetition is the fraction of overlapping ngram windows beyond the number of distinct windows. Full response text, token IDs and metadata are retained only in private storage. CPU-only recomputation verified every public statistic and token count against those records; frozen run files still match their manifest.

The worker exited 0 and released the GPU. This run reserved 193.175201310 GPU seconds, including import, load, first-call compilation, generation, diagnostics and teardown. All I-DLM attempts total 570.660168628 seconds; all new-model attempts total 916.209686782 seconds. The first request latency includes compilation and is not a steady-state performance result. A failed prelaunch byte-transfer check is retained separately and launched no GPU worker. Runs 001 and 002 remain unchanged.

Actual denoising-forward counts remain unavailable: the internal counter is cumulative, its per-request API exposure and graph capture/model-call coverage have not been verified. The pure-prefill path does increment that counter. No counter instrumentation was added, and actualN is not inferred from output lengths. This diagnostic demonstrates native stopping on these toy calls only; general baseline accuracy and GPU sparse integration remain unqualified.
