# V31 AIME26 GLOBAL budget sweep

Frozen sweep on the same AIME26 manifest and native sampler as the completed
V31 panel. Arms are one dense reference and three MAGE + LOCAL sparse variants:
GLOBAL budgets 1024, 2048, and 4096 tokens; LOCAL budget 512 tokens with the
`local_compact_triton_q64` consumer. Each arm has all 30 problems at seeds
42/43/44 (90 cells).

Thinking is enabled in the pinned manifest. The generation cap is 8192 tokens,
including reasoning and final answer. Native canvas length is 256, maximum 48
denoising calls per canvas, confidence threshold 0.005, stability threshold 1,
entropy bound 0.1, temperature schedule 0.8 -> 0.4, and EOS stopping.

The dense reference uses vLLM 0.30.0 with the PR #51994 mask fix. Sparse arms
use FA4 selection, fused logit statistics, chunked DP build, Triton KV copy and
LSE merge, qblock-max selection, first-call carry, settledness re-selection at
0.15, and sticky bonus 1.386. Cache pages and sparse tiles are 64 tokens.

Raw completions remain in the user-owned private run directory. Public records
contain only hashes, lengths, timing, and sanitized receipts.
