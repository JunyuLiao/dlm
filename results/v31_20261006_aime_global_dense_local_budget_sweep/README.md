# V31 AIME26 GLOBAL budget sweep with dense LOCAL

Frozen matched panel on the same AIME26 manifest, native sampler, and seeds 42/43/44 as the preceding V31 sweep. One dense reference is compared with MAGE GLOBAL budgets 1024, 2048, and 4096 tokens. All 25 LOCAL layers remain native dense: `LOCAL_KV_BUDGET` is intentionally unset and no local sparse consumer is installed.

Thinking is enabled. The native sampler uses generation cap 8192, canvas length 256, maximum 48 denoising calls per canvas, confidence threshold 0.005, stability threshold 1, entropy bound 0.1, temperature schedule 0.8 -> 0.4, and EOS stopping. GLOBAL and cache tiles are 64 tokens.

Raw completions and private manifests remain outside the repository. Public records contain sanitized hashes, lengths, timing, and receipts. Results are publishable only after `final_complete.json` is written.
