# CP1 RULER request correction

The initial CP1 capture used the numerical QK request default `thinking=True` for a RULER4K prompt. The frozen RULER request path in `experiments/diffusion_gemma_solattn_blasst_multibench/runner.py:_request` uses `thinking=False` and each task's original 30/32/50/120/128 token output budget. The numerical QK request path therefore changed the chat template and failed its exact prompt-token check after generation. This is a protocol mismatch, not evidence of a tokenizer or dataset mutation.

On the pinned mpk CPU tokenizer, the first RULER prompt's stored 3,949-token list matches `thinking=False` exactly; `thinking=True` produces 3,956 tokens and does not match. No answer or gold content was inspected in this check.

The corrected v18 generation manifests explicitly set `thinking=False` for RULER and `thinking=True` for AIME. They were written under the new private `private_v2` path and their draft protocols under `protocols_v2` in `/media/volume/dllm-1/dyh/junyu_frontier_v18_20260926`. Initial CP1 files and hashes were preserved. Calibration and evaluation must use the corrected v2 manifest byte hashes and a new immutable deploy; the failed capture is not a valid generation result.
