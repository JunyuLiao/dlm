# Native-mask support audit (Phase A1)

Function under test: `dllm.attention.blasst.core._attention_validity` (the
exact call used by the numerical adapter and fresh Junyu T), not a
reimplementation. Geometry: `sliding_window=1024`, `layer_types` has 25/30
local (`sliding_attention`) and 5/30 global (`full_attention`) layers, from
the actually executed model config
(`f7f5b7f5fa82ffc52addd066915886d497f5517b/config.json`, `text_config`).
Canvas length 256 matches the runner's request `block_size`.

Native ground truth = `sliding_window=None`: the installed Transformers 5.11
`sdpa_attention_forward` (`integrations/sdpa_attention.py`) ignores the
`sliding_window` keyword entirely, and DiffusionGemma's decoder attention is
bidirectional (`is_causal=False`, `modeling_diffusion_gemma.py`). Native
local-layer support is therefore every key already in the DynamicCache
(itself pre-capped to `sliding_window-1` tokens by
`DynamicSlidingWindowLayer.update`) plus the full canvas -- no further
per-query narrowing.

Legacy (`legacy_junyu_mask`, the only mode any prior result used) additionally
applies the query-relative bound `k_position >= q_position - sliding_window +
1`.

## Result (`native_mask_audit.json`)

| prefix | native legal pairs | legacy legal pairs | excluded prefix pairs | last-query excluded prefix keys (of prefix) |
|---|---|---|---|---|
| 0..768  | equal | equal | 0 | 0 |
| 1000 | 321536 | 294508 | 27028 | 232 / 1000 |
| 1023 | 327424 | 294784 | 32640 | 255 / 1023 |

Global layers (5/30) are unaffected in every row (no `sliding_window`).

## Interpretation

The two supports are **identical** whenever the committed local-layer prefix
is at most `sliding_window - canvas = 768` tokens -- true for early canvases.
Once a local layer's capped prefix exceeds 768 tokens (i.e. the running
document context has filled the 1024-token local window past one canvas'
worth of headroom), legacy silently drops up to ~25% of the native-legal
prefix for the *last* query of the canvas, growing canvas-by-canvas toward
that ceiling. This is a real, quantified support mismatch on 25/30 layers.

It is **not** a sufficient explanation for M1/M3's 0/4 quality collapse: T
(fresh Junyu, current QK every call) inherits the identical legacy support
and still scores 3/4. Comparing native_mask against legacy_junyu_mask
isolates one genuine correctness gap to fix, but the M1/M3 failure mode has
to be explained by something T does not share -- i.e. staleness/support
churn under the score cache, addressed separately in Phase B.

## Implementation

`experiments/numerical_qk_reuse/integration.py`: `Attention(..., support=
'legacy_junyu_mask'|'native_mask')`. `native_mask` passes `sliding_window=None`
into both the physical crop computation and `observe_scores`'s
`_attention_validity` call; `legacy_junyu_mask` (default) is byte-for-byte the
prior behavior. `runner.py --support` and `native_reuse_smoke_batch.py
--support` thread the choice through to the frozen run config (so it appears
in the config fingerprint). Regression: `tests/test_numerical_reuse_native_mask.py`.

Reproduce: `PYTHONPATH=src:. python scripts/native_reuse_mask_audit.py
--output results/numerical_qk_reuse_recovery_20260924/native_mask_audit.json`
(pure function of shape/position; no GPU or model load required).
