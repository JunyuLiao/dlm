"""v27 long-context execution helpers (32K-128K prompts on one H100).

``prefill_dense64`` replaces, for the duration of a request, the causal GLOBAL prefill
and the sliding-window LOCAL prefill with the memory-linear 64-row kernel (native SDPA
at head_dim 512 materializes O(n^2) scores/masks and does not fit at 64K), and chunks the
per-token MoE experts over the token axis (exact). With ``canvas=True`` (state capture
only) canvas GLOBAL calls also use the dense kernel. It is a common execution change
applied to every arm; it is not part of any method.
"""
from contextlib import contextmanager


@contextmanager
def prefill_dense64(model, enabled, canvas=False):
    """v27: memory-linear causal GLOBAL prefill for 32K-128K prompts (native SDPA at head_dim 512
    materializes O(n^2) scores and does not fit on one H100 at 64K). Canvas calls are untouched."""
    if not enabled:
        yield
        return
    import importlib
    from experiments.numerical_qk_reuse.v27_consumer64 import dense64
    registry = importlib.import_module(type(model).__module__.replace('generation_', 'modeling_')).ALL_ATTENTION_FUNCTIONS
    inner = registry['sdpa']
    def expected(window, q, k):
        # spot-check rows of an explicit prefill mask against causal / sliding-causal semantics
        import torch
        rows = torch.tensor(sorted({0, q // 3, q // 2, q - 1}), device=attention_mask_holder[0].device)
        cols = torch.arange(k, device=rows.device)
        pos = rows[:, None] + (k - q)
        want = cols[None, :] <= pos
        if window:
            want = want & (cols[None, :] > pos - window)
        return rows, want
    attention_mask_holder = [None]
    def attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kw):
        nq, nk = query.shape[2], key.shape[2]
        if (canvas and nq <= 256 and query.shape[-1] == 512 and attention_mask is None and is_causal is False
                and query.shape[0] == 1 and not dropout and not getattr(module, 'is_sliding', False)):
            # long-context capture: canvas GLOBAL calls through the memory-linear kernel too
            scale = scaling if scaling is not None else query.shape[-1] ** -.5
            return dense64(query, key, value, scale, splits=2), None
        if nq > 256 and query.shape[0] == 1 and not dropout:
            window = int(kw.get('sliding_window') or 0) if getattr(module, 'is_sliding', False) else 0
            if attention_mask is not None:
                import torch
                attention_mask_holder[0] = attention_mask
                rows, want = expected(window, nq, nk)
                got = attention_mask[0, 0, rows, :nk]
                got = got if got.dtype == torch.bool else got == 0
                if got.shape != want.shape or not bool((got == want).all()):
                    raise RuntimeError('long-context prefill mask is not plain causal/sliding-causal')
            elif not is_causal:
                return inner(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                             is_causal=is_causal, **kw)
            scale = scaling if scaling is not None else query.shape[-1] ** -.5
            return dense64(query, key, value, scale, splits=1, causal=True, window=window), None
        return inner(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                     is_causal=is_causal, **kw)
    # MoE experts are per-token: chunking the token axis is the same computation with bounded
    # activation memory (grouped_mm over 64K tokens at once does not fit next to the weights).
    import torch
    patched = []
    for name, mod in model.named_modules():
        if name.endswith('.experts') and hasattr(mod, 'forward'):
            original = mod.forward
            def chunked(hidden, top_k_index, top_k_weights, *a, _orig=original, **kw):
                if hidden.shape[0] <= 16384:
                    return _orig(hidden, top_k_index, top_k_weights, *a, **kw)
                return torch.cat([_orig(hidden[i:i + 16384], top_k_index[i:i + 16384],
                                        top_k_weights[i:i + 16384], *a, **kw)
                                  for i in range(0, hidden.shape[0], 16384)], 0)
            mod.forward = chunked
            patched.append((mod, original))
    registry['sdpa'] = attention
    try:
        yield
    finally:
        registry['sdpa'] = inner
        for mod, original in patched:
            mod.forward = original


