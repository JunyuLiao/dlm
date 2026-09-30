"""v27 long-context execution helpers (32K-128K prompts on one H100).

``prefill_dense64`` replaces, for the duration of a request, the causal GLOBAL prefill
and the sliding-window LOCAL prefill with the memory-linear 64-row kernel (native SDPA
at head_dim 512 materializes O(n^2) scores/masks and does not fit at 64K), and chunks the
per-token MoE experts over the token axis (exact). With ``canvas=True`` (state capture
only) canvas GLOBAL calls also use the dense kernel. It is a common execution change
applied to every arm; it is not part of any method.
"""
from contextlib import contextmanager

_SEMANTICS_CHECKED = []   # the model's prefill mask builder, verified once per process


@contextmanager
def prefill_dense64(model, enabled, canvas=False, kernel='dense64'):
    """v27: memory-linear causal GLOBAL prefill for 32K-128K prompts (native SDPA at head_dim 512
    materializes O(n^2) scores and does not fit on one H100 at 64K). Canvas calls are untouched."""
    if not enabled:
        yield
        return
    if kernel not in ('dense64', 'fa4'):
        raise ValueError(kernel)
    import importlib
    from experiments.numerical_qk_reuse.v27_consumer64 import dense64
    registry = importlib.import_module(type(model).__module__.replace('generation_', 'modeling_')).ALL_ATTENTION_FUNCTIONS
    inner = registry['sdpa']
    def expected(window, q, k, device):
        # spot-check rows of an explicit prefill mask against causal / sliding-causal semantics
        import torch
        rows = torch.tensor(sorted({0, q // 3, q // 2, q - 1}), device=device)
        cols = torch.arange(k, device=rows.device)
        pos = rows[:, None] + (k - q)
        want = cols[None, :] <= pos
        if window:
            want = want & (cols[None, :] > pos - window)
        return rows, want
    # v27b: generate() builds the prefill mask mapping (two [1,1,n,n] bool masks, 7.9 GiB at 64K)
    # and keeps it in a local until the NEXT canvas, i.e. through the whole first canvas. The
    # kernel above never reads mask values, so for a batch-1, unpadded, empty-cache prefill the
    # mapping is elided -- after the model's own builder has been checked once per process, on a
    # small synthetic prefill, to produce exactly causal / sliding-causal masks.
    elided = [False]
    encoder = next((m for m in model.modules() if type(m).__name__ == 'DiffusionGemmaEncoderModel'), None)
    builder = getattr(encoder, 'create_masks_for_generate', None) if encoder is not None else None

    def masks_for_generate(*args, **kw):
        import torch
        embeds, mask, cache = kw.get('inputs_embeds'), kw.get('attention_mask'), kw.get('past_key_values')
        if (args or embeds is None or embeds.shape[0] != 1 or embeds.shape[1] <= 256 or mask is None
                or not bool(mask.all()) or cache is None or cache.get_seq_length() != 0):
            return builder(*args, **kw)
        if not _SEMANTICS_CHECKED:
            # (a prompt shorter than the 1500-token probe is its own probe: a fixed n=1500 failed the shape check
            # for every 257-1499-token prompt -- aime26/17 in the v27 quality preview)
            n = min(1500, embeds.shape[1])
            small = builder(**dict(kw, inputs_embeds=embeds[:, :n], attention_mask=mask[:, :n],
                                   position_ids=(kw['position_ids'][:, :n] if kw.get('position_ids') is not None
                                                 else None)))
            window = int(getattr(kw['config'].get_text_config(), 'sliding_window', 0) or 0)
            for key, w in (('full_attention', 0), ('sliding_attention', window)):
                got = small[key]
                got = got if got.dtype == torch.bool else got == 0
                rows, want = expected(w, n, n, got.device)
                if tuple(got.shape[-2:]) != (n, n) or not bool((got[0, 0, rows, :n] == want).all()):
                    raise RuntimeError('model prefill masks are not plain causal / sliding-causal')
            _SEMANTICS_CHECKED.append(True)
        elided[0] = True
        return {'full_attention': None, 'sliding_attention': None}

    def attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kw):
        nq, nk = query.shape[2], key.shape[2]
        if (canvas and nq <= 256 and query.shape[-1] == 512 and attention_mask is None and is_causal is False
                and query.shape[0] == 1 and not dropout and not getattr(module, 'is_sliding', False)):
            # long-context capture: canvas GLOBAL calls through the memory-linear kernel too
            scale = scaling if scaling is not None else query.shape[-1] ** -.5
            return dense64(query, key, value, scale, splits=2), None
        if (kernel == 'fa4' and nq <= 256 and nk > nq and query.shape[-1] == 512 and query.shape[0] == 1
                and not dropout and not getattr(module, 'is_sliding', False) and attention_mask is not None):
            # piecewise_v2: the encoder's canvas append on GLOBAL layers through official FA4 (causal, bottom-right
            # aligned), after spot-checking that the model's own mask is exactly that
            import torch
            rows, want = expected(0, nq, nk, attention_mask.device)
            got = attention_mask[0, 0, rows, :nk]
            got = got if got.dtype == torch.bool else got == 0
            if got.shape == want.shape and bool((got == want).all()):
                from experiments.numerical_qk_reuse import v27_fa4
                scale = scaling if scaling is not None else query.shape[-1] ** -.5
                return v27_fa4.causal(query, key, value, scale), None
        if nq > 256 and query.shape[0] == 1 and not dropout:
            window = int(kw.get('sliding_window') or 0) if getattr(module, 'is_sliding', False) else 0
            if attention_mask is not None:
                import torch
                # (v27b: no reference to the mask is kept; holding it pinned an O(n^2) bool
                # mask, 4 GiB at 64K, for the whole request)
                rows, want = expected(window, nq, nk, attention_mask.device)
                got = attention_mask[0, 0, rows, :nk]
                got = got if got.dtype == torch.bool else got == 0
                if got.shape != want.shape or not bool((got == want).all()):
                    raise RuntimeError('long-context prefill mask is not plain causal/sliding-causal')
            elif is_causal is False or not (is_causal or elided[0]):
                return inner(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                             is_causal=is_causal, **kw)
            scale = scaling if scaling is not None else query.shape[-1] ** -.5
            if kernel == 'fa4':
                from experiments.numerical_qk_reuse import v27_fa4
                return v27_fa4.causal(query, key, value, scale, window), None
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
    if builder is not None:
        encoder.create_masks_for_generate = masks_for_generate
    try:
        yield
    finally:
        registry['sdpa'] = inner
        if builder is not None:
            del encoder.create_masks_for_generate
        for mod, original in patched:
            mod.forward = original


