"""v27 execution substrate for DiffusionGemma decoding.

  'eager'        -- the HF eager path (every v21-v27 panel before 2026-09-30). Host-bound: GPU savings in GLOBAL
                    attention do not reach wall time (results/m1_m2_m3_frontier_v27_20260929/substrate/README.md).
  'piecewise_v1' -- the vLLM split. The decoder forward is torch.compile(mode='reduce-overhead', dynamic=False)
                    (inductor + CUDA graphs) with graph boundaries at the attention modules that must stay eager:
                      * the 5 GLOBAL attention modules always (torch._dynamo.disable): the variable-length KV
                        concat, FA4 dense/sparse and all M1/M2/M3 state logic run exactly as in eager;
                      * LOCAL attention modules are either inside the graphs -- bound, under a private registry
                        key, to the same native SDPA function they already reach in every GLOBAL-scope arm, so no
                        plugin state is traced -- or eager boundaries too, for arms whose scope routes LOCAL layers
                        (set per request with set_local);
                    after every encoder call the LOCAL sliding-window cache is compacted (HF keeps it as a slice of the full
                    concatenation: prompt-length-dependent strides and the whole prompt's LOCAL K/V retained);
                    the runtime-capture pre-hooks the attention binding registers on the base model run eager; the
                    sampler's accept/renoise are compiled exactly as the official DiffusionGemma compiled path does
                    (generation_diffusion_gemma._compile_functions), cached on the model.
Requires torch with grouped_mm (>= 2.8). On torch 2.6, transformers' MoE falls back to a per-expert loop with a
host sync per call, which cannot be graph-captured.
"""
from __future__ import annotations

SUBSTRATES = ('eager', 'piecewise_v1', 'piecewise_v2', 'piecewise_v3', 'piecewise_v4')
# piecewise_v2 = piecewise_v1 + the shared (all-arm) prompt prefill and the GLOBAL canvas append on official FA4
# causal instead of the 64-row Triton kernel / HF SDPA (substrate/prefill_append_bench.jsonl)
# piecewise_v3 = piecewise_v2 + two value-identical, all-arm host/copy fixes (substrate/time_breakdown.json):
#   * GLOBAL decoder attention no longer torch.cat-s the whole encoder cache with the 256 canvas K/V on every call
#     (modeling_diffusion_gemma.py:450-451; 2.15 ms per forward at 32K, growing with context). Each GLOBAL layer
#     keeps one [prefix + canvas] K and V buffer: the encoder cache is copied in once per canvas (when the encoder
#     cache tensor changes) and each call writes only its canvas rows. Same values, same contiguous layout, same
#     attention kernels.
#   * the denoising step's `if finished_denoising.any():` host sync is skipped for batch size 1: with one item, a
#     finished item ends the canvas loop before the next step, so the guarded torch.where never changes anything.
V3_SUBSTRATES = ('piecewise_v3', 'piecewise_v4')
# piecewise_v4 = piecewise_v3 + one static LOCAL shape for the compiled decoder (all-arm; 2026-09-30):
#   while the encoder's sliding-window cache holds fewer than sliding_window - 1 tokens (a short prompt's first
#   canvases), the decoder sees it LEFT-PADDED to sliding_window - 1 with zero K/V, and the LOCAL attention gets a
#   persistent key mask (pad slots excluded, updated in place). Without this every prompt-dependent cache length is a
#   new decoder-layer shape: on AIME the per-shape recompiles exhausted dynamo's recompile limit and the decoder layers
#   fell back to eager (~110 ms instead of ~23 ms per forward), invisibly to the new-graph counter. Once the window is
#   full there is no padding and no mask, so long prompts run exactly the v3 graphs. The encoder (prefill and canvas
#   appends) always sees the real, unpadded cache; LOCAL-routing arms (local mode 'eager') are never padded.
V4_SUBSTRATES = ('piecewise_v4',)


def prefill_kernel(substrate: str) -> str:
    return 'fa4' if substrate in ('piecewise_v2', 'piecewise_v3', 'piecewise_v4') else 'dense64'


def joined_kv(attn, encoder_keys, encoder_values, keys, values):
    """[encoder cache | canvas] K and V, value-identical to torch.cat(dim=2), from a per-layer persistent buffer.
    The buffer is rebuilt whenever the encoder cache tensors are different objects (new canvas, new request, a
    copied cache) or the geometry differs; otherwise only the canvas rows are written."""
    import weakref

    import torch
    n, m = encoder_keys.shape[2], keys.shape[2]
    state = attn.__dict__.get('_v27_kv')
    fresh = (state is None or state['keys_src']() is not encoder_keys or state['values_src']() is not encoder_values
             or state['n'] != n or state['k'].shape[2] != n + m or state['k'].dtype != keys.dtype
             or state['k'].shape[:2] != encoder_keys.shape[:2] or state['k'].shape[3] != encoder_keys.shape[3]
             or state['v'].shape[3] != encoder_values.shape[3])
    if fresh:
        attn.__dict__['_v27_kv'] = None                     # release the old buffers before allocating
        k = torch.empty((*encoder_keys.shape[:2], n + m, encoder_keys.shape[3]), dtype=keys.dtype,
                        device=keys.device)
        v = torch.empty((*encoder_values.shape[:2], n + m, encoder_values.shape[3]), dtype=values.dtype,
                        device=values.device)
        k[:, :, :n].copy_(encoder_keys)
        v[:, :, :n].copy_(encoder_values)
        state = dict(keys_src=weakref.ref(encoder_keys), values_src=weakref.ref(encoder_values), n=n, k=k, v=v)
        attn.__dict__['_v27_kv'] = state
        attn.__dict__['_v27_kv_builds'] = attn.__dict__.get('_v27_kv_builds', 0) + 1
    state['k'][:, :, n:].copy_(keys)
    state['v'][:, :, n:].copy_(values)
    return state['k'], state['v']


def denoising_step_batch1(model):
    """generation_diffusion_gemma._denoising_step (transformers 5.x), line for line, minus the host sync of
    `if finished_denoising.any():` for batch size 1. With one item, generate() leaves the canvas loop as soon as
    the item finishes (`if torch.all(finished_denoising): break`), so every step starts with it unfinished and the
    guarded torch.where calls would be identities. Batch > 1 falls back to the official step."""
    import torch
    official = model._denoising_step

    def step(decoder_forward, current_canvas, argmax_canvas, input_ids, decoder_position_ids,
             self_conditioning_logits, mask_mapping, past_key_values, finished_denoising, cur_step, sampler,
             logits_processor, diffusion_stopping_criteria, **model_kwargs):
        if current_canvas.shape[0] != 1:
            return official(decoder_forward=decoder_forward, current_canvas=current_canvas,
                            argmax_canvas=argmax_canvas, input_ids=input_ids,
                            decoder_position_ids=decoder_position_ids,
                            self_conditioning_logits=self_conditioning_logits, mask_mapping=mask_mapping,
                            past_key_values=past_key_values, finished_denoising=finished_denoising,
                            cur_step=cur_step, sampler=sampler, logits_processor=logits_processor,
                            diffusion_stopping_criteria=diffusion_stopping_criteria, **model_kwargs)
        cur_step = torch.tensor(cur_step, device=current_canvas.device, dtype=torch.int32)
        torch.compiler.cudagraph_mark_step_begin()
        decoder_outputs = decoder_forward(
            decoder_input_ids=current_canvas, self_conditioning_logits=self_conditioning_logits,
            decoder_attention_mask=mask_mapping, past_key_values=past_key_values,
            decoder_position_ids=decoder_position_ids, **model_kwargs)
        raw_logits = decoder_outputs.logits
        processed_logits = logits_processor(input_ids, raw_logits, cur_step=cur_step)
        probs = torch.softmax(processed_logits, dim=-1, dtype=torch.float32)
        vocab_size = model.config.text_config.vocab_size
        batch_size, canvas_length = current_canvas.shape
        denoiser_canvas = torch.multinomial(probs.view(-1, vocab_size), num_samples=1)
        denoiser_canvas = denoiser_canvas.squeeze(-1).view(batch_size, canvas_length)
        new_argmax_canvas = torch.argmax(processed_logits, dim=-1)
        accepted_canvas = sampler.accept_canvas(current_canvas, denoiser_canvas, processed_logits, cur_step)
        accepted_canvas = accepted_canvas.clone()
        new_current_canvas = sampler.renoise_canvas(accepted_canvas, cur_step)
        new_current_canvas = new_current_canvas.clone()
        if diffusion_stopping_criteria is not None:
            finished_denoising |= diffusion_stopping_criteria(new_argmax_canvas, processed_logits)
        embeddings_dtype = model.model.decoder.embed_tokens.weight.dtype
        self_conditioning_logits = processed_logits.to(embeddings_dtype)
        return new_current_canvas, new_argmax_canvas, self_conditioning_logits, finished_denoising
    return step


def no_concat_forward(attn, modeling):
    """DiffusionGemmaDecoderTextAttention.forward (transformers 5.x), line for line, except that the encoder
    cache and the canvas K/V are joined by joined_kv instead of torch.cat."""
    rotary = modeling.apply_rotary_pos_emb
    fallback = modeling.eager_attention_forward

    def forward(hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)
        cos, sin = position_embeddings
        query_states = attn.q_proj(hidden_states).view(hidden_shape)
        query_states = attn.q_norm(query_states)
        query_states = rotary(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)
        key_states = attn.k_proj(hidden_states).view(hidden_shape)
        value_states = attn.v_proj(hidden_states).view(hidden_shape) if attn.v_proj is not None else key_states
        key_states = attn.k_norm(key_states)
        key_states = rotary(key_states, cos, sin, unsqueeze_dim=2)
        key_states = key_states.transpose(1, 2)
        value_states = attn.v_norm(value_states)
        value_states = value_states.transpose(1, 2)
        if past_key_values is not None:
            layer = past_key_values.layers[attn.layer_idx]
            key_states, value_states = joined_kv(attn, layer.keys, layer.values, key_states, value_states)
        interface = modeling.ALL_ATTENTION_FUNCTIONS.get_interface(attn.config._attn_implementation, fallback)
        attn_output, attn_weights = interface(
            attn, query_states, key_states, value_states, attention_mask,
            dropout=attn.attention_dropout if attn.training else 0.0, scaling=attn.scaling,
            sliding_window=attn.sliding_window, is_causal=attn.is_causal, **kwargs)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn.o_proj(attn_output)
        return attn_output, attn_weights
    return forward
LOCAL_KEY = 'v27_local_sdpa'


def local_mode_for(config: dict) -> str:
    """'eager' when the arm's attention binding covers LOCAL layers (ALL_NATIVE_LEGAL scope), else 'graph'."""
    scopes = {config.get('v20_scope'), (config.get('parent_config') or {}).get('v20_scope'),
              (config.get('parent_config') or {}).get('v21_scope'), config.get('v21_scope')}
    return 'eager' if 'ALL_NATIVE_LEGAL' in scopes else 'graph'


def mirrored_accept(first, sampler, compiled_accept):
    """The compiled accept/renoise are bound to the first sampler and keep their state (accepted_token_mask) there,
    exactly like the official compiled path; mirror it onto this request's sampler after every accept so observers
    of the current sampler (diagnostics, the v27 density gate) read this step's mask, never None or a stale one."""
    def accept_canvas(*a, **k):
        out = compiled_accept(*a, **k)
        sampler.accepted_token_mask = first.accepted_token_mask
        return out
    return accept_canvas


def compact_sliding_cache(cache) -> int:
    """Replace every sliding-window layer's key/value slice view by a compact copy (value-identical)."""
    import torch
    copied = 0
    for layer in getattr(cache, 'layers', ()):
        if getattr(layer, 'is_sliding', False) and isinstance(getattr(layer, 'keys', None), torch.Tensor):
            if not layer.keys.is_contiguous():
                layer.keys, copied = layer.keys.contiguous(), copied + 1
            if not layer.values.is_contiguous():
                layer.values, copied = layer.values.contiguous(), copied + 1
    return copied


def pad_sliding_cache(cache, full: int) -> int:
    """v4: left-pad every sliding layer's cached K/V to ``full`` (= sliding_window - 1) tokens for the decoder, keeping
    the real tensors aside. Returns the pad length shared by all sliding layers (0: the window is full, untouched)."""
    import torch
    pad = None
    for layer in getattr(cache, 'layers', ()):
        if not getattr(layer, 'is_sliding', False) or not isinstance(getattr(layer, 'keys', None), torch.Tensor):
            continue
        if '_v27_real' in layer.__dict__:
            raise RuntimeError('sliding cache padded twice')
        n = layer.keys.shape[-2]
        if n > full:
            raise ValueError(f'sliding cache holds {n} > {full} tokens')
        if pad is None:
            pad = full - n
        elif pad != full - n:
            raise ValueError('sliding layers disagree on their cached length')
        if pad:
            layer._v27_real = (layer.keys, layer.values)
            keys, values = layer.keys, layer.values
            layer.keys = torch.cat([keys.new_zeros((*keys.shape[:2], pad, keys.shape[3])), keys], dim=-2)
            layer.values = torch.cat([values.new_zeros((*values.shape[:2], pad, values.shape[3])), values], dim=-2)
    return pad or 0


def unpad_sliding_cache(cache) -> None:
    """v4: restore the real (unpadded) sliding K/V before anything but the decoder's LOCAL attention reads them."""
    for layer in getattr(cache, 'layers', ()):
        real = layer.__dict__.pop('_v27_real', None)
        if real is not None:
            layer.keys, layer.values = real


def local_attention_v4(sdpa):
    """LOCAL decoder attention for v4: the native SDPA function, with the persistent pad mask while padding is active
    (keys = [pad | real window | canvas]; the canvas part is always attended, as in HF's diffusion decoder mask)."""
    def attention(module, query, key, value, attention_mask, **kwargs):
        import torch
        state = getattr(module, '_v27_pad', None)
        if state is not None and state['active']:
            encoder_mask = state['mask']
            canvas = encoder_mask.new_ones((1, 1, 1, key.shape[-2] - encoder_mask.shape[-1]))
            attention_mask = torch.cat([encoder_mask, canvas], dim=-1)
        return sdpa(module, query, key, value, attention_mask, **kwargs)
    return attention


def install(model, backend: str = 'inductor', name: str = 'piecewise_v1') -> dict:
    """Idempotent. Returns the substrate identity recorded in receipts. backend='eager' (dynamo capture and the
    same graph split, no codegen, no CUDA graphs) exists only for the equivalence diagnostic."""
    if backend not in ('inductor', 'eager') or name not in SUBSTRATES[1:]:
        raise ValueError((backend, name))
    state = getattr(model, '_v27_substrate', None)
    if state is not None:
        return identity(model)
    import copy
    import importlib

    import torch
    import torch._dynamo as dynamo
    if not (hasattr(torch.nn.functional, 'grouped_mm') or hasattr(torch, '_grouped_mm')):
        raise RuntimeError('piecewise_v1 needs torch grouped_mm (>= 2.8); this torch would run the per-expert '
                           'MoE fallback with host syncs')
    for option in ('recompile_limit', 'cache_size_limit'):      # (never reuse `name`: it is the substrate id)
        if hasattr(dynamo.config, option):
            setattr(dynamo.config, option, 256)
    if hasattr(dynamo.config, 'accumulated_recompile_limit'):
        dynamo.config.accumulated_recompile_limit = 4096
    decoder = model.model.decoder
    modeling = importlib.import_module(type(decoder).__module__)
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    modeling.ALL_ATTENTION_FUNCTIONS[LOCAL_KEY] = (local_attention_v4(sdpa_attention_forward) if name in V4_SUBSTRATES
                                                  else sdpa_attention_forward)
    pad_state = dict(active=False, mask=None, full=None)
    global_layers, local_layers = [], []
    for layer in decoder.layers:
        attn = layer.self_attn
        attn._v27_forward = attn.forward                          # bound class method
        attn._v27_eager_forward = dynamo.disable(attn._v27_forward)
        if attn.is_sliding:
            attn._v27_config = attn.config
            graph_config = copy.copy(attn.config)
            graph_config._attn_implementation = LOCAL_KEY
            if graph_config._attn_implementation != LOCAL_KEY:
                raise RuntimeError('could not bind LOCAL attention implementation')
            attn._v27_graph_config = graph_config
            if name in V4_SUBSTRATES:
                attn._v27_pad = pad_state
                window = int(attn.sliding_window)
                pad_state['full'] = window - 1 if pad_state['full'] is None else pad_state['full']
                if pad_state['full'] != window - 1:
                    raise RuntimeError('LOCAL layers disagree on the sliding window')
            local_layers.append(int(attn.layer_idx))
        else:
            if name in V3_SUBSTRATES:
                attn._v27_eager_forward = dynamo.disable(no_concat_forward(attn, modeling))
            attn.forward = attn._v27_eager_forward
            global_layers.append(int(attn.layer_idx))
    encoder = model.model.encoder
    encoder_forward = encoder.forward

    global_attention = [layer.self_attn for layer in decoder.layers if not layer.self_attn.is_sliding]

    def compact_cache_forward(*a, **k):
        # HF's DynamicSlidingWindowLayer stores keys/values as a SLICE of the full concatenation, so after the
        # prefill every LOCAL layer keeps the whole prompt's K/V alive (about 15 GiB at 75K tokens) and exposes a
        # prompt-length-dependent stride to the compiled decoder (a recompile per new prompt). Compacting them is
        # value-identical and frees that memory before the first denoising call.
        for attn in global_attention:
            # piecewise_v3: every encoder call replaces the encoder cache tensors, so the joined K/V buffers are
            # stale from here on; releasing them now keeps them out of the prefill's memory peak (+1.5 GiB at 75K)
            attn.__dict__.pop('_v27_kv', None)
        if name in V4_SUBSTRATES:
            unpad_sliding_cache(k.get('past_key_values'))
        out = encoder_forward(*a, **k)
        cache = getattr(out, 'past_key_values', None) or k.get('past_key_values')
        compact_sliding_cache(cache)
        if name in V4_SUBSTRATES:
            pad = pad_sliding_cache(cache, pad_state['full']) if model._v27_substrate['local'] == 'graph' else 0
            if pad:
                if pad_state['mask'] is None:
                    pad_state['mask'] = torch.ones((1, 1, 1, pad_state['full']), dtype=torch.bool,
                                                   device=next(model.parameters()).device)
                pad_state['mask'][..., :pad] = False      # in place: one persistent tensor, one compiled shape
                pad_state['mask'][..., pad:] = True
            pad_state['active'] = bool(pad)
        return out
    encoder.forward = compact_cache_forward
    base = model.model
    register = base.register_forward_pre_hook

    def register_eager(hook, *a, **k):
        return register(dynamo.disable(hook), *a, **k)
    base.register_forward_pre_hook = register_eager
    model._v27_eager_model_forward = model.forward
    model.forward = (torch.compile(model.forward, mode='reduce-overhead', dynamic=False) if backend == 'inductor'
                     else torch.compile(model.forward, backend='eager', dynamic=False))
    prepare_sampler, prepare_stop = model._prepare_sampler, model._prepare_diffusion_stopping_criteria

    def signature(obj):
        return repr(sorted((key, repr(value)) for key, value in vars(obj).items()
                           if not callable(value) and not isinstance(value, torch.Tensor)))

    def compiled_sampler(*a, **k):
        sampler = prepare_sampler(*a, **k)
        if not hasattr(model, '_v27_accept'):
            model._v27_accept = torch.compile(sampler.accept_canvas, mode='reduce-overhead', fullgraph=True)
            model._v27_renoise = torch.compile(sampler.renoise_canvas, mode='reduce-overhead', fullgraph=True)
            model._v27_sampler_signature, model._v27_first_sampler = signature(sampler), sampler
        elif signature(sampler) != model._v27_sampler_signature:
            # the compiled methods are bound to the FIRST sampler (as in the official _compile_functions); a request
            # with different sampler settings would silently run with the first one's
            raise RuntimeError('piecewise_v1 compiled sampler is bound to a sampler with different settings')
        sampler.accept_canvas = mirrored_accept(model._v27_first_sampler, sampler, model._v27_accept)
        sampler.renoise_canvas = model._v27_renoise
        return sampler

    def compiled_stop(*a, **k):
        stop = prepare_stop(*a, **k)
        if stop is not None:
            if not hasattr(model, '_v27_stop'):
                model._v27_stop = torch.compile(stop.__call__, mode='reduce-overhead', fullgraph=True)
            stop.__call__ = model._v27_stop    # assigned exactly as the official _compile_functions does
        return stop
    model._prepare_sampler, model._prepare_diffusion_stopping_criteria = compiled_sampler, compiled_stop
    if name in V3_SUBSTRATES:
        model._v27_official_denoising_step = model._denoising_step
        model._denoising_step = denoising_step_batch1(model)
    model._v27_substrate = dict(substrate=name if backend == 'inductor' else name + '_eager_backend',
                                global_layers=global_layers, local_layers=local_layers,
                                local=None, torch=torch.__version__)
    set_local(model, 'graph')
    return identity(model)


def set_local(model, mode: str) -> None:
    state = getattr(model, '_v27_substrate', None)
    if state is None or mode == state['local']:
        return
    if mode not in ('graph', 'eager'):
        raise ValueError(mode)
    for layer in model.model.decoder.layers:
        attn = layer.self_attn
        if not attn.is_sliding:
            continue
        if mode == 'graph':
            attn.config = attn._v27_graph_config
            attn.__dict__.pop('forward', None)
        else:
            attn.config = attn._v27_config
            attn.forward = attn._v27_eager_forward
    state['local'] = mode


def identity(model) -> dict:
    state = getattr(model, '_v27_substrate', None)
    if state is None:
        return dict(substrate='eager')
    try:
        from torch._dynamo.utils import counters
        graphs = int(counters['stats'].get('unique_graphs', 0))
    except Exception:
        graphs = None
    return dict(substrate=state['substrate'], torch=state['torch'], local=state['local'],
                eager_global_layers=len(state['global_layers']), local_layers=len(state['local_layers']),
                dynamo_unique_graphs=graphs)
