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
                    the runtime-capture pre-hooks the attention binding registers on the base model run eager; the
                    sampler's accept/renoise are compiled exactly as the official DiffusionGemma compiled path does
                    (generation_diffusion_gemma._compile_functions), cached on the model.
Requires torch with grouped_mm (>= 2.8). On torch 2.6, transformers' MoE falls back to a per-expert loop with a
host sync per call, which cannot be graph-captured.
"""
from __future__ import annotations

SUBSTRATES = ('eager', 'piecewise_v1')
LOCAL_KEY = 'v27_local_sdpa'


def local_mode_for(config: dict) -> str:
    """'eager' when the arm's attention binding covers LOCAL layers (ALL_NATIVE_LEGAL scope), else 'graph'."""
    scopes = {config.get('v20_scope'), (config.get('parent_config') or {}).get('v20_scope'),
              (config.get('parent_config') or {}).get('v21_scope'), config.get('v21_scope')}
    return 'eager' if 'ALL_NATIVE_LEGAL' in scopes else 'graph'


def install(model) -> dict:
    """Idempotent. Returns the substrate identity recorded in receipts."""
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
    for name in ('recompile_limit', 'cache_size_limit'):
        if hasattr(dynamo.config, name):
            setattr(dynamo.config, name, 256)
    if hasattr(dynamo.config, 'accumulated_recompile_limit'):
        dynamo.config.accumulated_recompile_limit = 4096
    decoder = model.model.decoder
    modeling = importlib.import_module(type(decoder).__module__)
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    modeling.ALL_ATTENTION_FUNCTIONS[LOCAL_KEY] = sdpa_attention_forward
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
            local_layers.append(int(attn.layer_idx))
        else:
            attn.forward = attn._v27_eager_forward
            global_layers.append(int(attn.layer_idx))
    base = model.model
    register = base.register_forward_pre_hook

    def register_eager(hook, *a, **k):
        return register(dynamo.disable(hook), *a, **k)
    base.register_forward_pre_hook = register_eager
    model._v27_eager_model_forward = model.forward
    model.forward = torch.compile(model.forward, mode='reduce-overhead', dynamic=False)
    prepare_sampler, prepare_stop = model._prepare_sampler, model._prepare_diffusion_stopping_criteria

    def compiled_sampler(*a, **k):
        sampler = prepare_sampler(*a, **k)
        if not hasattr(model, '_v27_accept'):
            model._v27_accept = torch.compile(sampler.accept_canvas, mode='reduce-overhead', fullgraph=True)
            model._v27_renoise = torch.compile(sampler.renoise_canvas, mode='reduce-overhead', fullgraph=True)
        sampler.accept_canvas, sampler.renoise_canvas = model._v27_accept, model._v27_renoise
        return sampler

    def compiled_stop(*a, **k):
        stop = prepare_stop(*a, **k)
        if stop is not None:
            if not hasattr(model, '_v27_stop'):
                model._v27_stop = torch.compile(stop.__call__, mode='reduce-overhead', fullgraph=True)
            stop.__call__ = model._v27_stop    # assigned exactly as the official _compile_functions does
        return stop
    model._prepare_sampler, model._prepare_diffusion_stopping_criteria = compiled_sampler, compiled_stop
    model._v27_substrate = dict(substrate='piecewise_v1', global_layers=global_layers, local_layers=local_layers,
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
