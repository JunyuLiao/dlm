"""v12 GLOBAL-only scope: temporal methods on the 5 full-attention layers, the
25 LOCAL (sliding) layers completely native.

Mechanism: a proxy adapter whose ``is_blasst_attention_module`` selects only
the GLOBAL decoder attention modules is handed to the EXISTING installers
(``_install_dense`` binding, numerical ``Attention`` router, Junyu's router,
and their ``Sketches`` leases). LOCAL modules are therefore never tagged,
never hooked, never given score history/summaries/sketches; the BLASST
registry dispatcher sends an untagged call straight to the captured original
native SDPA function with the original arguments (one Python ``getattr``
check per LOCAL call is the only added work; disclosed, measured by spies).

Arms (runner conditions, all require this plug-in):
  global_T   fresh Junyu T on GLOBAL layers (collect off by default)
  global_M1  historical-QK / current-projected-V M1, decision every step
  global_M3  same, decision interval 2
  global_B8  same score anchors + anchor selector, decision held until the
             next real A8 score observation / canvas reset (decision interval
             8 == score period); no summaries (legacy selector); same consumer
GLOBAL legality is the model's own (unrestricted decoder domain, mask None);
a non-None mask on a routed GLOBAL call fails explicitly.
"""
from contextlib import contextmanager

import torch

SCOPE = 'global_only_native_local'
CONDITIONS = ('global_T', 'global_M1', 'global_M3', 'global_B8')
DECISION_INTERVAL = {'global_M1': 1, 'global_M3': 2, 'global_B8': 8}


class GlobalScopeAdapter:
    """Delegates everything to the real adapter except module selection."""

    def __init__(self, adapter):
        self._adapter = adapter
        self.active_layers = sorted(int(m.layer_idx) for n, m in adapter.model.named_modules()
                                    if adapter.is_blasst_attention_module(n, m) and not m.is_sliding)
        local = [int(m.layer_idx) for n, m in adapter.model.named_modules()
                 if adapter.is_blasst_attention_module(n, m) and m.is_sliding]
        if not self.active_layers or len(self.active_layers) + len(local) != len(set(self.active_layers) | set(local)):
            raise ValueError('could not partition decoder attention into GLOBAL/LOCAL')
        self.local_layers = sorted(local)

    def __getattr__(self, name):
        return getattr(self._adapter, name)

    def is_blasst_attention_module(self, name, module):
        return self._adapter.is_blasst_attention_module(name, module) and not module.is_sliding


class _MaskGuard:
    """Routed GLOBAL calls must carry the model's own mask=None domain."""

    def __init__(self, router):
        self.router = router

    def __call__(self, module, q, k, v, mask, **kwargs):
        if mask is not None or module.is_sliding:
            raise ValueError('GLOBAL-only scope: unsupported mask or a LOCAL call reached the router')
        return self.router(module, q, k, v, mask, **kwargs)

    def __getattr__(self, name):
        return getattr(self.router, name)


@contextmanager
def install(adapter, config, condition):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if getattr(adapter.model, '_value_direction_lease', False):
        raise RuntimeError('Concurrent request binding unsupported')
    scoped = GlobalScopeAdapter(adapter)
    effective = dict(scope=SCOPE, condition=condition, active_layers=scoped.active_layers,
                     native_local_layers=scoped.local_layers)
    if condition == 'global_T':
        from experiments.diffusion_gemma_jl_output_aware.projections import Projections
        from experiments.value_direction_hopper.integration import install as install_t
        from experiments.value_direction_hopper.query_adaptive import State
        collect = bool(config.get('collect', False))
        with install_t(scoped, config['library'], config['policy'], mode='value', projections=Projections(),
                       torch_library=config['torch_library'], collect=collect) as (binding, router):
            binding.runtime.attention_override = _MaskGuard(router)
            state = State('T', router, m_ref=config['m_ref'], beta=config['beta'], gamma=config['gamma'],
                          diagnostics=bool(config['diagnostic']))
            effective.update(method='fresh Junyu T', collect=collect, precision=router.precision, tma=router.tma,
                             projection='fused' if router.cache.fused else 'torch', library=config['library'],
                             bound_modules=sorted(int(m.layer_idx) for m in binding.modules))
            yield dict(binding=binding, router=router, state=state, counters=lambda: dict(effective, calls=router.calls))
        return
    from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _install_dense
    from .integration import Attention, NativeReuseState
    interval = DECISION_INTERVAL[condition]
    if int(config['decision_interval']) != interval:
        raise ValueError(f'{condition} requires decision interval {interval}')
    selector = config.get('selector', 'legacy_recompute')
    if condition == 'global_B8' and selector != 'legacy_recompute':
        raise ValueError('B8 never consumes summaries; it must not build them')
    adapter.model._value_direction_lease = True
    binding = router = None
    try:
        binding = _install_dense(scoped)
        router = Attention(scoped, config['policy'], score_period=config['score_refresh_period'],
                           decision_interval=interval, trace=False, support='legacy_junyu_mask',
                           output_mode=config.get('output_mode', 'historical_route_preqk_current_output'),
                           selector=selector, selector_layers=config.get('selector_layers', 'all'),
                           kernel_variant=config.get('kernel_variant', 'generic'),
                           telemetry=config.get('telemetry', 'minimal'),
                           guard_mode=config.get('guard_mode', 'fused'),
                           consumer=config.get('consumer', 'hopper'), support_build=config.get('support_build'))
        binding.runtime.attention_override = _MaskGuard(router)
        state = NativeReuseState('T', router, m_ref=config['m_ref'], beta=config['beta'], gamma=config['gamma'],
                                 diagnostics=config['diagnostic'])
        effective.update(method={'global_M1': 'M1 historical-QK + current projected V, decision every step',
                                 'global_M3': 'M3 decision interval 2',
                                 'global_B8': 'anchor bitmap held to the next A8 observation / canvas reset'}[condition],
                         bound_modules=sorted(int(m.layer_idx) for m in binding.modules))

        def counters():
            c = router.counters()
            c.update(effective, score_period=router.cache.score_period, decision_interval=router.cache.decision_interval,
                     history_layers=sorted(router.cache.entries), sketch_layers=sorted(router.sketches.entries),
                     summary_layers=sorted(router.summaries))
            return c
        yield dict(binding=binding, router=router, state=state, counters=counters)
    finally:
        if router is not None:
            router.close()
        if binding is not None:
            binding.close()
        if getattr(adapter.model, '_value_direction_lease', False):
            del adapter.model._value_direction_lease
