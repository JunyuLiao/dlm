"""v18 opt-in all-layer scope using the native SDPA legal support domain.

LOCAL/GLOBAL remains the native policy kind. The fresh value kernel sees all
keys allowed by native SDPA (including LOCAL layers whose native call has no
explicit mask); this scope does not add the historical Junyu sliding window.
"""
from contextlib import contextmanager
import math

from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _install_dense

from .integration import Attention
from .query_adaptive import State


CONDITION = 'native_legal_all_layers'
ARMS = {'D_matched': 'kernel_dense', 'U50': 'unweighted', 'U60': 'unweighted',
        'U70': 'unweighted', 'T50': 'T', 'T60': 'T', 'T70': 'T'}


def runtime_policy(policy, method):
    if set(policy) != {'global', 'local'} or any('log_threshold' not in policy[k] for k in policy):
        raise ValueError('both calibrated GLOBAL/LOCAL thresholds required')
    if method == 'kernel_dense':
        if any(policy[k]['log_threshold'] != '-inf' for k in policy):
            raise ValueError('D_matched must retain every legal tile')
        return {k: {**policy[k], 'log_threshold': -float('inf')} for k in policy}
    if any(type(policy[k]['log_threshold']) not in (int, float) or
           not math.isfinite(policy[k]['log_threshold']) for k in policy):
        raise ValueError('sparse calibration thresholds must be finite numbers')
    return policy


@contextmanager
def install(adapter, config, condition):
    if condition != CONDITION or config.get('frontier_arm') not in ARMS:
        raise ValueError('native_legal_all_layers requires a named frontier arm')
    arm = config['frontier_arm']
    method = ARMS[arm]
    if config.get('method') != method:
        raise ValueError(f'{arm} requires method={method}')
    if config.get('diagnostic'):
        raise ValueError('v18 frontier production arms require diagnostics off')
    if getattr(adapter.model, '_value_direction_lease', False):
        raise RuntimeError('Concurrent request binding unsupported')
    policy = runtime_policy(config['policy'], method)
    if method != 'kernel_dense' and config.get('target') != int(arm[1:]):
        raise ValueError('arm target and frozen calibration disagree')
    adapter.model._value_direction_lease = True
    binding = router = None
    try:
        binding = _install_dense(adapter)
        router = Attention(adapter, config['library'], policy, torch_library=config['torch_library'],
                           mode='value', collect=bool(config.get('collect', False)),
                           support_geometry='native_legal')
        binding.runtime.attention_override = router
        state = State(method, router, m_ref=config['m_ref'], beta=config['beta'], gamma=config['gamma'],
                      diagnostics=False, fast_t=method == 'T')

        def counters():
            return dict(scope=CONDITION, frontier_arm=arm, method=method,
                        support_geometry=router.support_geometry, routed_calls=router.calls,
                        active_layers=sorted(int(m.layer_idx) for name, m in adapter.model.named_modules()
                                             if adapter.is_blasst_attention_module(name, m)),
                        projected_tokens=router.cache.projected_tokens,
                        reused_tokens=router.cache.reused_tokens)

        yield dict(binding=binding, router=router, state=state, counters=counters)
    finally:
        if router is not None:
            router.cache.close()
        if binding is not None:
            binding.close()
        if getattr(adapter.model, '_value_direction_lease', False):
            del adapter.model._value_direction_lease
