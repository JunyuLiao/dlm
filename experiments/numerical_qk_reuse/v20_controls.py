"""v20 same-consumer dense, fresh T and explicitly ported held-bitmap controls.

G75L30_nativeQ128 preserves the old mass-ranking intent, NOT its vLLM grouping:
each consumer's physical head/Q128/KV64 tile is decided jointly. Step0 is dense,
step1 observes actual current scores, then the bitmap is held within the canvas.
Full current-canvas and partial-prefix boundary tiles are mandatory. There is no
Junyu value/T rule in this reference and no claim of historical bit equivalence.
"""
from contextlib import contextmanager

SCOPES = ('ALL_NATIVE_LEGAL', 'GLOBAL_ONLY_NATIVE_LOCAL')


def mass_bitmap(scores, prefix, fraction):
    """Current BF16-transformed scores -> deterministic physical support.

    Full native legal mask only. Rank each physical tile by its maximum row
    probability mass; preserve every row's first argmax tile. Mandatory blocks
    are removed from the deletion ranking, and the integer target is an upper
    bound when mandatory blocks leave fewer candidates.
    """
    import torch
    b, h, nq, nk = scores.shape
    if not 0 <= fraction < 1 or not 0 <= prefix <= nk - nq:
        raise ValueError('invalid held-map geometry or deletion fraction')
    torch._assert_async(torch.isfinite(scores).all(), 'held-map scores must be finite')
    qb, kt = (nq + 127) // 128, (nk + 63) // 64
    probability = scores.softmax(-1)
    mass = torch.nn.functional.pad(probability, (0, kt * 64 - nk)).reshape(b, h, nq, kt, 64).sum(-1)
    mass = torch.nn.functional.pad(mass, (0, 0, 0, qb * 128 - nq))
    tile_mass = mass.reshape(b, h, qb, 128, kt).amax(-2)
    winners = scores.argmax(-1) // 64
    mandatory = torch.zeros((b, h, nq, kt), device=scores.device, dtype=torch.bool)
    mandatory.scatter_(-1, winners[..., None], True)
    mandatory = torch.nn.functional.pad(mandatory, (0, 0, 0, qb * 128 - nq))
    mandatory = mandatory.reshape(b, h, qb, 128, kt).any(-2)
    # Only wholly immutable prefix tiles can be deleted. Current canvas and
    # unaligned boundary are kept even when ranked low.
    prunable = torch.arange(kt, device=scores.device) < prefix // 64
    candidates = prunable & ~mandatory
    target = int((prefix // 64) * fraction)
    ranked = torch.argsort(tile_mass.masked_fill(~candidates, float('inf')), dim=-1, stable=True)
    rank = torch.empty_like(ranked)
    rank.scatter_(-1, ranked, torch.arange(kt, device=scores.device).expand_as(ranked))
    skipped = (candidates & (rank < target)).contiguous()
    eligible = torch.ones_like(skipped)
    return skipped, eligible


class Consumer:
    def __init__(self, adapter, config, held=False):
        from .integration import Attention
        # Reuse the qualified lifecycle hooks/identities, without invoking its
        # historical selector. This also keeps prefix ownership checks identical.
        self.owner = Attention(adapter, config['policy'], support='native_mask',
                               output_mode='historical_route_preqk_current_output',
                               kernel_variant='generic', telemetry='minimal', guard_mode='fused',
                               consumer=config['consumer'], support_build=config.get('support_build'))
        self.held, self.maps, self.map_canvas = held, {}, -1
        self.query_sensitivity = self.policy_selector = None
        self.calls = self.bootstrap_calls = self.observation_calls = self.held_calls = 0
        self.phase_counts = {'A': 0, 'D': 0, 'H': 0}

    def begin_step(self, canvas, step):
        self.owner.begin_step(canvas, step)
        if canvas != self.map_canvas:
            self.maps.clear()
            self.map_canvas = canvas

    def invalidate(self, *args):
        self.owner.invalidate(*args)
        self.maps.clear()

    def __call__(self, module, q, k, v, mask, *, dropout=0., scaling=None,
                 is_causal=None, sliding_window=None, **kwargs):
        import torch
        from .cached_executor import attention, fused_guard
        if mask is not None or is_causal is not False or dropout or module.training:
            raise ValueError('v20 control requires observed bidirectional native mask=None inference')
        b, h, nq, d = q.shape
        layer, nk = int(module.layer_idx), k.shape[-2]
        source = self.owner.sources[layer]
        prefix = source[3]
        if nk != prefix + nq or v.shape != k.shape or b != 1:
            raise ValueError('v20 control native geometry mismatch')
        identity = (self.owner.canvas, self.owner.epoch, id(source[0]), nk, nq, h, k.shape[1], d)
        scale = float(scaling) if scaling is not None else d ** -.5
        old = self.maps.get(layer)
        step = self.owner.step
        observe = self.held and step >= 1 and (old is None or old[0] != identity)
        if observe:
            scores = self.owner.observe_scores(q, k, None, scale, False, None, 0)
            fraction = .30 if module.is_sliding else .75
            skipped, eligible = mass_bitmap(scores, prefix, fraction)
            self.maps[layer] = (identity, skipped, eligible)
            result = attention(scores, v.contiguous(), skipped=skipped, eligible=eligible,
                               trace=False, variant='generic')
            self.observation_calls += 1
            self.phase_counts['A'] += 1
        else:
            if self.held and step >= 1 and old is not None:
                _, skipped, eligible = old
                self.held_calls += 1
                self.phase_counts['H'] += 1
            else:
                shape = (b, h, (nq + 127) // 128, (nk + 63) // 64)
                eligible = torch.ones(shape, dtype=torch.bool, device=q.device)
                skipped = torch.zeros_like(eligible)
                self.bootstrap_calls += 1
                self.phase_counts['A'] += 1
            self.owner._mask_present = False
            result = self.owner._consume(q, k, v, skipped, eligible, scale, None, False)
        returned = result.output.transpose(1, 2).contiguous()
        fused_guard(returned, result.invalid_scores)
        self.calls += 1
        return returned, None

    def counters(self):
        return dict(attention_calls=self.calls, bootstrap_calls=self.bootstrap_calls,
                    bitmap_observation_calls=self.observation_calls, held_decision_calls=self.held_calls,
                    phase_counts=self.phase_counts, support='native_mask',
                    consumer=self.owner.consumer, current_output=True,
                    selector='mass_max_Q128_per_query_head' if self.held else 'all_legal_kept',
                    actual_qk_pv_counts=None, counters_note='counter twin required for pair-weighted physical work')

    def close(self):
        self.maps.clear()
        self.owner.close()


@contextmanager
def install(adapter, config, condition):
    from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _install_dense
    from experiments.value_direction_hopper.integration import Attention as Fresh
    from experiments.value_direction_hopper.query_adaptive import State
    from .global_scope import GlobalScopeAdapter
    from .integration import NativeReuseState
    if condition not in ('v20_dense_consumer', 'v20_fresh_T', 'v20_G75L30_nativeQ128'):
        raise ValueError(condition)
    scope = config.get('v20_scope')
    if scope not in SCOPES or config.get('diagnostic') is not False:
        raise ValueError('explicit scope and clean production configuration required')
    if getattr(adapter.model, '_value_direction_lease', False):
        raise RuntimeError('concurrent request binding')
    scoped = GlobalScopeAdapter(adapter) if scope == SCOPES[1] else adapter
    binding = router = None
    adapter.model._value_direction_lease = True
    try:
        binding = _install_dense(scoped)
        if condition == 'v20_fresh_T':
            router = Fresh(scoped, config['library'], config['policy'], torch_library=config['torch_library'],
                           mode='value', collect=False, support_geometry='native_legal')
            state = State('T', router, m_ref=config['m_ref'], beta=config['beta'], gamma=config['gamma'],
                          diagnostics=False, fast_t=True)
            counters = lambda: dict(scope=scope, method='fresh_current_QK_current_V_T', fast_t=True,
                                    attention_calls=router.calls, support_geometry='native_legal')
        else:
            router = Consumer(scoped, config, held=condition == 'v20_G75L30_nativeQ128')
            state = NativeReuseState('kernel_dense', router, m_ref=config['m_ref'], diagnostics=False)
            counters = lambda: dict(router.counters(), scope=scope)
        binding.runtime.attention_override = router
        yield dict(binding=binding, router=router, state=state, counters=counters)
    finally:
        if router is not None:
            if hasattr(router, 'close'): router.close()
            else: router.cache.close()
        if binding is not None: binding.close()
        if getattr(adapter.model, '_value_direction_lease', False): del adapter.model._value_direction_lease
