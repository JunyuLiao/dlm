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


G75_LOCAL_FRACTIONS = (0.0, 0.15, 0.30)   # v27 port of the vLLM-era F (G75L0) / S15 / S30 configurations

# v27 PORT of SparseD (Wang et al., ICLR 2026, arXiv 2509.24014; github.com/INV-WZQ/SparseD): full attention for the
# first skip% of the denoising steps, then ONE sparse pattern from the current attention scores, average-pooled per
# (query block, key block), keeping the top select% key blocks per query block and head (prefill and generation keys
# selected separately), reused for the rest of the canvas. Port details (labelled): blocks are the FA4 kernel tiles
# (Q128 x KV64; SparseD's long-context block_size is 128 on both sides), the current canvas keys are always kept
# (SparseD applies the same ratio to generation keys; 4 tiles here), the ratio applies to the prefix keys, and the
# canvas has an adaptive length, so skip is a number of steps: 10 = 20% of the 48-step cap (SparseD's default
# skip=0.2) or 1 (matched to B/G75). Kernel: FA4 block-sparse (SparseD uses FlexAttention).
SPARSED_KEEPS = (0.1, 0.2, 0.3, 0.5, 0.6, 0.7)   # 0.5-0.7: target sparsity 30-50% (AIME comparison)
SPARSED_SKIPS = (1, 10)


def sparsed_bitmap(scores, prefix, keep):
    """SparseD selection on current scores: per (head, Q128 block), keep the top ``keep`` fraction of the
    wholly-prefix KV64 tiles by average-pooled attention probability; canvas and boundary tiles are always kept."""
    import math
    import torch
    b, h, nq, nk = scores.shape
    if not 0 < keep < 1 or not 0 <= prefix <= nk - nq:
        raise ValueError('invalid SparseD geometry or keep ratio')
    torch._assert_async(torch.isfinite(scores).all(), 'SparseD scores must be finite')
    qb, kt = (nq + 127) // 128, (nk + 63) // 64
    probability = scores.softmax(-1)
    probability = torch.nn.functional.pad(probability, (0, kt * 64 - nk, 0, qb * 128 - nq))
    pooled = probability.reshape(b, h, qb, 128, kt, 64).mean((3, 5))       # avgpool over each tile
    prefix_tiles = prefix // 64
    prunable = torch.arange(kt, device=scores.device) < prefix_tiles
    n_keep = min(prefix_tiles, max(1, math.ceil(keep * prefix_tiles))) if prefix_tiles else 0
    ranked = torch.argsort(pooled.masked_fill(~prunable, float('-inf')), dim=-1, descending=True, stable=True)
    rank = torch.empty_like(ranked)
    rank.scatter_(-1, ranked, torch.arange(kt, device=scores.device).expand_as(ranked))
    skipped = (prunable & (rank >= n_keep)).contiguous()
    return skipped, torch.ones_like(skipped)



class Consumer:
    def __init__(self, adapter, config, held=False, local_fraction=.30, c64=False, fa4=False, sparsed=None):
        from .integration import Attention
        # Reuse the qualified lifecycle hooks/identities, without invoking its
        # historical selector. This also keeps prefix ownership checks identical.
        self.owner = Attention(adapter, config['policy'], support='native_mask',
                               output_mode='historical_route_preqk_current_output',
                               kernel_variant='generic', telemetry='minimal', guard_mode='fused',
                               consumer='triton' if (c64 or fa4) else config['consumer'],
                               support_build=config.get('support_build'))
        if c64:
            # v27 port: every consumer call (bootstrap, observation output, held) runs the 64-row kernel
            self.owner.consumer, self.owner.c64_splits, self.owner.output_layout = 'triton64', 2, 'model_major'
        if fa4:
            # v27 FA4 port: every consumer call runs FlashAttention-4 through its block-sparse interface
            # (bootstrap = all tiles kept), exactly the M1/M2/M3 FA4 consumer path
            if c64:
                raise ValueError('choose one G75 consumer')
            c64 = True   # same call routing as the 64-row port below; only the kernel differs
            self.owner.consumer, self.owner.c64_splits, self.owner.output_layout = 'fa4', 2, 'model_major'
        if local_fraction not in G75_LOCAL_FRACTIONS:
            raise ValueError('G75 local deletion fraction must be a named vLLM-era point')
        self.local_fraction, self.c64 = local_fraction, c64
        # sparsed = (keep, skip_steps) for the SparseD port; None for G75
        if sparsed is not None and (sparsed[0] not in SPARSED_KEEPS or sparsed[1] not in SPARSED_SKIPS):
            raise ValueError('SparseD port needs a named keep ratio and skip')
        self.sparsed = sparsed
        self.observe_step = sparsed[1] if sparsed is not None else 1
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
        observe = self.held and step >= self.observe_step and (old is None or old[0] != identity)
        if observe and self.sparsed is not None:
            # SparseD: the pattern comes from the last full-attention step, whose output stays dense; it is
            # applied from the next step on
            scores = self.owner.observe_scores(q, k, None, scale, False, None, 0)
            skipped, eligible = sparsed_bitmap(scores, prefix, self.sparsed[0])
            del scores
            self.maps[layer] = (identity, skipped, eligible)
            from types import SimpleNamespace
            from . import v27_fa4
            out = v27_fa4.dense(q, k, v, scale)
            result = SimpleNamespace(output=out.transpose(1, 2), invalid_scores=torch.zeros(
                (b, h, nq), dtype=torch.bool, device=q.device))
            self.observation_calls += 1
            self.phase_counts['A'] += 1
        elif observe:
            scores = self.owner.observe_scores(q, k, None, scale, False, None, 0)
            fraction = self.local_fraction if module.is_sliding else .75
            skipped, eligible = mass_bitmap(scores, prefix, fraction)
            self.maps[layer] = (identity, skipped, eligible)
            if self.c64:
                del scores
                self.owner._mask_present = False
                result = self.owner._consume(q, k, v, skipped, eligible, scale, None, False)
            else:
                result = attention(scores, v.contiguous(), skipped=skipped, eligible=eligible,
                                   trace=False, variant='generic')
            self.observation_calls += 1
            self.phase_counts['A'] += 1
        else:
            bootstrap = False
            if self.held and step >= self.observe_step and old is not None:
                _, skipped, eligible = old
                self.held_calls += 1
                self.phase_counts['H'] += 1
            else:
                shape = (b, h, (nq + 127) // 128, (nk + 63) // 64)
                eligible = torch.ones(shape, dtype=torch.bool, device=q.device)
                skipped = torch.zeros_like(eligible)
                self.bootstrap_calls += 1
                self.phase_counts['A'] += 1
                bootstrap = True
            self.owner._mask_present = False
            if self.owner.consumer == 'fa4' and bootstrap:
                # step-0 bootstrap: FA4 with every tile kept through its cached all-kept lists (bitwise identical
                # to a freshly built all-kept list; no per-call list build)
                from types import SimpleNamespace
                from . import v27_fa4
                out = v27_fa4.dense(q, k, v, scale)
                result = SimpleNamespace(output=out.transpose(1, 2), invalid_scores=torch.zeros(
                    (b, h, nq), dtype=torch.bool, device=q.device))
            else:
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
                    selector=('sparsed_avgpool_topk_port' if self.sparsed is not None else
                              'mass_max_Q128_per_query_head' if self.held else 'all_legal_kept'),
                    g75_local_fraction=self.local_fraction if self.held and self.sparsed is None else None,
                    sparsed_keep=None if self.sparsed is None else self.sparsed[0],
                    sparsed_skip_steps=None if self.sparsed is None else self.sparsed[1],
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
    if condition not in ('v20_dense_consumer', 'v20_fresh_T', 'v20_G75L30_nativeQ128', 'v27_G75_c64', 'v27_G75_fa4',
                         'v27_sparsed_fa4'):
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
            if condition == 'v27_G75_c64':
                if config.get('consumer') != 'triton64':
                    raise ValueError('v27 G75 port runs on the 64-row consumer')
                router = Consumer(scoped, config, held=True, local_fraction=config['g75_local_fraction'], c64=True)
            elif condition == 'v27_sparsed_fa4':
                if config.get('consumer') != 'fa4' or scope != SCOPES[1]:
                    raise ValueError('the SparseD port runs GLOBAL-only on the FA4 consumer')
                router = Consumer(scoped, config, held=True, local_fraction=0.0, fa4=True,
                                  sparsed=(config['sparsed_keep'], config['sparsed_skip_steps']))
            elif condition == 'v27_G75_fa4':
                if config.get('consumer') != 'fa4':
                    raise ValueError('v27 G75 FA4 port runs on the FA4 consumer')
                router = Consumer(scoped, config, held=True, local_fraction=config['g75_local_fraction'], fa4=True)
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
