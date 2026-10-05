"""v10 CP1 qualification: length-generic kernels vs the static reference.

Executable, not textual, evidence (plus one textual guard that the bodies did
not drift): every production entry point (anchor fused route+PV, ordinary
route_only + preqk consumer, summary STORE then LOAD, held bitmap path with
_held_eligible and _pv) must give BIT-IDENTICAL bitmaps, eligibility, malformed
flags, BF16 outputs, LSE, invalid flags, summary buffers and physical
counters, across model-shaped LOCAL/GLOBAL geometries and length boundaries,
and the generic variant must stop compiling new variants as lengths grow.
"""
import math
import re
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
ROOT = Path(__file__).resolve().parents[1] / 'experiments/numerical_qk_reuse'
LOCAL = dict(h=16, hk=8, d=256, window=1024, threshold=-1.0099318265914916)
GLOBAL = dict(h=16, hk=2, d=512, window=None, threshold=-3.1366905212402343)


def test_generic_bodies_are_textually_the_static_bodies():
    static = (ROOT / 'cached_executor.py').read_text()
    generic = (ROOT / 'generic_kernels.py').read_text()
    for name in ('_route', '_held_eligible', '_pv', '_preqk_pv'):
        a = static[static.index(f'def {name}('):]
        a = a[:a.index('\n\n\n')] if '\n\n\n' in a else a
        b = generic[generic.index(f'def {name}_generic('):]
        b = b[:b.index('\n\n\n')] if '\n\n\n' in b else b
        # the ONLY permitted additions: the KDIV alignment hint parameter and line
        b = b.replace(', KDIV: tl.constexpr', '')
        b = re.sub(r'\n *K = K \* KDIV[^\n]*', '', b)
        norm = lambda text: re.sub(r'\s+', ' ', re.sub(r'(?<![A-Za-z_])(K|KT|PREFIX_TILES): tl\.constexpr', r'\1',
                                                       text.replace(f'{name}_generic(', f'{name}('))).strip()
        assert norm(a) == norm(b), name


def inputs(seed, geometry, prefix, canvas=256, poison=None):
    from experiments.numerical_qk_reuse.integration import Attention
    g = torch.Generator(device='cuda').manual_seed(seed)
    h, hk, d, window = geometry['h'], geometry['hk'], geometry['d'], geometry['window']
    crop = max(0, prefix - window + 1) // 64 * 64 if window else 0
    nk = prefix + canvas - crop
    q = (torch.randn(1, canvas, h, d, generator=g, device='cuda', dtype=torch.bfloat16) * .3).transpose(1, 2)
    kv = torch.randn(1, nk, hk, 2 * d, generator=g, device='cuda', dtype=torch.bfloat16) * .3
    k, v = kv[..., :d].transpose(1, 2), kv[..., d:].transpose(1, 2).contiguous()
    scores = Attention.observe_scores(q, k, None, d ** -.5, False, window, crop)
    if poison == 'nan':
        scores[0, 3, 5, 70] = float('nan')
    if poison == 'masked_rows':
        scores[0, :, 17:40, :] = -math.inf
    z = torch.randn(1, hk, nk, 32, generator=g, device='cuda', dtype=torch.float32).contiguous()
    ref = (torch.rand(1, hk, generator=g, device='cuda', dtype=torch.float32) + .5).contiguous()
    sens = (1. + 3. * torch.rand(1, canvas, generator=g, device='cuda', dtype=torch.float32)).contiguous()
    return dict(q=q, k=k, v=v, scores=scores, z=z, ref=ref, sens=sens, nk=nk, prefix_tiles=(prefix - crop) // 64,
                window=window, d=d)


def pipeline(x, variant, threshold):
    """Every production entry point on one state; returns all observable tensors."""
    from experiments.numerical_qk_reuse.cached_executor import (allocate_summary, attention,
                                                                preqk_attention, route_only)
    b, h, nq, nk = x['scores'].shape
    qb, kt = (nq + 127) // 128, (nk + 63) // 64
    out = {}
    summary = allocate_summary(1, h, qb, kt, x['prefix_tiles'], 32, x['scores'].device, ('id',)) \
        if x['prefix_tiles'] > 0 else None
    anchor = attention(x['scores'], x['v'], x['z'], x['ref'], sensitivity=x['sens'], log_threshold=threshold,
                       trace=True, summary=summary, store_summary=summary is not None, variant=variant)
    for name in ('output', 'skipped', 'eligible', 'log_normalizer', 'projected_state', 'risk',
                 'invalid_scores', 'counters'):
        out[f'anchor.{name}'] = getattr(anchor, name)
    if summary is not None:
        for name in ('z', 'mu', 'active', 'bad'):
            out[f'summary.{name}'] = getattr(summary, name).clone()
    sens2 = (x['sens'] * .7 + .4).contiguous()          # live T changes between anchor and reuse
    for mode, summ in (('legacy', None), ('loaded', summary)):
        route = route_only(x['scores'], x['z'], x['ref'], sensitivity=sens2, log_threshold=threshold,
                           summary=summ, store_summary=False, variant=variant)
        out[f'{mode}.skipped'], out[f'{mode}.eligible'], out[f'{mode}.bad'] = \
            route.skipped, route.eligible, route.invalid_tiles
        for trace in (False, True):                       # production runs trace=False
            consumer = preqk_attention(x['q'], x['k'], x['v'], route.skipped, route.eligible,
                                       scale=x['d'] ** -.5, window=x['window'], trace=trace, variant=variant)
            for name in ('output', 'log_normalizer', 'invalid_scores') + (('counters',) if trace else ()):
                out[f'{mode}.preqk{int(trace)}.{name}'] = getattr(consumer, name)
    held = attention(x['scores'], x['v'], skipped=anchor.skipped.clone(), trace=True, variant=variant)
    for name in ('output', 'eligible', 'log_normalizer', 'invalid_scores', 'counters'):
        out[f'held.{name}'] = getattr(held, name)
    torch.cuda.synchronize()
    return out


def bit_equal(a, b):
    if a.dtype.is_floating_point:
        view = {torch.float32: torch.int32, torch.bfloat16: torch.int16}[a.dtype]
        return torch.equal(a.contiguous().view(view), b.contiguous().view(view))
    return torch.equal(a, b)


CASES = ([('local', LOCAL, p, None) for p in (63, 64, 65, 127, 128, 129, 365, 1023, 1645)]
         + [('global', GLOBAL, p, None) for p in (63, 64, 65, 129, 365, 1645, 3181, 7900, 8300)]
         # K = prefix + 256: odd, %2, %4, %8, %16 alignment classes
         + [('global_align', GLOBAL, p, None) for p in (1645, 1646, 1648, 1652, 1656, 5004, 5006)]
         + [('local_align', LOCAL, p, None) for p in (1102, 1104, 1108, 1112)]
         + [('local_nan', LOCAL, 700, 'nan'), ('global_masked_rows', GLOBAL, 900, 'masked_rows'),
            ('local_all_kept', dict(LOCAL, threshold=-math.inf), 400, None),
            ('global_sparse', dict(GLOBAL, threshold=5.0), 2000, None)])


@pytest.mark.parametrize('name,geometry,prefix,poison', CASES, ids=[f'{c[0]}-{c[2]}' for c in CASES])
def test_generic_is_bit_identical_to_static(name, geometry, prefix, poison):
    x = inputs(prefix, geometry, prefix, poison=poison)
    static = pipeline(x, 'static', geometry['threshold'])
    generic = pipeline(x, 'generic', geometry['threshold'])
    assert static.keys() == generic.keys()
    diffs = [key for key in static if not bit_equal(static[key], generic[key])]
    assert not diffs, diffs
    # the summary really stored and reloaded, and loading changed no decision
    if x['prefix_tiles'] > 0:
        assert torch.equal(static['legacy.skipped'], static['loaded.skipped'])
    if poison == 'nan':
        assert static['legacy.bad'].any() and static['anchor.invalid_scores'].any()
    if name == 'local_all_kept':
        assert not static['anchor.skipped'].any()
    if name == 'global_sparse':
        assert static['anchor.skipped'].any()


def compiled(kernel):
    return sum(len(v) for v in kernel.cache.values())


def test_generic_variant_count_is_bounded_while_static_grows_with_length():
    from experiments.numerical_qk_reuse import generic_kernels as g
    from experiments.numerical_qk_reuse import cached_executor as c
    kernels = dict(generic=[g._route_generic, g._pv_generic, g._preqk_pv_generic, g._held_eligible_generic],
                   static=[c._route, c._pv, c._preqk_pv, c._held_eligible])
    # unseen global lengths; 2816 gives K=3072 (K%16==0) so both alignment classes initialize
    lengths = [1901 + 256 * i for i in range(6)] + [2816, 2420, 5003, 5004, 5006, 5000]
    counts = {}
    for variant in ('generic', 'static'):
        before = [compiled(k) for k in kernels[variant]]
        history = []
        for prefix in lengths + lengths:                  # sequence, then repeated
            x = inputs(prefix, GLOBAL, prefix)
            pipeline(x, variant, GLOBAL['threshold'])
            history.append(sum(compiled(k) for k in kernels[variant]) - sum(before))
        counts[variant] = history
    generic, static = counts['generic'], counts['static']
    # generic: once the alignment classes present in the sequence are initialized, nothing new compiles
    assert generic[-1] == max(generic) and generic[len(lengths):] == [generic[-1]] * len(lengths)
    assert generic[-1] <= 5 * 7, generic          # <= 5 alignment classes x 7 distinct mode launches
    assert static[len(lengths) - 1] > generic[len(lengths) - 1], (static, generic)
    # an extra unseen length after initialization compiles nothing for generic
    before = sum(compiled(k) for k in kernels['generic'])
    for prefix in (6007, 6016, 6018, 6020, 6024):
        pipeline(inputs(prefix, GLOBAL, prefix), 'generic', GLOBAL['threshold'])
    assert sum(compiled(k) for k in kernels['generic']) == before
