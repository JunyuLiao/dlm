from dataclasses import replace
import math
import pytest
import torch

from experiments.diffusion_gemma_jl_output_aware.config import Config, PROJECTED, FAMILIES, DIMENSIONS
from experiments.diffusion_gemma_jl_output_aware.projections import Projections, SketchCache, digest
from experiments.diffusion_gemma_jl_output_aware import reference


def state(values, scores=None, valid=None):
    n = values.shape[-2]
    scores = torch.zeros((1, 1, 2, n)) if scores is None else scores
    valid = torch.ones_like(scores, dtype=torch.bool) if valid is None else valid
    return reference.block_statistics(scores, valid, values)


def test_all_six_predeclared():
    assert all(f'jl_{f}_r{r}' in PROJECTED for f in FAMILIES for r in DIMENSIONS)


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('rank', DIMENSIONS)
def test_projection_deterministic_independent_heads(family, rank):
    p = Projections()
    a = p.get(5, 2, 64, family, rank, 1729, 'cpu')
    b = Projections().get(5, 2, 64, family, rank, 1729, 'cpu')
    assert torch.equal(a, b) and not torch.equal(a[0], a[1])
    assert not torch.equal(a, p.get(6, 2, 64, family, rank, 1729, 'cpu'))
    assert not torch.equal(a, p.get(5, 2, 64, family, rank, 2718, 'cpu'))
    if family == 'sign':
        torch.testing.assert_close(a.abs(), torch.full_like(a, 1/math.sqrt(rank)))
    assert all(d['sha256'] == digest(b[i]) for i, d in enumerate(next(iter(p.manifest.values()))))


def test_projection_linearity_and_gqa():
    torch.manual_seed(1)
    v = torch.randn(1, 2, 130, 32)
    r = Projections().get(3, 2, 32, 'sign', 16, 1729, 'cpu')
    s = torch.randn(1, 4, 3, 130)
    valid = torch.rand_like(s) > .2
    full = reference.block_statistics(s, valid, v.repeat_interleave(2, 1))
    small = reference.block_statistics(s, valid, (v@r).repeat_interleave(2, 1))
    expected = torch.einsum('bhqtd,hdr->bhqtr', full['mu'], r.repeat_interleave(2, 0))
    torch.testing.assert_close(small['mu'], expected, atol=1e-6, rtol=2e-5)


def test_identity_matches_full_and_normalized_online_update():
    torch.manual_seed(3)
    values = torch.randn(1, 1, 150, 8)
    scores = torch.randn(1, 1, 2, 150)
    s = state(values, scores)
    mask, trace = reference.route(s, torch.ones(1, 1), Config(family='identity'), True)
    assert not mask.any()
    final = trace[-1]['previous']+trace[-1]['delta']
    torch.testing.assert_close(final, scores.softmax(-1)@values, atol=2e-6, rtol=2e-5)


def test_redundant_high_mass_can_skip_and_ties_retained():
    v = torch.ones(1, 1, 128, 8)
    s = state(v)
    mask = reference.route(s, torch.ones(1, 1), Config(log_threshold=math.log(.01)))
    assert mask.tolist() == [[[False, True]]]
    tied = reference.route(s, torch.ones(1, 1), Config(log_threshold=-math.inf))
    assert not tied.any()


def test_internal_cancellation_is_not_harmless_deletion():
    v = torch.ones(1, 1, 128, 8)
    v[..., 64:96, :] = 2
    v[..., 96:, :] = -2
    s = state(v)
    centered = reference.route(s, torch.ones(1, 1), Config(log_threshold=math.log(.1)))
    contribution = reference.route(s, torch.ones(1, 1), Config(method='contribution', log_threshold=math.log(.1)))
    assert not centered[..., 1].item() and contribution[..., 1].item()
    assert s['mu'][..., 1, :].abs().sum() == 0


def test_structural_padding_first_support_and_ineligible():
    v = torch.zeros(1, 1, 129, 8)
    valid = torch.zeros(1, 1, 2, 129, dtype=torch.bool)
    valid[..., 0, 64:128] = True
    valid[..., 1, 128] = True
    s = state(v, valid=valid)
    mask = reference.route(s, torch.ones(1, 1), Config(log_threshold=20.))
    assert not mask.any()  # Each eligible tile supplies a previously empty row.
    assert s['eligible'].tolist() == [[[False, True, True]]]


def test_skips_do_not_update_state():
    v = torch.ones(1, 1, 192, 8)
    v[..., 128:, :] = 2
    s = state(v)
    mask, traces = reference.route(s, torch.ones(1, 1), Config(log_threshold=math.log(.1)), True)
    assert mask[..., 1].item()
    torch.testing.assert_close(traces[2]['previous'], torch.ones(1, 1, 2, 8))
    torch.testing.assert_close(traces[2]['alpha'], torch.full((1, 1, 2), .5))


def test_cache_invalidation_boundary_and_native_heads():
    v = torch.randn(1, 2, 150, 32)
    valid = torch.ones(v.shape[:-1], dtype=torch.bool)
    cache = SketchCache(Config(rank=8))
    first = cache.get(1, v, valid, 90)
    saved = first['z'].clone()
    second = cache.get(1, v, valid, 90)
    assert cache.work['reused_tokens'] == 128
    torch.testing.assert_close(second['z'], saved)
    v[..., 70, :] += 1  # Mixed boundary must refresh even though within prefix.
    third = cache.get(1, v, valid, 90)
    assert not torch.equal(third['z'][..., 70, :], saved[..., 70, :])
    v[..., 5, :] += 1
    fourth = cache.get(1, v, valid, 90)
    assert not torch.equal(fourth['z'][..., 5, :], saved[..., 5, :])
    valid[..., 5] = False
    fifth = cache.get(1, v, valid, 90)
    expected = (v.square().sum(-1).masked_fill(~valid, 0.).sum(-1)/valid.sum(-1)).sqrt()
    torch.testing.assert_close(fifth['ref'], expected)


def test_diagnostic_cancellation_and_adaptive_support():
    from experiments.diffusion_gemma_jl_output_aware.diagnostics import paired_risk_diagnostics
    v = torch.ones(1, 1, 192, 8)
    v[..., 64:96, :] = 2
    v[..., 96:128, :] = -2
    v[..., 128:, :] = 1
    scores = torch.zeros(1, 1, 2, 192)
    valid = torch.ones_like(scores, dtype=torch.bool)
    c = Config(method='contribution', log_threshold=math.log(.1))
    d = paired_risk_diagnostics(scores, valid, v, v, torch.ones(1, 1), c)
    assert d['dangerous_underestimates'] >= 2
    assert d['types']['internally_cancelling']['skipped_tiles'] == 1
    assert d['types']['internally_cancelling']['full_update_sq_sum'] > 0
    assert d['eligible_tiles'] == 3 and d['skipped_tiles'] == 1


def test_compressed_raw_shards_preserve_counts(tmp_path):
    from experiments.diffusion_gemma_jl_output_aware.runner import write_output, load_output
    path = tmp_path/'sample.json'
    original = dict(prediction='test', records=[dict(layer=0, head=1, eligible=123456789, skipped=98765432,
        mass_sum=.1234567890123456)])
    write_output(path, original)
    loaded = load_output(path)
    assert loaded['records'] == original['records']
    assert loaded['prediction'] == original['prediction']
    source = path.with_suffix('.records.json.gz')
    source.write_bytes(source.read_bytes()+b'corruption')
    with pytest.raises(ValueError, match='changed'):
        load_output(path)


@pytest.mark.parametrize('rank', (8, 16, 32, 128, 256, 512))
@pytest.mark.parametrize('method', ('centered', 'contribution', 'mass_exact', 'cancellation_guard'))
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA kernel validation')
def test_gpu_statistics_and_mask_reference(rank, method):
    from experiments.diffusion_gemma_jl_output_aware import kernels
    if method == 'cancellation_guard' and rank > 32:
        pytest.skip('Guard is only a low-dimensional control')
    torch.manual_seed(31)
    family = 'identity' if rank > 32 else 'gaussian'
    c = Config(method=method, rank=rank, family=family)
    width = 2*rank if method == 'cancellation_guard' else rank
    s = torch.randn(1, 4, 131, 193, device='cuda').bfloat16()
    valid = torch.rand(s.shape, device='cuda') > .2
    valid[..., 0, :] = False
    z = torch.randn(1, 2, 193, width, device='cuda')
    norm = z[..., :rank].norm(dim=-1) if method == 'cancellation_guard' else z.norm(dim=-1)
    actual = kernels.block_statistics(s, valid, z, norm)
    expected = reference.block_statistics(s, valid, z.repeat_interleave(2, 1), norm.repeat_interleave(2, 1))
    for key in ('b', 'logz', 'mu', 'mean_norm'):
        torch.testing.assert_close(actual[key], expected[key], atol=4e-6, rtol=2e-5)
    assert torch.equal(actual['count'], expected['count'])
    ref = torch.ones(1, 2, device='cuda')*math.sqrt(width)
    thresholds = [-math.inf, -3., -1., 0.]
    masks, margin, risks = kernels.route(actual, ref, c, thresholds, True)
    for pi, threshold in enumerate(thresholds):
        for qb, start in enumerate(range(0, s.shape[-2], 128)):
            st = {k: v[..., start:start+128, :] for k, v in expected.items() if k not in ('mu', 'eligible')}
            st['mu'] = expected['mu'][..., start:start+128, :, :]
            st['eligible'] = st['active'].any(-2)
            mask = reference.route(st, ref.repeat_interleave(2, 1), replace(c, log_threshold=threshold))
            assert torch.equal(mask[0], masks[pi, :, qb, :])


@pytest.mark.parametrize('method,rank', [('centered', 8), ('centered', 512),
    ('contribution', 16), ('mass_exact', 1), ('cancellation_guard', 16)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA diagnostic replay validation')
def test_gpu_forced_support_trace(method, rank):
    from experiments.diffusion_gemma_jl_output_aware.trace_kernels import trace
    torch.manual_seed(84)
    c = Config(method=method, rank=rank, family='identity' if rank in (1, 512) else 'gaussian', log_threshold=-2.)
    width = rank*2 if method == 'cancellation_guard' else rank
    s = torch.randn(1, 2, 17, 193, device='cuda')
    valid = torch.rand_like(s) > .15
    valid[..., 0, :] = False
    valid[..., 1, :64] = False
    z = torch.randn(1, 2, 193, width, device='cuda')
    st = reference.block_statistics(s, valid, z, z[..., :rank].norm(dim=-1))
    ref = torch.ones(1, 2, device='cuda')*math.sqrt(rank)
    # A common externally chosen support tests skipped-state consistency.
    skip = torch.zeros(1, 2, 4, dtype=torch.bool, device='cuda')
    skip[..., 2] = True
    _, expected = reference.route(st, ref, c, True, forced_skip=skip)
    actual = trace(st, ref, skip, c)
    for j, row in enumerate(expected):
        supported = row['support'] & row['active']
        assert torch.equal(actual['supported'][..., j].bool(), supported)
        assert torch.equal(actual['active'][..., j].bool(), row['active'])
        torch.testing.assert_close(actual['alpha'][..., j], row['alpha'], atol=2e-6, rtol=3e-5)
        distance = (row['mu']-row['previous'])[..., :rank].norm(dim=-1)/ref[..., None]
        centered = row['delta'][..., :rank].norm(dim=-1)/ref[..., None]
        kappa = row['mu'][..., :rank].norm(dim=-1)/(st['mean_norm'][..., j]+1e-12)
        for key, value in (('distance',distance), ('centered',centered), ('kappa',kappa)):
            torch.testing.assert_close(actual[key][..., j], value, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(actual['risk'][..., j][supported], row['log_risk'].exp()[supported], atol=3e-6, rtol=3e-5)


def test_preserved_ids_nemo_budget_and_calibration_isolation():
    import json
    from experiments.diffusion_gemma_jl_output_aware.protocol import ROOT, PARENT, audit
    from experiments.diffusion_gemma_jl_output_aware.calibration import rows_for
    from copy import deepcopy
    setup = json.loads((ROOT/'setup.json').read_text())
    prior = json.loads((PARENT/'setup.json').read_text())
    assert audit(setup)['passed']
    assert {r['id'] for r in setup['final']} == {r['id'] for r in prior['final']}
    assert len(setup['conditions']) == 37
    assert all(r['generation_budget'] == 4096 for r in setup['final'] if r['benchmark'] == 'longbench_v2')
    frozen = deepcopy(setup)
    for benchmark in ('aime26', 'longbench_v2'):
        rows = rows_for(setup, benchmark)
        assert len(rows) == 6 and all(r['generation_budget'] == 512 for r in rows)
        assert {r['id'] for r in rows}.isdisjoint(r['id'] for r in setup['final'] if not r['calibration'])
    assert setup == frozen
    changed = deepcopy(setup)
    next(r for r in changed['final'] if r['benchmark'] == 'longbench_v2')['generation_budget'] = 128
    with pytest.raises(ValueError, match='4096'):
        audit(changed)


def test_supporting_audit_cannot_pass_without_checks(tmp_path):
    from experiments.diffusion_gemma_jl_output_aware.report import supporting_audit
    missing, violations, sources = supporting_audit(tmp_path, {}, {'fingerprint':'none'}, {})
    assert len(missing) == 4 and not violations and not sources


def test_shared_selection_covers_every_layer_without_selecting_by_error():
    from experiments.diffusion_gemma_jl_output_aware.shared_analysis import selected_sources
    sources = [dict(id=f'{b}/{i}', layer=l, step=s) for b in ('aime26','longbench_v2')
        for i in (2,1) for l in range(30) for s in (0,4,12,24)]
    selected = selected_sources(sources)
    assert len(selected) == 72
    assert all(row['id'].endswith('/1') for row in selected.values())
    assert all(row['step'] == 0 or row['layer'] in (0,5) for row in selected.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA shared-state pipeline validation')
def test_shared_screen_and_all_candidate_diagnostics(tmp_path):
    import json
    from experiments.diffusion_gemma_jl_output_aware.screen import shared_screen, distributions
    from experiments.diffusion_gemma_jl_output_aware.shared_analysis import analyze
    from experiments.diffusion_gemma_jl_output_aware.config import BASELINES, TARGETS
    from experiments.diffusion_gemma_jl_output_aware.protocol import sha
    torch.manual_seed(93)
    sources=[]
    for layer,kind in ((0,'local'),(5,'global')):
        path=tmp_path/f'state{layer}.pt'
        s=torch.randn(1,1,17,193,device='cuda').bfloat16()
        valid=torch.rand_like(s.float())>.1
        valid[...,0,:]=False
        values=torch.randn(1,1,193,32,device='cuda').bfloat16()
        ident=dict(id='aime26/test',split='calibration',layer=layer,step=0,head=0,
            native_kv_head=1,native_head_count=2,attention_type=kind,prefix=90)
        torch.save(dict(identity=ident,scores=s,valid=valid,values=values,kv_valid=valid.any(-2)),path)
        sources.append(dict(path=str(path),sha256=sha(path.read_bytes()),**ident))
    contract={'fingerprint':'gpu-test'}
    outputs=shared_screen(tmp_path,sources,contract)
    assert len(outputs)==2
    for name in {**BASELINES,**PROJECTED}:
        arrays,source=distributions(tmp_path,name,'aime26')
        assert all(len(a)==4 for a in arrays.values()) and len(source)==2
        for target in TARGETS:
            path=tmp_path/'policies/aime26'/f'{name}_s{int(target*100)}.json'
            path.parent.mkdir(exist_ok=True,parents=True)
            path.write_text(json.dumps(dict(policy={k:dict(log_threshold=-2.) for k in ('local','global')})))
    diagnostics=analyze(tmp_path,sources,contract)
    assert len(diagnostics)==72
    for entry in diagnostics:
        result=json.loads(open(entry['path']).read())['metrics']
        assert result['eligible_tiles']==4
        assert 0<=result['skipped_tiles']<=3
        assert math.isfinite(result['error_sq'])
        assert result['valid_query_rows']==16
        assert sum(r['eligible_tiles'] for r in result['regions'].values())==4
    assert diagnostics==analyze(tmp_path,sources,contract)
