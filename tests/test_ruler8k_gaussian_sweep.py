from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from experiments import diffusion_gemma_ruler8k_gaussian_sweep as study
from experiments import diffusion_gemma_ruler8k_projection_diagnostics as diagnostics
from experiments import diffusion_gemma_ruler8k_gaussian_report as reporting
from experiments.diffusion_gemma_jl_output_aware.config import Config
from experiments.diffusion_gemma_jl_output_aware.projections import Projections, SketchCache
from experiments.diffusion_gemma_jl_output_aware import reference


@pytest.mark.parametrize('rank',study.RANKS)
def test_requested_rank_projection_linearity_and_exact_cache_invalidation(rank):
    torch.manual_seed(73); v=torch.randn(1,2,135,64); valid=torch.ones(v.shape[:-1],dtype=torch.bool)
    bank=Projections(); matrix=bank.get(5,2,64,'gaussian',rank,1729,'cpu')
    assert torch.equal(matrix,Projections().get(5,2,64,'gaussian',rank,1729,'cpu'))
    assert not torch.equal(matrix,Projections().get(5,2,64,'gaussian',rank,2718,'cpu'))
    s=torch.randn(1,4,3,135); mask=torch.rand_like(s)>.15
    full=reference.block_statistics(s,mask,v.repeat_interleave(2,1))
    small=reference.block_statistics(s,mask,(v@matrix).repeat_interleave(2,1))
    expected=torch.einsum('bhqtd,hdr->bhqtr',full['mu'],matrix.repeat_interleave(2,0))
    torch.testing.assert_close(small['mu'],expected,atol=5e-6,rtol=5e-5)
    with study.scope():
        cache=SketchCache(Config(rank=rank));old=cache.get(5,v,valid,91)['z'].clone()
        assert torch.equal(cache.get(5,v,valid,91)['z'],old)
        for pos in (5,75,130):
            v[...,pos,:]+=1
            updated=cache.get(5,v,valid,91)['z'].clone()
            assert not torch.equal(updated[...,pos,:],old[...,pos,:]);old=updated


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA gate')
@pytest.mark.parametrize('rank',study.RANKS)
def test_rank_gpu_native_output_and_physical_masks_match_trusted(rank):
    from experiments.diffusion_gemma_jl_output_aware.routing import Attention
    torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(47)
    q=torch.randn(1,4,129,32,device='cuda',dtype=torch.bfloat16)
    k,v=[torch.randn(1,2,270,32,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
    valid=torch.ones(1,1,129,270,device='cuda',dtype=torch.bool);valid[...,0,:]=False;valid[...,-1]=False
    module=SimpleNamespace(num_key_value_groups=2,layer_idx=0,training=False,is_sliding=True,
        _blasst_2d_runtime=SimpleNamespace(current_denoising_iteration=0))
    cfg=dict(family='gaussian',rank=rank)
    with study.scope():
        for threshold in (-1000., -1.):
            policy={k:dict(log_threshold=threshold) for k in study.base.KINDS}
            fast=Attention(cfg,policy,validate=True);trusted=Attention(cfg,policy,trusted=True)
            got=fast(module,q,k,v,valid,scaling=.5,is_causal=False,sliding_window=256)[0]
            expected=trusted(module,q,k,v,valid,scaling=.5,is_causal=False,sliding_window=256)[0]
            assert torch.equal(got,expected) and torch.isfinite(got).all()
            for field in ('eligible','skipped','prefix_eligible','canvas_eligible','boundary_eligible'):
                assert sum(r[field] for r in fast.records)==sum(r[field] for r in trusted.records)
            if threshold==-1000.: assert not sum(r['skipped'] for r in fast.records)


def test_scope_restores_previous_dispatch_and_allows_all_six_ranks():
    old_dims=study.base.configuration.DIMENSIONS;old_score=study.base.runner.score
    with study.active():
        assert study.base.CONFIGS==study.CONFIGS and study.base.EXPECTED==2470
        for rank in study.DIAGNOSTIC_RANKS:assert Config(rank=rank).rank==rank
    assert study.base.configuration.DIMENSIONS==old_dims and study.base.runner.score is old_score
    assert len(study.base.CONFIGS)==4


def test_balanced_state_selection_is_score_blind_and_calibration_only():
    states=study.base.read(study.previous.ROOT/'shared_state_index.json')
    selected=study.selected_sources(states)
    assert len(selected)==390 and selected==study.selected_sources(list(reversed(states)))
    assert len({key[1] for key in selected})==13 and {s['layer'] for s in selected.values()}==set(range(30))
    bad=deepcopy(states);bad[0]['split']='final'
    with pytest.raises(ValueError):study.selected_sources(bad)


def test_preparation_preserves_all_examples_budgets_and_baselines(tmp_path):
    setup=study.prepare(tmp_path);old=study.base.read(study.previous.ROOT/'setup.json')
    for split in ('calibration','final','development'):assert setup[split]==old[split]
    assert len(setup['conditions'])==19 and setup['expected']==2470
    assert setup['expected_new_final']==1300 and setup['expected_reused_final']==1170
    assert all(study.CONFIGS[k]==v for k,v in study.OLD_CONFIGS.items())
    assert study.prepare(tmp_path)==setup


def test_imported_policy_cannot_change_threshold_or_examples(tmp_path,monkeypatch):
    base=study.base;setup=base.read(study.previous.ROOT/'setup.json');contract=dict(fingerprint='new')
    source=study.previous.ROOT/'policies'/base.BENCHMARK/'full_centered_s50.json';old=base.read(source)
    p=dict(old,fingerprint='new',imported_policy=dict(path=str(source),sha256=base.sha(source.read_bytes())))
    calls=[];monkeypatch.setattr(study,'BASE_AUDIT',lambda *args:calls.append(args))
    study.audit_policy(tmp_path,p,setup,contract);assert len(calls)==1 and calls[0][1]==old
    bad=deepcopy(p);bad['policy']['local']['log_threshold']+=.01
    with pytest.raises(ValueError):study.audit_policy(tmp_path,bad,setup,contract)
    changed=deepcopy(setup);changed['final'][0]['generation_budget']+=1
    with pytest.raises(ValueError):study.audit_policy(tmp_path,p,changed,contract)


def trace_fixture(scale=1.):
    values=torch.tensor([[[[.1,.4,.8],[.1,.6,1.2]]]])
    active=torch.ones_like(values);supported=active.clone();supported[...,0]=0
    return dict(centered=values*scale,risk=values*scale,active=active,supported=supported,
        alpha=torch.ones_like(values)*.5,distance=values*scale*2,kappa=torch.ones_like(values)*.5)


def test_common_support_identity_has_no_distortion_or_disagreement():
    full=trace_fixture();skip=torch.tensor([[[False,False,False]]])
    metrics=diagnostics.common_support_metrics(full,full,skip,-1.,128)
    assert metrics['norm_relative_error_sq_sum']==0 and metrics['norm_ratio_sum']==4
    assert metrics['counterfactual_disagreement_tiles']==0 and metrics['factor_two_underestimate_rows']==0
    assert metrics['dangerous_underestimates']==0


def test_common_support_detects_underestimation_without_changing_support():
    full=trace_fixture();projected=trace_fixture(.1);skip=torch.tensor([[[False,False,False]]]);old=skip.clone()
    m=diagnostics.common_support_metrics(projected,full,skip,-1.,128)
    assert torch.equal(skip,old) and m['counterfactual_unsafe_skip_tiles']==2
    assert m['factor_two_underestimate_rows']==4 and m['dangerous_underestimates']==4
    assert m['full_above_threshold_rows']==4 and m['skipped_tiles']==0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA diagnostic gate')
@pytest.mark.parametrize('rank',study.RANKS)
def test_common_support_gpu_trace_equals_projected_full_update(rank):
    from experiments.diffusion_gemma_jl_output_aware import kernels
    from experiments.diffusion_gemma_jl_output_aware.trace_kernels import trace
    torch.manual_seed(87);torch.backends.cuda.matmul.allow_tf32=False
    s=torch.randn(1,1,5,192,device='cuda');valid=torch.ones_like(s,dtype=torch.bool)
    valid[...,0,:64]=False
    v=torch.randn(1,1,192,32,device='cuda');matrix=Projections().get(0,1,32,'gaussian',rank,1729,'cuda')
    full=kernels.block_statistics(s,valid,v)
    small=kernels.block_statistics(s,valid,v@matrix)
    skip=torch.tensor([[[False,True,False]]],device='cuda');ref=torch.ones(1,1,device='cuda')
    with study.scope():out=trace(small,ref,skip,Config(rank=rank))
    logz=torch.full((1,1,5),-torch.inf,device='cuda');previous=torch.zeros(1,1,5,32,device='cuda')
    expected=[]
    for tile in range(3):
        z=full['logz'][...,tile];mu=full['mu'][...,tile,:]
        combined=torch.logaddexp(logz,z);safe=combined.masked_fill(~torch.isfinite(combined),0.)
        alpha=torch.exp(z-safe);delta=alpha[...,None]*(mu-previous)
        expected.append((delta@matrix).norm(dim=-1))
        if not skip[0,0,tile]:
            previous=torch.exp(logz-safe)[...,None]*previous+alpha[...,None]*mu;logz=combined
    torch.testing.assert_close(out['centered'],torch.stack(expected,-1),atol=3e-6,rtol=3e-5)
    assert not out['supported'][...,0].any() and not out['supported'][...,0,1]


def test_rank_report_preserves_primary_metric_and_exposes_budget_sensitivity(tmp_path):
    raw=[dict(condition='dense',id=str(i),task='vt' if i<10 else 'qa_1',accuracy=float(i>=10),dense_accuracy=float(i>=10)) for i in range(20)]
    result=reporting.sensitivity(raw)
    assert result[0]['count']==10 and result[0]['accuracy']==1. and result[0]['paired_ci95']==[0.,0.]
    assert len(raw)==20 and sum(r['accuracy'] for r in raw)/len(raw)==.5
    setup=study.base.read(study.previous.ROOT/'setup.json')
    reporting.write_report(tmp_path,setup,[],[],[],[],[],dict(completed=0,complete=False,missing=[],violations=[]),[])
    text=(tmp_path/'report.md').read_text()
    assert '0/2470' in text and 'not fresh held-out' in text and 'not greedy' in text and 'All10 dense VT' in text
