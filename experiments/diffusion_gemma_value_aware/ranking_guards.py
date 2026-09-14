"""Signal-only nonempty repair for Sol and diagnostic rankings.

The initial shared screen's generic rescue consulted exact dense attention
mass. This occurred on30.3% of AIME rows at Sol Gaussian90%, so it is not a
negligible implementation detail. Preserve that screen as historical evidence,
but make final proxy/ranking selection use ONLY its declared signal.
The main streaming BLASST/value operators and their frozen caches are unchanged.
"""
import json
from pathlib import Path
from unittest.mock import patch
import torch
import torch.nn.functional as F
from . import routing as base
from . import refinements
from .operators import Config,proxy_keep,sol_proxy,select
from .protocol import sha

GUARD='signal_nonempty'


def source_sha():return sha(Path(__file__).read_bytes())


def is_guarded(config):
    return config.get('method') in ('sol','diagnostic') and config.get('aggregation')==GUARD


def signal_rescue(keep,active,signal):
    """Retain the highest declared-signal tile supporting each empty row.

    Ties use lowest KV index, as in the primary stable ranking. If structural
    support differs, several physical tiles may be restored; all are counted.
    No attention probability, V, or current dense output is consulted here.
    """
    empty=active.any(-1)&~(active&keep[...,None,:]).any(-1)
    ranked=signal[...,None,:].expand_as(active).masked_fill(~active,-torch.inf)
    restore=F.one_hot(ranked.argmax(-1),keep.shape[-1]).bool()&empty[...,None]
    return keep|restore.any(-2),empty


def diagnostic_signal(state,name):
    if name=='qk':return state['b'].amax(-2)
    if name=='mass':return state['mass'].sum(-2)
    if name=='contribution':return state['contrib'].square().sum((-1,-3)).sqrt()
    raise ValueError(name)


class RankingAttention(base.Attention):
    def __call__(self,module,q,k,v,mask,**kwargs):
        if self.screen:raise ValueError('use JointScreen for shared dense-state observation')
        if not is_guarded(self.config.__dict__):raise ValueError('explicit signal-only guard configuration required')
        self.current_proxy=None
        original=base.proxy_keep
        def capture(proxy,eligible,config):
            self.current_proxy=proxy
            return original(proxy,eligible,config)
        def repair(keep,state):
            signal=self.current_proxy if self.config.method=='sol' else diagnostic_signal(state,self.config.pooling)
            if signal is None:raise RuntimeError('proxy not computed before nonempty repair')
            return signal_rescue(keep,state['active'],signal)
        with patch.object(base,'proxy_keep',capture),patch.object(base,'rescue',repair):
            result=super().__call__(module,q,k,v,mask,**kwargs)
        if self.config.method=='diagnostic' and self.config.pooling in ('mass','contribution'):
            step=int(module._blasst_2d_runtime.current_denoising_iteration);layer=int(module.layer_idx)
            for head in range(q.shape[1]):self.buckets[layer,head,step,'execution']['softmax_skipped']=0.
        return result


class JointScreen(refinements.RefinedAttention):
    def _screen(self,state,meta,q,k,valid,scale,layer,step,kind,prefix,length):
        super()._screen(state,meta,q,k,valid,scale,layer,step,kind,prefix,length)
        for value in (False,True):
            proxy=sol_proxy(q,k,valid,meta,scale,value)
            for mode in ('gaussian','topk'):
                for target in (.25,.5,.75,.9):
                    c=Config(method='sol',value_proxy=value,mode=mode,amount=target,aggregation=GUARD)
                    keep,_,_=proxy_keep(proxy,state['eligible'],c)
                    keep,empty=signal_rescue(keep,state['active'],proxy)
                    self.record(state,state['eligible']&~keep,c,meta,layer,step,kind,
                        f'guarded_sol/{"value" if value else "plain"}/{mode}/s{int(target*100)}',prefix,empty)
        for name in ('qk','mass','contribution'):
            signal=diagnostic_signal(state,name)
            for target in (.25,.5,.75,.9):
                c=Config(method='diagnostic',pooling=name,mode='topk',amount=target,aggregation=GUARD)
                keep,empty=signal_rescue(select(signal,state['eligible'],'topk',target),state['active'],signal)
                self.record(state,state['eligible']&~keep,c,meta,layer,step,kind,
                    f'guarded_diagnostic/{name}/s{int(target*100)}',prefix,empty)
                if name in ('mass','contribution'):
                    for head in range(q.shape[1]):self.buckets[layer,head,step,f'guarded_diagnostic/{name}/s{int(target*100)}']['softmax_skipped']=0.


def _check_cache(path):
    if path.exists() and json.loads(path.read_text()).get('ranking_guard_sha256')!=source_sha():
        raise RuntimeError('ranking-guard source mismatch; do not reuse this phase cache')


def cache_guarded(adapter,row,path,fp,config=None,thresholds=None,screen=False,root=None):
    from . import run
    if not is_guarded(config or {}) or screen:raise ValueError('guarded execution config required')
    _check_cache(path);original=run.generate
    def generate(*args,**kwargs):
        with patch.object(run,'Attention',RankingAttention):result,arrays=original(*args,**kwargs)
        result['ranking_guard_sha256']=source_sha()
        result['nonempty_rule']='highest declared proxy/ranking signal among tiles supporting each empty row; no exact-mass rescue'
        return result,arrays
    kwargs=dict(config=config,thresholds=thresholds,screen=False)
    if root is not None:kwargs['root']=root
    with patch.object(run,'generate',generate):return run.cache(adapter,row,path,fp,**kwargs)


def cache_joint_screen(adapter,row,path,fp,root):
    from . import run
    _check_cache(path);original=run.generate
    def generate(*args,**kwargs):
        result,arrays=original(*args,**kwargs)
        result['ranking_guard_sha256']=source_sha()
        result['screen_nonempty_rule']='guarded_sol and guarded_diagnostic use their declared signals, not exact-mass fallback'
        return result,arrays
    # The existing refinement cache installs its factory around this generator;
    # both scoped injections restore even on exceptions. One replay, not two.
    with patch.object(run,'generate',generate),patch.object(refinements,'RefinedAttention',JointScreen):
        return refinements.cache_refined(adapter,row,path,fp,screen=True,root=root)


def smoke(adapter,setup,root,fp):
    from .run import shard_path
    from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
    path=root/'ranking_guard_smoke.json';code=source_sha()
    if path.exists():
        d=json.loads(path.read_text());assert d['passed'] and d['fingerprint']==fp and d['ranking_guard_sha256']==code;return
    tests=[]
    cases=[('sol_gaussian90',dict(method='sol',mode='gaussian',amount=.9)),
        ('sol_value_gaussian90',dict(method='sol',mode='gaussian',amount=.9,value_proxy=True)),
        ('sol_unpruned',dict(method='sol',mode='topk',amount=0.)),
        ('qk_unpruned',dict(method='diagnostic',pooling='qk',mode='topk',amount=0.)),
        ('qk_topk90',dict(method='diagnostic',pooling='qk',mode='topk',amount=.9)),
        ('mass_topp95',dict(method='diagnostic',pooling='mass',mode='topp',amount=.95)),
        ('contribution_topk90',dict(method='diagnostic',pooling='contribution',mode='topk',amount=.9))]
    for benchmark in ('aime26','longbench'):
        row=dict(next(r for r in setup['calibration'] if r['benchmark']==benchmark),generation_budget=32)
        # This exact-parity unpruned refinement was just verified against dense
        # by the stage's preceding smoke. Reuse its tokens; no extra dense run.
        dense_path=shard_path(root,'refinement_smoke','mass_exact_unpruned',row['id'])
        dense=json.loads(dense_path.read_text())
        assert dense['fingerprint']==fp and dense['refinement_sha256']==refinements.source_sha()
        assert all(dense[k]==row[k] for k in ('id','prompt_hash','seed','generation_budget'))
        for name,config in cases:
            config=dict(config,aggregation=GUARD)
            out=cache_guarded(adapter,row,shard_path(root,'ranking_guard_smoke',name,row['id']),fp,config,root=root)
            parity=name.endswith('unpruned')
            if parity:assert out['completion_tokens']==dense['completion_tokens'],'ranking guard dense parity'
            tests.append(dict(benchmark=benchmark,case=name,passed=True,dense_parity=True if parity else None,
                rescued_rows=sum(r['rescued_rows'] for r in out['records']),finite_calls=out['finite_calls']))
    _write(path,dict(passed=True,fingerprint=fp,ranking_guard_sha256=code,tests=tests))
