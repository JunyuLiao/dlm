"""Resumable trajectory-aware calibration of the existing Gaussian-32 router.

Only the local/global threshold schedule and causal query sensitivity differ
from the archived query-adaptive studies. The CUDA kernel and decoder are not
modified. See TRAJECTORY_GUARDRAIL_PLAN.md for preregistered constraints.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import fcntl
import gzip
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from dllm.models import create_adapter
from experiments import diffusion_gemma_ruler4k_gaussian_sweep as pool_source
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _request, _set_context
from experiments.diffusion_gemma_ruler8k_jl import score
from .experiment import atomic, fingerprint, sha, shard_path
from .integration import install
from .query_adaptive import State, observe
from .query_adaptive_study import aggregate, empty_counts
from .query_adaptive_tile_controls import tile_broadcast
from .sparsity_steps import routing_counts


ROOT = Path(__file__).resolve().parents[2]/'results'/'query_adaptive_guardrail_v1'
PARENT = Path(__file__).resolve().parents[2]/'results'/'query_adaptive_allocation_v1'
ARCHIVE = Path(__file__).resolve().parents[2]/'results'/'query_adaptive_v3'
RANK_SWEEP = Path(__file__).resolve().parents[2]/'results'/'diffusion_gemma_ruler4k_gaussian_rank_sweep_v16'
TARGETS = (50, 70)
METHODS = ('unweighted', 'C', 'M', 'T', 'T_uniform', 'T_tile_mean', 'T_shuffle')
TASKS = tuple(pool_source.base.official.PAPER_TASKS)


def label(method, target):
    return f'{method}_s{target}'


def method_specs():
    return {
        'native_dense': dict(method='native_dense', allocation='normal', tile_mode=None),
        'kernel_dense': dict(method='kernel_dense', allocation='normal', tile_mode=None),
        'unweighted': dict(method='unweighted', allocation='normal', tile_mode=None),
        'C': dict(method='C', allocation='normal', tile_mode=None),
        'M': dict(method='M', allocation='normal', tile_mode=None),
        'T': dict(method='T', allocation='normal', tile_mode=None),
        'T_uniform': dict(method='T', allocation='uniform', tile_mode=None),
        'T_tile_mean': dict(method='T', allocation='normal', tile_mode='mean'),
        'T_shuffle': dict(method='T', allocation='shuffle', tile_mode=None),
    }


def phase_policy(early, late=None):
    late = early if late is None else late
    return {phase: {kind: dict(log_threshold=float(pair[kind]['log_threshold']))
                    for kind in ('local', 'global')}
            for phase, pair in (('early', early), ('late', late))}


def pair(local, global_):
    return {'local': dict(log_threshold=float(local)),
            'global': dict(log_threshold=float(global_))}


def _exposed_rows():
    paths = [ARCHIVE/'configs'/f'{split}_manifest.json' for split in ('final', 'calibration')]
    paths += [PARENT/'configs'/f'{split}_manifest.json'
              for split in ('final', 'calibration', 'development')]
    paths += [RANK_SWEEP/f'{split}_manifest.json'
              for split in ('final', 'calibration', 'development')]
    return [row for path in paths for row in json.loads(path.read_text())], paths


def _pool_selection():
    exposed, paths = _exposed_rows()
    used = {field: {row[field] for row in exposed if field in row}
            for field in ('source_id', 'prompt_hash')}
    chosen = {split: [] for split in ('calibration', 'validation', 'smoke')}
    rows = pool_source._pool_rows()
    for task in TASKS:
        local = [row for row in rows if row['task'] == task]
        local.sort(key=lambda row: pool_source._sha(
            f'42|{task}|{row["_pool_index"]}|{pool_source._sha(row["input"])}'))
        available = []
        for rank, row in enumerate(local):
            source_id = f'ruler_4096_{task}_p{int(row["_pool_index"]):04d}'
            if source_id in used['source_id'] or pool_source._sha(row['input']) in used['prompt_hash']:
                continue
            available.append(dict(row, _task_rank=rank))
        if len(available) < 5:
            raise ValueError(f'Only {len(available)} unused pool examples remain for {task}')
        for split, subset in (('calibration', available[:2]),
                              ('validation', available[2:4]),
                              ('smoke', available[4:5])):
            chosen[split].extend(dict(row, split=split) for row in subset)
    return chosen, paths


def _row(raw, split, tokenizer):
    prompt = raw['input']
    tokens = tokenizer.encode_prompt(prompt, {'thinking': False})
    source_id = f'ruler_4096_{raw["task"]}_p{int(raw["_pool_index"]):04d}'
    return dict(id=f'ruler4k/{source_id}', source_id=source_id, benchmark='ruler4k',
        task=raw['task'], task_base=pool_source._task_base(raw['task']),
        outputs=raw['outputs'], official_index=int(raw['index']),
        generator_seed=int(raw['index']), split=split, calibration=split=='calibration',
        prompt=prompt, prompt_hash=pool_source._sha(prompt), prompt_tokens=tokens,
        prompt_token_count=len(tokens), task_local_rank=int(raw['_task_rank']),
        generation_budget=int({'vt':30, 'cwe':120, 'fwe':50, 'qa_1':32,
                               'qa_2':32}.get(raw['task'],128)), seed=42)


def audit_manifests(manifests):
    exposed, _ = _exposed_rows()
    excluded = {field: {row[field] for row in exposed if field in row}
                for field in ('id', 'source_id', 'prompt_hash')}
    for split, per_task in (('calibration', 2), ('validation', 2), ('smoke', 1)):
        rows = manifests[split]
        if Counter(row['task'] for row in rows) != dict.fromkeys(TASKS, per_task):
            raise ValueError(f'Incorrect {split} task quotas')
        for field in excluded:
            values = [row[field] for row in rows]
            if len(set(values)) != len(values) or set(values) & excluded[field]:
                raise ValueError(f'Duplicate or exposed {split} {field}')
            excluded[field].update(values)
        for row in rows:
            if row['split'] != split or row['prompt_hash'] != pool_source._sha(row['prompt']):
                raise ValueError('Manifest prompt metadata mismatch')
    final = manifests['final']
    parent = json.loads((PARENT/'configs/final_manifest.json').read_text())
    if len(final) != 130 or any(x['id'] != y['id'] or x['prompt_hash'] != y['prompt_hash']
                                or x['seed'] != y['seed'] for x, y in zip(final, parent)):
        raise ValueError('Final manifest differs from the frozen fresh-prompt cohort')


def prepare(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    parent = json.loads((PARENT/'configs/configuration.json').read_text())
    selected, exposed_paths = _pool_selection()
    inputs = [Path(__file__), Path(__file__).with_name('query_adaptive.py'),
              Path(__file__).with_name('query_adaptive_study.py'),
              Path(__file__).with_name('query_adaptive_tile_controls.py'),
              Path(__file__).with_name('integration.py'),
              Path(__file__).with_name('cuda.py'), Path(parent['library']),
              Path(parent['torch_library']),
              Path('src/dllm/models/adapters/diffusion_gemma.py'),
              Path('experiments/diffusion_gemma_jl_output_aware/projections.py'),
              pool_source.POOL/'manifest.json', pool_source.POOL/'samples.jsonl',
              PARENT/'configs/configuration.json', PARENT/'configs/final_manifest.json',
              ARCHIVE/'configs/configuration.json', *exposed_paths]
    from transformers.models.diffusion_gemma import generation_diffusion_gemma
    inputs.append(Path(generation_diffusion_gemma.__file__))
    hashes = {str(path.resolve()): sha(path) for path in inputs}
    config_path = root/'configs/configuration.json'
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        if hashes != cfg['source_hashes']:
            raise ValueError('Frozen source changed: create a new versioned result root')
        manifests = {split: json.loads((root/f'configs/{split}_manifest.json').read_text())
                     for split in ('calibration', 'validation', 'smoke', 'final')}
        audit_manifests(manifests)
        return cfg, manifests
    tokenizer = create_adapter('diffusion_gemma', parent['model'], device='cpu',
        precision='float32', revision=parent['revision']).load_tokenizer()
    manifests = {split: [_row(row, split, tokenizer) for row in rows]
                 for split, rows in selected.items()}
    manifests['final'] = json.loads((PARENT/'configs/final_manifest.json').read_text())
    audit_manifests(manifests)
    old_frozen = json.loads((ARCHIVE/'configs/frozen_policies.json').read_text())
    cfg = dict(schema='query_adaptive_trajectory_guardrail_v1',
        source_hashes=hashes, model=parent['model'], revision=parent['revision'],
        library=parent['library'], torch_library=parent['torch_library'],
        methods=method_specs(), m_ref=parent['m_ref'], beta=3., gamma=.5,
        seed=42, confirmation_seeds=[43,44], projection_seed=1729, rank=32,
        canvas=256, max_steps=48, physical_tile=[128,64],
        temperature=parent['temperature'], early_calls=2,
        early_sensitivity='unit weights for every method, then own causal history',
        target_tolerance=.02, validation_tolerance=.03,
        guards={'50': dict(early_whole=.55, early_stratum=.57,
                           mean_steps=5.5, p90_steps=8., max_caps=0),
                '70': dict(early_whole=.75, early_stratum=.77,
                           mean_steps=8., p90_steps=12., max_caps=1)},
        source_pool=str(pool_source.POOL.resolve()),
        old_t70=old_frozen['policies']['T_s70'],
        parent_fingerprint=parent['fingerprint'],
        exposure='Final 130 previously examined; new calibration/validation/smoke are disjoint from prior manifests',
        selection='calibration: first feasible early/calls guardrails, then max whole/G/L target error, then calls/tiles; validation before final',
        sparsity='pooled skipped eligible 128x64 tiles / pooled eligible over all native decoder calls; prefix excluded')
    cfg['fingerprint'] = fingerprint(cfg)
    atomic(config_path, cfg)
    for split, rows in manifests.items():
        atomic(root/f'configs/{split}_manifest.json', rows)
    atomic(root/'configs/dataset_audit.json', dict(passed=True,
        counts={split:len(rows) for split,rows in manifests.items()},
        exposed_manifest_paths=[str(path.resolve()) for path in exposed_paths],
        source_hashes=hashes))
    return cfg, manifests


class PhaseState(State):
    """Phase thresholds plus optional tile-mean T weights, with native sampler."""
    def __init__(self, *args, phase_thresholds, tile_mode=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_thresholds = phase_thresholds
        self.tile_mode = tile_mode
        if tile_mode not in (None, 'mean'):
            raise ValueError(tile_mode)

    def begin(self, cur_step, canvas):
        super().begin(cur_step, canvas)
        if self.router is None:
            return
        if self.iteration <= 2:
            self.used_weights = None
            self.router.query_sensitivity = None
            self.current['weight_active'] = False
            for key in ('sensitivity_mean','sensitivity_p10','sensitivity_p50','sensitivity_p90'):
                self.current.pop(key,None)
        elif self.tile_mode == 'mean' and self.used_weights is not None:
            self.used_weights = tile_broadcast(self.used_weights, 'mean')
            self.router.query_sensitivity = self.used_weights
            if self.diagnostics:
                x = self.used_weights.detach().float().flatten()
                q = torch.quantile(x, torch.tensor([.1,.5,.9], device=x.device))
                self.current.update(sensitivity_mean=float(x.mean()),
                    sensitivity_p10=float(q[0]), sensitivity_p50=float(q[1]),
                    sensitivity_p90=float(q[2]))
        # Newer trajectory calibration can distinguish call 1, call 2, and
        # later calls.  Preserve the legacy early/late schedule as a fallback
        # for existing experiments.
        if self.iteration == 1 and 'call1' in self.phase_thresholds:
            phase = 'call1'
        elif self.iteration == 2 and 'call2' in self.phase_thresholds:
            phase = 'call2'
        elif self.iteration <= 2 and 'early' in self.phase_thresholds:
            phase = 'early'
        else:
            phase = 'late'
        self.router.policy_selector = lambda _iteration, kind: self.phase_thresholds[phase][kind]
        self.current['threshold_phase'] = phase


def generate(adapter, row, spec, policy, cfg, projections, *, diagnostics=True):
    native = spec['method'] == 'native_dense'
    ctx = nullcontext((None,None)) if native else install(adapter,cfg['library'],
        policy['late'],mode='value',projections=projections,
        torch_library=cfg['torch_library'],collect=True)
    with ctx as (binding, router):
        if binding:
            _set_context(binding,row)
        state = PhaseState(spec['method'],router,m_ref=cfg['m_ref'],beta=cfg['beta'],
            gamma=cfg['gamma'],allocation=spec['allocation'],bootstrap=False,
            seed=row['seed'],diagnostics=diagnostics,
            phase_thresholds=policy,tile_mode=spec['tile_mode'])
        with observe(adapter.model,state):
            torch.cuda.synchronize()
            started=time.perf_counter()
            output=adapter.generate(_request(row))
            torch.cuda.synchronize()
            seconds=time.perf_counter()-started
        routing=[] if router is None else router.records()
    calls=output.metadata['actual_denoising_step_count']
    if diagnostics and len(state.steps)!=calls:
        raise AssertionError('Native decoder call count mismatch')
    if output.metadata['native_canvas_length']!=256 or diagnostics and len(state.canvases)!=1:
        raise ValueError('Expected one native 256-position canvas')
    counts=routing_counts(routing)
    per_step=defaultdict(empty_counts)
    for item in routing:
        index=item['step']+1
        if not 1<=index<=calls:
            raise ValueError('Routing call index out of range')
        for kind in ('whole',item['attention_type']):
            per_step[index][kind]['eligible']+=item['eligible']
            per_step[index][kind]['skipped']+=item['skipped']
    if diagnostics:
        for step in state.steps:
            step['counts']=per_step[step['iteration']]
        canvas=dict(state.canvases[0],counts=counts,
            generated_tokens=len(output.completion_tokens),
            returned_tokens=len(output.completion_tokens))
    else:
        canvas=dict(canvas_index=0,iterations=calls,counts=counts)
    result=dict(id=row['id'],task=row['task'],prompt_hash=row['prompt_hash'],
        seed=row['seed'],method=spec['method'],score=score(row,output.text),
        prediction=output.text,completion_tokens=output.completion_tokens,
        metadata=output.metadata,steps=calls,seconds=seconds,counts=counts,
        canvas=canvas,step_records=state.steps,
        projection_manifest=projections.manifest if not native else {})
    return result,routing


def cached(adapter, root, stage, method, target, row, policy, cfg, projections):
    condition=method if target is None else label(method,target)
    spec=cfg['methods'][method]
    identity=fingerprint([cfg['fingerprint'],stage,condition,spec,policy,
                          cfg['m_ref'],row['id'],row['prompt_hash'],row['seed']])
    dest=shard_path(Path(root)/stage,condition,row)
    if dest.exists():
        result=json.loads(dest.read_text())
        if result['identity']!=identity or result.get('status')!='complete':
            raise ValueError(f'Invalid cached identity: {dest}')
        if result.get('routing_path') and sha(result['routing_path'])!=result['routing_sha256']:
            raise ValueError(f'Corrupt routing shard: {dest}')
        return result
    atomic(Path(root)/'status.json',dict(stage=stage,condition=condition,
        id=row['id'],pid=os.getpid(),updated=time.time()))
    result,routing=generate(adapter,row,spec,policy,cfg,projections)
    result.update(identity=identity,condition=condition,target=target,
                  policy=('all-retained' if method=='kernel_dense' else
                          None if method=='native_dense' else policy),status='complete')
    if routing:
        dest.parent.mkdir(parents=True,exist_ok=True)
        raw=dest.with_suffix('.routing.json.gz')
        temporary=raw.with_suffix('.tmp')
        with gzip.open(temporary,'wt',compresslevel=3) as stream:
            json.dump(routing,stream)
        os.replace(temporary,raw)
        result.update(routing_path=str(raw.resolve()),routing_sha256=sha(raw))
    atomic(dest,result)
    print(json.dumps(dict(event='complete',stage=stage,condition=condition,
        id=row['id'],steps=result['steps'],seconds=round(result['seconds'],3),
        sparsity=round(result['counts']['whole']['skipped']/max(1,result['counts']['whole']['eligible']),4))),flush=True)
    return result


def profile(rows):
    overall, counts=aggregate(rows)
    grouped={phase:empty_counts() for phase in ('step1','step2','later')}
    accepted={phase:[] for phase in ('step1','step2')}
    entropy={phase:[] for phase in ('step1','step2')}
    for row in rows:
        for step in row['step_records']:
            phase='step1' if step['iteration']==1 else 'step2' if step['iteration']==2 else 'later'
            for kind in ('whole','local','global'):
                for field in ('eligible','skipped'):
                    grouped[phase][kind][field]+=step['counts'][kind][field]
            if phase in accepted:
                accepted[phase].append(step['accepted'])
                entropy[phase].append(step['processed_entropy_mean'])
    rates={phase:{kind:(data['skipped']/data['eligible'] if data['eligible'] else 0.)
                  for kind,data in groups.items()}
           for phase,groups in grouped.items()}
    steps=sorted(row['steps'] for row in rows)
    return dict(n=len(rows),overall=overall,counts=counts,phase_sparsity=rates,
        phase_counts=grouped,mean_steps=float(statistics.mean(steps)),
        median_steps=float(statistics.median(steps)),
        p90_steps=float(np.quantile(steps,.9)),p95_steps=float(np.quantile(steps,.95)),
        cap_count=sum(step>=48 for step in steps),total_steps=sum(steps),
        accuracy=float(np.mean([row['score'] for row in rows])),
        accepted_mean={k:float(np.mean(v)) if v else None for k,v in accepted.items()},
        entropy_mean={k:float(np.mean(v)) if v else None for k,v in entropy.items()},
        executed_tiles=counts['whole']['eligible']-counts['whole']['skipped'])


def violations(info,target,*,validation=False,goal=None):
    rule={'50':dict(early_whole=.55,early_stratum=.57,mean_steps=5.5,p90_steps=8.,max_caps=0),
          '70':dict(early_whole=.75,early_stratum=.77,mean_steps=8.,p90_steps=12.,max_caps=1)}[str(target)]
    tolerance=.03 if validation else .02
    goal=target/100 if goal is None else goal
    if isinstance(goal,float):goal={kind:goal for kind in ('whole','local','global')}
    problems=[]
    for kind in ('whole','local','global'):
        if abs(info['overall'][kind]-goal[kind])>tolerance:
            problems.append(f'overall_{kind}')
    for phase in ('step1','step2'):
        if info['phase_sparsity'][phase]['whole']>rule['early_whole']:
            problems.append(f'{phase}_whole')
        for kind in ('local','global'):
            if info['phase_sparsity'][phase][kind]>rule['early_stratum']:
                problems.append(f'{phase}_{kind}')
    if info['mean_steps']>rule['mean_steps']:
        problems.append('mean_steps')
    if info['p90_steps']>rule['p90_steps']:
        problems.append('p90_steps')
    if info['cap_count']>rule['max_caps']:
        problems.append('cap_count')
    return problems


def goal_error(info,goal):
    if isinstance(goal,float):goal={kind:goal for kind in ('whole','local','global')}
    return max(abs(info['overall'][kind]-goal[kind]) for kind in ('whole','local','global'))


def _logs(policy,phase='late'):
    return tuple(policy[phase][kind]['log_threshold'] for kind in ('local','global'))


def _unique_policies(items):
    seen=set();out=[]
    for policy in items:
        key=tuple(round(x,6) for phase in ('early','late') for x in _logs(policy,phase))
        if key not in seen:
            seen.add(key);out.append(policy)
    return out


def initial_policies(target):
    old=json.loads((ARCHIVE/'configs/thresholds'/f'T_s{target}.json').read_text())['policy']
    fresh=json.loads((PARENT/'configs/thresholds'/f'T_s{target}.json').read_text())['policy']
    ol,og=(old[k]['log_threshold'] for k in ('local','global'))
    nl,ng=(fresh[k]['log_threshold'] for k in ('local','global'))
    candidates=[pair(ol,og),pair((ol+nl)/2,(og+ng)/2),pair(nl,ng),
                pair(ol,ng),pair(nl,og),pair(ol+.15,og+.15),
                pair(ol-.15,og-.15)]
    return _unique_policies([phase_policy(item) for item in candidates])


def late_policies(target,early):
    old=json.loads((ARCHIVE/'configs/thresholds'/f'T_s{target}.json').read_text())['policy']
    fresh=json.loads((PARENT/'configs/thresholds'/f'T_s{target}.json').read_text())['policy']
    ol,og=(old[k]['log_threshold'] for k in ('local','global'))
    nl,ng=(fresh[k]['log_threshold'] for k in ('local','global'))
    candidates=[old, pair((ol+nl)/2,(og+ng)/2),fresh,
                pair(ol,ng),pair(nl,og),pair(ol+.15,og+.15),
                pair(nl+.15,ng+.15)]
    return _unique_policies([phase_policy(early,item) for item in candidates])


def rank_point(point,goal):
    info=point['metrics']
    return (len(point['violations'])!=0,
            goal_error(info,goal),info['mean_steps'],info['executed_tiles'])


def _evaluate_policy(adapter,root,cfg,rows,projections,stage,method,target,policy):
    key=fingerprint([method,target,policy])[:16]
    results=[cached(adapter,root,f'{stage}/{label(method,target)}/{key}',method,target,
                    row,policy,cfg,projections) for row in rows]
    info=profile(results)
    return dict(key=key,policy=policy,metrics=info,
        violations=violations(info,target),stage=stage,
        ids=[row['id'] for row in rows])


def _write_trace(root,condition,points):
    atomic(Path(root)/'calibration_traces'/f'{condition}.json',points)


def calibrate_t(adapter,root,cfg,manifests,projections,target):
    condition=label('T',target)
    dest=Path(root)/'configs/thresholds'/f'{condition}.json'
    if dest.exists():
        frozen=json.loads(dest.read_text())
        if frozen['fingerprint']!=cfg['fingerprint']:
            raise ValueError('Frozen T policy fingerprint mismatch')
        return frozen
    cal=manifests['calibration'];val=manifests['validation'];goal=target/100
    points=[]
    for policy in initial_policies(target):
        item=_evaluate_policy(adapter,root,cfg,cal,projections,'calibration',
                              'T',target,policy)
        points.append(item);_write_trace(root,condition,points)
        print(json.dumps(dict(event='calibration_point',condition=condition,
            key=item['key'],metrics=item['metrics'],violations=item['violations'])),flush=True)
    feasible=sorted((p for p in points if not p['violations']),key=lambda p:rank_point(p,goal))
    if not feasible:
        # Freeze the safest early pair before searching only late thresholds.
        early_rank=sorted(points,key=lambda p:(
            sum(v.startswith(('step1','step2')) for v in p['violations']),
            abs(p['metrics']['phase_sparsity']['step1']['whole']-min(goal+.02,.73))+
            abs(p['metrics']['phase_sparsity']['step2']['whole']-min(goal+.02,.73)),
            p['metrics']['mean_steps']))
        for anchor in early_rank[:2]:
            for policy in late_policies(target,anchor['policy']['early']):
                if any(p['policy']==policy for p in points):continue
                item=_evaluate_policy(adapter,root,cfg,cal,projections,'calibration',
                                      'T',target,policy)
                points.append(item);_write_trace(root,condition,points)
                print(json.dumps(dict(event='calibration_point',condition=condition,
                    key=item['key'],metrics=item['metrics'],violations=item['violations'])),flush=True)
            feasible=sorted((p for p in points if not p['violations']),
                            key=lambda p:rank_point(p,goal))
            if feasible:break
    ranked=feasible if feasible else sorted(points,key=lambda p:rank_point(p,goal))
    unpruned=phase_policy(pair(-math.inf,-math.inf))
    dense=[cached(adapter,root,'validation_dense','kernel_dense',None,row,
                  unpruned,cfg,projections) for row in val]
    dense_profile=profile(dense)
    atomic(Path(root)/'configs/validation_dense.json',dict(
        ids=[row['id'] for row in val],metrics=dense_profile,
        source='same matched v4 H100 kernel with all eligible tiles retained'))
    validation=[];selected=None
    for point in ranked[:4]:
        item=_evaluate_policy(adapter,root,cfg,val,projections,'validation',
                              'T',target,point['policy'])
        item['violations']=violations(item['metrics'],target,validation=True)
        if item['metrics']['accuracy'] < dense_profile['accuracy']-.03:
            item['violations'].append('accuracy_drop_gt_3pp')
        validation.append(item)
        if not item['violations']:
            selected=point;break
    if selected is None:
        selected=ranked[0]
    status='attained' if not selected['violations'] and any(
        not x['violations'] and x['policy']==selected['policy'] for x in validation) else 'unattainable_under_guardrails'
    frozen=dict(fingerprint=cfg['fingerprint'],condition=condition,target=target,
        method='T',policy=selected['policy'],status=status,
        calibration=selected,validation=validation,
        calibration_ids=[r['id'] for r in cal],validation_ids=[r['id'] for r in val],
        selection='first validation-passing calibration-feasible point ranked by max O/G/L sparsity error then steps; otherwise report unattainable')
    atomic(dest,frozen)
    return frozen


def _adapter(cfg):
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return create_adapter('diffusion_gemma',cfg['model'],device='cuda',
        precision='bfloat16',revision=cfg['revision']).load()


def smoke(adapter,root,cfg,manifests,projections):
    dest=Path(root)/'smoke.json'
    if dest.exists():
        saved=json.loads(dest.read_text())
        if saved['fingerprint']!=cfg['fingerprint'] or not saved['passed']:
            raise ValueError('Invalid smoke provenance')
        return
    base=json.loads((ARCHIVE/'configs/thresholds/T_s70.json').read_text())['policy']
    scheduled=phase_policy(base,pair(base['local']['log_threshold']+.1,
                                      base['global']['log_threshold']+.1))
    unpruned=phase_policy(pair(-math.inf,-math.inf))
    checks=[]
    for row in manifests['smoke'][:2]:
        for method in ('native_dense','kernel_dense','unweighted','T','C','M',
                       'T_uniform','T_tile_mean'):
            policy=unpruned if method in ('native_dense','kernel_dense') else scheduled
            first,_=generate(adapter,row,cfg['methods'][method],policy,cfg,projections)
            second,_=generate(adapter,row,cfg['methods'][method],policy,cfg,projections,
                              diagnostics=False)
            if first['completion_tokens']!=second['completion_tokens'] or first['steps']!=second['steps']:
                raise AssertionError(f'Instrumentation changed {method}')
            if method!='native_dense' and first['step_records'][0]['threshold_phase']!='early':
                raise AssertionError('First decoder call did not select early policy')
            if any(s['weight_active'] for s in first['step_records'][:2]):
                raise AssertionError('Early shared unit-weight phase failed')
            if method=='kernel_dense' and first['counts']['whole']['skipped']!=0:
                raise AssertionError('Unpruned kernel skipped tiles')
            checks.append(dict(method=method,id=row['id'],parity=True,steps=first['steps']))
        variant=dict(cfg,beta=0.)
        weighted,_=generate(adapter,row,cfg['methods']['T'],scheduled,variant,projections)
        plain,_=generate(adapter,row,cfg['methods']['unweighted'],scheduled,cfg,projections)
        if weighted['completion_tokens']!=plain['completion_tokens'] or weighted['counts']!=plain['counts']:
            raise AssertionError('Unit T sensitivity differs from unweighted routing')
        checks.append(dict(method='T_beta0',id=row['id'],unit_weight_parity=True))
    parent_policy=json.loads((PARENT/'configs/frozen_policies.json').read_text())['policies']['T_s70']
    row=manifests['final'][0]
    expected=json.loads(shard_path(PARENT/'final','T_s70',row).read_text())
    reproduced,_=generate(adapter,row,cfg['methods']['T'],phase_policy(parent_policy),
                          cfg,projections)
    if (reproduced['completion_tokens']!=expected['completion_tokens'] or
        reproduced['steps']!=expected['steps'] or reproduced['counts']!=expected['counts']):
        raise AssertionError('Frozen parent T70 output/steps/physical counts did not reproduce')
    checks.append(dict(method='parent_T_s70',id=row['id'],exact_parity=True,
                       steps=reproduced['steps']))
    atomic(dest,dict(passed=True,fingerprint=cfg['fingerprint'],checks=checks))


def command(root,stage):
    root=Path(root)
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg,manifests=prepare(root)
        adapter=_adapter(cfg);projections=Projections()
        smoke(adapter,root,cfg,manifests,projections)
        if stage=='smoke':return
        if stage=='calibrate-t':
            for target in (70,50):
                calibrate_t(adapter,root,cfg,manifests,projections,target)
        elif stage=='run-t':
            frozen=calibrate_t(adapter,root,cfg,manifests,projections,70)
            if frozen['status']!='attained':
                raise ValueError('T70 failed calibration/validation guardrails; no final run launched')
            policy=frozen['policy'];failures=[]
            for row in manifests['final']:
                try:cached(adapter,root,'final','T',70,row,policy,cfg,projections)
                except Exception as error:
                    failures.append(dict(id=row['id'],error=repr(error),
                        traceback=traceback.format_exc()))
                    atomic(root/'failures.json',failures)
                    if 'illegal memory' in str(error) or 'device-side assert' in str(error):raise
                    torch.cuda.empty_cache()
            atomic(root/'status.json',dict(stage='run-t-complete' if not failures else 'run-t-incomplete',
                completed=130-len(failures),errors=len(failures),updated=time.time()))
        else:raise ValueError(stage)


def launch(root,stage):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH='src:.',CUDA_VISIBLE_DEVICES='0',
        HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',OMP_NUM_THREADS='4')
    cmd=[sys.executable,'-m','experiments.value_direction_hopper.query_adaptive_guardrail',
         stage,'--root',str(root)]
    logfile=root/f'{stage}_{int(time.time())}.log'
    with logfile.open('xb') as out:
        worker=subprocess.Popen(cmd,cwd=Path(__file__).resolve().parents[2],env=env,
            stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,
            start_new_session=True,close_fds=True)
    atomic(root/f'{stage}_launch.json',dict(pid=worker.pid,log=str(logfile),
        command=cmd,started=time.time()))
    print(json.dumps(dict(pid=worker.pid,log=str(logfile))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('prepare','smoke','calibrate-t','run-t',
                                         'launch-smoke','launch-calibrate-t','launch-run-t'))
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    if args.stage.startswith('launch-'):
        launch(args.root,args.stage.removeprefix('launch-'))
    elif args.stage=='prepare':
        prepare(args.root)
    else:
        command(args.root,args.stage)
