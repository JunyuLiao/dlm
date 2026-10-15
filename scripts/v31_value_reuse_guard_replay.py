"""Causal residual-trust replay after unconditional correction showed tail harm."""
import argparse
from collections import defaultdict
from pathlib import Path
import json
import subprocess
import time

import torch

from scripts import v31_value_decision_replay as base
from experiments.numerical_qk_reuse.v31_value_decision import fixed_topk, mass_score, moments
from experiments.numerical_qk_reuse.v31_value_reuse_guard import residual_trust


def prepare(study):
    parent=base.verify(study.parent)
    if not (study.parent/'whole_layer/operator_complete.json').exists():
        raise RuntimeError('whole-layer screen must finish first')
    source=dict(parent['source_sha256'])
    for p in [Path(__file__),base.ROOT/'experiments/numerical_qk_reuse/v31_value_reuse_guard.py']:
        source[str(p.relative_to(base.ROOT))]=base.sha(p)
    protocol=dict(parent,schema='value_residual_trust_replay_v1',source_sha256=source,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),heads=list(range(16)),query_tiles=[0,1],
        stage='whole_layer_causal_residual_guard',
        refinement_basis='unconditional correction lowered mean error but increased worst-row errors',
        modes=['mass_anchor','residual','residual_direction_guard','residual_norm_guard'],
        causal_inputs='current selected output, anchor selected/dense outputs, real O projection',
        future_information_used_for_selection=False,
        inference_limit='surrogate assumes stationary dense anchor; no certified current-error bound',timing='paused')
    base.write(study/'operator_protocol.json',protocol,True)


@torch.inference_mode()
def run(study):
    p=base.verify(study)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('H100 occupied')
    for rel,digest in base.read(base.SOURCE/'specs/real_snapshot_inventory.json')['sha256'].items():
        assert base.sha(base.SOURCE/rel)==digest
    for rel,digest in p['model_shards_sha256'].items():
        assert base.sha(base.MODEL/rel)==digest
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(8)
    from safetensors import safe_open
    index=base.read(base.MODEL/'model.safetensors.index.json')['weight_map']; weights={}
    for layer in (5,11,17,23,29):
        name=f'model.decoder.layers.{layer}.self_attn.o_proj.weight'
        with safe_open(str(base.MODEL/index[name]),framework='pt',device='cpu') as f:weights[layer]=f.get_tensor(name).float().cuda()
    captures=[0];original=torch.cuda.CUDAGraph.capture_begin
    def capture(graph,*a,**kw):
        captures[0]+=1;return original(graph,*a,**kw)
    torch.cuda.CUDAGraph.capture_begin=capture
    last=time.monotonic()
    for group,files in sorted(base.group_paths().items()):
        name='_'.join(group);family=study/'operator'/name
        done=sorted(family.glob('attempt*/complete.json'))
        if done:
            marker=base.read(done[-1]);assert marker['protocol_sha256']==base.sha(study/'operator_protocol.json')
            assert marker['records_sha256']==base.sha(done[-1].parent/'records.json');continue
        used=[int(x.name[7:]) for x in family.glob('attempt*') if x.is_dir()]
        leaf=family/f'attempt{max(used,default=0)+1:03d}';leaf.mkdir(parents=True)
        base.write(study/'state.json',dict(stage='running',group=name,timing='paused'))
        prefix=None;selected={};anchor_dense=anchor_sparse=None;age=0;rows=[];start=time.monotonic()
        for path in files:
            d=torch.load(path,map_location='cpu',weights_only=True)
            if prefix is None:prefix=torch.load(path.parent/d['prefix_file'],map_location='cuda',weights_only=True)
            if not selected and not d['selection_event']:continue
            keys=torch.cat((prefix['k'],d['tail_k'].cuda()),2)[0];values=torch.cat((prefix['v'],d['tail_v'].cuda()),2)[0]
            w=weights[d['layer']];dense=torch.zeros(d['n'],w.shape[0],device='cuda');sparse=torch.zeros_like(dense)
            event=d['selection_event'];age=0 if event else age+1
            for h in range(16):
                for qt in (0,1):
                    lo,hi=qt*128,min(d['n'],(qt+1)*128)
                    if hi<=lo:continue
                    m=moments(d['q'][0,h,lo:hi].cuda(),keys[h//8],values[h//8],d['prefix'],d['scale'])
                    budget=min(len(m.mass),27 if d['dataset']=='aime26' else 128)
                    if event:
                        if d['dataset']=='aime26':
                            score=mass_score(m).clamp_min(1e-30).log()
                            if (h,qt) in selected:score+=1.386*selected[h,qt]
                            selected[h,qt]=fixed_topk(score,budget)
                        else:selected[h,qt]=d['next'][0,h,qt,:len(m.mass)].cuda().bool()
                    mask=selected[h,qt];assert int(mask.sum())==budget
                    dense[lo:hi]+=m.dense()@w[:,h*512:(h+1)*512].T
                    sparse[lo:hi]+=m.output(mask)@w[:,h*512:(h+1)*512].T
                    del m
            if event:
                anchor_dense,anchor_sparse=dense.clone(),sparse.clone()
            residual=anchor_dense-anchor_sparse
            alpha={'mass_anchor':torch.zeros(d['n'],device='cuda'),'residual':torch.ones(d['n'],device='cuda'),
                'residual_direction_guard':residual_trust(sparse,anchor_sparse,anchor_dense,kind='direction'),
                'residual_norm_guard':residual_trust(sparse,anchor_sparse,anchor_dense,kind='norm')}
            for arm,a in alpha.items():
                err=(sparse+a[:,None]*residual-dense).norm(dim=-1)
                rows.append(dict(request=group[0],dataset=d['dataset'],canvas=d['canvas'],call=d['call'],layer=d['layer'],age=age,
                    budget=budget,prefix_tiles=d['prefix']//64,arm=arm,mean_post_o_error=float(err.mean()),
                    max_post_o_error=float(err.max()),p95_post_o_error=float(err.quantile(.95)),alpha_mean=float(a.mean())))
            if time.monotonic()-last>=1800:
                print(json.dumps(dict(event='progress',group=name,call=d['call'])),flush=True);last=time.monotonic()
        base.write(leaf/'records.json',rows,True)
        base.write(leaf/'complete.json',dict(status='complete',records_sha256=base.sha(leaf/'records.json'),
            protocol_sha256=base.sha(study/'operator_protocol.json'),input_frames=len(files),records=len(rows),
            cuda_captures=captures[0],budget_errors=0,gpu_reservation_s=time.monotonic()-start),True)
        print(json.dumps(dict(event='group_complete',group=name,records=len(rows))),flush=True)
        del prefix,selected
        torch.cuda.empty_cache()
    summarize(study)


def summarize(study):
    p=base.verify(study);rows=[];groups=0
    for group in p['groups']:
        done=sorted((study/'operator'/group).glob('attempt*/complete.json'))
        if not done:continue
        marker=base.read(done[-1]);assert marker['protocol_sha256']==base.sha(study/'operator_protocol.json')
        assert marker['records_sha256']==base.sha(done[-1].parent/'records.json')
        rows+=base.read(done[-1].parent/'records.json');groups+=1
    refs={(r['request'],r['canvas'],r['call'],r['layer']):r for r in rows if r['arm']=='mass_anchor'}
    buckets=defaultdict(list)
    for r in rows:
        if r['age']>0 and r['prefix_tiles']>r['budget']:buckets[r['request'],r['arm']].append(r)
    metrics=[]
    for (request,arm),rr in sorted(buckets.items()):
        control=[refs[r['request'],r['canvas'],r['call'],r['layer']] for r in rr]
        metrics.append(dict(request=request,arm=arm,frames=len(rr),
            mean_error_ratio=sum(r['mean_post_o_error'] for r in rr)/sum(r['mean_post_o_error'] for r in control),
            mean_harm_frames=sum(r['mean_post_o_error']>c['mean_post_o_error']+1e-6 for r,c in zip(rr,control)),
            max_harm_frames=sum(r['max_post_o_error']>c['max_post_o_error']*1.02+1e-6 for r,c in zip(rr,control)),
            max_error=max(r['max_post_o_error'] for r in rr),alpha_mean=sum(r['alpha_mean'] for r in rr)/len(rr)))
    base.write(study/'summary.json',dict(status='complete' if groups==len(p['groups']) else 'partial',groups=groups,
        metrics=metrics,model_decision_gate='not evaluated',timing='paused'))
    if groups==len(p['groups']):base.write(study/'operator_complete.json',dict(status='complete',
        protocol_sha256=base.sha(study/'operator_protocol.json'),summary_sha256=base.sha(study/'summary.json'),whole_study_complete=False),True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['prepare','run','summary'],required=True)
    a.add_argument('--study',type=Path,default=base.DEFAULT/'reuse_guard')
    args=a.parse_args();{'prepare':prepare,'run':run,'summary':summarize}[args.stage](args.study.resolve())
