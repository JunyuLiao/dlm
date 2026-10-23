"""Bounded full-dimensional real-state attention diagnostics; no native rollout.

Inputs are inherited V31 trajectories, not dense census states. All counterfactual
attention uses current Q/K/V. Prefix and current canvas remain explicitly distinct.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

REPO=Path('/home/exouser/ljy/dlm')
SOURCE=REPO/'results/v31_cached_refresh_stable_20261013'
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def probability(q,k,scale):
    return (q@k.T*scale).softmax(-1)

def output(p,v,support):
    retained=p*support[None]
    mass=retained.sum(-1)
    if bool((mass<=0).any()):raise ValueError('empty row support')
    normalized=retained/mass[:,None]
    return normalized@v,normalized,mass

def differential(p,q,k,v,dq,dk,dv,scale):
    mean=p@v
    def covariance(ds):
        weights=p*ds
        return weights@v-weights.sum(-1,keepdim=True)*mean
    query=covariance(dq@k.T*scale)
    key=covariance(q@dk.T*scale)
    value=p@dv
    return dict(query=query,key=key,value=value,total=query+key+value)

def token_support(mask,length,block=64):
    n=len(mask)*block
    return torch.cat((mask.repeat_interleave(block),torch.ones(length-n,dtype=torch.bool,device=mask.device)))

def candidate_bank(p,v,prefix,budget):
    blocks=prefix//64
    tile_mass=p[:,:blocks*64].reshape(len(p),blocks,64).sum(-1)
    prefix_mass=p[:,:prefix].sum(-1).clamp_min(1e-30)
    mass_score=(tile_mass/prefix_mass[:,None]).amax(0)
    dense=p@v
    contributions=torch.stack([p[:,j*64:(j+1)*64]@v[j*64:(j+1)*64] for j in range(blocks)],1)
    singleton=((contributions-tile_mass[...,None]*dense[:,None]).norm(dim=-1)/(1-tile_mass).clamp_min(1e-30)).amax(0)
    masks={}
    def top(score):
        mask=torch.zeros(blocks,dtype=torch.bool,device=p.device)
        mask[score.argsort(descending=True,stable=True)[:budget]]=True
        return mask
    masks['mass']=top(mass_score);masks['singleton']=top(singleton)
    order=mass_score.argsort(descending=True,stable=True)
    # Fixed deterministic bounded swaps; candidates do not use future state.
    for count in (1,2,4,8):
        if count>min(budget,blocks-budget):continue
        for offset in (0,1):
            mask=masks['mass'].clone()
            drops=order[budget-count:budget]
            start=min(budget+offset*count,blocks-count)
            adds=order[start:start+count]
            if bool(mask[adds].any()):continue
            mask[drops]=False;mask[adds]=True
            masks[f'swap{count:02d}_offset{offset}']=mask
    masks['position_sham']=top(-torch.remainder(torch.arange(blocks,device=p.device)*73+19,max(blocks,1)).float())
    for mask in masks.values():
        if int(mask.sum())!=budget:raise RuntimeError('physical budget violation')
    # Labels can produce the same physical support, especially at short prefixes.
    unique={};seen=set()
    for label,mask in masks.items():
        identity=mask.detach().cpu().numpy().tobytes()
        if identity not in seen:
            unique[label]=mask;seen.add(identity)
    return unique

def retained_bound(anchor,dq,keys,scale,all_kept=False):
    radius=dq.norm(dim=-1)*keys.norm(dim=-1).max()*scale
    # Prefix-only certificate, fixed prefix K/V; reset by construction at every canvas.
    if all_kept:return torch.ones_like(anchor),torch.zeros_like(radius)
    # Exact endpoint limits; do not degrade u=1 by arbitrary clamping.
    u=anchor.clamp(0,1)
    bound=(torch.logit(u)-2*radius).sigmoid()
    return bound,radius

def choose_frames():
    inventory=json.loads((SOURCE/'specs/real_snapshot_inventory.json').read_text())
    selected=[];files={}
    for request in ('request001','request002','request003','request004'):
        frames=sorted(SOURCE/p for p in inventory['sha256'] if request in p and 'canvas00_call' in p and p.endswith('_layer29.pt'))
        if len(frames)<4:raise RuntimeError('insufficient preserved trajectory')
        for phase,index in [('early',1),('middle',(len(frames)-2)//2),('late',len(frames)-2)]:
            age_indices=sorted({min(index+age,len(frames)-1) for age in (1,2,4)})
            paths=[frames[index-1],frames[index]]+[frames[i] for i in age_indices]
            # Prefix file is fixed by the canvas and layer.
            prefix_path=frames[0].with_name('canvas00_layer29_prefix.pt')
            paths.append(prefix_path)
            for path in paths:
                relative=str(path.relative_to(SOURCE));expected=inventory['sha256'][relative]
                if sha(path)!=expected:raise RuntimeError('historical input hash mismatch')
                files[str(path)]=expected
            selected.append(dict(request=request,phase=phase,index=index,anchor=str(frames[index]),
                previous=str(frames[index-1]),future=[str(frames[i]) for i in age_indices],
                prefix=str(prefix_path),split='heldout' if request in ('request002','request004') else 'development'))
    return selected,files

def freeze():
    selected,files=choose_frames()
    write(ROOT/'protocol.json',dict(schema='response_operator_diagnostic_v1',
        source_scope='Preserved inherited V31 trajectories; dense attention is recomputed at identical state. Not a dense on-policy state distribution.',
        heads=[0,8],query_rows=[0,32,64,96],layer=29,canvas=0,
        prefix_budget_fractions=[.25,.5,.75,.9],epsilon=1e-3,
        phases=['early','middle','late'],selected=selected,input_sha256=files,
        source_sha256=sha(__file__),torch_dtype='float64',
        candidate_accounting='deduplicate identical physical supports within each state and budget',
        numerical_gate='finite_difference absolute norm error <=1e-5 +1e-3*analytic norm',
        supersedes_preserved_operator_protocol_sha256=sha(ROOT.with_name('operator_diagnostic001')/'protocol.json'),
        response_direction='causal preceding Q/K/V displacement; mask-induced differential versus dense',
        reuse_ages=[1,2,4],canvas_policy='mandatory boundary/current canvas, no carry, certificate resets at canvas transition',
        matched_pair=dict(deleted_mass_absolute_tolerance=.01,relative_output_error_ratio_tolerance=.1),
        guard=dict(max_whole_deleted_mass=.2,max_relative_output_bias=.1,max_canvas_amplification=1.25),
        promotion='operator evidence only; no native promotion without same-state native continuation and held-out transition prediction',
        prohibited_native_actions=['model forwards','sampler draws','stopping changes','residual compensation'],
        max_gpu_reservation_seconds=900))

def rank_correlation(x,y):
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return None
    from scipy.stats import spearmanr
    value=float(spearmanr(x,y).statistic)
    return value if math.isfinite(value) else None

@torch.inference_mode()
def run(device):
    protocol=json.loads((ROOT/'protocol.json').read_text())
    if sha(__file__)!=protocol['source_sha256']:raise RuntimeError('frozen source drift')
    for path,expected in protocol['input_sha256'].items():
        if sha(path)!=expected:raise RuntimeError('frozen input drift')
    started=time.monotonic();records=[];attention_evaluations=0
    for state in protocol['selected']:
        pref=torch.load(state['prefix'],map_location='cpu',weights_only=True)
        anchor=torch.load(state['anchor'],map_location='cpu',weights_only=True)
        previous=torch.load(state['previous'],map_location='cpu',weights_only=True)
        future=[torch.load(path,map_location='cpu',weights_only=True) for path in state['future']]
        for head in protocol['heads']:
            kvhead=head//8;rows=protocol['query_rows']
            def tensors(frame):
                q=frame['q'][0,head,rows].to(device=device,dtype=torch.float64)
                k=torch.cat((pref['k'][0,kvhead],frame['tail_k'][0,kvhead])).to(device=device,dtype=torch.float64)
                v=torch.cat((pref['v'][0,kvhead],frame['tail_v'][0,kvhead])).to(device=device,dtype=torch.float64)
                return q,k,v
            q,k,v=tensors(anchor);pq,pk,pv=tensors(previous)
            dq,dk,dv=q-pq,k-pk,v-pv;scale=float(anchor['scale']);prefix=int(anchor['prefix'])
            p=probability(q,k,scale);dense=p@v
            d_dense=differential(p,q,k,v,dq,dk,dv,scale)
            future_tensors=[tensors(frame) for frame in future]
            future_p=[probability(fq,fk,scale) for fq,fk,fv in future_tensors]
            eps=protocol['epsilon']
            pplus=probability(q+eps*dq,k+eps*dk,scale)
            pminus=probability(q-eps*dq,k-eps*dk,scale)
            fd_dense=(pplus@(v+eps*dv)-pminus@(v-eps*dv))/(2*eps)
            blocks=prefix//64
            for fraction in protocol['prefix_budget_fractions']:
                budget=max(1,min(blocks,int(math.ceil(blocks*fraction))))
                masks=candidate_bank(p,v,prefix,budget)
                for label,mask in masks.items():
                    support=token_support(mask,len(k));out,ps,mass=output(p,v,support)
                    alpha=1-mass;bias=out-dense
                    norm=dense.norm(dim=-1).clamp_min(1e-12)
                    response=differential(ps,q,k,v,dq,dk,dv,scale)
                    gains={name:float(((response[name]-d_dense[name]).norm(dim=-1)/norm).max()) for name in response}
                    plus=output(pplus,v+eps*dv,support)[0];minus=output(pminus,v-eps*dv,support)[0]
                    fd=(plus-minus)/(2*eps)-fd_dense
                    analytic=response['total']-d_dense['total']
                    fd_error=float((fd-analytic).norm()/analytic.norm().clamp_min(1e-6))
                    fd_absolute=float((fd-analytic).norm())
                    fd_gate=fd_absolute<=1e-5+1e-3*float(analytic.norm())
                    whole_prefix=p[:,:prefix].sum(-1)
                    removed_prefix=(p[:,:prefix]*(~support[:prefix])[None]).sum(-1)
                    canvas_dense=p[:,prefix:].sum(-1)
                    canvas_sparse=ps[:,prefix:].sum(-1)
                    renorm_error=float((canvas_sparse-canvas_dense/mass).abs().max())
                    anchor_retained=(p[:,:prefix]*support[:prefix][None]).sum(-1)/whole_prefix.clamp_min(1e-30)
                    reuse=[]
                    for frame,(fq,fk,fv),fp in zip(future,future_tensors,future_p):
                        next_out,fps,next_mass=output(fp,fv,support);next_dense=fp@fv
                        next_bias=next_out-next_dense
                        actual_retained=(fp[:,:prefix]*support[:prefix][None]).sum(-1)/fp[:,:prefix].sum(-1).clamp_min(1e-30)
                        bound,radius=retained_bound(anchor_retained,fq-q,k[:prefix],scale,all_kept=bool(mask.all()))
                        reuse.append(dict(age=int(frame['call'])-int(anchor['call']),
                            retained_prefix_mass_min=float(actual_retained.min()),bound_min=float(bound.min()),
                            bound_violation=float((bound-actual_retained).max()),query_drift_l2_max=float((fq-q).norm(dim=-1).max()),
                            output_bias_relative_max=float((next_bias.norm(dim=-1)/next_dense.norm(dim=-1).clamp_min(1e-12)).max()),
                            bias_motion_relative_max=float(((next_bias-bias).norm(dim=-1)/norm).max()),
                            conservative_radius_max=float(radius.max()),whole_deleted_mass_max=float((1-next_mass).max())))
                    relative_bias=float((bias.norm(dim=-1)/norm).max());max_alpha=float(alpha.max())
                    records.append(dict(request=state['request'],split=state['split'],phase=state['phase'],
                        dataset=anchor['dataset'],head=head,layer=29,canvas=0,call=int(anchor['call']),
                        sampled_rows=len(rows),query_block=0,prefix_tokens=prefix,prefix_blocks=blocks,
                        budget_fraction=fraction,kept_prefix_blocks=budget,eligible_physical_tiles=math.ceil(len(k)/64),
                        kept_physical_tiles=budget+math.ceil(len(k)/64)-blocks,candidate=label,
                        whole_deleted_mass_max=max_alpha,prefix_deleted_mass_max=float((removed_prefix/whole_prefix.clamp_min(1e-30)).max()),
                        current_canvas_deleted_mass=0.,canvas_amplification_max=float((1/mass).max()),
                        renormalization_identity_max_abs=renorm_error,output_bias_relative_max=relative_bias,
                        response_gain=gains,finite_difference_relative_error=fd_error,
                        finite_difference_absolute_error=fd_absolute,finite_difference_gate=fd_gate,
                        support_sha256=hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest(),
                        support_allkept=bool(mask.all()),reuse=reuse,
                        response_guard_feasible=max_alpha<=.2 and relative_bias<=.1 and float((1/mass).max())<=1.25,
                        downstream_logits=None,native_acceptance=None,native_stopping=None,task_outcome=None))
                    attention_evaluations+=3+len(reuse)
        print(json.dumps(dict(completed_state=state['request']+'/'+state['phase'],records=len(records))),flush=True)
    # Matched candidates differ only in support, with strict predeclared tolerances.
    pairs=[]
    for i,a in enumerate(records):
        for b in records[i+1:]:
            keys=('request','phase','head','budget_fraction')
            if any(a[k]!=b[k] for k in keys):continue
            if abs(a['whole_deleted_mass_max']-b['whole_deleted_mass_max'])>.01:continue
            if abs(a['output_bias_relative_max']-b['output_bias_relative_max'])>.1*max(a['output_bias_relative_max'],b['output_bias_relative_max'],1e-12):continue
            ga,gb=a['response_gain']['total'],b['response_gain']['total']
            if abs(ga-gb)<=1e-8:continue
            low,high=(a,b) if ga<gb else (b,a)
            pairs.append(dict(request=a['request'],split=a['split'],dataset=a['dataset'],phase=a['phase'],head=a['head'],
                budget_fraction=a['budget_fraction'],low=low['candidate'],high=high['candidate'],
                response_ratio=low['response_gain']['total']/max(high['response_gain']['total'],1e-12),
                lower_response_has_lower_future_bias_motion=low['reuse'][0]['bias_motion_relative_max']<high['reuse'][0]['bias_motion_relative_max']))
    aggregates={}
    for split in ('development','heldout'):
        for dataset in sorted({r['dataset'] for r in records}):
            selected=[r for r in records if r['split']==split and r['dataset']==dataset]
            matched=[p for p in pairs if p['split']==split and p['dataset']==dataset]
            aggregates[split+'/'+dataset]=dict(records=len(selected),matched_pairs=len(matched),
                matched_pair_concordance=sum(p['lower_response_has_lower_future_bias_motion'] for p in matched)/len(matched) if matched else None,
                response_vs_future_bias_motion_spearman=rank_correlation([r['response_gain']['total'] for r in selected],[r['reuse'][0]['bias_motion_relative_max'] for r in selected]),
                current_error_vs_future_bias_motion_spearman=rank_correlation([r['output_bias_relative_max'] for r in selected],[r['reuse'][0]['bias_motion_relative_max'] for r in selected]),
                feasible_fraction=sum(r['response_guard_feasible'] for r in selected)/len(selected))
    partial_pairs=[]
    for pair in pairs:
        candidates=[r for r in records if r['request']==pair['request'] and r['phase']==pair['phase'] and r['head']==pair['head'] and r['budget_fraction']==pair['budget_fraction']]
        if all(not r['support_allkept'] for r in candidates):partial_pairs.append(pair)
    write(ROOT/'records.json',records);write(ROOT/'matched_pairs.json',pairs)
    summary=dict(protocol_sha256=sha(ROOT/'protocol.json'),records=len(records),aggregates=aggregates,
        finite_difference_relative_error_median=float(np.median([r['finite_difference_relative_error'] for r in records])),
        finite_difference_gate_failures=sum(not r['finite_difference_gate'] for r in records),
        finite_difference_absolute_error_max=max(r['finite_difference_absolute_error'] for r in records),
        unique_support_records=len(records),allkept_records=sum(r['support_allkept'] for r in records),
        renormalization_identity_max_abs=max(r['renormalization_identity_max_abs'] for r in records),
        validity_bound_max_violation=max(u['bound_violation'] for r in records for u in r['reuse']),
        validity_bound_below_half_fraction=sum(u['bound_min']<.5 for r in records for u in r['reuse'])/sum(len(r['reuse']) for r in records),
        partial_support_bound_below_half_fraction=sum(u['bound_min']<.5 for r in records if not r['support_allkept'] for u in r['reuse'])/sum(len(r['reuse']) for r in records if not r['support_allkept']),
        partial_support_bound_checks=sum(len(r['reuse']) for r in records if not r['support_allkept']),
        native_promotion=False,go_no_go='NO_GO: native continuation and held-out logit/correction evidence are absent; operator correlations alone cannot promote a selector',
        native_model_forwards=0,native_rng_draws=0,diagnostic_attention_evaluations=attention_evaluations,
        resource_seconds=time.monotonic()-started,clean_timing=False,
        limitations=['Inherited V31 on-policy states, not new dense census states','One layer, two heads, four sampled rows, first canvas only',
            'Q/K/V changes are observed displacements, not future predictors','No downstream logits, acceptance or task outcome supervision',
            'Pairwise masks are correlated within requests; descriptive only'])
    write(ROOT/'summary.json',summary);write(ROOT/'operator_complete.json',dict(complete=True,protocol_sha256=sha(ROOT/'protocol.json'),summary_sha256=sha(ROOT/'summary.json'),native_promotion=False))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--device',default='cuda');args=parser.parse_args()
    if args.freeze:freeze()
    else:run(args.device)
