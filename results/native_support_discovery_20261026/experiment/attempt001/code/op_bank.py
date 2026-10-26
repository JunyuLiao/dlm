"""Finite CURRENT-operator bank; no model/suffix forward or native RNG draws."""
from collections import Counter
import hashlib,itertools,json,math
import numpy as np
import torch

UNITS=[(0,0),(0,1),(8,0),(8,1)]

def support_hash(units):
    return hashlib.sha256(json.dumps(units,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def budget(blocks,fraction):return max(1,math.floor(fraction*blocks))

def matched(a,b):
    return abs(a['current_error']-b['current_error'])/max(a['current_error'],b['current_error'],1e-6)<=.05 and abs(a['deleted_mass']-b['deleted_mass'])<=.01

def bias_metrics(sparse,dense,math_bias=None):
    """Actual applied consumer difference, distinct from total-output GQA error."""
    b=sparse.float()-dense.float();den=dense.float();n=float(b.norm());base=float(den.norm());floor=max(1e-12,base*1e-6)
    out=dict(current_error=float((b.norm(dim=-1)/den.norm(dim=-1).clamp_min(1e-8)).max()),bias_Frobenius=n,
        bias_relative_Frobenius=n/max(base,1e-12),bias_direction_available=n>=floor,
        position_relative_bias=(b.norm(dim=-1)/den.norm(dim=-1).clamp_min(1e-8)).cpu().tolist(),nonfinite=int((~torch.isfinite(b)).sum()))
    if math_bias is not None:
        ref=math_bias.float();rn=float(ref.norm());valid=n>=floor and rn>=floor
        out.update(FP32_bias_Frobenius=rn,actual_vs_FP32_cosine=float(torch.nn.functional.cosine_similarity(b.flatten(),ref.flatten(),dim=0)) if valid else None,
            actual_vs_FP32_relative_residual=float((b-ref).norm())/max(rn,floor),roundoff_or_tiny_bias=not valid)
    return out,b

def primitive_proposals(p,v,blocks,drop,probe,seed,limit=64):
    """Mass heuristics, values, independent sets and substantial swaps."""
    proposals=[];seen=set()
    def add(x):
        z=tuple(sorted(set(int(i) for i in x)))
        if len(z)==drop and z not in seen and min(z)>=0 and max(z)<blocks:seen.add(z);proposals.append(z)
    if math.comb(blocks,drop)<=limit:
        return list(itertools.combinations(range(blocks),drop))
    # Sum of per-key row maxima is a proposal upper bound, not deleted mass.
    mass=p[:,:blocks*64].amax(0).reshape(blocks,64).sum(1)
    value=(v[:blocks*64].float()@probe.mean(0)).reshape(blocks,64).mean(1)
    low=torch.argsort(mass,stable=True).cpu().tolist();signed=torch.argsort(value,stable=True).cpu().tolist()
    add(low[:drop]);add(range(drop));add(range(blocks-drop,blocks));add(np.linspace(0,blocks-1,drop,dtype=int))
    add(signed[:drop]);add(signed[-drop:]);add(range(max(0,(blocks-drop)//2),max(0,(blocks-drop)//2)+drop))
    rng=np.random.default_rng(seed)
    for step in range(3*limit):
        if len(proposals)>=limit:break
        if step%3==0:
            base=low[:drop];amount=max(1,(drop+1)//2);retained=list(rng.choice(base,size=drop-amount,replace=False)) if drop>amount else []
            outside=[i for i in range(blocks) if i not in retained];add(retained+list(rng.choice(outside,size=amount,replace=False)))
        else:add(rng.choice(blocks,size=drop,replace=False))
    return proposals

def gather_units(output):
    return torch.stack([output[qb*128:(qb+1)*128,h] for h,qb in UNITS])

def make_probe(model,layer,prior,current_step):
    """Approximate fixed previous-pair linear readout, no model invocation."""
    weight=model.model.layers[layer].self_attn.o_proj.weight.detach()
    head=model.lm_head.weight.detach()
    assert weight.shape==(2816,8192) and head.shape[1]==2816
    device=weight.device
    a=prior['winner'].to(device);b=prior['competitor'].to(device)
    wa=head.index_select(0,a).float();wb=head.index_select(0,b).float()
    previous_temperature=.4+.4*(48-(current_step-2))/48
    temperature=.4+.4*(48-(current_step-1))/48
    cap=getattr(model,'final_logit_softcapping',None)
    if cap is not None:
        pa=(1-(prior['winner_logit'].to(device).float()*previous_temperature/float(cap)).square()).clamp(0,1)
        pb=(1-(prior['competitor_logit'].to(device).float()*previous_temperature/float(cap)).square()).clamp(0,1)
        wa=wa*pa[:,None];wb=wb*pb[:,None]
    difference=(wa-wb)/temperature
    probes=torch.stack([difference[qb*128:(qb+1)*128]@weight[:,h*512:(h+1)*512].float() for h,qb in UNITS])
    entropy=prior['entropy'].to(device).float();entropy=entropy/entropy.sum().clamp_min(1e-12)
    e=torch.stack([entropy[qb*128:(qb+1)*128] for h,qb in UNITS])
    stable=torch.stack([prior['stable_positions'][qb*128:(qb+1)*128] for h,qb in UNITS]).to(device)
    base=torch.stack([prior['margin'][qb*128:(qb+1)*128] for h,qb in UNITS]).to(device)*previous_temperature/temperature
    return probes,e,stable,base,dict(linear_projection_multiply_adds=4*128*2816*512,head_weight_rows=512,temperature=temperature,previous_temperature=previous_temperature,
        omitted_Jacobians=['post_attention_RMS','MLP_MoE','final_RMS','remaining_layers'],uses_current_downstream_logits=False)

def build(q,k,v,prefix,scale,fraction,probe,entropy_weight,protected,base_margin,native_operator,identity):
    blocks=prefix//64;drop=budget(blocks,fraction);assert blocks>=4
    work=Counter();primitive=[];math_private=[];whole=math.ceil(len(k)/64)
    for unit,(h,qb) in enumerate(UNITS):
        qu=q[qb*128:(qb+1)*128,h].float();ku=k[:,h//8].float();vu=v[:,h//8].float()
        p=(qu@ku.T*scale).softmax(-1);o=p@vu
        seed=int(hashlib.sha256(f'new-discovery-v1|{identity}|{drop}|{h}|{qb}'.encode()).hexdigest()[:16],16)
        proposals=primitive_proposals(p,vu,blocks,drop,probe[unit],seed)
        work.update(QK_rectangles=whole,PV_rectangles=whole,value_projection_key_rows=blocks*64)
        metrics=[];biases=[];masses=[]
        for tiles in proposals:
            ids=(torch.tensor(tiles,device=q.device)[:,None]*64+torch.arange(64,device=q.device)[None]).flatten()
            mass=p[:,ids].sum(-1);removed=p[:,ids]@vu[ids];bias=(mass[:,None]*o-removed)/(1-mass).clamp_min(1e-12)[:,None]
            error=float((bias.norm(dim=-1)/o.norm(dim=-1).clamp_min(1e-8)).max());maximum=float(mass.max())
            metrics.append(dict(tiles=list(tiles),current_error=error,deleted_mass=maximum,feasible=error<=.1 and maximum<=.2,
                projected_gain=float(((bias*probe[unit]).sum(-1)*entropy_weight[unit]).sum()),mean_tile=float(np.mean(tiles))))
            biases.append(bias);masses.append(mass);work['PV_rectangles']+=drop
        primitive.append(dict(head=h,qb=qb,blocks=blocks,drop_count=drop,candidates=metrics))
        math_private.append(dict(dense=o,biases=biases,masses=masses,prefix_mass=p[:,:blocks*64].sum(-1)))
    feasible=[[i for i,x in enumerate(u['candidates']) if x['feasible']] for u in primitive]
    if not all(feasible):
        return dict(blocks=blocks,drop_count=drop,primitive=primitive,candidates=[],selections={},screen=[],reason='no_four_unit_math_feasible_support',diagnostic_work=dict(work)),dict(),dict(work)
    ce=[min(ids,key=lambda i:(primitive[u]['candidates'][i]['current_error'],primitive[u]['candidates'][i]['tiles'])) for u,ids in enumerate(feasible)]
    composites=[];seen=set()
    def add(indices,label):
        z=tuple(indices)
        if z not in seen and len(composites)<24:seen.add(z);composites.append((z,label))
    add(ce,'unit_error_minimum')
    for key,reverse,label in [('deleted_mass',False,'unit_mass_minimum'),('projected_gain',True,'value_readout_positive'),('projected_gain',False,'value_readout_negative'),('mean_tile',True,'near'),('mean_tile',False,'far')]:
        add([min(ids,key=lambda i:((-1 if reverse else 1)*primitive[u]['candidates'][i][key],primitive[u]['candidates'][i]['tiles'])) for u,ids in enumerate(feasible)],label)
    ce_error=max(primitive[u]['candidates'][i]['current_error'] for u,i in enumerate(ce));ce_mass=max(primitive[u]['candidates'][i]['deleted_mass'] for u,i in enumerate(ce))
    narrow=[[i for i in ids if primitive[u]['candidates'][i]['current_error']<=ce_error+.05*max(ce_error,1e-6) and primitive[u]['candidates'][i]['deleted_mass']<=ce_mass+.01] for u,ids in enumerate(feasible)]
    rng=np.random.default_rng(int(hashlib.sha256(f'composite-v1|{identity}|{drop}'.encode()).hexdigest()[:16],16))
    for _ in range(8):add([int(rng.choice(ids)) for ids in narrow],'CE_error_band_mix')
    for _ in range(128):
        if len(composites)>=24:break
        add([int(rng.choice(ids)) for ids in feasible],'independent_feasible_mix')
    dense=native_operator(None);repeat=native_operator(None)
    assert torch.equal(dense,repeat),'all-kept operator repeat changed'
    work['native_A_QK_rectangles']+=2*32*whole;work['native_A_PV_rectangles']+=2*32*whole
    a=gather_units(dense);records=[];vectors=[];private_candidates=[]
    for indices,label in composites:
        units=[dict(head=h,qb=qb,tiles=primitive[u]['candidates'][idx]['tiles']) for u,((h,qb),idx) in enumerate(zip(UNITS,indices))]
        sparse=native_operator(units);work['native_S_QK_rectangles']+=32*whole-4*drop;work['native_S_PV_rectangles']+=32*whole-4*drop
        unaffected=sparse.clone();reference=dense.clone()
        for h,qb in UNITS:unaffected[qb*128:(qb+1)*128,h]=0;reference[qb*128:(qb+1)*128,h]=0
        assert torch.equal(unaffected,reference),'unselected heads changed in diagnostic consumer'
        mb=torch.stack([math_private[u]['biases'][i] for u,i in enumerate(indices)])
        masses=torch.stack([math_private[u]['masses'][i] for u,i in enumerate(indices)])
        prefix_attention_mass=torch.stack([u['prefix_mass'] for u in math_private])
        measured,bias=bias_metrics(gather_units(sparse),a,mb);mass=float(masses.max());math_error=max(primitive[u]['candidates'][i]['current_error'] for u,i in enumerate(indices))
        projected_units=(bias*probe).sum(-1)
        projected=torch.cat([projected_units[0]+projected_units[2],projected_units[1]+projected_units[3]])
        base=torch.cat([base_margin[0],base_margin[1]]);protect=torch.cat([protected[0],protected[1]])
        ew=torch.cat([entropy_weight[0],entropy_weight[1]])
        crossings=int(((base+projected<0)&protect).sum())
        FP32_dense=torch.stack([u['dense'] for u in math_private]);reference_S=FP32_dense+mb
        GQA=float(((gather_units(sparse).float()-reference_S).norm(dim=-1)/reference_S.norm(dim=-1).clamp_min(1e-8)).max())
        assert GQA<=.03,'Same-support GQA/consumer total-output error'
        rec=dict(units=units,support_sha256=support_hash(units),proposal_family=label,source_indices=list(indices),**measured,deleted_mass=mass,
            deleted_mass_by_position=masses.cpu().tolist(),math_current_error=math_error,feasible=measured['current_error']<=.1 and mass<=.2 and math_error<=.1,
            prefix_only_deleted_mass_max=float((masses/prefix_attention_mass.clamp_min(1e-12)).max()),renormalization_max=float((1/(1-masses)).max()),removed_current_canvas_mass=0.,
            projected_stable_crossings=crossings,projected_entropy_weighted_gain=float((projected*ew).sum()),projected_margin_change_by_position=projected.cpu().tolist(),GQA_error=GQA,
            bias_entropy_concentration=float((bias.norm(dim=-1)*entropy_weight).sum()/bias.norm(dim=-1).sum().clamp_min(1e-12)),
            mean_tile=float(np.mean([t for u in units for t in u['tiles']])))
        assert rec['nonfinite']==0
        records.append(rec);vectors.append(bias.flatten());private_candidates.append(dict(sparse=gather_units(sparse).detach().cpu(),actual_bias=bias.detach().cpu(),FP32_bias=mb.detach().cpu(),mass=masses.detach().cpu()))
        work['candidate_readout_multiply_adds']+=4*128*q.shape[-1]
    feasible=[i for i,r in enumerate(records) if r['feasible']];selection={};screen=[]
    if feasible:
        ce=min(feasible,key=lambda i:(records[i]['current_error'],records[i]['support_sha256']))
        same=[i for i in feasible if matched(records[i],records[ce])]
        alternatives=[i for i in same if i!=ce]
        sham=min(alternatives or [ce],key=lambda i:hashlib.sha256(('support-sham-v1|'+identity+'|'+records[i]['support_sha256']).encode()).hexdigest())
        readout=min(same,key=lambda i:(records[i]['projected_stable_crossings'],-records[i]['projected_entropy_weighted_gain'],records[i]['current_error'],records[i]['support_sha256']))
        selection=dict(current_error=ce,mass=min(feasible,key=lambda i:(records[i]['deleted_mass'],records[i]['support_sha256'])),support_sham=sham,
            near=min(feasible,key=lambda i:(-records[i]['mean_tile'],records[i]['support_sha256'])),far=min(feasible,key=lambda i:(records[i]['mean_tile'],records[i]['support_sha256'])),readout_margin=readout)
        for name,idx in selection.items():
            if idx not in screen:screen.append(idx)
        while len(screen)<min(8,len(feasible)):
            remaining=[i for i in feasible if i not in screen]
            def distance(i):
                if not records[i]['bias_direction_available']:return 0.
                values=[1-float(torch.nn.functional.cosine_similarity(vectors[i],vectors[j],dim=0)) for j in screen if records[j]['bias_direction_available']]
                return min(values) if values else 0.
            idx=min(remaining,key=lambda i:(-distance(i),records[i]['support_sha256']));screen.append(idx)
    result=dict(blocks=blocks,drop_count=drop,primitive=primitive,candidates=records,selections=selection,screen=screen,
        reason=None if feasible else 'no_actual_consumer_feasible_support',diagnostic_work=dict(work),current_QKV_only=True,
        scope=dict(layer=29,heads=[0,8],queryblocks=[0,1],Q=128,KV=64),mass_guard=.2,error_guard=.1,protected_current_canvas_deleted_mass=0.,
        tiny_direction_floor_relative=1e-6,allkept_repeat_exact=True,previous_protected_positions=int(torch.cat([protected[0],protected[1]]).sum()),selection_uses_future=False)
    private=dict(dense=a.detach().cpu(),FP32_dense=FP32_dense.detach().cpu(),prefix_attention_mass=prefix_attention_mass.detach().cpu(),candidates=private_candidates,readout_probe=probe.detach().cpu(),entropy_weight=entropy_weight.detach().cpu(),protected=protected.detach().cpu(),base_margin=base_margin.detach().cpu())
    return result,private,dict(work)
