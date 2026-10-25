"""Current-only signed-bias bank for ONE pulse; no attention-motion objective."""
import hashlib
import math
import torch
from collections import Counter
import importlib.util
from pathlib import Path

REPO=Path(__file__).resolve().parents[5]
SPEC=importlib.util.spec_from_file_location('a1_proposals',REPO/'results/denoising_mechanism_stage_a_20261025/attempt002/code/operators.py')
PROPOSALS=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(PROPOSALS)


def matching(a,b):
    return abs(a['current_error']-b['current_error'])/max(a['current_error'],b['current_error'],1e-6)<=.05 and abs(a['deleted_mass']-b['deleted_mass'])<=.01


def build(q,k,v,prefix,scale,audit=True):
    work=Counter();public=[];private=[];best_pair=None
    blocks=prefix//64;drop=max(1,math.floor(.05*blocks));assert blocks>=4
    for h in [0,8]:
        for qb in [0,1]:
            qu=q[qb*128:(qb+1)*128,h].float();ku=k[:,h//8].float();vu=v[:,h//8].float()
            p=(qu@ku.T*scale).softmax(-1);o=p@vu;whole=math.ceil(len(ku)/64)
            work.update(QK_rectangles=whole,PV_rectangles=whole)
            # Reuse A1's frozen current-only diverse proposal function, with its
            # per-key-row-max upper-bound mass heuristic explicitly labeled.
            candidates=PROPOSALS.support_bank(p,vu,blocks,drop,h*2+qb,48)
            if math.comb(blocks,drop)>48:work.update(PV_rectangles=whole,value_projection_key_rows=blocks*64)
            records=[];biases=[]
            for tiles in candidates:
                idx=(torch.tensor(tiles,device=q.device)[:,None]*64+torch.arange(64,device=q.device)[None]).flatten()
                mass=p[:,idx].sum(-1);contribution=p[:,idx]@vu[idx];bias=(mass[:,None]*o-contribution)/(1-mass).clamp_min(1e-12)[:,None]
                work.update(PV_rectangles=drop)
                error=float((bias.norm(dim=-1)/o.norm(dim=-1).clamp_min(1e-8)).max());removed=float(mass.max())
                records.append(dict(tiles=list(tiles),current_error=error,deleted_mass=removed,feasible=error<=.1 and removed<=.2,
                    bias_relative_Frobenius=float(bias.norm()/o.norm().clamp_min(1e-8)),
                    signed_output_alignment=float((bias*o).sum()/o.square().sum().clamp_min(1e-16)),
                    position_bias_magnitude=(bias.norm(dim=-1)/o.norm(dim=-1).clamp_min(1e-8)).cpu().tolist() if audit else None))
                biases.append(bias)
            feasible=[i for i,x in enumerate(records) if x['feasible']]
            ce=min(feasible,key=lambda i:(records[i]['current_error'],records[i]['tiles'])) if feasible else None
            sham_pool=[i for i in feasible if ce is not None and matching(records[i],records[ce])]
            sham=min(sham_pool,key=lambda i:hashlib.sha256(f'H20261025|{h}|{qb}|{records[i]["tiles"]}'.encode()).digest()) if sham_pool else None
            normalized=torch.stack([x.flatten()/x.norm().clamp_min(1e-12) for x in biases]);cos=normalized@normalized.T
            work['bias_vector_inner_products']+=len(biases)**2
            pairs=[]
            for i in feasible:
                for j in feasible:
                    if i>=j or not matching(records[i],records[j]):continue
                    value=float(cos[i,j]);nonzero=min(records[i]['bias_relative_Frobenius'],records[j]['bias_relative_Frobenius'])>=1e-6
                    pairs.append(dict(a=i,b=j,cosine=value,nonzero=nonzero))
                    key=(value,h,qb,records[i]['tiles'],records[j]['tiles'])
                    if nonzero and value<=.5 and (best_pair is None or key<best_pair[0]):best_pair=(key,len(public),i,j)
            public.append(dict(head=h,qb=qb,blocks=blocks,drop_count=drop,candidates=records,feasible=len(feasible),current_error=ce,support_sham=sham,matched_pairs=pairs))
            private.append(dict(dense_output=o.detach().cpu() if audit else None,all_biases=torch.stack(biases)))
    feasible_common=all(x['current_error'] is not None for x in public)
    qualified=feasible_common and best_pair is not None
    selections={};direction=None;indices={}
    if feasible_common:
        base=[x['current_error'] for x in public];sham=[x['support_sham'] for x in public]
        indices=dict(current_error=base,support_sham=sham)
        if qualified:
            _,unit,i,j=best_pair;ra,rb=public[unit]['candidates'][i],public[unit]['candidates'][j]
            hi,lo=(i,j) if (ra['signed_output_alignment'],ra['tiles'])>(rb['signed_output_alignment'],rb['tiles']) else (j,i)
            indices.update(direction_high=[hi if u==unit else x for u,x in enumerate(base)],direction_low=[lo if u==unit else x for u,x in enumerate(base)])
        for name,chosen in indices.items():
            units=[dict(head=row['head'],qb=row['qb'],tiles=row['candidates'][index]['tiles']) for row,index in zip(public,chosen)]
            selections[name]=dict(units=units,source_indices=chosen,support_sha256=hashlib.sha256(str(units).encode()).hexdigest(),
                current_error=max(row['candidates'][index]['current_error'] for row,index in zip(public,chosen)),
                deleted_mass=max(row['candidates'][index]['deleted_mass'] for row,index in zip(public,chosen)))
        assert matching(selections['current_error'],selections['support_sham'])
        if qualified:
            assert matching(selections['direction_high'],selections['direction_low'])
            if audit:
                full={name:torch.cat([private[u]['all_biases'][index].cpu().flatten() for u,index in enumerate(chosen)]) for name,chosen in indices.items()}
                whole_cosine=float(torch.nn.functional.cosine_similarity(full['direction_high'],full['direction_low'],dim=0))
            else:whole_cosine=None
            direction=dict(unit=unit,head=public[unit]['head'],qb=public[unit]['qb'],a=i,b=j,cosine=best_pair[0][0],
                per_unit_error_mass_matched=True,whole_support_error_mass_matched=True,whole_bias_cosine=whole_cosine,other_units_fixed_to_current_error=True,
                high_vs_current_error_matched=matching(selections['direction_high'],selections['current_error']),
                low_vs_current_error_matched=matching(selections['direction_low'],selections['current_error']))
    if audit:private=[{k:v.detach().cpu() for k,v in row.items()} for row in private]
    else:private=[]
    return dict(feasible_common=feasible_common,qualified=qualified,reason=None if qualified else 'no_feasible_four_unit_support_or_no_matched_nonzero_cosine_at_most_0.5_pair',
        prefix=prefix,blocks=blocks,drop_count=drop,units=public,selections=selections,directional_contrast=direction,
        method='signed_bias_direction_pulse_not_A1_response',proposal_mass_semantics='sum of per-key row maxima, a proposal upper bound; exact row deleted mass guards'),private,dict(work)
