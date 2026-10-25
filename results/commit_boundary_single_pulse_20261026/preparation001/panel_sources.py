from pathlib import Path
import shutil
R=Path(__file__).resolve().parents[1];OLD=R.parent/'response_behavior_and_stopping_headroom_20261025/track_h/attempt001/code';C=R/'panel/attempt001/code'
for n in ['measure.py']:shutil.copyfile(OLD/n,C/n)
p=C/'bank.py';s=(OLD/'bank.py').read_text().replace('def build(q,k,v,prefix,scale):','def build(q,k,v,prefix,scale,audit=True):')
s=s.replace('position_bias_magnitude=(bias.norm(dim=-1)/o.norm(dim=-1).clamp_min(1e-8)).cpu().tolist()', 'position_bias_magnitude=(bias.norm(dim=-1)/o.norm(dim=-1).clamp_min(1e-8)).cpu().tolist() if audit else None')
s=s.replace("private.append(dict(dense_output=o.detach().cpu(),all_biases=torch.stack(biases).detach().cpu()))", "private.append(dict(dense_output=o.detach().cpu() if audit else None,all_biases=torch.stack(biases)))")
start=s.index('    qualified=all(');end=s.index('    return dict(',start)
s=s[:start]+'''    feasible_common=all(x['current_error'] is not None for x in public)
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
''' +s[end:]
s=s.replace('return dict(qualified=qualified,reason=', 'return dict(feasible_common=feasible_common,qualified=qualified,reason=')
p.write_text(s)
