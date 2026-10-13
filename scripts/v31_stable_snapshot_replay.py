"""Sequential real-Q/K/V replay; all dense work is offline and diagnostic."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from experiments.numerical_qk_reuse.v31_stable_refresh import StableConfig, StableState, mass_scores, stable_topk
from experiments.numerical_qk_reuse.v31_stable_observe import observe, tile_ids
from experiments.numerical_qk_reuse import v27_fa4
from experiments.numerical_qk_reuse.v31_fa4_observe import observe_dense


def main():
    p=argparse.ArgumentParser(); p.add_argument('--study',type=Path,required=True)
    args=p.parse_args(); study=args.study.resolve(); output=study/'operator/real_replay'
    if output.exists(): raise RuntimeError('replay output already exists; preserve it and choose a new study/attempt')
    output.mkdir(parents=True)
    configs={'raw_s2':(StableConfig(mode='raw',hysteresis=0),2,True),
             'hold_no_progress':(StableConfig(mode='raw',updates_enabled=False),0,False),
             **{f'raw_no_progress_s{s}':(StableConfig(mode='raw',hysteresis=0),s,False) for s in (1,4,8)}}
    for li,lam in enumerate([0.,math.log(1.1),math.log(1.25),math.log(1.5)]):
        for gi,guard in enumerate([.005,.01,.025]):
            configs[f'stable_l{li}_g{gi}_s2']=(StableConfig(hysteresis=lam,removal_tolerance=guard),2,True)
    # Probe extension is explicitly offline; later native wiring is conditional.
    configs['stable_probe4_s2']=(StableConfig(hysteresis=math.log(1.25),removal_tolerance=.01,probes=4),2,True)
    configs.update({f'stable_no_progress_s{s}':(StableConfig(),s,False) for s in (1,4,8)})
    binding=json.loads((study/'operator/trajectory_binding.json').read_text())
    trajectory_leaf=study/binding['completed_leaf']
    for name in ('receipt','config'):
        if hashlib.sha256((trajectory_leaf/f'{name}.json').read_bytes()).hexdigest()!=binding[f'{name}_sha256']:
            raise RuntimeError('trajectory source identity mismatch')
    sources=sorted((trajectory_leaf/'private/trajectories').glob('request*/canvas*_call*_layer*.pt'))
    if not sources: raise RuntimeError('real trajectories unavailable')
    groups={}
    for f in sources:
        key=(f.parent.name,f.name.split('_')[0],f.name.split('_')[-1])
        groups.setdefault(key,[]).append(f)
    records=[]; coverage=set(); start=time.perf_counter(); max_stat_error=0.
    for group,files in sorted(groups.items()):
        states={}; pending={}; clocks={}; anchor_age={}; prefix=None
        for f in sorted(files):
            data=torch.load(f,map_location='cpu',weights_only=True)
            coverage.add((data['dataset'],data['layer']))
            if prefix is None:
                prefix=torch.load(f.parent/data['prefix_file'],map_location='cuda',weights_only=True)
            q=data['q'].cuda(); k=torch.cat([prefix['k'],data['tail_k'].cuda()],2); v=torch.cat([prefix['v'],data['tail_v'].cuda()],2)
            n,pt,h=data['n'],data['prefix']//64,q.shape[1]; qb=(n+127)//128
            # Native FA4 full-support statistics and output, exclusively offline.
            dense_z=torch.full((h,qb,pt,128),-math.inf,device='cuda')
            dense=observe_dense(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),data['scale'],dense_z).float()
            z=dense_z.reshape(h*qb,pt,128)
            tail_ids=torch.arange(pt,(k.shape[2]+63)//64,device='cuda')[None].expand(h*qb,-1).contiguous()
            tail=observe(q,k,data['scale'],tail_ids,n).logsumexp(1)
            dense_norm=torch.logaddexp(z.logsumexp(1),tail)
            def fullmask(sel):
                out=torch.ones(1,h,qb,(k.shape[2]+63)//64,dtype=torch.bool,device='cuda')
                out[0,:,:,:pt]=sel.reshape(h,qb,pt); return out
            def output_for(sel):
                return v27_fa4.sparse_lists(q,k,v,v27_fa4.block_sparse_tensors(fullmask(sel)),data['scale']).float()
            for name,(cfg,period,progress) in configs.items():
                selected_event=data['selection_event'] and (progress or name not in states)
                if selected_event:
                    current=(stable_topk(mass_scores(z)+1.386*states[name].selected,states[name].k)
                             if name in states else data['next'].cuda()[0,:,:,:pt].reshape(h*qb,pt))
                    states[name]=StableState(z,current,cfg); clocks[name]=0; anchor_age[name]=0; pending.pop(name,None)
                    continue
                if name not in states: continue
                state=states[name]; anchor_age[name]+=1; clocks[name]+=1
                before=state.selected.clone()
                used=output_for(before)
                valid=(torch.arange(qb*128,device='cuda').reshape(qb,128)<n).expand(h,-1,-1).reshape(h*qb,128)
                rowerr=(used-dense).norm(dim=-1).reshape(-1)
                rel=rowerr/dense.norm(dim=-1).reshape(-1).clamp_min(1e-8)
                kept_log=torch.logaddexp(z.masked_fill(~before[:,:,None],-math.inf).logsumexp(1),tail)
                true_mass=(kept_log-dense_norm).exp()[valid]
                immediate=0.; realized=0.; predicted=None
                if name in pending:
                    old,predicted=pending[name]
                    held_output=output_for(old)
                    immediate=float((used-held_output).norm(dim=-1).amax())
                    oldmass=(torch.logaddexp(z.masked_fill(~old[:,:,None],-math.inf).logsumexp(1),tail)-dense_norm).exp()[valid]
                    realized=float((true_mass-oldmass).mean())
                due=period>0 and clocks[name]%period==0 and state.k<pt
                swaps=0; rejected=0; probe_work=0; selector_ms=0.
                outside_err=0.
                outside=~before[:,:,None]&torch.isfinite(state.anchor)&torch.isfinite(z)
                if outside.any(): outside_err=float((z-state.anchor).abs()[outside].mean())
                if due:
                    # Actual partial-observation parity, independent of the offline dense evaluator.
                    selected_ids=tile_ids(before)
                    fresh_part=observe(q,k,data['scale'],selected_ids,n)
                    dense_part=z.gather(1,selected_ids[:,:,None].expand(-1,-1,128))
                    delta=torch.nan_to_num((fresh_part-dense_part).abs()).amax()
                    max_stat_error=max(max_stat_error,float(delta))
                    fresh=torch.full_like(z,-math.inf)
                    fresh.scatter_(1,selected_ids[:,:,None].expand(-1,-1,128),fresh_part)
                    probe=None
                    if cfg.probes:
                        score=mass_scores(torch.where(before[:,:,None],fresh,state.anchor)).masked_fill(before,-math.inf)
                        probe_ids=score.argsort(dim=-1,descending=True,stable=True)[:,:min(cfg.probes,pt-state.k)]
                        probe=torch.zeros_like(before).scatter_(1,probe_ids,True)
                        fresh.scatter_(1,probe_ids[:,:,None].expand(-1,-1,128),observe(q,k,data['scale'],probe_ids,n))
                        probe_work=probe_ids.numel()
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); a.record()
                    selected,info=state.observe(fresh,tail,probe); b.record(); b.synchronize(); selector_ms=a.elapsed_time(b)
                    swaps=int(info['swaps'].sum()); rejected=int(info['rejected_guard'].sum())
                    hybrid=torch.where((before|probe if probe is not None else before)[:,:,None],fresh,state.anchor)
                    predict_norm=torch.logaddexp(hybrid.logsumexp(1),tail)
                    pnew=(torch.logaddexp(hybrid.masked_fill(~selected[:,:,None],-math.inf).logsumexp(1),tail)-predict_norm).exp()[valid]
                    pold=(torch.logaddexp(hybrid.masked_fill(~before[:,:,None],-math.inf).logsumexp(1),tail)-predict_norm).exp()[valid]
                    pending[name]=(before,float((pnew-pold).mean()))
                else: pending.pop(name,None)
                records.append(dict(arm=name,dataset=data['dataset'],layer=data['layer'],canvas=data['canvas'],call=data['call'],anchor_age=anchor_age[name],
                    true_retained_mass_mean=float(true_mass.mean()),row_error_mean=float(rowerr.mean()),row_error_p95=float(torch.quantile(rowerr,.95)),row_error_max=float(rowerr.max()),relative_row_error_mean=float(rel.mean()),
                    immediate_output_change_max=immediate,realized_mass_change=realized,predicted_mass_change=predicted,
                    observation=due,swaps=swaps,cumulative_max=int(state.cumulative.max()),guard_rejections=rejected,
                    iou_anchor=float((before&state.initial).sum()/((before|state.initial).sum().clamp_min(1))),
                    outside_logmass_prediction_error=outside_err,selector_ms=selector_ms,probe_rectangles=probe_work,
                    selector_rectangles=(before.sum().item()+tail_ids.numel()) if due else 0))
            del q,k,v,z,dense
        print(json.dumps({'group':list(group),'completed_snapshots':len(files),'records':len(records)}),flush=True)
    (output/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    aggregates={}
    for name in configs:
        rr=[r for r in records if r['arm']==name]
        aggregates[name]={k:sum(r[k] for r in rr)/max(1,len(rr)) for k in ('true_retained_mass_mean','row_error_p95','immediate_output_change_max','outside_logmass_prediction_error','selector_ms')}
        aggregates[name].update(records=len(rr),swaps=sum(r['swaps'] for r in rr),max_cumulative=max((r['cumulative_max'] for r in rr),default=0),
            false_admissions=sum(r['predicted_mass_change'] is not None and r['predicted_mass_change']>0 and r['realized_mass_change']<0 for r in rr))
    raw=aggregates['raw_s2']; hold=aggregates['hold_no_progress']
    acceptable=[(name,x) for name,x in aggregates.items() if name.startswith('stable_l') and
        x['immediate_output_change_max']<=.5*raw['immediate_output_change_max'] and x['true_retained_mass_mean']>=hold['true_retained_mass_mean']]
    selected=(max(acceptable,key=lambda p:(p[1]['true_retained_mass_mean'],-p[1]['swaps']))[0] if acceptable else 'stable_l3_g0_s2')
    cfg=configs[selected][0]
    probe=aggregates['stable_probe4_s2']; reference=aggregates['stable_l2_g1_s2']
    probe_good=(selected=='stable_l2_g1_s2' and reference['false_admissions']>0 and probe['false_admissions']<=.5*reference['false_admissions']
                and probe['immediate_output_change_max']<=.8*reference['immediate_output_change_max'])
    parameters={'schema':'operator_only_parameter_selection_v1','selector':vars(cfg),'selected_grid_arm':selected,'screen_passed':bool(acceptable),
        'selection_data':'real frozen Q/K/V; no target answers','conditional_probe_promoted':probe_good,
        'probe_gate':'at least 50% fewer false admissions and 20% lower output perturbation than equal-config stable reference'}
    (study/'specs/parameters.json').write_text(json.dumps(parameters,indent=2)+'\n')
    summary={'schema':'real_sequential_operator_replay_v1','coverage':sorted(coverage),'snapshots':len(sources),'groups':len(groups),
        'max_retained_statistic_error':max_stat_error,'selector_grid':aggregates,'parameters':parameters,'gpu_seconds':time.perf_counter()-start,
        'snapshot_sha256':{str(f.relative_to(study)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},'offline_dense_only':True}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if max_stat_error>3e-4: raise RuntimeError('retained statistic parity failed')
    expected={(ds,l) for ds in ('longbench_v2_0shot_think','aime26') for l in (5,11,17,23,29)}
    if not expected.issubset(coverage): raise RuntimeError('workload/layer coverage incomplete')
    (output/'final_complete.json').write_text(json.dumps({'status':'complete','snapshots':len(sources),'records':len(records)})+'\n')


if __name__=='__main__': main()
