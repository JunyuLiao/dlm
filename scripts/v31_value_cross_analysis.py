"""Sanitized two-question implementation/seed cross; not benchmark inference."""
import argparse
import json
from pathlib import Path
import statistics
from scripts.v31_value_pause_analysis import read, sha, require, lines


def mean(rows,field):return statistics.mean(r[field] for r in rows)


def run(study,output):
    require(not output.exists(),'preserve report')
    marker=read(study/'final_complete.json')
    require(marker['protocol_sha256']==sha(study/'protocol.json'),'protocol hash')
    rows=[];bindings={};warnings=[]
    for rel,digest in marker['engines'].items():
        path=study/rel;require(sha(path)==digest,'engine marker changed');receipt=read(path)
        for file,h in receipt['files_sha256'].items():require(sha(path.parent/file)==h,'engine record changed')
        rr=read(path.parent/'scored.json');rows.extend(rr);bindings[rel]=digest
        warm=4;completed=0;jit=0;oom=0
        for line in (path.parent/'private/worker.log').read_text().splitlines():
            oom+='memory allocation failed with OOM' in line
            if completed>=warm and 'JIT compilation during inference' in line:jit+=1
            try:r=json.loads(line)
            except ValueError:continue
            if isinstance(r,dict) and 'denoise_forwards' in r and 'wall_s' in r:completed+=1
        warnings.append(dict(engine=rr[0]['engine'],measured_jit=jit,all_log_allocator_oom=oom))
    require(len(rows)==96,'complete predeclared cross required')
    groups=[]
    for engine in range(1,5):
        for index in (0,49):
            for seed in (1,42):
                group={r['label']:r for r in rows if r['engine']==engine and r['index']==index and r['panel_seed']==seed}
                require(len(group)==6,'missing group')
                c=[group['control_before'],group['control_after']]
                comparisons={}
                for method in ('full_swap8','residual_direction_guard'):
                    ref,dev=[group[b+'_'+method] for b in ('reference','device_exact')]
                    comparisons[method]=dict(reference_device_hash_equal=ref['output_hash']==dev['output_hash'],
                        reference_device_N_delta=dev['denoise_forwards']-ref['denoise_forwards'],
                        reference_device_score_delta=dev['score']-ref['score'],
                        reference_N=ref['denoise_forwards'],device_N=dev['denoise_forwards'],
                        device_minus_control_N=dev['denoise_forwards']-mean(c,'denoise_forwards'),
                        reference_minus_control_N=ref['denoise_forwards']-mean(c,'denoise_forwards'))
                groups.append(dict(engine=engine,pass_name=c[0]['pass_name'],index=index,panel_seed=seed,
                    control_N=mean(c,'denoise_forwards'),control_score=mean(c,'score'),
                    control_hash_equal=c[0]['output_hash']==c[1]['output_hash'],methods=comparisons))
    aggregates={}
    for pass_name in ('clean','audit'):
        aggregates[pass_name]={}
        for seed in (1,42):
            subset=[r for r in rows if r['pass_name']==pass_name and r['panel_seed']==seed]
            labels=['control_mean']+sorted({r['label'] for r in subset if r['backend']!='control'})
            aggregates[pass_name][str(seed)]={}
            for label in labels:
                rr=[r for r in subset if (r['backend']=='control' if label=='control_mean' else r['label']==label)]
                aggregates[pass_name][str(seed)][label]={k:mean(rr,k) for k in ('score','denoise_forwards','canvases','output_tokens','decode_s','wall_s')}
    report=dict(status='complete_diagnostic',scope='Only predeclared questions0/49, seeds1/42; two engines per pass; no benchmark accuracy or timing inference',
        backend_checks={m:dict(pairs=16,hash_disagreements=sum(not g['methods'][m]['reference_device_hash_equal'] for g in groups),
            score_disagreements=sum(g['methods'][m]['reference_device_score_delta']!=0 for g in groups),
            N_disagreements=sum(g['methods'][m]['reference_device_N_delta']!=0 for g in groups)) for m in ('full_swap8','residual_direction_guard')},
        control_bookend_hash_disagreements=sum(not g['control_hash_equal'] for g in groups),groups=groups,aggregates=aggregates,warnings=warnings,
        gpu_reservation_s=marker['gpu_reservation_s'])
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    (output/'final_complete.json').write_text(json.dumps(dict(status='complete',source_sha256=sha(__file__),summary_sha256=sha(output/'summary.json'),engine_bindings=bindings),indent=2)+'\n')
    print(json.dumps(dict(backend_checks=report['backend_checks'],bookend_disagreements=report['control_bookend_hash_disagreements'],aggregates=aggregates),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.study,a.output)
