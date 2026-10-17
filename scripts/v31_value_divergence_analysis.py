"""CPU summaries of bounded private first-divergence traces."""
import argparse
import json
from pathlib import Path
import torch
from scripts import v31_official as official
from scripts.v31_value_pause_analysis import read, sha, require, lines


def equal(a,b):
    if a is None or b is None:return a is b
    return torch.equal(a,b)


def differences(a,b):
    if a is None or b is None:return None
    if a.shape!=b.shape:return None
    return int((a!=b).sum())


def run(study,cross,output):
    require(not output.exists(),'preserve report')
    marker=read(study/'final_complete.json')
    for rel,digest in marker['files_sha256'].items():require(sha(study/rel)==digest,'trace binding')
    require(marker['protocol_sha256']==sha(study/'protocol.json'),'protocol binding')
    protocol=read(study/'protocol.json');labels=protocol['trace_labels']
    first=protocol['trace_first_request']
    frames={i:[torch.load(p,map_location='cpu',weights_only=True) for p in sorted((study/f'private/trace/request{i:03d}').glob('call*.pt'))] for i in range(first,first+4)}
    comparisons={}
    pairs=[(first,first+1,labels[1]),(first,first+2,labels[2]),(first,first+3,'control_after')]
    if protocol.get('backend_localization'):pairs.append((first+1,first+2,'reference_vs_device_guard'))
    for baseline,i,label in pairs:
        rows=[]
        for call,(a,b) in enumerate(zip(frames[baseline],frames[i])):
            aa,bb=a['attention'],b['attention']
            require(aa is not None and bb is not None,'missing native attention trace')
            selected=[aa['selected_support'],bb['selected_support']]
            support_diff=differences(*selected)
            output_delta=(aa['output'].float()-bb['output'].float()).abs() if aa['output'] is not None and bb['output'] is not None else None
            r=dict(call=call,query_equal=aa['q_hash']==bb['q_hash'],observed=[aa['observed'],bb['observed']],
                canvas_ids=[aa['canvas'],bb['canvas']],prefix_tokens=[aa['prefix'],bb['prefix']],
                native_steps_before=[int(a['before']['step'][0]),int(b['before']['step'][0])],
                prior_support_equal=equal(aa['prior_support'],bb['prior_support']),selected_support_equal=equal(*selected),selected_support_differences=support_diff,
                layer29_output_equal=aa['output_hash']==bb['output_hash'],layer29_output_max_abs=float(output_delta.max()) if output_delta is not None else None,
                logits_equal=a['before']['logits_hash']==b['before']['logits_hash'] if a['before']['logits_hash'] is not None and b['before']['logits_hash'] is not None else None,
                before_cuda_rng_equal=equal(a['before']['cuda_rng'],b['before']['cuda_rng']),after_cuda_rng_equal=equal(a['after']['cuda_rng'],b['after']['cuda_rng']),
                before_canvas_disagreements=differences(a['before']['canvas'],b['before']['canvas']),
                after_canvas_disagreements=differences(a['after']['canvas'],b['after']['canvas']),
                argmax_disagreements=differences(a['after']['argmax'],b['after']['argmax']),
                reconstructed_acceptance_disagreements=differences(a['after']['reconstructed_acceptance'],b['after']['reconstructed_acceptance']),
                accepted_counts=[int(a['after']['reconstructed_acceptance'].sum()),int(b['after']['reconstructed_acceptance'].sum())],
                mean_entropy=[float(a['after']['entropy'].mean()),float(b['after']['entropy'].mean())],
                commit_next=[bool(a['after']['encoder'].any()),bool(b['after']['encoder'].any())],
                confident=[bool(a['after']['confident'].any()),bool(b['after']['confident'].any())],
                self_conditioning_equal=a['after']['self_conditioning_hash']==b['after']['self_conditioning_hash'])
            if aa.get('cache') is not None and bb.get('cache') is not None:
                r['cache_comparison']={k:dict(equal=equal(aa['cache'][k],bb['cache'][k]),max_abs=float((aa['cache'][k]-bb['cache'][k]).abs().max())) for k in aa['cache']}
            for key in ('k_hash','v_hash'):
                r[key+'_equal']=aa[key]==bb[key] if key in aa and key in bb else None
            rows.append(r)
        comparisons[label]=dict(rows=rows,first_different={k:next((r['call'] for r in rows if r[k] is False),None) for k in
            ('query_equal','selected_support_equal','layer29_output_equal','logits_equal','before_cuda_rng_equal','after_cuda_rng_equal','self_conditioning_equal')},
            first_different_cache=next((r['call'] for r in rows if 'cache_comparison' in r and any(not v['equal'] for v in r['cache_comparison'].values())),None),
            first_changed_commit=next((r['call'] for r in rows if r['commit_next'][0]!=r['commit_next'][1]),None),
            first_changed_canvas=next((r['call'] for r in rows if r['after_canvas_disagreements']),None),
            first_changed_acceptance=next((r['call'] for r in rows if r['reconstructed_acceptance_disagreements']),None))
    public=lines(study/'records.jsonl')[protocol['explicit_warmups']:];private=lines(study/'private/generation.private.jsonl')[protocol['explicit_warmups']:]
    pool=study/'private/pools/longbench_v2';pools=official.Pools(pool,fields=('thinking',));gold=read(pool/'longbench_v2_0shot_think_gold.json');final,_=official.load_final_response()
    cross_rows=[r for p in cross.glob('engine*/attempt001/scored.json') for r in read(p)]
    trials=[]
    for label,r,t in zip(labels,public,private):
        meta,_,verified=official.bind(t,pools);require(verified,'private binding')
        cross_label=label if label.startswith(('control','reference_','device_exact_')) else 'device_exact_'+label
        rr=[x for x in cross_rows if x['index']==0 and x['panel_seed']==protocol['panel_seed'] and x['label']==cross_label]
        trials.append(dict(label=label,score=int(official.lb_extract_answer(final(t['completion'],bool(meta['thinking'])))==gold[t['id']]),
            N=r['denoise_forwards'],C=r['canvases'],T=r['output_tokens'],output_hash=r['output_hash'],
            untraced_cross_comparisons=len(rr),untraced_hash_matches=sum(x['output_hash']==r['output_hash'] for x in rr)))
    result=dict(status='complete_diagnostic',comparisons=comparisons,trials=trials,
        captured_calls={str(i):len(v) for i,v in frames.items()},panel_seed=protocol['panel_seed'],
        limitations='One previously examined native question. Exact full-logit hashes only first6 calls; later logits equality is unknown. Entropy acceptance is recomputed with the sampler formula; actual native canvas/argmax/RNG/state are captured. No extra model forward or RNG draw. Observed/unobserved matches are conditional on this case, not general observer equivalence.',
        gpu_reservation_s=marker['gpu_reservation_s'])
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (output/'final_complete.json').write_text(json.dumps(dict(status='complete',source_sha256=sha(__file__),summary_sha256=sha(output/'summary.json'),trace_complete_sha256=sha(study/'final_complete.json')),indent=2)+'\n')
    print(json.dumps(dict(first={k:{x:v[x] for x in ('first_different','first_changed_canvas','first_changed_acceptance')} for k,v in comparisons.items()},trials=trials),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--cross',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);run(a.study,a.cross,a.output)
