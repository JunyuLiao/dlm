"""Official matched-cohort scoring and supplemental comparison publication."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from scripts.trado_ruler32k_baselines import ARMS,BASE,PREVIOUS,source_files
from scripts.multi_model_score import final_text

ROOT=Path(__file__).resolve().parents[1]
LABELS={'dense_native':'Native dense','allkept_fa4':'All-kept FA4','mass_topk_reuse_r4':'V31',
        'value_v3a_singleton_reuse_r4':'V3a','value_v2_current_tile_init_gaussian32_reuse_r4':'V2 current-tile init (existing)'}
V2='value_v2_current_tile_init_gaussian32_reuse_r4'


def read(p):return json.loads(Path(p).read_text())
def rows(p):return list(map(json.loads,Path(p).read_text().splitlines()))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')


def aggregate(rs):
    count=Counter()
    for r in rs:count.update(r['counts'])
    n,c=count['denoising_forwards'],count['canvases'];e,k=count['eligible_tiles'],count['kept_tiles']
    return {'requests':len(rs),'official_score':statistics.mean(r['quality'] for r in rs),'full_credit':sum(r['quality']==1. for r in rs),
            'N':n,'C':c,'T':sum(r['output_tokens'] for r in rs),'N_per_request':n/len(rs),'C_per_request':c/len(rs),
            'N_per_C':n/c,'mean_request_N_per_C':statistics.mean(r['counts']['denoising_forwards']/r['counts']['canvases'] for r in rs),
            'mean_output_tokens':statistics.mean(r['output_tokens'] for r in rs),'median_output_tokens':statistics.median(r['output_tokens'] for r in rs),
            'eligible_tiles':e,'kept_tiles':k,'sparsity':1-k/e,'eos_finished':sum(r['finished'] for r in rs),
            'cap_hits':sum(not r['finished'] and r['output_tokens']==128 for r in rs),'counts':dict(count)}


def contrast(candidate,reference,task_of):
    ref={(r['id'],r['seed']):r for r in reference};assert len(ref)==130
    delta=[];task=defaultdict(list);wins=losses=0
    for r in candidate:
        other=ref[r['id'],r['seed']]
        for field in ('prompt_sha256','prompt_tokens','generation_cap'):assert r[field]==other[field]
        d=r['quality']-other['quality'];delta.append(d);task[task_of[r['id']]].append(d)
        wins+=d>0;losses+=d<0
    x=np.array(delta);rng=np.random.default_rng(738)
    boot=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
    t=np.array([statistics.mean(v) for v in task.values()]);rng=np.random.default_rng(738)
    tb=t[rng.integers(len(t),size=(10000,len(t)))].mean(1)
    return {'score_delta':float(x.mean()),'problem_ci95':np.quantile(boot,[.025,.975]).tolist(),
            'task_clusters':len(t),'task_cluster_ci95':np.quantile(tb,[.025,.975]).tolist(),'wins':int(wins),'losses':int(losses),
            'bootstrap_resamples':10000,'bootstrap_seed':738}


def main(root):
    from dllm.evaluation.ruler.official import load_scorers,verify_checkout
    protocol=read(root/'protocol.json');assert protocol['source_sha256']=={k:sha(v) for k,v in source_files().items()}
    generation=read(root/'generation_complete.json');assert generation['requests']==520 and generation['v2_generation_requests']==0
    assert read(root/'qualification/kernel_receipt.json')['status']=='pass'
    assert read(root/'qualification/control_receipt.json')['exact_output_and_counts']
    ruler=ROOT/'reference/RULER';scorer_identity=verify_checkout(ruler);scorers=load_scorers(ruler)
    data=read(source_files()['manifest']);byid={r['id']:r for r in data};expected={(r['id'],42) for r in data}
    attempt=1
    while (root/f'analysis{attempt:03}').exists():attempt+=1
    out=root/f'analysis{attempt:03}';out.mkdir(exist_ok=False)
    allrows={};sources=[]
    def score(r,priv):
        item=byid[r['id']]
        text=item.get('answer_prefix','')+final_text(priv['text'])
        return scorers[item['task_base']]([text],[item['gold']])/100.
    for arm in ARMS:
        rs=[]
        leaves=sorted((root/'runs'/arm).glob('shard*/attempt001'));assert len(leaves)==4
        for leaf in leaves:
            receipt=read(leaf/'receipt.json');config=read(leaf/'config.json');raw=rows(leaf/'records.jsonl')
            assert receipt['records_sha256']==sha(leaf/'records.jsonl')
            assert read(leaf/'final_complete.json')=={'status':'complete','requests':len(raw)}
            assert config['sources']==protocol['source_sha256'] and config['protocol_sha256']==sha(root/'protocol.json')
            private={(r['id'],r['seed']):r for r in rows(leaf/'private/generation.private.jsonl')}
            assert len(private)==len(raw)
            for r in raw:
                p=private[r['id'],r['seed']]
                assert r['arm']==arm and r['seed']==42 and r['generation_cap']==128
                for field in ('counts','output_sha256','output_tokens','prompt_sha256','prompt_tokens','finished'):
                    assert r[field]==p[field]
                assert hashlib.sha256(json.dumps(p['tokens']).encode()).hexdigest()==r['output_sha256']
                assert r['protocol_sha256']==config['protocol_sha256'] and r['manifest_sha256']==config['manifest_sha256']
                assert r['counts']['eligible_tiles']%1152==0
                assert r['counts']['eligible_tiles']>=r['counts']['kept_tiles']>=1152*r['counts']['denoising_forwards']
                if arm in ARMS[2:]:assert r['counts']['audited_half_budget_units']==32*r['counts']['selection_layer_calls']>0
                else:assert r['counts']['eligible_tiles']==r['counts']['kept_tiles']
                rs.append(dict(r,quality=score(r,p),score_detail='official_ruler_fractional'))
            sources.append({'arm':arm,'shard':config['shard'],'records_sha256':sha(leaf/'records.jsonl'),'receipt_sha256':sha(leaf/'receipt.json')})
        assert len(rs)==130 and {(r['id'],r['seed']) for r in rs}==expected
        allrows[arm]=rs
    # Re-score the existing V2 completions on CPU and require exact agreement.
    old=[r for r in rows(source_files()['reference_scores']) if (r['model'],r['suite'])==('trado','ruler32k')]
    assert len(old)==130 and {(r['id'],r['seed']) for r in old}==expected
    refleaf=PREVIOUS/'panels/trado/ruler32k/attempt001'
    private={r['id']:r for p in refleaf.glob('runs/shard*/attempt001/private/generation.private.jsonl') for r in rows(p)}
    assert len(private)==130
    for r in old:
        assert r['quality']==score(r,private[r['id']])
    allrows[V2]=old
    # Reference source identity, model, host, sampler and manifest match the new arms.
    refprotocol=read(source_files()['reference_protocol'])
    assert refprotocol['model_revision']==protocol['model_revision']
    for short,long in [('runtime','multi_model_runtime'),('attention','multi_model_attention'),('base_worker','worker')]:
        assert refprotocol['source_sha256'][short]==protocol['source_sha256'][long]
    assert refprotocol['manifest_sha256']==protocol['source_sha256']['manifest']
    summary={arm:aggregate(rs) for arm,rs in allrows.items()}
    for arm in ARMS:
        assert .48<=summary[arm]['sparsity']<=.52 if arm in ARMS[2:] else summary[arm]['sparsity']==0.
    prior=read(source_files()['reference_summary'])['trado/ruler32k']
    assert (summary[V2]['N'],summary[V2]['C'],summary[V2]['T'])==(prior['N'],prior['C'],prior['T'])
    tasks={}
    for arm,rs in allrows.items():
        tasks[arm]={t:{'requests':len(part:=[r for r in rs if byid[r['id']]['task']==t]),
                        'score':statistics.mean(r['quality'] for r in part),'full_credit':sum(r['quality']==1. for r in part),
                        'N':sum(r['counts']['denoising_forwards'] for r in part),'C':sum(r['counts']['canvases'] for r in part),
                        'T':sum(r['output_tokens'] for r in part)} for t in sorted({r['task'] for r in data})}
    paired={}
    task_of={r['id']:r['task'] for r in data}
    for arm in allrows:
        for ref in allrows:
            if arm!=ref:paired[arm+'_vs_'+ref]=contrast(allrows[arm],allrows[ref],task_of)
    write(out/'summary.json',summary);write(out/'task_scores.json',tasks);write(out/'paired.json',paired)
    write(out/'scoring_identity.json',{'runner_sha256':sha(__file__),'ruler':scorer_identity,'v2_reference_rescored':True,
                                     'v2_reference_source_sha256':protocol['source_sha256']['reference_scores']})
    with (out/'scored_records.jsonl').open('x') as f:
        for arm,rs in allrows.items():
            for r in rs:f.write(json.dumps(r)+'\n')
    lines=['# TraDo-8B-Instruct RULER32K: completed baseline comparison','',
           'Completed four new arms /520requests on the same130-problem cohort. The existing current-tile V2 result is reused and officially re-scored on CPU; zero V2 generation requests. Prior reports and experiments remain unchanged.','',
           '| Method | Official retrieval score | Full-credit /130 | Mean output tokens | N total | C total | N/request | C/request | N/C | Tile sparsity | EOS / caps |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm,s in summary.items():
        lines.append(f"| {LABELS[arm]} | {100*s['official_score']:.2f}% | {s['full_credit']} | {s['mean_output_tokens']:.1f} | {s['N']} | {s['C']} | {s['N_per_request']:.2f} | {s['C_per_request']:.2f} | {s['N_per_C']:.3f} | {100*s['sparsity']:.2f}% | {s['eos_finished']} / {s['cap_hits']} |")
    lines+=['','Official RULER score is fractional target-retrieval credit, averaged over the balanced13tasks; full-credit count is the number of completely correct samples. N is denoising forwards,C native canvases; N/C is pooled. Caps count non-EOS outputs at128tokens.','',
      'Setup:130problems,10/task across13official tasks,seed42. Same frozen RULER32K prompts as prior V2; prompt lengths29798–32472,128output cap,32768context limit,no truncation. Original TraDo revision2d37bd3e and8c0d14b0 source; BF16,batch1,oneH100. Native4-token block,4denoising steps,temperature1,top-p1,top-k0,dynamic confidence0.9,native EOS. All36denoising layers eligible; dense prefill and commits.','',
      'V31/V3a keep max(round(J/2),mandatory_count) physical KV64tiles per query head/Q128block, including mandatory current tiles; ties-to-even rounding follows effective context length at each selection. V31 ranks attention mass; V3a uses Gaussian32 full-information singleton risk(seed1729+layer). Existing FA4 current-full-V consumer,refresh every fourth call,no carry. No threshold calibration is required for these fixed budgets.','',
      'Sparsity is1−sum(kept)/sum(eligible), measured on actual denoising consumer trajectories including mandatory tiles and selection/reuse calls, excluding selector discovery,prefill,commits. It is geometric tile sparsity,not whole-model FLOPs. Dense native SDPA and all-kept FA4 remain separate controls. Instrumented worker times are accounting,not clean latency.','',
      'Paired differences (candidate minus reference,percentage points;10000problem-bootstrap resamples):']
    for arm,ref in [('mass_topk_reuse_r4','dense_native'),('value_v3a_singleton_reuse_r4','mass_topk_reuse_r4'),(V2,'mass_topk_reuse_r4'),(V2,'value_v3a_singleton_reuse_r4'),(V2,'dense_native')]:
        c=paired[arm+'_vs_'+ref];lo,hi=c['problem_ci95'];tl,th=c['task_cluster_ci95']
        lines.append(f"- {LABELS[arm]} vs {LABELS[ref]}: {100*c['score_delta']:+.2f}pp,problem95%CI[{100*lo:+.2f},{100*hi:+.2f}],13-task-cluster95%CI[{100*tl:+.2f},{100*th:+.2f}].")
    lines+=['','Per-task retrieval scores:','',
            '| Task | Native dense | All-kept FA4 | V31 | V3a | Existing current-tile V2 |',
            '|---|---:|---:|---:|---:|---:|']
    for task in sorted(tasks[ARMS[0]]):lines.append('| '+task+' | '+' | '.join(f"{100*tasks[arm][task]['score']:.2f}%" for arm in (*ARMS,V2))+' |')
    lines+=['','Validation:five focused CPU tests,independent CPU/CUDA selector masks and FP32 masked-attention checks,exact historical all-kept output/count replay,16complete new shards,520unique source-bound new cells,per-selection exact budget/mandatory/eligibility assertions. Existing130V2scores independently re-score exactly. Same prompt/seed/cap/runtime/model identities are verified. No V2 worker or previous Gemma campaign is resumed.','',
            'The cohort was previously examined and all methods are exploratory. Intervals describe observed paired sample/task uncertainty,not seed variability or noninferiority. Raw prompts,gold,tokens and generations remain private.','']
    (out/'README.md').write_text('\n'.join(lines))
    # Publish a new supplemental combined table; never rewrite the old report.
    prior_metrics=read(PREVIOUS/'publication001/comparison_metrics.json')
    merged=prior_metrics[:]
    for arm in ARMS:
        s=summary[arm]
        merged.append({'model':'trado','suite':'ruler32k','method':LABELS[arm],'accuracy':s['official_score'],
                       **{k:s[k] for k in ('requests','N','C','T','N_per_C','mean_output_tokens','sparsity','eos_finished','cap_hits')}})
    write(out/'combined_35_rows.json',merged);assert len(merged)==35
    compact=['# Completed two-model accuracy table,including TraDo RULER32K baselines','',
             '| Model | Benchmark | Native dense | All-kept FA4 | V31 | Original V2 | V3a | Current-tile V2 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for model,suite in [('llada22','aime26'),('llada22','humaneval'),('llada22','ruler128k'),('trado','aime26'),('trado','humaneval'),('trado','ruler32k')]:
        part={x['method']:x for x in merged if (x['model'],x['suite'])==(model,suite)}
        values=['—' if label not in part else f"{100*part[label]['accuracy']:.2f}%" for label in ('Native dense','All-kept FA4','V31','Original V2','V3a','V2 current-tile init')]
        compact.append('| '+model+' | '+suite+' | '+' | '.join(values)+' |')
    compact+=['','Existing results remain tied to their frozen historical artifacts. No original-V2 TraDo RULER32K evaluation exists. See README.md for work,scope,scoring and uncertainty.','']
    (out/'combined_accuracy.md').write_text('\n'.join(compact))
    for key,path in source_files().items():assert sha(path)==protocol['source_sha256'][key],('source/reference drift',key)
    receipt={'status':'pass','new_requests':520,'existing_v2_requests_rescored':130,'new_arms':list(ARMS),'shards':sources,
             'protocol_sha256':sha(root/'protocol.json'),'summary_sha256':sha(out/'summary.json'),
             'report_sha256':sha(out/'README.md'),'scored_records_sha256':sha(out/'scored_records.jsonl'),
             'new_v2_generation_requests':0,'prior_reference_hashes_unchanged':True,
             'kernel_receipt_sha256':sha(root/'qualification/kernel_receipt.json'),'control_receipt_sha256':sha(root/'qualification/control_receipt.json')}
    write(out/'receipt.json',receipt)
    write(root/'final_complete.json',{'status':'complete','new_panels':4,'new_requests':520,'comparison_requests':650,
                                    'analysis':out.name,'receipt_sha256':sha(out/'receipt.json')})
    print(json.dumps({'status':'complete','summary':summary}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();main(a.root.resolve())
