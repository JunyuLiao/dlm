"""Official scores and immutable historical comparisons for nine V2 panels."""
from __future__ import annotations
from collections import Counter,defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

from scripts import v27_humaneval as he,v31_official as op
from scripts.multi_model_score import final_text
from scripts.current_tile_campaign import ORDER

REPO=Path(__file__).resolve().parents[1]
HF_REPORT=REPO/'results/multi_model_value_report_20261102/attempt002'
DG_REPORT=REPO/'results/v2_execution_results_20261023/attempt001'
DG_SHORT=REPO/'results/optimized_refresh_results_20261023/attempt003'


def read(p):return json.loads(Path(p).read_text())
def rows(p):return list(map(json.loads,Path(p).read_text().splitlines()))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')


def aggregate(rs):
    c=Counter()
    for r in rs:c.update(r['counts'])
    n,canvas=c['denoising_forwards'],c['canvases'];e,k=c['eligible_tiles'],c['kept_tiles']
    return {'requests':len(rs),'accuracy':statistics.mean(r['quality'] for r in rs),'full_credit':sum(r['quality']==1. for r in rs),
            'N':n,'C':canvas,'N_per_C':n/canvas,'N_per_request':n/len(rs),'C_per_request':canvas/len(rs),
            'T':sum(r['output_tokens'] for r in rs),'mean_output_tokens':statistics.mean(r['output_tokens'] for r in rs),
            'median_output_tokens':statistics.median(r['output_tokens'] for r in rs),
            'eligible_tiles':e,'kept_tiles':k,'consumer_sparsity':1-k/e,'finished':sum(r['finished'] for r in rs),
            'caps':sum(not r['finished'] and r['output_tokens']>=r['generation_cap'] for r in rs),'counts':dict(c)}


def score_hf(root,model,suite,leaf,numeric,scorers):
    dataset=read(root/'private/data'/model/f'{suite}.json');byid={r['id']:r for r in dataset}
    expected={(r['id'],s) for r in dataset for s in r['seeds']};result=[]
    for shard in sorted((leaf/'runs').glob('shard*/attempt001')):
        receipt=read(shard/'receipt.json');public=rows(shard/'records.jsonl')
        assert sha(shard/'records.jsonl')==receipt['records_sha256'] and (shard/'final_complete.json').exists()
        private={(r['id'],r['seed']):r for r in rows(shard/'private/generation.private.jsonl')}
        for r in public:
            p=private[r['id'],r['seed']];row=byid[r['id']]
            for key in ('counts','output_tokens','output_sha256','prompt_sha256','prompt_tokens','finished','generation_cap'):
                assert r[key]==p[key]
            assert hashlib.sha256(json.dumps(p['tokens']).encode()).hexdigest()==r['output_sha256']
            assert r['counts']['current_tile_init_selection_layer_calls']==r['counts']['selection_layer_calls']>0
            text=final_text(p['text'])
            if suite=='humaneval':
                code,detail=he.extract_code('<channel|>'+text,row['gold']['entry_point'])
                quality=float(he.run_test(he.program_for(row['gold'],code))[0]) if code else 0.
            elif suite=='aime26':quality=float(numeric(text,row['gold'])['correct']);detail='official_aime_exact'
            else:quality=scorers[row['task_base']]([row.get('answer_prefix','')+text],[row['gold']])/100.;detail='official_ruler_fractional'
            result.append(dict(r,quality=quality,score_detail=detail))
    assert len(result)==len(expected) and {(r['id'],r['seed']) for r in result}==expected
    return result


def score_gemma(root,suite,leaf,numeric):
    from scripts import v31_score_humaneval as sh,v31_score_aime as sa,v31_score_ruler as sr
    folder=root/'private/pools/diffusiongemma'/suite if suite!='ruler128k' else leaf/'private/pool'
    fields=('thinking',) if suite=='humaneval' else ('source_id','thinking') if suite=='aime26' else sr.FIELDS
    pools=op.Pools(folder,fields);final,_=op.load_final_response()
    staged=[];public={};meta={};sources=[]
    for shard in sorted((leaf/'runs').glob('shard*/attempt001')):
        receipt=read(shard/'receipt.json');assert receipt['records_sha256']==sha(shard/'records.jsonl')
        plan=read(shard/'private/cells.json');pr=rows(shard/'records.jsonl');priv=rows(shard/'private/generation.private.jsonl')
        assert len(pr)==len(priv)==len(plan)
        for r in pr:
            key=op.cell_key(r);assert key not in public;public[key]=r
        for r in priv:
            key=op.cell_key(r);op.bind(r,pools)
            staged.append(('current_tile_v2',key,r));meta[key]=r
        sources.append({'records_sha256':receipt['records_sha256'],'receipt_sha256':sha(shard/'receipt.json')})
    assert len(public)==len(staged)==(90 if suite=='aime26' else 164 if suite=='humaneval' else 130)
    if suite=='humaneval':
        gold=read(folder/'gold.json');official,_,binding,diag,_=sh.score(staged,gold,he,pools,workers=4)
    elif suite=='aime26':
        answers={str(x['id']):str(x['answer']) for x in rows(folder/'aime2026_answers.jsonl')}
        official,_,binding,diag=sa.score(staged,answers,pools,final,numeric)
    else:
        metrics=op.ruler_metrics(REPO/'reference/RULER')
        official,_,binding,diag=sr.score(staged,metrics,{},pools,
            lambda ds:read(folder/f'{ds}_gold.json'),final)
    result=[]
    for key,value in official['current_tile_v2'].items():
        r=public[key];a=r['receipts']['adapter'];p=meta[key]
        assert a['native_local_skipped_tiles']==0 and a['current_canvas_initialization']
        assert a['global_calls']==5*r['denoise_forwards'] and a['native_local_calls_audited']==25*r['denoise_forwards']
        counts={'denoising_forwards':r['denoise_forwards'],'canvases':r['canvases'],
                'eligible_tiles':a['global_eligible_tiles'],'kept_tiles':a['global_kept_tiles'],
                'local_eligible_rectangles':a['native_local_eligible_rectangles'],'local_skipped_tiles':0,
                'overall_eligible_rectangles':a['overall_eligible_rectangles'],
                'current_canvas_selector_calls':a['current_canvas_selector_calls']}
        result.append({'id':p['id'],'index':r['index'],'seed':r['panel_seed'],'model':'diffusiongemma','suite':suite,
                       'arm':'current_canvas_init_v2','counts':counts,'output_tokens':r['output_tokens'],
                       'output_sha256':r['output_hash'],'prompt_sha256':r['prompt_sha256'],'generation_cap':r['budget'],
                       'finished':op.finish_class(r['finish_reason'])=='finished','quality':float(value)/(100. if suite=='ruler128k' else 1.),
                       'official_cell_key':key})
    return result,{'official':official,'diagnostics':{k:dict(v) for k,v in diag.items()},'source_shards':sources}


def contrasts(new,old):
    ref={(r['id'],r['seed']):r for r in old};clusters=defaultdict(list)
    if set(ref)!={(r['id'],r['seed']) for r in new}:return None
    for r in new:
        b=ref[r['id'],r['seed']]
        if r['prompt_sha256']!=b['prompt_sha256'] or r['generation_cap']!=b['generation_cap']:return None
        clusters[r['id']].append(r['quality']-b['quality'])
    x=np.array([statistics.mean(v) for v in clusters.values()]);rng=np.random.default_rng(738)
    boot=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
    return {'problem_clusters':len(x),'accuracy_delta':float(x.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist()}


def main(root):
    index=1
    while (root/f'analysis{index:03}').exists():index+=1
    out=root/f'analysis{index:03}';out.mkdir(exist_ok=False)
    he.sandbox_preflight();numeric,ns=op.aime_numeric_score()
    from dllm.evaluation.ruler.official import load_scorers,verify_checkout
    ruler_identity=verify_checkout(REPO/'reference/RULER');scorers=load_scorers(REPO/'reference/RULER')
    old_rows=rows(HF_REPORT/'scored_records.jsonl');old_summary=read(HF_REPORT/'summary.json')
    results={};scored=[];pair={};attempts={};dg_scores={}
    for model,suite in ORDER:
        parent=root/'panels'/model/suite
        markers=sorted(parent.glob('attempt*/generation_complete.json'))
        attempts[f'{model}/{suite}']=[read(f) for f in markers]
        chosen=[f for f in markers if read(f)['accepted']];assert len(chosen)==1
        leaf=chosen[0].parent;protocol=read(leaf/'protocol.json')
        if model=='diffusiongemma':rs,evidence=score_gemma(root,suite,leaf,numeric);dg_scores[suite]=evidence
        else:rs=score_hf(root,model,suite,leaf,numeric,scorers)
        summary=aggregate(rs);marker=read(chosen[0])
        assert (summary['eligible_tiles'],summary['kept_tiles'])==(marker['eligible_tiles'],marker['kept_tiles'])
        assert .4<=summary['consumer_sparsity']<=.6
        summary.update(threshold=marker['threshold'],attempt=marker['attempt'],protocol_sha256=sha(leaf/'protocol.json'),
                       calibration=read(leaf/'calibration/selected_threshold.json'),generation_marker_sha256=sha(chosen[0]))
        if model!='diffusiongemma':summary['control_receipt']=read(leaf/'control_receipt.json')
        if model=='diffusiongemma':
            summary['overall_sparsity']=(summary['eligible_tiles']-summary['kept_tiles'])/summary['counts']['overall_eligible_rectangles']
        results[f'{model}/{suite}']=summary;scored+=rs
        if model!='diffusiongemma':
            for arm in ('dense_native','allkept_fa4','mass_topk_reuse_r4','value_v2_gaussian32_reuse_r4','value_v3a_singleton_reuse_r4'):
                old=[r for r in old_rows if (r['model'],r['suite'],r['arm'])==(model,suite,arm)]
                if old:pair[f'{model}/{suite}/vs_{arm}']=contrasts(rs,old)
    assert sum(r['requests'] for r in results.values())==1152
    historical={k:v for k,v in old_summary.items() if k.startswith(('llada22/','trado/'))}
    dg_short=read(DG_SHORT/'summary.json');dg_follow=read(DG_REPORT/'summary.json')
    historical['diffusiongemma/humaneval']=dg_short['humaneval']
    historical['diffusiongemma/aime26']=dg_follow['aime26_k1920']['arms']
    historical['diffusiongemma/ruler128k_104_only']=dg_follow['ruler128k_s45']['arms']
    write(out/'summary.json',results);write(out/'attempts.json',attempts);write(out/'paired.json',pair)
    write(out/'historical_results_unchanged.json',historical);write(out/'gemma_official_scoring.json',dg_scores)
    with (out/'scored_records.jsonl').open('x') as f:
        for r in scored:f.write(json.dumps(r)+'\n')
    write(out/'scoring_identity.json',{'source_sha256':sha(__file__),'humaneval_sha256':sha(he.__file__),
        'humaneval_program':he.PROGRAM,'humaneval_extractor':he.EXTRACTOR,'humaneval_sandbox':he.SANDBOX,
        'aime_numeric':ns,'ruler':ruler_identity,'bootstrap_resamples':10000,'bootstrap_seed':738})
    text=['# Current-tile V2: nine completed model/benchmark panels','',
          'Only the new V2 variants were benchmarked. Prior methods remain unchanged. All selected complete panels achieve40–60% physical eligible denoising-consumer sparsity after small count-only calibration; any rejected completed attempt is retained in attempts.json.','',
          '| Model | Benchmark | Score | Requests | Mean output | Steps/request | Canvases/request | Steps/canvas | Sparsity | Threshold | Attempt |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for (model,suite) in ORDER:
        s=results[f'{model}/{suite}']
        text.append(f"| {model} | {suite} | {100*s['accuracy']:.2f}% | {s['requests']} | {s['mean_output_tokens']:.1f} | {s['N_per_request']:.2f} | {s['C_per_request']:.2f} | {s['N_per_C']:.3f} | {100*s['consumer_sparsity']:.2f}% | {s['threshold']:.9g} | {s['attempt']} |")
    text+=['','AIME:30 problems x seeds42/43/44. HumanEval:164 problems,seed42. RULER:10 samples/task across13 official tasks,seed42; fractional retrieval score. RULER128K on LLaDA/Gemma,32K on TraDo.','',
           'LLaDA and TraDo preserve their native32/4-token samplers, all-layer sparse scope, Q128/KV64, Gaussian32 and four-call reuse. Gemma preserves its256-token native sampler, GLOBAL-only layers5/11/17/23/29, dense25 LOCAL layers, dense prefill/commits, and original carry/progress/sticky lifecycle. Only initialization/order changes. Per-model output caps and thinking settings match the original setups.','',
           'Sparsity includes mandatory/current tiles and all native denoising consumer phases, excludes discovery/prefill/commits. Gemma also reports overall GLOBAL+LOCAL sparsity in summary.json. Short-benchmark calibration uses six fixed prompt-length strata of examined benchmark prompts, count-only and transductive; retrieval uses13 separately generated development prompts. No gold or score chooses a threshold. A completed outside40–60% panel triggers a new calibration/attempt; inside the range it is retained without retuning.','',
           'Historical references are in historical_results_unchanged.json. The old Gemma RULER cohort contains104 items (8/task); its score is a separate cohort reference, not a paired baseline for the new130. TraDo32K has no historical matched panel. Paired intervals are emitted only after prompt/seed/cap identity checks; repeated examined benchmarks are exploratory. No clean latency or universal accuracy-preservation claim.','']
    (out/'README.md').write_text('\n'.join(text))
    receipt={'status':'pass','panels':9,'selected_requests':1152,'protocol_sha256':sha(root/'protocol.json'),
             'summary_sha256':sha(out/'summary.json'),'scored_records_sha256':sha(out/'scored_records.jsonl'),
             'scoring_identity_sha256':sha(out/'scoring_identity.json'),'all_final_sparsities_in_40_60':True}
    protected=root/'baseline_readonly_inventory.json'
    if protected.exists():
        for name,digest in read(protected).items():assert sha(REPO/name)==digest,('historical evidence changed',name)
        receipt['historical_protected_hashes_pass']=True
    write(out/'receipt.json',receipt)
    write(root/'final_complete.json',{'status':'complete','panels':9,'requests':1152,'analysis':out.name,'receipt_sha256':sha(out/'receipt.json')})
    print(json.dumps({'status':'complete','panels':9,'requests':1152}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();main(a.root.resolve())
