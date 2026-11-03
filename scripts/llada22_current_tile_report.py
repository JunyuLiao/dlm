"""Official score and source/accounting audit for current-tile V2 HumanEval."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from scripts.llada22_v2_humaneval_diagnose import token_metrics
from scripts.multi_model_score import final_text
from scripts import v27_humaneval as he

ARM = 'value_v2_current_tile_init_gaussian32_reuse_r4'
BASE_ARMS = ('dense_native','allkept_fa4','mass_topk_reuse_r4',
             'value_v2_gaussian32_reuse_r4','value_v3a_singleton_reuse_r4')


def read(p):
    return json.loads(p.read_text())


def rows(p):
    return list(map(json.loads,p.read_text().splitlines()))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,v):
    with p.open('x') as f:
        json.dump(v,f,indent=2);f.write('\n')


def describe(records,private,gold):
    count=Counter();complete=Counter();length=[]
    for r in records:
        count.update(r['counts']);length.append(r['output_tokens'])
        text=final_text(private[r['id']]['text'])
        _,reason=he.extract_code('<channel|>'+text,gold[r['id']]['entry_point'])
        complete[reason]+=1
        for key,value in token_metrics(text,gold[r['id']]['entry_point']).items():
            if key.startswith('entry_point_'):complete[key]+=int(value)
        complete['eos_finished']+=int(r['finished'])
        complete['cap_hits']+=int(not r['finished'] and r['output_tokens']==1024)
    n,c=count['denoising_forwards'],count['canvases']
    e,k=count['eligible_tiles'],count['kept_tiles']
    return dict(requests=len(records),correct=sum(r['quality'] for r in records),
                accuracy=statistics.mean(r['quality'] for r in records),N=n,C=c,T=sum(length),
                N_per_request=n/164,C_per_request=c/164,N_per_C=n/c,
                mean_request_N_per_C=statistics.mean(r['counts']['denoising_forwards']/r['counts']['canvases'] for r in records),
                mean_output_tokens=statistics.mean(length),median_output_tokens=statistics.median(length),
                max_output_tokens=max(length),eligible_tiles=e,kept_tiles=k,consumer_sparsity=1-k/e,
                mean_kept_tiles_per_head_call=k/(320*n),mean_available_tiles_per_head_call=e/(320*n),
                counts=dict(count),completeness=dict(complete))


def main():
    p=argparse.ArgumentParser();p.add_argument('base',type=Path);p.add_argument('run',type=Path)
    p.add_argument('--published',type=Path,default=Path('results/multi_model_value_report_20261102/attempt002'))
    a=p.parse_args();base,run=a.base.resolve(),a.run.resolve()
    assert read(run/'generation_complete.json')['requests']==164
    protocol=read(run/'protocol.json')
    assert read(run/'qualification/kernel_receipt.json')['status']=='pass'
    assert read(run/'qualification/control_receipt.json')['status']=='pass'
    calibration=read(run/'calibration/selected_threshold.json')
    out=run/'analysis001';out.mkdir(exist_ok=False)
    he.sandbox_preflight()
    dataset=read(base/'private/data/llada22/humaneval.json');gold={r['id']:r['gold'] for r in dataset}
    expected={(r['id'],42) for r in dataset};assert len(expected)==164
    scoring={'source_sha256':sha(Path(__file__)),'humaneval_source_sha256':sha(Path(he.__file__)),
             'extractor':he.EXTRACTOR,'program':he.PROGRAM,'sandbox':he.SANDBOX,
             'bootstrap_resamples':10000,'bootstrap_seed':738,'unit':'HumanEval problem',
             'final_text_rule':'same as original multi_model_score.final_text'}
    old_scoring=read(a.published/'receipt.json')
    # Sources and item identities are checked independently below; preserve the
    # actual scorer fingerprint rather than assuming a compatible installation.
    old_identity=read(base/'analysis_scope30_20261102_attempt001/scoring_identity.json')
    assert old_identity['humaneval_source_sha256']==scoring['humaneval_source_sha256']
    write(out/'scoring_identity.json',scoring)
    fresh=[];sources=[];raw_private={};seconds=0.
    for leaf in sorted((run/'runs/llada22/humaneval'/ARM).glob('shard*/attempt001')):
        config=read(leaf/'config.json');receipt=read(leaf/'receipt.json');part=rows(leaf/'records.jsonl')
        assert read(leaf/'final_complete.json')=={'status':'complete','cells':41}
        assert len(part)==41 and sha(leaf/'records.jsonl')==receipt['records_sha256']
        assert config['sources']==protocol['source_sha256']
        assert config['protocol_sha256']==sha(run/'protocol.json')
        assert config['threshold']==calibration['threshold']
        private={(r['id'],r['seed']):r for r in rows(leaf/'private/generation.private.jsonl')}
        for r in part:
            assert r['arm']==ARM and r['seed']==42 and r['generation_cap']==1024
            assert r['protocol_sha256']==config['protocol_sha256'] and r['threshold']==config['threshold']
            assert r['counts']['current_tile_init_selection_layer_calls']==r['counts']['selection_layer_calls']>0
            n=r['counts']['denoising_forwards'];e=r['counts']['eligible_tiles'];k=r['counts']['kept_tiles']
            assert e%320==0 and e>=k>=320*n
            priv=private[r['id'],r['seed']]
            for key in ('counts','output_sha256','output_tokens','finished','prompt_tokens','prompt_sha256'):
                assert r[key]==priv[key]
            assert hashlib.sha256(json.dumps(priv['tokens']).encode()).hexdigest()==r['output_sha256']
            code,detail=he.extract_code('<channel|>'+final_text(priv['text']),gold[r['id']]['entry_point'])
            quality=float(he.run_test(he.program_for(gold[r['id']],code))[0]) if code else 0.
            fresh.append(dict(r,quality=quality,score_detail=detail))
            raw_private[r['id']]=priv
        sources.append({'config_sha256':sha(leaf/'config.json'),'records_sha256':receipt['records_sha256'],
                        'receipt_sha256':sha(leaf/'receipt.json'),'shard':config['shard']})
        seconds+=receipt['gpu_worker_seconds']
    assert len(sources)==4 and len(fresh)==164 and {(r['id'],r['seed']) for r in fresh}==expected
    previous=rows(a.published/'scored_records.jsonl')
    records={arm:[r for r in previous if (r['model'],r['suite'],r['arm'])==('llada22','humaneval',arm)] for arm in BASE_ARMS}
    records[ARM]=fresh;summaries={}
    for arm,rs in records.items():
        assert len(rs)==164 and {(r['id'],r['seed']) for r in rs}==expected
        priv=raw_private if arm==ARM else {r['id']:r for p in (base/'runs/llada22/humaneval'/arm).glob('shard*/attempt001/private/generation.private.jsonl') for r in rows(p)}
        summaries[arm]=describe(rs,priv,gold)
        baseline={r['id']:r for r in rs}
        for r in fresh:
            for key in ('prompt_sha256','prompt_tokens','generation_cap'):
                assert baseline[r['id']][key]==r[key],(arm,r['id'],key)
    new={r['id']:r for r in fresh};paired={}
    for arm in BASE_ARMS:
        delta=np.array([new[r['id']]['quality']-r['quality'] for r in records[arm]])
        rng=np.random.default_rng(738)
        samples=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)
        paired[arm]={'accuracy_delta':float(delta.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),
                     'wins':int((delta>0).sum()),'losses':int((delta<0).sum()),'ties':int((delta==0).sum())}
    result={'summaries':summaries,'paired_new_vs_reference':paired,'calibration':calibration,
            'allocated_benchmark_worker_seconds':seconds,'no_post_benchmark_recalibration':True,
            'target_met':.48<=summaries[ARM]['consumer_sparsity']<=.52,'previously_examined':True}
    write(out/'summary.json',result)
    with (out/'scored_records.jsonl').open('x') as f:
        for r in fresh:f.write(json.dumps(r)+'\n')
    labels=['Native dense','All-kept FA4','V31','Original V2','V3a','V2 current-tile init']
    lines=['# LLaDA2.2-mini HumanEval: current-tile initialization','',
           'Exploratory follow-up on the previously examined full164-problem cohort, seed42. Original native sampler/EOS/editing rules and1024-token cap; all20 layers, Gaussian32, Q128/KV64, refresh every fourth call. Only V2 initialization/order changes: current mandatory tile first, then chronological tile0 through J-2.','',
           '| Method | Correct /164 | Accuracy | Mean output tokens | N/request | C/request | N/C | Tile sparsity | EOS / cap | Syntax errors |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm,label in zip((*BASE_ARMS,ARM),labels):
        s=summaries[arm];co=s['completeness']
        lines.append(f"| {label} | {s['correct']:.0f} | {100*s['accuracy']:.2f}% | {s['mean_output_tokens']:.1f} | {s['N_per_request']:.2f} | {s['C_per_request']:.2f} | {s['N_per_C']:.3f} | {100*s['consumer_sparsity']:.2f}% | {co['eos_finished']} / {co['cap_hits']} | {co.get('syntax_error',0)} |")
    lines+=['',f"Calibration: six handwritten nonbenchmark development prompts, seed142, {len(calibration['trials'])} threshold candidates; frozen threshold {calibration['threshold']:.12g}, confirmed sparsity {100*calibration['confirmed_sparsity']:.2f}%. Calibration uses tile counts only. Full benchmark runs once, without retuning based on its achieved sparsity or accuracy.",'',
            'Tile sparsity is1-sum(kept)/sum(eligible) on actual denoising consumers, including mandatory tiles, excluding selector discovery, dense prefill and commits. N/C is pooled; per-request statistics and complete counters are in summary.json. Times are instrumented accounting, not clean latency.','',
            'Paired accuracy differences for the new variant (95% problem-bootstrap intervals,10000 draws):']
    for arm,label in zip(BASE_ARMS,labels):
        d=paired[arm];lo,hi=d['ci95']
        lines.append(f"- Versus {label}: {100*d['accuracy_delta']:+.2f}pp [{100*lo:+.2f}, {100*hi:+.2f}], {d['wins']} wins/{d['losses']} losses.")
    lines+=['','Validation: independent CPU tests and30 CUDA oracle/consumer cases;12 exact original-control replays; all164 new records source-bound and reconciled with private token hashes, four complete41-cell shards, same prompt identities/caps and official scorer fingerprint. Native dense and all-kept FA4 are separate historical references. Repeated HumanEval is exploratory; no held-out confirmation or universal repair claim. Raw prompts, gold and generations remain private.','']
    (out/'README.md').write_text('\n'.join(lines))
    receipt={'status':'pass','requests':164,'shards':sources,'protocol_sha256':sha(run/'protocol.json'),
             'generation_marker_sha256':sha(run/'generation_complete.json'),'summary_sha256':sha(out/'summary.json'),
             'scoring_identity_sha256':sha(out/'scoring_identity.json'),'scored_records_sha256':sha(out/'scored_records.jsonl'),
             'qualification_sha256':sha(run/'qualification/kernel_receipt.json'),'controls_sha256':sha(run/'qualification/control_receipt.json')}
    write(out/'receipt.json',receipt)
    write(run/'final_complete.json',{'status':'complete','requests':164,'receipt_sha256':sha(out/'receipt.json'),
                                    'summary_sha256':sha(out/'summary.json')})
    print(json.dumps({'status':'complete','new':summaries[ARM],'paired':paired}),flush=True)


if __name__=='__main__':main()
