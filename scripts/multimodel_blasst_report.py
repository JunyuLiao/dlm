"""Official six-panel aggressive-BLASST scores with preserved paired controls."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.multimodel_blasst_campaign import ORDER,ARM,PREVIOUS,read,rows,sha,write,verify
from scripts.current_tile_campaign_report import aggregate,contrasts
from scripts.multi_model_score import final_text
from scripts import v27_humaneval as he,v31_official as op


def main(root):
    p=verify(root);out=root/'analysis001';assert not out.exists();out.mkdir()
    he.sandbox_preflight();numeric,numeric_source=op.aime_numeric_score()
    from dllm.evaluation.ruler.official import load_scorers,verify_checkout
    ruler_source=verify_checkout(ROOT/'reference/RULER');scorers=load_scorers(ROOT/'reference/RULER')
    result={};sources=[];scored=[]
    for model,suite in ORDER:
        data=read(root/'private/data'/model/f'{suite}.json');byid={r['id']:r for r in data};expected={(r['id'],s) for r in data for s in r['seeds']}
        accepted=[f for f in (root/'panels'/model/suite).glob('attempt*/generation_complete.json') if read(f)['accepted']]
        assert len(accepted)==1,(model,suite,'accepted count');leaf=accepted[0].parent;panel=read(leaf/'protocol.json')
        assert panel['root_protocol_sha256']==sha(root/'protocol.json') and panel['manifest_sha256']==p['input_sha256'][model+'/'+suite]
        selected=read(leaf/'calibration/selected_threshold.json');assert not selected['accuracy_used'] and not selected['v2_threshold_used']
        rs=[]
        for shard in range(1,5):
            dest=leaf/'runs'/f'shard{shard:03}'/'attempt001';config=read(dest/'config.json');receipt=read(dest/'receipt.json')
            assert read(dest/'final_complete.json')['status']=='complete'
            assert receipt['records_sha256']==sha(dest/'records.jsonl') and receipt['config_sha256']==sha(dest/'config.json')
            assert config['model_revision']==p['model_revisions'][model] and config['log_kappa']==selected['log_kappa']
            assert config['panel_protocol_sha256']==sha(leaf/'protocol.json') and config['root_protocol_sha256']==sha(root/'protocol.json')
            public=rows(dest/'records.jsonl');private=rows(dest/'private/generation.private.jsonl');assert len(public)==len(private)==receipt['requests']
            for r,pr in zip(public,private):
                row=byid[r['id']]
                for key in ('id','seed','counts','output_tokens','output_sha256','prompt_sha256','prompt_tokens','finished','generation_cap'):assert r[key]==pr[key]
                assert hashlib.sha256(json.dumps(pr['tokens']).encode()).hexdigest()==r['output_sha256']
                assert hashlib.sha256(row['prompt'].encode()).hexdigest()==r['prompt_sha256']
                assert r['panel_protocol_sha256']==sha(leaf/'protocol.json') and r['root_protocol_sha256']==sha(root/'protocol.json')
                assert r['generation_cap']==p['caps'][model][suite] and r['log_kappa']==selected['log_kappa']
                c=r['counts'];assert c['blasst_selection_layer_calls']==c['selection_layer_calls']>0 and not c.get('projected_value_tokens',0)
                assert 0<c['kept_tiles']<=c['eligible_tiles'] and 0<r['lambda_min']<=r['lambda_max']
                text=final_text(pr['text'])
                if suite=='humaneval':
                    code,detail=he.extract_code('<channel|>'+text,row['gold']['entry_point'])
                    quality=float(he.run_test(he.program_for(row['gold'],code))[0]) if code else 0.
                elif suite=='aime26':quality=float(numeric(text,row['gold'])['correct']);detail='official_aime_exact'
                else:quality=scorers[row['task_base']]([row.get('answer_prefix','')+text],[row['gold']])/100.;detail='official_ruler_fractional'
                rs.append(dict(r,quality=quality,score_detail=detail))
            sources.append({'model':model,'suite':suite,'shard':shard,'path':str(dest.relative_to(root)),
                'records_sha256':sha(dest/'records.jsonl'),'receipt_sha256':sha(dest/'receipt.json'),'gpu_uuid':config['gpu_uuid']})
        assert len(rs)==len(expected) and {(r['id'],r['seed']) for r in rs}==expected
        summary=aggregate(rs);summary.update(log_kappa=selected['log_kappa'],lambda_min=min(r['lambda_min'] for r in rs),
            lambda_max=max(r['lambda_max'] for r in rs),attempt=leaf.name,calibration_sparsity=selected['sparsity'],
            instrumented_seconds=sum(r['instrumented_seconds'] for r in rs),calibration_transductive=panel['transductive'])
        assert .4<=summary['consumer_sparsity']<=.6
        result[model+'/'+suite]=summary;scored.extend(rs)
    assert len({s['gpu_uuid'] for s in sources})==1 and len(scored)==768
    historical=rows(ROOT/'results/multi_model_value_report_20261102/attempt002/scored_records.jsonl')
    cti=rows(PREVIOUS/'analysis001/scored_records.jsonl')
    ruler=rows(ROOT/'results/trado_ruler32k_baselines_20261104/attempt002/analysis002/scored_records.jsonl')
    paired={};comparison={}
    for model,suite in ORDER:
        fresh=[r for r in scored if (r['model'],r['suite'])==(model,suite)]
        old=ruler if (model,suite)==('trado','ruler32k') else historical
        arms={arm:[r for r in old if (r['model'],r['suite'],r['arm'])==(model,suite,arm)] for arm in
            ('dense_native','allkept_fa4','mass_topk_reuse_r4','value_v3a_singleton_reuse_r4')}
        arms['current_tile_v2']=[r for r in cti if (r['model'],r['suite'])==(model,suite)]
        paired[model+'/'+suite]={}
        for arm,rs in arms.items():
            contrast=contrasts(fresh,rs);assert contrast is not None,(model,suite,arm,'pair identity')
            paired[model+'/'+suite][arm]=contrast
        comparison[model+'/'+suite]={arm:aggregate(rs) for arm,rs in arms.items()}
        comparison[model+'/'+suite][ARM]=result[model+'/'+suite]
    write(out/'summary.json',result);write(out/'paired.json',paired);write(out/'comparison.json',comparison)
    with (out/'scored_records.jsonl').open('x') as f:
        for r in scored:f.write(json.dumps(r)+'\n')
    lines=['# Aggressive BLASST on LLaDA2.2-mini and TraDo-8B-Instruct','',
        'Six complete new panels, 768 benchmark requests. Existing baseline, V31/V3a and V2-CTI results are preserved.','',
        '| Model | Benchmark | Method | Score | Full credit | Mean output | Steps/request | Canvases/request | Steps/canvas | Sparsity | EOS / caps |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key,arms in comparison.items():
        model,suite=key.split('/')
        for arm,x in arms.items():
            lines.append(f"| {model} | {suite} | {arm} | {100*x['accuracy']:.2f}% | {x['full_credit']}/{x['requests']} | {x['mean_output_tokens']:.1f} | {x['N_per_request']:.2f} | {x['C_per_request']:.2f} | {x['N_per_C']:.3f} | {100*x['consumer_sparsity']:.2f}% | {x['finished']} / {x['caps']} |")
    lines+=['','Calibration and achieved lambda ranges:','',
        '| Model/task | log kappa | Calibration sparsity | Full sparsity | Minimum lambda | Maximum lambda | Calibration exposure |',
        '|---|---:|---:|---:|---:|---:|---|']
    for key,x in result.items():
        lines.append(f"| {key} | {x['log_kappa']:.8g} | {100*x['calibration_sparsity']:.2f}% | {100*x['consumer_sparsity']:.2f}% | {x['lambda_min']:.8g} | {x['lambda_max']:.8g} | {'transductive count-only' if x['calibration_transductive'] else 'independent retrieval development'} |")
    lines+=['','AIME uses 30 problems × seeds 42/43/44; HumanEval uses all 164 and seed 42; RULER uses 130 balanced problems and seed 42, 128K on LLaDA and 32K on TraDo. RULER score is fractional official retrieval credit. Prompt, cap, seed and scorer identities match the prior experiments.','',
        'Aggressive BLASST uses the preceding maximum over all seen tiles, including skipped tiles; strict all-active-row skipping, first/current protections and lambda(L)=exp(log_kappa)/L. Lambda may exceed 1. This is the existing aggressive extension, not literal paper pseudocode above lambda=1. The two model ports use FP32 accumulated scaled QK scores consistent with their FA4 attention path. No V2 value projection, CTI initialization or V2 threshold is used.','',
        'All denoising layers, Q128/KV64 tiles, four-call map reuse and no cross-canvas carry; full-dimensional FA4 consumer; dense prefill and commits. Native samplers, EOS, editing and output caps are unchanged. The earlier BLASST rule and Yuhan inherited map reuse remain distinct provenance.','',
        'Fresh thresholds use only small count-only calibration trajectories: six prompt-length strata per short benchmark, or 13 independent retrieval development items. Target 48–52%, final acceptance 40–60%; all rejected attempts are retained. Exact calibration points, full-panel drift and lambda ranges are recorded, with no accuracy-driven threshold selection.','',
        'Sparsity is pooled skipped/eligible physical denoising tiles, including mandatory and held calls, excluding selector discovery and dense prefill/commit. Dense QK discovery and instrumented accounting remain separate; no clean latency or speedup claim.','',
        'Independent CPU/CUDA masks, FP32 attention consumer checks, historical all-kept replay, exact shard/source/private-token-hash joins, coverage and fairness audits pass. Paired intervals in paired.json use question clusters (including AIME seeds), not independent seed-level samples. Nonsignificance does not establish noninferiority.','']
    (out/'README.md').write_text('\n'.join(lines))
    for path,digest in read(root/'baseline_readonly_inventory.json').items():assert sha(ROOT/path)==digest,('historical evidence drift',path)
    verify(root)
    write(out/'receipt.json',{'status':'pass','requests':768,'panels':6,'source_sha256':sha(__file__),
        'protocol_sha256':sha(root/'protocol.json'),'summary_sha256':sha(out/'summary.json'),
        'report_sha256':sha(out/'README.md'),'scored_records_sha256':sha(out/'scored_records.jsonl'),
        'historical_hashes_unchanged':True,'shards':sources,'aime_scorer':numeric_source,'ruler_scorer':ruler_source,
        'humaneval_scorer_sha256':sha(Path(he.__file__)),'private_token_hashes_verified':True})
    write(root/'final_complete.json',{'status':'complete','panels':6,'requests':768,'analysis':'analysis001','receipt_sha256':sha(out/'receipt.json')})
    print(json.dumps({'status':'complete','summary':result}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();main(a.root.resolve())
