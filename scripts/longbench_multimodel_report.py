"""Official LongBench-v2 extraction with immutable coverage/source audits."""
from __future__ import annotations
import argparse
import ast
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.longbench_multimodel import FIXED,CTI,PINS,OFFICIAL,read,rows,sha,write,verify_sources

LABELS=dict(zip((*FIXED,CTI),('Native dense','All-kept FA4','V31 K512','V3a K512','V2-CTI')))


def extractor():
    tree=ast.parse((OFFICIAL/'pred.py').read_text())
    function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='extract_answer')
    namespace={'re':re};exec(compile(ast.Module(body=[function],type_ignores=[]),'<pinned_official_extract_answer>','exec'),namespace)
    return namespace['extract_answer']


def aggregate(rs):
    n=len(rs);N=sum(r['counts']['denoising_forwards'] for r in rs);C=sum(r['counts']['canvases'] for r in rs)
    T=sum(r['output_tokens'] for r in rs);e=sum(r['counts']['eligible_tiles'] for r in rs);k=sum(r['counts']['kept_tiles'] for r in rs)
    seconds=sum(r['instrumented_seconds'] for r in rs)
    return {'requests':n,'correct':sum(r['correct'] for r in rs),'accuracy':statistics.mean(r['correct'] for r in rs),
        'unparsed':sum(not r['parsed'] for r in rs),'N':N,'C':C,'T':T,'N_per_request':N/n,'C_per_request':C/n,
        'N_per_C':N/C,'mean_output_tokens':T/n,'eligible_tiles':e,'kept_tiles':k,'skipped_tiles':e-k,'sparsity':1-k/e,
        'eos':sum(r['finished'] for r in rs),'cap_hits':sum(not r['finished'] and r['output_tokens']>=128 for r in rs),
        'truncated_requests':sum(r['truncated'] for r in rs),'instrumented_seconds':seconds,'instrumented_seconds_per_N':seconds/N,
        'discovery_qk_pairs':sum(r['counts'].get('discovery_qk_pairs',0) for r in rs),
        'projected_value_tokens':sum(r['counts'].get('projected_value_tokens',0) for r in rs)}


def paired(a,b):
    import numpy as np
    x={r['id']:r for r in a};y={r['id']:r for r in b};assert x.keys()==y.keys()
    d=np.array([float(x[k]['correct'])-float(y[k]['correct']) for k in sorted(x)])
    gen=np.random.default_rng(20261104);means=d[gen.integers(len(d),size=(10000,len(d)))].mean(1)
    return {'delta':float(d.mean()),'ci95':list(map(float,np.quantile(means,[.025,.975]))),
        'a_only_correct':int((d>0).sum()),'b_only_correct':int((d<0).sum()),'resamples':10000,'unit':'problem'}


def main(root):
    protocol=verify_sources(root);out=root/'analysis001';assert not out.exists();out.mkdir()
    extract=extractor();scored={};sources=[]
    for model in PINS:
        data=read(root/'private/data'/f'{model}.json');byid={r['id']:r for r in data};scored[model]={}
        for phase in ('fixed','cti'):
            candidates=sorted(p for p in (root/'panels'/model/phase).glob('attempt*/generation_complete.json') if read(p)['accepted'])
            assert len(candidates)==1,(model,phase,'accepted attempt count')
            leaf=candidates[0].parent;done=read(candidates[0]);p=read(leaf/'protocol.json')
            assert p['root_protocol_sha256']==sha(root/'protocol.json') and p['source_sha256']==protocol['source_sha256']
            assert done['panel_protocol_sha256']==sha(leaf/'protocol.json')
            for arm in (FIXED if phase=='fixed' else (CTI,)):
                records=[]
                for shard in range(1,9):
                    dest=leaf/'runs'/arm/f'shard{shard:03}'/'attempt001'
                    assert read(dest/'final_complete.json')['status']=='complete'
                    config=read(dest/'config.json');receipt=read(dest/'receipt.json')
                    assert receipt['records_sha256']==sha(dest/'records.jsonl') and receipt['config_sha256']==sha(dest/'config.json')
                    assert config['root_protocol_sha256']==sha(root/'protocol.json') and config['panel_protocol_sha256']==sha(leaf/'protocol.json')
                    assert config['manifest_sha256']==protocol['manifests'][model]['sha256'] and config['seed']==42
                    public=rows(dest/'records.jsonl');private=rows(dest/'private/generation.private.jsonl')
                    assert len(public)==len(private)==receipt['requests']==config['requests']
                    expected={r['id'] for i,r in enumerate(data) if i%8==shard-1}
                    assert {r['id'] for r in public}==expected
                    for r,pr in zip(public,private):
                        item=byid[r['id']]
                        for key in ('id','seed','counts','output_tokens','output_sha256','prompt_tokens','finished'):assert r[key]==pr[key]
                        assert hashlib.sha256(json.dumps(pr['tokens']).encode()).hexdigest()==r['output_sha256']
                        assert hashlib.sha256(item['prompt'].encode()).hexdigest()==r['prompt_sha256']
                        assert r['input_sha256']==item['input_sha256'] and r['prompt_tokens']==item['prompt_tokens']
                        assert r['panel_protocol_sha256']==sha(leaf/'protocol.json') and r['root_protocol_sha256']==sha(root/'protocol.json')
                        assert (r['model'],r['arm'],r['seed'],r['generation_cap'])==(model,arm,42,128)
                        c=r['counts'];assert 0<c['kept_tiles']<=c['eligible_tiles']
                        if arm in FIXED[:2]:assert c['kept_tiles']==c['eligible_tiles']
                        elif arm in FIXED[2:]:assert c['audited_k512_head_units']>0
                        else:assert c['current_tile_init_selection_layer_calls']==c['selection_layer_calls']>0
                        prediction=extract(pr['text'].strip())
                        records.append(dict(r,correct=prediction==item['gold'],parsed=prediction is not None))
                    sources.append({'model':model,'arm':arm,'shard':shard,'path':str(dest.relative_to(root)),
                        'records_sha256':sha(dest/'records.jsonl'),'receipt_sha256':sha(dest/'receipt.json'),'gpu_uuid':config['gpu_uuid']})
                assert len(records)==len(data) and len({r['id'] for r in records})==len(data)
                scored[model][arm]=records
                assert math.isclose(aggregate(records)['sparsity'],done['arms'][arm]['sparsity'],abs_tol=1e-14)
    assert len({s['gpu_uuid'] for s in sources})==1
    summary={m:{a:aggregate(rs) for a,rs in arms.items()} for m,arms in scored.items()}
    slices={}
    for m,arms in scored.items():
        slices[m]={}
        for arm,rs in arms.items():
            groups={}
            for field in ('length','difficulty','truncated','domain'):
                for value in sorted({r[field] for r in rs},key=str):groups[f'{field}/{value}']=aggregate([r for r in rs if r[field]==value])
            slices[m][arm]=groups
    intervals={m:{a+'_vs_'+b:paired(arms[a],arms[b]) for a in (*FIXED[2:],CTI) for b in (*FIXED[:2],FIXED[2]) if a!=b} for m,arms in scored.items()}
    write(out/'summary.json',summary);write(out/'slices.json',slices);write(out/'paired.json',intervals)
    with (out/'scored_records.jsonl').open('x') as f:
        for arms in scored.values():
            for rs in arms.values():
                for r in rs:f.write(json.dumps(r)+'\n')
    lines=['# LongBench v2: LLaDA2.2-mini and TraDo-8B-Instruct','',
        'Seed 42; official zero-shot prompt and exact official answer extraction. Invalid/unparsed answers count wrong. No chance compensation.','',
        '| Model | Length groups | Method | Correct | Accuracy | Mean output tokens | Steps/request | Canvases/request | Steps/canvas | Actual sparsity | EOS / caps | Unparsed |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model,arms in summary.items():
        for arm,x in arms.items():
            lines.append(f"| {model} | {' + '.join(protocol['groups'][model])} | {LABELS[arm]} | {x['correct']}/{x['requests']} | {100*x['accuracy']:.2f}% | {x['mean_output_tokens']:.1f} | {x['N_per_request']:.2f} | {x['C_per_request']:.2f} | {x['N_per_C']:.3f} | {100*x['sparsity']:.3f}% | {x['eos']} / {x['cap_hits']} | {x['unparsed']} |")
    lines+=['','Official length-group accuracy:','',
        '| Model | Group | Native dense | All-kept FA4 | V31 K512 | V3a K512 | V2-CTI |',
        '|---|---|---:|---:|---:|---:|---:|']
    for model in PINS:
        for group in protocol['groups'][model]:
            lines.append('| '+model+' | '+group+' | '+' | '.join(f"{100*slices[model][arm]['length/'+group]['accuracy']:.2f}%" for arm in (*FIXED,CTI))+' |')
    lines+=['','Input/truncation accounting (identical across methods within each model):','']
    for model,manifest in protocol['manifests'].items():
        lines.append(f"- {model}: {manifest['requests']} problems, {manifest['truncated']} middle-truncated; actual prompt range {manifest['prompt_min']}–{manifest['prompt_max']} tokens. Context limit {protocol['context_limits'][model]}, including the 128-token output reserve.")
    lines+=['','Setup and interpretation:','',
        'Each model retains its existing chat template and native sampler. LLaDA uses 32-token blocks, 32 steps, threshold 0.5, editing threshold 0, max post steps 16, temperature 0 and native early EOS. TraDo uses 4-token blocks, 4 steps, temperature 1, top-p 1, top-k 0 and confidence 0.9 with native EOS. BF16, batch 1, one H100.','',
        'All denoising layers are eligible; prefill and commit forwards stay dense. Native dense uses SDPA; all-kept FA4 controls for the sparse consumer. V31 ranks attention mass; V3a uses Gaussian32 full-information singleton risk. K512 means eight KV64 tiles, including mandatory current tiles, per query head/Q128 block. Four-call map reuse, no cross-canvas carry. Junyu selectors and current-tile initialization are distinct from Yuhan inherited map reuse.','',
        'Sparsity is one minus pooled actual kept/eligible denoising tiles, including mandatory tiles and selection/reuse consumer calls; selector discovery, dense prefill and commits are outside that denominator. Discovery QK pairs and projected-value work appear in summary.json. Sparsity does not imply whole-model speedup. Timings are instrumented accounting only.','',
        'V2-CTI thresholds are selected after both fixed-budget runs, on six count-only prompt-length strata per requested length group. This is transductive calibration: no correctness, gold or parsing metric is used. Exact reference and achieved calibration points are in each selected_threshold.json. Achieved full-panel sparsity is reported above; it is not assumed equal from requested budgets.','',
        'Paired item-bootstrap differences in percentage points (10,000 resamples):','']
    for model,pairs in intervals.items():
        for key,c in pairs.items():
            a,b=key.split('_vs_');lo,hi=c['ci95']
            lines.append(f"- {model}, {LABELS[a]} − {LABELS[b]}: {100*c['delta']:+.2f} pp, 95% CI [{100*lo:+.2f}, {100*hi:+.2f}].")
    lines+=['','Single-seed exploratory comparison; these intervals do not estimate seed variability or establish noninferiority. Official word-length groups do not guarantee token fit; truncated and untruncated, difficulty and domain slices are retained in slices.json. Prompts, gold, token IDs and completions stay private.','']
    (out/'README.md').write_text('\n'.join(lines))
    verify_sources(root)
    write(out/'receipt.json',{'status':'pass','runner_sha256':sha(__file__),'root_protocol_sha256':sha(root/'protocol.json'),
        'summary_sha256':sha(out/'summary.json'),'report_sha256':sha(out/'README.md'),'scored_records_sha256':sha(out/'scored_records.jsonl'),
        'requests':sum(x['requests'] for arms in summary.values() for x in arms.values()),'shards':sources,
        'official_extraction_sha256':sha(OFFICIAL/'pred.py'),'private_token_hashes_verified':True,'identical_prompt_seed_cap_per_model':True})
    write(root/'final_complete.json',{'status':'complete','models':list(PINS),'panels':10,'requests':2875,
        'analysis':'analysis001','receipt_sha256':sha(out/'receipt.json')})
    print(json.dumps({'status':'complete','summary':summary}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();main(a.root.resolve())
