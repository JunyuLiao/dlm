"""Official scoring and sanitized technical-reproducibility diagnostics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from scripts import v31_humaneval_repro_study as study
from scripts import v31_humaneval_value_study as old
from scripts import v31_official as official
from scripts.v31_score_humaneval import run_all
from scripts import v27_humaneval as he


FIELDS=('output_hash','denoise_forwards','output_tokens','canvases')


def analyze(root,out):
    if out.exists():raise RuntimeError('new immutable analysis attempt required')
    leaves=sorted(p.parent for p in (root/'diagnosis').glob('*/attempt_complete.json'))
    jobs=[];identities=[];records={};configs={};markers={};worker_hashes={}
    he.sandbox_preflight()
    for leaf in leaves:
        m=study.read(leaf/'attempt_complete.json');c=study.read(leaf/'config.json')
        for rel in ('records.jsonl','private/generation.private.jsonl','config.json','private/cells.json'):
            if study.sha(leaf/rel)!=m['files_sha256'][rel]:raise RuntimeError('artifact drift')
        pool=official.Pools(root/f"private/pools/{c['mode']}",('thinking',))
        gold=study.read(root/f"private/pools/{c['mode']}/gold.json")
        raw=study.base.jsonl(leaf/'records.jsonl');private=study.base.jsonl(leaf/'private/generation.private.jsonl')
        if len(raw)!=len(private):raise RuntimeError('private/public mismatch')
        records[leaf.name]=raw;configs[leaf.name]=c;markers[leaf.name]=study.sha(leaf/'attempt_complete.json')
        launch=study.read(leaf/'private/launch.json')
        worker_hashes[leaf.name]=study.sha(launch['command'][1])
        for i,(r,p) in enumerate(zip(raw,private)):
            _,_,bound=official.bind(p,pool,False)
            if not bound or any(r[k]!=p[k] for k in ('rng_seed','index','panel_seed','prompt_sha256','manifest_sha256')):
                raise RuntimeError('binding mismatch')
            code,why=old.extract(p['completion'],gold[p['id']]['entry_point'],c['mode']=='thinking_on')
            jobs.append((gold[p['id']],code));identities.append((leaf.name,i,why))
    scored,t1=run_all(jobs,he)
    rescored,t2=run_all(jobs,he)
    if scored!=rescored:raise RuntimeError('independent official rescore disagreement')
    per_item=[]
    for (name,i,why),(passed,outcome) in zip(identities,scored):
        r=records[name][i];c=configs[name]
        variant=(c['variants'][(i-8)%len(c['variants'])] if c.get('variants') and i>=8
                 else 'warmup' if i<8 else 'native')
        per_item.append(dict(job=name,position=i,index=r['index'],seed=r['panel_seed'],
            mode=c['mode'],family=c['family'],pass_name=c['pass_name'],variant=variant,
            warmup=i<8,score=bool(passed),scorer_outcome=outcome,extractor_outcome=why,
            **{k:r[k] for k in (*FIELDS,'rng_seed','prompt_tokens','wall_s','decode_s','cuda_graph_captures')}))
    scores={(r['job'],r['position']):r['score'] for r in per_item}
    def policy(c):
        return {k:v for k,v in c.get('overrides',{}).items()
                if k not in ('HUMANEVAL_REPRO_TRACE','HUMANEVAL_REPRO_MODEL_IDENTITY',
                             'HUMANEVAL_REPRO_FACTOR_RECEIPT','HUMANEVAL_KERNEL_INVENTORY')}
    pairs=[];within=[]
    for i,a in enumerate(leaves):
        for b in leaves[i+1:]:
            x,y=records[a.name],records[b.name];ca,cb=configs[a.name],configs[b.name]
            if len(x)!=len(y) or ca['mode']!=cb['mode']:
                continue
            if [(r['index'],r['rng_seed'],r['prompt_sha256']) for r in x]!=[(r['index'],r['rng_seed'],r['prompt_sha256']) for r in y]:
                continue
            same_worker=worker_hashes[a.name]==worker_hashes[b.name]
            kind=('selector_or_consumer_contrast' if ca['family']!=cb['family'] else
                  'runtime_policy_contrast' if policy(ca)!=policy(cb) else
                  'observer_source_contrast_process_confounded' if not same_worker or ca.get('trace')!=cb.get('trace') else
                  'audit_clean_equivalence' if ca['pass_name']!=cb['pass_name'] else
                  'same_method_technical_repeat')
            deltas=[u['denoise_forwards']-v['denoise_forwards'] for u,v in zip(x[8:],y[8:])]
            pairs.append(dict(left=a.name,right=b.name,kind=kind,measured_records=len(x)-8,
                unique_questions=len({r['index'] for r in x[8:]}),
                differences={f:sum(u[f]!=v[f] for u,v in zip(x[8:],y[8:])) for f in FIELDS},
                score_differences=sum(scores[(a.name,j)]!=scores[(b.name,j)] for j in range(8,len(x))),
                mean_N_delta=sum(deltas)/len(deltas),max_abs_N_delta=max(map(abs,deltas))))
    for name,c in configs.items():
        variants=c.get('variants')
        if not variants:continue
        rows=records[name];groups=[]
        for i in range(8,len(rows),len(variants)):
            base=rows[i]
            groups.append(dict(index=base['index'],comparisons=[dict(variant=variant,
                differences=[f for f in FIELDS if rows[i+j][f]!=base[f]],
                N_delta=rows[i+j]['denoise_forwards']-base['denoise_forwards'],
                score_changed=scores[(name,i+j)]!=scores[(name,i)]) for j,variant in enumerate(variants)]))
        within.append(dict(job=name,variants=variants,groups=groups,
            groups_with_any_difference=sum(any(x['differences'] for x in g['comparisons']) for g in groups)))
    out.mkdir(parents=True)
    (out/'per_item_metrics.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in per_item))
    summary=dict(status='diagnostic_partial_unqualified',cause='unresolved',full_runs_qualified=False,
        protocol_sha256=study.sha(root/'protocol.json'),engines=len(leaves),records=len(per_item),
        official_rescore_mismatches=0,timeouts_first=t1,timeouts_independent=t2,
        completed_markers_sha256=markers,comparisons=pairs,within_engine=within,
        seed_variation='seed42 only; additional-seed confirmation remains gated',
        accounting='diagnostic request times only; no clean performance claim')
    study.write(out/'summary.json',summary,True)
    lines=['# HumanEval technical reproducibility — diagnostic, unqualified','',
           f'{len(leaves)} completed engines, {len(per_item)} officially scored/rescored records. Cause unresolved.',
           'No full K256/K128 rerun or reliable method comparison is released.','',
           '| Type | Left | Right | Unique questions / pairs | Output differences | N differences | Score differences | Max abs N delta |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for r in pairs:
        if r['kind'] not in ('same_method_technical_repeat','audit_clean_equivalence'):continue
        lines.append(f"| {r['kind']} | {r['left']} | {r['right']} | {r['unique_questions']} / {r['measured_records']} | {r['differences']['output_hash']} | {r['differences']['denoise_forwards']} | {r['score_differences']} | {r['max_abs_N_delta']} |")
    lines+=['','Within-engine interventions (not latency measurements):','',
        '| Engine | Unique question groups | Groups with any output/N/C/T change |','|---|---:|---:|']
    for r in within:lines.append(f"| {r['job']} | {len(r['groups'])} | {r['groups_with_any_difference']} |")
    lines+=['','All source/runtime/consumer contrasts remain in summary.json. Different methods and consumers are not required to agree.',
            'Repeated requests do not increase the unique-question count. No seed-robust superiority or reliable method ranking is inferred.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    study.write(out/'analysis_complete.json',dict(analysis_complete=True,campaign_complete=False,
        files_sha256={p.name:study.sha(p) for p in out.iterdir() if p.is_file()}),True)
    print(json.dumps(dict(engines=len(leaves),records=len(per_item),rescore_mismatches=0)))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--study',type=Path,default=study.DEFAULT)
    a.add_argument('--out',type=Path,required=True);args=a.parse_args();analyze(args.study,args.out)
