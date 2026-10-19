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
    jobs=[];identities=[];records={};configs={};markers={}
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
    pairs=[]
    for i,a in enumerate(leaves):
        for b in leaves[i+1:]:
            x,y=records[a.name],records[b.name];ca,cb=configs[a.name],configs[b.name]
            if len(x)!=len(y) or ca['family']!=cb['family'] or ca['mode']!=cb['mode']:
                continue
            if [(r['index'],r['rng_seed'],r['prompt_sha256']) for r in x]!=[(r['index'],r['rng_seed'],r['prompt_sha256']) for r in y]:
                continue
            pairs.append(dict(left=a.name,right=b.name,measured_records=len(x)-8,
                differences={f:sum(u[f]!=v[f] for u,v in zip(x[8:],y[8:])) for f in FIELDS}))
    out.mkdir(parents=True)
    (out/'per_item_metrics.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in per_item))
    summary=dict(status='diagnostic_partial_unqualified',cause='unresolved',full_runs_qualified=False,
        protocol_sha256=study.sha(root/'protocol.json'),engines=len(leaves),records=len(per_item),
        official_rescore_mismatches=0,timeouts_first=t1,timeouts_independent=t2,
        completed_markers_sha256=markers,comparisons=pairs,
        accounting='diagnostic request times only; no clean performance claim')
    study.write(out/'summary.json',summary,True)
    lines=['# HumanEval technical reproducibility — diagnostic, unqualified','',
           f'{len(leaves)} completed engines, {len(per_item)} officially scored/rescored records. Cause unresolved.',
           'No full K256/K128 rerun or reliable method comparison is released.','',
           '| Left | Right | Measured pairs | Output differences | N differences |',
           '|---|---|---:|---:|---:|']
    for r in pairs:
        lines.append(f"| {r['left']} | {r['right']} | {r['measured_records']} | {r['differences']['output_hash']} | {r['differences']['denoise_forwards']} |")
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    study.write(out/'analysis_complete.json',dict(analysis_complete=True,campaign_complete=False,
        files_sha256={p.name:study.sha(p) for p in out.iterdir() if p.is_file()}),True)
    print(json.dumps(dict(engines=len(leaves),records=len(per_item),rescore_mismatches=0)))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--study',type=Path,default=study.DEFAULT)
    a.add_argument('--out',type=Path,required=True);args=a.parse_args();analyze(args.study,args.out)
