"""Targeted refinement stage, separate cache/provenance and no heldout data."""
import argparse
import json
import time
import traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from dllm.models import create_adapter
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write,_append,_fingerprint
from .protocol import ROOT,MODEL,REVISION,prepare,fingerprint,frozen_write,sha
from .run import shard_path,generate
from .refinements import CONFIGS,cache_refined,source_sha
from .calibration import quantile_threshold,aggregate
from .report import csv_write,table
from .ranking_guards import cache_joint_screen,source_sha as guard_sha,smoke as guard_smoke


def screen(root):
    setup=prepare(root);fp=fingerprint(root);rs=source_sha()
    assert json.loads((root/'screen_audit.json').read_text())['complete']
    frozen_write(root/'refinement_plan.json',dict(fingerprint=fp,refinement_sha256=rs,ranking_guard_sha256=guard_sha(),
        reason='Matched-budget screen found conservative mass and partial-running-output centered estimates worse than BLASST; isolate exact block mass and previous-step output',
        methods=CONFIGS,calibration_ids=[r['id'] for r in setup['calibration']],development_ids=[r['id'] for r in setup['development']],
        heldout_used=False,exact_mass_work='Requires block softmax, can omit PV only',
        previous_output_work='Cache one Q×D attention output per head/layer; invalidate on canvas/prefix/step discontinuity; running-output warmup fallback',
        dense_replay_reason='Existing screen lacked previous-step/exact-mass/row-inflation statistics; same replay also corrects Sol and diagnostic nonempty repair to use only their declared signal; exact token parity is checked against cached dense'))
    adapter=create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
    tests=[]
    smoke_path=root/'refinement_smoke.json'
    if smoke_path.exists():
        existing=json.loads(smoke_path.read_text());assert existing['passed'] and existing['refinement_sha256']==rs and existing['fingerprint']==fp
    else:
        for benchmark in ('aime26','longbench'):
            row=dict(next(r for r in setup['calibration'] if r['benchmark']==benchmark),generation_budget=32)
            dense,_=generate(adapter,row)
            for name,c in CONFIGS.items():
                for label,threshold in [('unpruned',-100.),('sparse',-.7)]:
                    config=dict(c,log_threshold=threshold)
                    path=shard_path(root,'refinement_smoke',f'{name}_{label}',row['id'])
                    out=cache_refined(adapter,row,path,fp,config,root=root)
                    if label=='unpruned':assert out['completion_tokens']==dense['completion_tokens'],'refined dense parity'
                    if 'exact_mass' in c['mode']:assert all(r['softmax_skipped']==0 for r in out['records'])
                    tests.append(dict(benchmark=benchmark,method=name,label=label,passed=True,coverage=out['previous_output_coverage']))
        _write(smoke_path,dict(passed=True,fingerprint=fp,refinement_sha256=rs,tests=tests))
    guard_smoke(adapter,setup,root,fp)
    failures=[]
    for row in sorted(setup['calibration']+setup['development'],key=lambda r:(r['generation_budget'],r['id'])):
        try:
            out=cache_joint_screen(adapter,row,shard_path(root,'refinement_screen','dense',row['id']),fp,root)
            original=json.loads(shard_path(root,'screen','dense',row['id']).read_text())
            assert original['completion_tokens']==out['completion_tokens'],'refinement replay parity'
        except Exception:
            failure=dict(stage='refinement_screen',id=row['id'],traceback=traceback.format_exc());failures.append(failure)
            _append(root/'failures.jsonl',failure);print(failure,flush=True)
        finally:torch.cuda.empty_cache()
    _write(root/'refinement_screen_status.json',dict(finished=True,failures=failures,fingerprint=fp,refinement_sha256=rs,ranking_guard_sha256=guard_sha()))


def summarize(root):
    setup=prepare(root);fp=fingerprint(root);rs=source_sha();parts=defaultdict(list);groups=defaultdict(list);diagnostics=[];sources={}
    for row in setup['calibration']+setup['development']:
        path=shard_path(root,'refinement_screen','dense',row['id']);d=json.loads(path.read_text())
        assert d['fingerprint']==fp and d['refinement_sha256']==rs and d['ranking_guard_sha256']==guard_sha()
        expected_probes={f'guarded_sol/{v}/{mode}/s{target}' for v in ('plain','value') for mode in ('gaussian','topk') for target in (25,50,75,90)}
        expected_probes|={f'guarded_diagnostic/{signal}/s{target}' for signal in ('qk','mass','contribution') for target in (25,50,75,90)}
        assert expected_probes<={r['probe'] for r in d['records']},'incomplete signal-only ranking screen'
        original=json.loads(shard_path(root,'screen','dense',row['id']).read_text())
        assert d['completion_tokens']==original['completion_tokens']
        sources[str(path)]=sha(path.read_bytes())
        for r in d['records']:
            for kind in ('overall',r['attention_type']):groups[row['benchmark'],row['split'],r['probe'],kind].append(r)
        diagnostics.extend(dict(id=row['id'],benchmark=row['benchmark'],split=row['split'],**r) for r in d['refinement_diagnostics'])
        if row['split']=='calibration':
            with np.load(path.with_suffix('.npz')) as arrays:
                for key in arrays:
                    name,kind=key.split('__');parts[row['benchmark'],name,kind].append(arrays[key])
    rows=[dict(benchmark=b,split=s,probe=p,attention_type=k,**aggregate(v)) for (b,s,p,k),v in groups.items()]
    csv_write(root/'refinement_screen_summary.csv',rows);_write(root/'refinement_screen_summary.json',rows)
    guarded=[r for r in rows if r['probe'].startswith(('guarded_sol/','guarded_diagnostic/'))]
    csv_write(root/'guarded_ranking_screen_summary.csv',guarded);_write(root/'guarded_ranking_screen_summary.json',guarded)
    _write(root/'refinement_failure_diagnostics.json',diagnostics)
    policies={}
    for (benchmark,name,kind),arrays in parts.items():
        policies.setdefault(benchmark,{}).setdefault(name,{})[kind]={str(t):quantile_threshold(arrays,t) for t in (.25,.5,.75,.9)}
    frozen_write(root/'refinement_threshold_proposals.json',dict(fingerprint=fp,refinement_sha256=rs,sources=sources,policies=policies,
        status='calibration-only proposals; sparse verification still required'))
    _write(root/'refinement_screen_audit.json',dict(complete=True,prompts=len(sources),fingerprint=fp,refinement_sha256=rs,ranking_guard_sha256=guard_sha(),
        all_dense_replays_exact=True,heldout_used=False))
    smoke=json.loads((root/'ranking_guard_smoke.json').read_text())
    assert smoke['passed'] and smoke['fingerprint']==fp and smoke['ranking_guard_sha256']==guard_sha()
    issue=root/'routing_review_issues.json'
    if issue.exists():
        d=json.loads(issue.read_text())
        resolved=[x for x in d['unresolved'] if x['id']=='ranking_exact_mass_nonempty_rescue']
        if resolved:
            _write(issue,dict(unresolved=[x for x in d['unresolved'] if x['id']!='ranking_exact_mass_nonempty_rescue'],
                resolved=d.get('resolved',[])+[dict(x,resolution='separate signal-only guard CUDA smoke passed; all21 dense replays exact; corrected probes complete',
                    ranking_guard_sha256=guard_sha(),screen_audit_sha256=sha((root/'refinement_screen_audit.json').read_bytes())) for x in resolved]))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['screen','summarize']);p.add_argument('--output',type=Path,default=ROOT)
    args=p.parse_args();globals()[args.command](args.output)


if __name__=='__main__':main()
