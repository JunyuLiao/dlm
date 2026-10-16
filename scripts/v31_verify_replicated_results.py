"""Independent stdlib audit of completed replicated native workload reports.

Does not import the generation or aggregation implementation. Reads only
numeric public records, scorer outputs, completion bindings and frozen config;
private generation files are hashed, never decoded or printed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

CONTROL='current_v31_control'
REFERENCES=('dense_full_fix51994','dense_piecewise','allkept_fa4')


def read(path):return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def require(condition,message):
    if not condition:raise RuntimeError(message)


def near(a,b):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10)


def verify(study,suite):
    protocol=read(study/'protocol.json');count=protocol['suites'][suite]['count']
    report=study/'reports'/suite;summary=read(report/'summary.json');done=read(report/'final_complete.json')
    require(done['summary_sha256']==sha(report/'summary.json'),'changed report')
    require(done['protocol_sha256']==sha(study/'protocol.json')==summary['protocol_sha256'],'protocol mismatch')
    markers=list((study/'scoring'/suite).glob('attempt*/scoring_complete.json'))
    require(len(markers)==1,'ambiguous official scoring');marker=markers[0];scoring=marker.parent;proof=read(marker)
    require(proof['protocol_sha256']==sha(study/'protocol.json'),'scoring protocol mismatch')
    for rel,digest in proof['inputs'].items():require(sha(study/rel)==digest,'changed scored generation')
    for rel,digest in proof['outputs'].items():require(sha(scoring/rel)==digest,'changed scorer output')
    scores=read(scoring/'official.official.json');coverage=read(scoring/'official.summary.json')['coverage']
    require(len(scores)==15,'unexpected score labels')
    require(all(c['scored']==count and not c['missing'] and not c['dropped'] for c in coverage.values()),'incomplete official coverage')
    rows={}
    for path in sorted((study/'shared_runs'/suite).glob('shard*/repeat*/attempt*/exports/*/records.jsonl')):
        rows.setdefault(path.parent.name,[]).extend(json.loads(x) for x in path.read_text().splitlines())
    for arm in REFERENCES:
        rows[arm]=[]
        for path in sorted((study/'runs'/suite/arm).glob('shard*/attempt*/records.jsonl')):
            rows[arm].extend(json.loads(x) for x in path.read_text().splitlines())
    require(set(rows)==set(scores),'public record labels differ')
    for label,records in rows.items():
        keys=[f"{r['dataset']}|{r['index']}|{r['panel_seed']}|{r['repeat']}" for r in records]
        require(len(keys)==len(set(keys))==count and set(keys)==set(scores[label]),'public and scored cohorts differ')
        require(all(r['cuda_graph_captures']==0 for r in records),'unexpected measured capture')
    if suite=='ruler128k':
        info=read(study/'private/pools/ruler128k/ruler128k_v33ofc_subset_rowinfo.json')['rows']
        task_of={r['index']:r['task'] for r in info};tasks={}
        for key in next(iter(scores.values())):tasks.setdefault(task_of[int(key.split('|')[1])],0);tasks[task_of[int(key.split('|')[1])]]+=1
        require(len(tasks)==13 and set(tasks.values())=={8},'RULER mean requires the frozen balanced task cohort')
    scale=1 if suite=='ruler128k' else 100
    for arm,result in summary['summaries'].items():
        labels=([f'r{i}_control_{pos}' for i in (1,2) for pos in ('before','after')] if arm==CONTROL else
                [arm] if arm in REFERENCES else [f'r{i}_{arm}' for i in (1,2)])
        rr=[r for label in labels for r in rows[label]];work=result['work']
        require(near(result['score_points'],scale*statistics.mean(float(v) for label in labels for v in scores[label].values())),'score arithmetic')
        require(result['problem_trials']==len(rr),'trial count')
        for field,metric in [('denoise_forwards','N'),('canvases','C'),('output_tokens','T')]:
            total=sum(r[field] for r in rr)
            require(work[metric]==total and near(work['mean_'+metric],total/len(rr)),'trajectory aggregation')
        require(near(work['N_per_C'],work['N']/work['C']),'calls per canvas')
        for field in ('global_eligible_tiles','global_kept_tiles','overall_eligible_rectangles','overall_skipped_rectangles'):
            require(work[field]==sum(r['receipts']['adapter'][field] for r in rr),'physical work aggregation')
        require(near(work['mean_global_kept_tiles'],work['global_kept_tiles']/len(rr)),'unequal repetition work normalization')
    changes=sum(scores[f'r{i}_control_before'][k]!=scores[f'r{i}_control_after'][k] for i in (1,2) for k in scores[f'r{i}_control_before'])
    return dict(status='passed',suite=suite,cells_per_label=count,score_labels=len(scores),
        bookend_score_disagreements=changes,summary_sha256=sha(report/'summary.json'),scoring_complete_sha256=sha(marker),
        verifier_sha256=sha(Path(__file__)),scope='independent completion/hash/coverage/repetition/N-C-T/physical-work audit',timing='paused')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--suite',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=verify(args.study.resolve(),args.suite)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():require(read(args.output)==result,'preserve existing verification')
    else:
        with args.output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(result))
