"""Export numeric evidence from a PRIVATE paused-run tar, without extracting it.

Never publish the input archive: it contains completions and private bindings.
Question labels are sequential anonymous IDs, not hashes of prompts or tokens.
This exporter does not score, validate a full panel, or filter failed requests.
"""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import re
import tarfile

NUMERIC_FIELDS = ('engine_seed', 'repeat', 'wall_s', 'prefill_s', 'decode_span_s',
    'denoise_forward_count', 'commit_forward_count', 'scheduler_denoise_forward_count',
    'scheduler_commit_forward_count', 'speculative_unused_denoising', 'scheduler_steps',
    'prefill_steps', 'output_tokens', 'graph_captures_timed', 'compilation_deltas')
ENUM_FIELDS = {'risk_state', 'scope', 'mu_mode', 'route_storage', 'bootstrap_policy',
    'threshold_shift', 'sensitivity', 'consumer', 'output_score_precision', 'output_layout',
    'observation_producer', 'selector', 'parent_v20_arm'}
FORBIDDEN_KEY = re.compile(r'prompt|gold|answer|completion|fingerprint|hash|path|credential|secret|logit|text|token',re.I)


def numeric(value):
    """Only numbers, booleans, null and nested numeric structure survive."""
    if value is None or type(value) in (bool,int):return value
    if type(value) is float:
        if not math.isfinite(value):raise ValueError('Non-finite measurement')
        return value
    if isinstance(value,dict):
        return {k:numeric(v) for k,v in value.items()
                if isinstance(k,str) and re.fullmatch(r'[A-Za-z0-9_]+',k)
                and not FORBIDDEN_KEY.search(k) and not isinstance(v,str)}
    if isinstance(value,list):
        return [numeric(v) for v in value if not isinstance(v,str)]
    raise ValueError('Unexpected nonnumeric measurement')


def sanitize(row, suite, worker, ordinal, question):
    if row.get('arm') not in ('dense','native','allkept','method'):raise ValueError('Unknown arm')
    if not re.fullmatch(r'[0-9a-f]{40}',row.get('deploy_commit','')):raise ValueError('Missing source commit')
    if not re.fullmatch(r'b[0-7]_(dense|native|allkept|method)',worker):raise ValueError('Unknown worker name')
    out=dict(suite=suite,worker=worker,ordinal=ordinal,question=question,arm=row['arm'],deploy_commit=row['deploy_commit'])
    for key in NUMERIC_FIELDS:
        if key in row:out[key]=numeric(row[key])
    n=row.get('denoise_forward_count',0)
    if n>0:out['S_per_N_s']=row['decode_span_s']/n
    rec=row.get('receipts') or {}
    out['receipt_numeric']=numeric(rec)
    method=(rec.get('method') or {}).get('effective_method')
    if method is not None:
        effective=numeric(method)
        for key in ENUM_FIELDS:
            value=method.get(key)
            if isinstance(value,str):
                if not re.fullmatch(r'[A-Za-z0-9_+-]{1,96}',value):raise ValueError('Unexpected method enum')
                effective[key]=value
        out['effective_method']=effective
    return out


def export(archive, output, suite):
    if suite not in ('longbench','aime','humaneval'):raise ValueError('Unknown suite')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with tarfile.open(archive,'r:gz') as tar:
        names=tar.getnames()
        state=json.load(tar.extractfile('status.private.json'))
        workers={}
        for name in sorted(names):
            if re.fullmatch(r'b[0-7]_(dense|native|allkept|method)/records.jsonl',name):
                rows=[json.loads(line) for line in tar.extractfile(name) if line.strip()]
                workers[name.split('/')[0]]=rows
        # Private question indices are used only to assign consistent anonymous
        # labels across arms/seeds. Neither indices nor their hashes are exported.
        keys=sorted({(r['dataset'],r['index']) for rows in workers.values() for r in rows})
        questions={key:'q'+str(i).zfill(3) for i,key in enumerate(keys)}
        jobs={j['name']:j for j in state['jobs']}
        summaries=[];total=0
        with gzip.open(output/'requests.jsonl.gz','wt',encoding='utf-8') as out:
            for worker,rows in workers.items():
                terminal_name=worker+'/terminal.json'
                terminal=json.load(tar.extractfile(terminal_name)) if terminal_name in names else {}
                complete=terminal.get('complete') is True and jobs.get(worker,{}).get('exit_code')==0
                for ordinal,row in enumerate(rows):
                    record=sanitize(row,suite,worker,ordinal,questions[(row['dataset'],row['index'])])
                    record['worker_complete']=complete
                    out.write(json.dumps(record,ensure_ascii=True,separators=(',',':'))+'\n');total+=1
                summaries.append(dict(worker=worker,complete=complete,timed_records=len(rows),
                    expected_timed=terminal.get('expected_timed'),gpu_reserved_seconds=jobs.get(worker,{}).get('gpu_reserved_seconds'),
                    W_sum_s=sum(r['wall_s'] for r in rows),S_sum_s=sum(r['decode_span_s'] for r in rows),
                    N_sum=sum(r['denoise_forward_count'] for r in rows)))
        with (output/'workers.csv').open('w',encoding='utf-8',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
        result=dict(suite=suite,records=total,anonymous_questions=len(questions),workers=summaries,
            generation_complete=state.get('generation_complete') is True,scored=False,
            note='Unscored pause snapshot; worker sums are accounting, not paired speed/accuracy estimates.')
        (output/'inventory.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return dict(suite=suite,records=total,anonymous_questions=len(questions),workers=len(summaries))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--suite',required=True)
    args=parser.parse_args();print(json.dumps(export(args.archive,args.output,args.suite)))
