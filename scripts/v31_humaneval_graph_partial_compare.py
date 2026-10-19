"""Analyze completed contrasts from an expired diagnostic, never qualify a panel."""
import argparse
import hashlib
import json
from pathlib import Path
from scripts import v31_humaneval_residual_repro_compare as helper

read,sha,jsonl=helper.read,helper.sha,helper.jsonl


def run(root,job,destination):
    p=read(root/'protocol.json');config=p['jobs'][job]
    leaf=root/'runs'/job/'attempt001';marker=read(leaf/'failed_attempt.json')
    assert marker['status']=='failed' and 'GPU reservation expired' in marker['errors']
    assert marker['protocol_sha256']==sha(root/'protocol.json')
    for name,digest in p['files_sha256'].items():assert sha(root/name)==digest
    for name,digest in marker['files_sha256'].items():assert sha(leaf/name)==digest
    rows=jsonl(leaf/'records.jsonl');assert len(rows)==marker['records']<config['requests']
    pairs=[]
    for group in config['bookend_groups']:
        a,b=group['original'],group['final']
        if max(a,b)>=len(rows):continue
        assert rows[a]['index']==rows[b]['index'] and rows[a]['panel_seed']==rows[b]['panel_seed']
        assert rows[a]['receipts']['adapter']['humaneval_live_arm']==rows[b]['receipts']['adapter']['humaneval_live_arm']
        full=read(leaf/f'private/trajectory/request{a+1:03d}.json')
        full_graph=read(leaf/f'private/graphs/request{a+1:03d}.json')
        light=read(leaf/f'private/graphs/request{b+1:03d}.json');assert light['mode']=='graph_only'
        x=[c['before']['logits_gpu128_sha256'] for c in full['calls']]
        y=helper.logit_digests(light)
        assert len(x)==rows[a]['sampler_calls'] and len(y)==rows[b]['sampler_calls']
        first=next((i for i,(u,v) in enumerate(zip(x,y)) if u!=v),None)
        first_full=next((i for i,(u,v) in enumerate(zip(x,y)) if u!=v and
            full['calls'][i]['before']['valid_canvas_len']==[256]),None)
        changed=[k for k in helper.FIELDS if rows[a][k]!=rows[b][k]]
        pairs.append(dict(label=group['label'],index=rows[a]['index'],seed=rows[a]['panel_seed'],
            native_fields_changed=changed,
            native_metrics=[{k:row[k] for k in helper.FIELDS} for row in (rows[a],rows[b])],
            before_logit_fingerprints_equal=x==y,first_raw_logit_changed_call=first,
            first_changed_full_canvas_reference_call=first_full,
            boundaries=helper.graph_contrast(full_graph,light,first_full if first_full is not None else first),
            before_first_changed_full_state=full['calls'][first_full]['before'] if first_full is not None else None))
    result=dict(status='partial_expired_diagnostic_only',full_accuracy_released=False,
        technical_repeatability_qualified=False,independent_engine_replication=False,
        protocol_sha256=sha(root/'protocol.json'),failure_marker_sha256=sha(leaf/'failed_attempt.json'),
        requests_complete=len(rows),requests_planned=config['requests'],contrasts_complete=len(pairs),contrasts=pairs,
        native_contrasts_changed=sum(bool(r['native_fields_changed']) for r in pairs),
        raw_logit_contrasts_changed=sum(not r['before_logit_fingerprints_equal'] for r in pairs),
        failure_errors=marker['errors'],gpu_reservation_s=marker['gpu_reservation_s'],
        limitations=['No completion/qualification marker; missing exit compiler inventory remains a real limitation',
            'Only contrasts with both public rows complete are analyzed; absent rows are not zero or agreement',
            'Partial tensor argument scope and full storage extents can include inactive values',
            'Hash localization requires actual tensor capture and exact-input operator replay',
            'Pure unobserved bookends and second engine did not complete; no observer equivalence is established'])
    destination.mkdir(parents=True,exist_ok=False)
    text=json.dumps(result,indent=2,sort_keys=True)+'\n'
    assert all(s not in text for s in ('/home/','"completion":','"prompt":','token_ids'))
    (destination/'summary.json').write_text(text)
    (destination/'analysis_complete.json').write_text(json.dumps(dict(status=result['status'],
        source_sha256=sha(Path(__file__)),helper_source_sha256=sha(Path(helper.__file__)),
        files_sha256={'summary.json':sha(destination/'summary.json')}),indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('contrasts','limitations')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--job',default='clean001')
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();run(args.study,args.job,args.destination)
