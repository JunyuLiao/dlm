"""Seal the bounded observing-FA4 numerical qualification, never full accuracy."""
import argparse
import hashlib
import json
from pathlib import Path

FIELDS=('output_hash','denoise_forwards','canvases','output_tokens','sampler_calls','native_interval_steps','rng_seed','prompt_sha256')


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def jsonl(path):return [json.loads(x) for x in path.read_text().splitlines()]


def qualify(root,analysis,destination):
    assert not destination.exists()
    p=read(root/'protocol.json');assert p['observing_fa4_no_overlap'] and not p['graph_boundary_fingerprints']
    for name,digest in p['files_sha256'].items():assert sha(root/name)==digest
    marker=read(analysis/'analysis_complete.json');assert marker['protocol_sha256']==sha(root/'protocol.json')
    for name,digest in marker['files_sha256'].items():assert sha(analysis/name)==digest
    report=read(analysis/'summary.json');assert report['protocol_sha256']==sha(root/'protocol.json')
    contrasts=report['contrasts'];errors=[];leaves={};records={};within=[]
    if len(contrasts)!=72:errors.append('missing independent observed-case coverage')
    for row in contrasts:
        if not row['all_logit_fingerprints_present']:errors.append('missing raw-logit coverage '+row['label'])
        if not (row['full_trajectories_equal'] or row['differences_confined_to_inactive_initial_argmax']):
            errors.append('independent observed native state mismatch '+row['label'])
        if not row['residual_fingerprints_equal']:errors.append('independent residual-cache mismatch '+row['label'])
    for job,config in p['jobs'].items():
        leaf=root/'runs'/job/'attempt001';receipt=read(leaf/'attempt_complete.json')
        assert receipt['status']=='complete' and receipt['protocol_sha256']==sha(root/'protocol.json')
        for name,digest in receipt['files_sha256'].items():assert sha(leaf/name)==digest
        data=jsonl(leaf/'records.jsonl');records[job]=data;leaves[job]=sha(leaf/'attempt_complete.json')
        if len(data)!=152 or len(config['bookend_groups'])!=72:errors.append('native coverage '+job)
        if receipt['observer_qualified'] is not True or len(receipt['bookends'])!=72:errors.append('observer equivalence '+job)
        for i,row in enumerate(data):
            a=(row.get('receipts') or {}).get('adapter') or {}
            if (row.get('cuda_graph_captures')!=0 or a.get('observing_fa4_intra_wg_overlap') is not False or
                a.get('observing_fa4_policy_sha256')!=sha(root/'private/observe_no_overlap.py') or
                a.get('observing_fa4_extra_forwards')!=0 or type(a.get('observing_fa4_calls')) is not int):
                errors.append(f'execution receipt {job} row{i}')
        for group in config['bookend_groups']:
            a,b=data[group['original']],data[group['final']]
            changed=[k for k in FIELDS if a[k]!=b[k]]
            within.append(dict(job=job,label=group['label'],passed=not changed,fields_changed=changed))
            if changed:errors.append('observed versus pure '+job+' '+group['label'])
    assert len(records)==2
    left,right=records;independent=[]
    for context in ('original','final'):
        groups=p['jobs'][left]['bookend_groups'];other=p['jobs'][right]['bookend_groups']
        assert [g['label'] for g in groups]==[g['label'] for g in other]
        for a,b in zip(groups,other):
            x,y=records[left][a[context]],records[right][b[context]]
            changed=[k for k in FIELDS if x[k]!=y[k]]
            independent.append(dict(context=context,label=a['label'],passed=not changed,fields_changed=changed))
            if changed:errors.append('independent '+context+' '+a['label'])
    warmup_changes=[i for i in range(8) if any(records[left][i][k]!=records[right][i][k] for k in FIELDS)]
    if warmup_changes:errors.append('independent warmup native metrics mismatch')
    result=dict(status='failed_native_no_overlap_qualification' if errors else 'native_no_overlap_qualified',
        errors=sorted(set(errors)),protocol_sha256=sha(root/'protocol.json'),source_commit=p['source_commit'],
        policy_source_sha256=sha(root/'private/observe_no_overlap.py'),leaf_markers_sha256=leaves,
        analysis_marker_sha256=sha(analysis/'analysis_complete.json'),analysis_source_sha256=marker['analysis_source_sha256'],
        qualifier_source_sha256=sha(Path(__file__)),observed_cases=72,within_engine_cases=within,independent_cases=independent,
        inactive_initial_argmax_cases=[c['label'] for c in contrasts if c['differences_confined_to_inactive_initial_argmax']],
        warmup_changed_rows=warmup_changes,full_accuracy_released=False,clean_latency_qualified=False,
        scope='six diagnostic questions, seed42/43, thinking on, K256; observed complete sampler/cache traces plus pure recorded native metrics',
        next_gate='corrected audit/clean and dense/all-kept controls; work-only V2 calibration before full accuracy')
    destination.mkdir(parents=True)
    with (destination/'qualification.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True)
    with (destination/'qualification_complete.json').open('x') as f:json.dump(dict(status=result['status'],files_sha256={'qualification.json':sha(destination/'qualification.json')}),f,indent=2)
    print(json.dumps(dict(status=result['status'],errors=result['errors'],full_accuracy_released=False)),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('study',type=Path);parser.add_argument('analysis',type=Path);parser.add_argument('destination',type=Path)
    args=parser.parse_args();qualify(args.study,args.analysis,args.destination)
