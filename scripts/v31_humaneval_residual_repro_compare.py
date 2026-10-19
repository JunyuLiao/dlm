"""Compare bounded diagnostics without reading generated programs into reports."""
import argparse
import hashlib
import json
from pathlib import Path

FIELDS=('output_hash','denoise_forwards','canvases','output_tokens','sampler_calls','native_interval_steps')


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def jsonl(path):return [json.loads(line) for line in path.read_text().splitlines()]


def compare(root,destination,workspace,quiet=False):
    p=read(root/'protocol.json');data={};inventory=[]
    for name,c in p['jobs'].items():
        leaf=root/'runs'/name/'attempt001';marker=read(leaf/'attempt_complete.json')
        for rel,digest in marker['files_sha256'].items():assert sha(leaf/rel)==digest
        rows=jsonl(leaf/'records.jsonl');old=jsonl(workspace/c['original']/'records.jsonl')
        differences=[]
        for i in range(c['observed_row']):
            changed=[key for key in FIELDS if rows[i][key]!=old[i][key]]
            if changed:differences.append(dict(row=i,index=rows[i]['index'],seed=rows[i]['panel_seed'],
                warmup=i<c['warmup_requests'],method=rows[i]['receipts']['adapter']['humaneval_live_arm'],fields=changed))
        inventory.append(dict(job=name,observer_qualified=marker['observer_qualified'],
            original_prefix_changes=differences,gpu_reservation_s=marker['gpu_reservation_s'],
            marker_sha256=sha(leaf/'attempt_complete.json')))
        groups=c.get('bookend_groups',[dict(observed=c['observed_row'],label='residual_direction_guard')])
        data[name]={}
        for group in groups:
            index=group['observed'];label=group.get('label',group.get('arm'))
            data[name][label]=dict(trajectory=read(leaf/f'private/trajectory/request{index+1:03d}.json'),
                residual=read(leaf/f'private/residual/request{index+1:03d}.json'))
    left,right=list(data);contrasts=[]
    assert set(data[left])==set(data[right])
    for label,x in data[left].items():
        y=data[right][label];a,b=x['trajectory'],y['trajectory']
        fingerprints=all('logits_gpu128_sha256' in trace['initial'] and all(
            'logits_gpu128_sha256' in call[side] for call in trace['calls'] for side in ('before','after'))
            for trace in (a,b))
        if p.get('mixed_group'):assert fingerprints
        first=next((i for i,(u,v) in enumerate(zip(a['calls'],b['calls'])) if u!=v),None)
        first_residual=next((i for i,(u,v) in enumerate(zip(x['residual']['events'],y['residual']['events'])) if u!=v),None)
        contrasts.append(dict(label=label,sampler_calls=[len(a['calls']),len(b['calls'])],
            all_logit_fingerprints_present=fingerprints,
            initial_fields_changed=[k for k in a['initial'] if a['initial'][k]!=b['initial'].get(k)],
            full_trajectories_equal=a==b,first_changed_call=first,
            first_changed_fields=[side+'.'+key for side in ('before','after')
                for key in a['calls'][first][side] if a['calls'][first][side][key]!=b['calls'][first][side].get(key)] if first is not None else [],
            residual_events=[len(x['residual']['events']),len(y['residual']['events'])],
            residual_fingerprints_equal=x['residual']==y['residual'],first_residual_event_changed=first_residual))
    result=dict(status='complete_diagnostic_reproducibility_unresolved',protocol_sha256=sha(root/'protocol.json'),
        jobs=inventory,contrasts=contrasts,full_accuracy_released=False,
        limitations=['instrumentation qualification is reported per job, not assumed',
            'equal fingerprints in a bounded pair do not repair original independent-prefix differences',
            'GPU integer logit fingerprints, if present, locate changes and require actual tensor confirmation'])
    destination.mkdir(parents=True,exist_ok=False)
    text=json.dumps(result,indent=2,sort_keys=True)+'\n'
    assert all(s not in text for s in ('/home/','"completion":','"prompt":','token_ids'))
    (destination/'summary.json').write_text(text)
    (destination/'analysis_complete.json').write_text(json.dumps(dict(status=result['status'],
        protocol_sha256=sha(root/'protocol.json'),analysis_source_sha256=sha(Path(__file__)),
        files_sha256={'summary.json':sha(destination/'summary.json')}),indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],contrasts=len(contrasts),
        changed_trajectories=sum(not r['full_trajectories_equal'] for r in contrasts),
        observer_qualified=[r['observer_qualified'] for r in inventory])) if quiet else text)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True);parser.add_argument('--workspace',type=Path,default=Path.cwd())
    parser.add_argument('--quiet',action='store_true')
    args=parser.parse_args();compare(args.study,args.destination,args.workspace,args.quiet)
