"""Report every fixed observation condition, including failures and bookends."""
import argparse
import hashlib
import json
from pathlib import Path

FIELDS=('output_hash','denoise_forwards','canvases','output_tokens','sampler_calls','native_interval_steps')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def compare(root,destination):
    p=read(root/'protocol.json');assert p['trace_crossover']
    leaf=root/'runs/crossover001/attempt001';done=read(leaf/'attempt_complete.json')
    for name,digest in done['files_sha256'].items():assert sha(leaf/name)==digest
    rows=[json.loads(line) for line in (leaf/'records.jsonl').read_text().splitlines()]
    conditions=p['jobs']['crossover001']['conditions'];reference=rows[conditions[0]['row']]
    metrics=[];traces={};groups=[]
    for number,condition in enumerate(conditions):
        index=condition['row'];r=rows[index]
        metrics.append(dict(cycle=number,condition=condition['condition'],row=index,
            **{key:r[key] for key in FIELDS},
            differences_from_first=[key for key in FIELDS if r[key]!=reference[key]]))
        first,last=rows[index-2],rows[index+3]
        groups.append(dict(cycle=number,condition=condition['condition'],
            v31_bookend_differences=[key for key in FIELDS if first[key]!=last[key]]))
        if condition['condition']!='unobserved':
            path=leaf/f'private/trajectory/request{index+1:03d}.json';trace=read(path)
            assert all('logits_gpu128_sha256' in call[side] for call in trace['calls'] for side in ('before','after'))
            traces[number]=trace
    pairs=[];first_index=next(iter(traces));left=traces[first_index]
    for number,right in traces.items():
        first=next((i for i,(a,b) in enumerate(zip(left['calls'],right['calls'])) if a!=b),None)
        pairs.append(dict(left_cycle=first_index,right_cycle=number,full_trace_equal=left==right,
            calls=[len(left['calls']),len(right['calls'])],first_changed_call=first,
            initial_changed_fields=[k for k in left['initial'] if left['initial'][k]!=right['initial'].get(k)],
            first_changed_fields=[side+'.'+k for side in ('before','after')
                for k in left['calls'][first][side] if left['calls'][first][side][k]!=right['calls'][first][side].get(k)] if first is not None else []))
    result=dict(status='complete_diagnostic_reproducibility_unresolved',
        protocol_sha256=sha(root/'protocol.json'),native_marker_sha256=sha(leaf/'attempt_complete.json'),
        observer_qualified=done['observer_qualified'],conditions=metrics,v31_bookends=groups,
        trace_comparisons=pairs,gpu_reservation_s=done['gpu_reservation_s'],
        full_accuracy_released=False,clean_timing=False,
        limitation='one examined question/seed, fixed condition order; timing and context effects are not isolated by output agreement alone')
    destination.mkdir(parents=True,exist_ok=False)
    (destination/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (destination/'analysis_complete.json').write_text(json.dumps(dict(status=result['status'],
        protocol_sha256=sha(root/'protocol.json'),source_sha256=sha(Path(__file__)),
        files_sha256={'summary.json':sha(destination/'summary.json')}),indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True);a=parser.parse_args();compare(a.study,a.destination)
