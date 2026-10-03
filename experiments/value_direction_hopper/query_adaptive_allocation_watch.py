"""Cheap filesystem-only 15-minute heartbeat for the allocation comparison."""
import argparse
import json
import os
from pathlib import Path
import time


def paths(root,subdirectory):
    directory=root/subdirectory
    return list(directory.glob('**/shards/*/*.json')) if directory.exists() else []


def snapshot(root):
    status_file=root/'status.json';status=json.loads(status_file.read_text()) if status_file.exists() else {}
    finals=paths(root,'final');calibration=paths(root,'calibration');confirmation=paths(root,'confirmation')
    failures=[]
    for name in ('failures.json','confirmation_failures.json'):
        path=root/name
        if path.exists():failures.extend(json.loads(path.read_text()))
    latest=sorted(finals or confirmation or calibration,key=lambda p:p.stat().st_mtime,reverse=True)[:10]
    recent=[]
    for path in latest:
        try:
            row=json.loads(path.read_text());whole=row['counts']['whole']
            recent.append((whole['skipped']/max(1,whole['eligible']),row['steps']))
        except (ValueError,KeyError):pass
    return dict(utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),
        stage=status.get('stage','not started'),condition=status.get('condition'),
        current_id=status.get('id'),calibration_shards=len(calibration),
        frozen_thresholds=len(list((root/'configs/thresholds').glob('*.json'))),
        final_shards=len(finals),final_expected=1300,
        confirmation_shards=len(confirmation),confirmation_expected=1560,
        recent_mean_sparsity=sum(x[0] for x in recent)/len(recent) if recent else None,
        recent_mean_steps=sum(x[1] for x in recent)/len(recent) if recent else None,
        failures=len(failures))


def write(root):
    data=snapshot(root);tmp=root/'progress.md.tmp';out=root/'progress.md'
    tmp.write_text('# T allocation comparison progress\n\n'
        f"Updated UTC: {data['utc']}\n"
        f"Stage: {data['stage']}; condition: {data['condition']}; example: {data['current_id']}\n"
        f"Calibration shards: {data['calibration_shards']}; frozen thresholds: {data['frozen_thresholds']}/8\n"
        f"Final examples: {data['final_shards']}/{data['final_expected']}\n"
        f"Confirmation examples: {data['confirmation_shards']}/{data['confirmation_expected']}\n"
        f"Recent ten: physical sparsity {data['recent_mean_sparsity']}; mean calls {data['recent_mean_steps']}\n"
        f"Errors: {data['failures']}\n")
    os.replace(tmp,out)
    tmp=root/'heartbeat.json.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n')
    os.replace(tmp,root/'heartbeat.json')
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('once','watch'))
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]/'results'/'query_adaptive_allocation_v1')
    args=parser.parse_args()
    if args.mode=='once':write(args.root)
    else:
        while True:
            data=write(args.root)
            if (data['final_shards']==1300 and data['confirmation_shards']==1560) or data['failures']:
                break
            time.sleep(900)
