"""Filesystem-only 15-minute heartbeat for the trajectory guardrail queue."""
import argparse
import json
import os
from pathlib import Path
import time


def _json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def snapshot(root):
    root=Path(root)
    status=_json(root/'status.json',{})
    calibration=list((root/'calibration').glob('**/shards/*/*.json'))
    validation=list((root/'validation').glob('**/shards/*/*.json'))
    final=list((root/'final').glob('**/shards/*/*.json'))
    confirm=list((root/'confirmation').glob('**/shards/*/*.json'))
    recent=sorted(calibration+validation+final+confirm,
                  key=lambda p:p.stat().st_mtime,reverse=True)[:10]
    rates=[];calls=[]
    for path in recent:
        row=_json(path,{})
        counts=row.get('counts',{}).get('whole',{})
        if counts.get('eligible'):
            rates.append(counts['skipped']/counts['eligible'])
            calls.append(row['steps'])
    failures=_json(root/'matrix_failures.json',[])+_json(root/'failures.json',[])
    return dict(utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),
        stage=status.get('stage'),condition=status.get('condition'),id=status.get('id'),
        calibration_shards=len(calibration),validation_shards=len(validation),
        frozen_policies=len(list((root/'configs/thresholds').glob('*.json'))),
        final_shards=len(final),confirmation_shards=len(confirm),
        recent_sparsity=sum(rates)/len(rates) if rates else None,
        recent_calls=sum(calls)/len(calls) if calls else None,
        failures=len(failures),latest_shard_age_seconds=(time.time()-recent[0].stat().st_mtime)
            if recent else None)


def write(root):
    root=Path(root);data=snapshot(root)
    report=(f'# Trajectory guardrail progress\n\nUpdated UTC: {data["utc"]}\n'
        f'Stage: {data["stage"]}; condition: {data["condition"]}; example: {data["id"]}\n'
        f'Calibration/validation shards: {data["calibration_shards"]}/{data["validation_shards"]}\n'
        f'Frozen policies: {data["frozen_policies"]}; final examples: {data["final_shards"]}; '
        f'confirmation examples: {data["confirmation_shards"]}\n'
        f'Recent ten: sparsity {data["recent_sparsity"]}; calls {data["recent_calls"]}\n'
        f'Errors: {data["failures"]}; newest shard age: {data["latest_shard_age_seconds"]}s\n')
    tmp=root/'progress.md.tmp';tmp.write_text(report);os.replace(tmp,root/'progress.md')
    tmp=root/'heartbeat.json.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n')
    os.replace(tmp,root/'heartbeat.json')
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('once','watch'))
    parser.add_argument('--root',type=Path,
        default=Path(__file__).resolve().parents[2]/'results'/'query_adaptive_guardrail_v1')
    args=parser.parse_args()
    if args.mode=='once':write(args.root)
    else:
        while True:
            write(args.root)
            time.sleep(900)
