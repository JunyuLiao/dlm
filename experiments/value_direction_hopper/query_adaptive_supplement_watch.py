"""Filesystem-only 15-minute heartbeat for the post-sweep GPU jobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def count(path):
    return sum(1 for _ in path.glob('**/shards/*/*.json')) if path.exists() else 0


def snapshot(root):
    confirmation=count(root/'confirmation')
    timing=count(root/'timing')
    replay=count(root/'diagnostic_replays')
    plan=root/'confirmation_plan.json'
    selected=json.loads(plan.read_text()).get('selected',[]) if plan.exists() else []
    confirmation_total=2*2*130*len(selected)
    timing_summary=root/'timing_summary.json'
    complete=timing_summary.exists() and json.loads(timing_summary.read_text()).get('complete',False)
    recent=[]
    for path in (root/'confirmation',root/'timing'):
        if path.exists():recent.extend(path.glob('**/shards/*/*.json'))
    latest=max((p.stat().st_mtime for p in recent),default=None)
    return dict(updated=time.time(),primary='2860/2860 audited',replay=f'{replay}/13',
        confirmation=f'{confirmation}/{confirmation_total}',timing_shards=timing,
        latest_shard_utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime(latest)) if latest else None,
        timing_complete=complete)


def write(root,data):
    target=root/'progress.md';tmp=root/'progress.md.tmp'
    tmp.write_text('# Query-adaptive experiment progress\n\n'
        f"Updated UTC: {time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime(data['updated']))}\n"
        f"Primary: {data['primary']}; same-state replay: {data['replay']}\n"
        f"Seed confirmation: {data['confirmation']}; timing shards: {data['timing_shards']}\n"
        f"Latest supplemental shard UTC: {data['latest_shard_utc']}\n"
        f"Complete uninstrumented timing: {data['timing_complete']}\n")
    os.replace(tmp,target)
    tmp=root/'supplement_heartbeat.json.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n')
    os.replace(tmp,root/'supplement_heartbeat.json')


def watch(root):
    while True:
        data=snapshot(root);write(root,data)
        if data['timing_complete']:return
        time.sleep(900)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('once','watch','launch'))
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]/'results'/'query_adaptive_v3')
    args=parser.parse_args()
    if args.mode=='launch':
        process=subprocess.Popen([sys.executable,'-m',
            'experiments.value_direction_hopper.query_adaptive_supplement_watch','watch','--root',str(args.root)],
            cwd=Path(__file__).resolve().parents[2],stdin=subprocess.DEVNULL,
            stdout=(args.root/'supplement_heartbeat.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
        print(process.pid)
    elif args.mode=='once':write(args.root,snapshot(args.root))
    else:watch(args.root)
