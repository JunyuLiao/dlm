"""Cheap filesystem-only 15-minute heartbeat; never touches the GPU worker."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def snapshot(root):
    status_path=root/'status.json';launch_path=root/'launch.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    launch=json.loads(launch_path.read_text()) if launch_path.exists() else {}
    shards=list((root/'final/shards').glob('*/*.json')) if (root/'final/shards').exists() else []
    calibration=list((root/'calibration').glob('**/shards/*/*.json')) if (root/'calibration').exists() else []
    errors=json.loads((root/'failures.json').read_text()) if (root/'failures.json').exists() else []
    pid=launch.get('pid');alive=False
    if pid:
        try:os.kill(pid,0);alive=True
        except ProcessLookupError:pass
    recent=[]
    for path in sorted(shards,key=lambda p:p.stat().st_mtime,reverse=True)[:10]:
        try:
            row=json.loads(path.read_text());counts=row['counts']['whole']
            recent.append(dict(iterations=row['steps'],sparsity=counts['skipped']/max(1,counts['eligible'])))
        except (ValueError,KeyError):pass
    elapsed=max(0,time.time()-launch.get('started',time.time()))
    estimate=None
    if len(shards)>100 and status.get('stage')=='final':estimate=(2860-len(shards))*elapsed/len(shards)
    result=dict(updated=time.time(),stage=status.get('stage','unknown'),condition=status.get('condition'),
        current_example=status.get('id'),completed_final=len(shards),total_final=2860,
        completed_calibration=len(calibration),elapsed_seconds=elapsed,eta_seconds=estimate,
        recent_mean_sparsity=sum(x['sparsity'] for x in recent)/len(recent) if recent else None,
        recent_mean_iterations=sum(x['iterations'] for x in recent)/len(recent) if recent else None,
        errors=len(errors),worker_alive=alive,pid=pid)
    return result


def write(root,data):
    tmp=root/'progress.md.tmp';out=root/'progress.md'
    eta='unknown' if data['eta_seconds'] is None else f"{data['eta_seconds']/60:.1f} min"
    lines=['# Query-adaptive experiment progress','',
        f"Updated UTC: {time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime(data['updated']))}",
        f"Stage: {data['stage']}; condition: {data['condition']}; current example: {data['current_example']}",
        f"Final examples: {data['completed_final']}/{data['total_final']}; calibration shards: {data['completed_calibration']}",
        f"Elapsed: {data['elapsed_seconds']/60:.1f} min; ETA: {eta}",
        f"Recent ten final shards: physical sparsity {data['recent_mean_sparsity']}, mean iterations {data['recent_mean_iterations']}",
        f"Worker alive: {data['worker_alive']}; errors: {data['errors']}",'']
    tmp.write_text('\n'.join(lines));os.replace(tmp,out)
    tmp_json=root/'heartbeat.json.tmp';tmp_json.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp_json,root/'heartbeat.json')


def watch(root):
    while True:
        data=snapshot(root);write(root,data)
        if data['stage'] in ('complete','incomplete') or (not data['worker_alive'] and data['pid']):return
        time.sleep(900)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('once','watch','launch'))
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]/'results'/'query_adaptive_v3')
    args=parser.parse_args()
    if args.mode=='launch':
        process=subprocess.Popen([sys.executable,'-m','experiments.value_direction_hopper.query_adaptive_watch','watch','--root',str(args.root)],
            cwd=Path(__file__).resolve().parents[2],stdin=subprocess.DEVNULL,
            stdout=(args.root/'heartbeat.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
        print(process.pid)
    elif args.mode=='once':write(args.root,snapshot(args.root))
    else:watch(args.root)
