"""Filesystem-only heartbeat for the versioned tile-broadcast extension."""
import argparse
import json
import os
from pathlib import Path
import time


def count(root,name):
    directory=root/name
    return sum(1 for _ in directory.glob('**/shards/*/*.json')) if directory.exists() else 0


def write(root):
    status_path=root/'status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    data=dict(updated_utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),
        stage=status.get('stage'),condition=status.get('condition'),id=status.get('id'),
        thresholds=len(list((root/'configs/thresholds').glob('*.json'))),
        calibration=count(root,'calibration'),final=count(root,'final'),confirmation=count(root,'confirmation'),
        errors=sum(len(json.loads((root/n).read_text())) for n in ('failures.json','confirmation_failures.json') if (root/n).exists()))
    lines=['# Tile-mean/max T comparison progress','',
        f"Updated UTC: {data['updated_utc']}",
        f"Stage: {data['stage']}; condition: {data['condition']}; example: {data['id']}",
        f"Thresholds: {data['thresholds']}/4; calibration shards: {data['calibration']}",
        f"Final examples: {data['final']}/520; confirmation: {data['confirmation']}/1040",
        f"Errors: {data['errors']}",'']
    tmp=root/'progress.md.tmp';tmp.write_text('\n'.join(lines));os.replace(tmp,root/'progress.md')
    tmp=root/'heartbeat.json.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,root/'heartbeat.json')
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('once','watch'))
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]/'results'/'query_adaptive_tile_controls_v2')
    args=parser.parse_args()
    if args.mode=='once':write(args.root)
    else:
        while True:
            data=write(args.root)
            if data['confirmation']==1040 or data['errors']:break
            time.sleep(900)
