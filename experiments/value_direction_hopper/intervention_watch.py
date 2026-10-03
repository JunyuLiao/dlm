"""Fifteen-minute health checks; no analysis or GPU inference."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time


def watch(root):
    launch=json.loads((root/'launch.json').read_text());pid=launch['pid']
    while True:
        state=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
        process=Path(f'/proc/{pid}/stat');alive=process.exists() and process.read_text().split(') ',1)[1].split()[0]!='Z'
        counts={stage:len(list((root/stage/'shards').glob('*/*.json'))) for stage in ('pilot','calibration','development','final')}
        policies=len(list((root/'policies').glob('*.json')))
        errors=json.loads((root/'failures.json').read_text()) if (root/'failures.json').exists() else []
        record=dict(time=datetime.now(timezone.utc).isoformat(),alive=alive,completed=counts,calibrated_policies=policies,
                    current=state.get('condition'),stage=state.get('stage'),errors=len(errors))
        with (root/'health.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        if not alive or state.get('stage') in ('complete','incomplete'):return
        for _ in range(15):time.sleep(60)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);watch(p.parse_args().root)
