"""Observe the sweep every fifteen minutes without issuing GPU work."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import time


def watch(root):
    launch=json.loads(sorted(root.glob('launch_*.json'))[-1].read_text())
    while True:
        state=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
        proc=Path(f'/proc/{launch["pid"]}/stat')
        alive=proc.exists() and proc.read_text().split(') ',1)[1].split()[0]!='Z'
        counts={regime:sum(1 for _ in (root/regime/'shards').glob('*/*.json')) for regime in ('adaptive','fixed4')}
        row=dict(time=datetime.now(timezone.utc).isoformat(),worker_alive=alive,counts=counts,state=state,
                 free_bytes=shutil.disk_usage(root).free)
        with (root/'health.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if not alive or state.get('stage') in ('complete','incomplete'):return
        for _ in range(15):time.sleep(60)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    watch(p.parse_args().root)
