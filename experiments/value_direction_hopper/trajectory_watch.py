"""Low-frequency health records for a single diagnostic worker."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time


def watch(root,interval=900):
    launch=json.loads(sorted(root.glob('launch_*.json'))[-1].read_text())
    while True:
        status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
        try:
            os.kill(launch['pid'],0);alive=True
        except ProcessLookupError:
            alive=False
        counts={regime:len(list((root/regime/'shards').glob('*/*.json'))) for regime in ('adaptive','fixed512')}
        record=dict(time=datetime.now(timezone.utc).isoformat(),alive=alive,counts=counts,
            status=status,free_bytes=shutil.disk_usage(root).free)
        with (root/'health.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        if not alive or status.get('status') in ('complete','incomplete'):
            return
        # Short interruptible waits; substantive health reads remain 15 minutes apart.
        deadline=time.monotonic()+interval
        while time.monotonic()<deadline:
            time.sleep(min(60,max(0,deadline-time.monotonic())))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    watch(p.parse_args().root)
