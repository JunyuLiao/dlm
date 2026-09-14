"""Passive 15-minute status logging; no model or data mutations."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

pid=int(sys.argv[1]);root=Path(sys.argv[2])
while True:
    proc=Path(f'/proc/{pid}/stat')
    try: alive=proc.read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError: alive=False
    row=dict(time=time.strftime('%FT%TZ',time.gmtime()),pid=pid,alive=alive,
        completed={d.name:len(list((d/'shards').glob('*.json'))) for d in (root/'conditions').glob('*') if d.is_dir()},
        failures=len((root/'failures.jsonl').read_text().splitlines()) if (root/'failures.jsonl').exists() else 0)
    if (root/'progress.json').exists():
        row['progress']=json.loads((root/'progress.json').read_text())
        row['current_sample_seconds']=round(time.time()-row['progress']['started'])
    row['gpu']=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    with (root/'monitor.jsonl').open('a') as f: f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(row),flush=True)
    if not alive: break
    time.sleep(900)
