"""Append a 15-minute status snapshot for the resumable full matrix queue."""
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
state=ROOT/'results/v31_value_matrix_20261011/queue_state.json'; out=ROOT/'results/v31_value_matrix_20261011/monitor.log'
while True:
    try:
        x=json.loads(state.read_text()) if state.exists() else {}
        counts={}
        for v in x.values(): counts[v.get('status','unknown')]=counts.get(v.get('status','unknown'),0)+1
        gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
        rec={'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'counts':counts,'gpu':gpu}
        for k,v in x.items():
            if v.get('status')=='running':
                a=Path(v.get('attempt','')); rec['running']={'id':k,'attempt':str(a),'records':sum(1 for _ in open(a/'records.jsonl')) if (a/'records.jsonl').exists() else 0}
        with out.open('a') as f: f.write(json.dumps(rec)+'\n')
        if counts.get('complete',0)>=len(x): break
    except Exception as e:
        with out.open('a') as f: f.write(json.dumps({'error':repr(e)})+'\n')
    time.sleep(900)
