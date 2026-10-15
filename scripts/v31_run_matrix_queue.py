"""Run the frozen full-matrix queue sequentially; completed attempts are skipped."""
import argparse,json,subprocess,time
from pathlib import Path

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--queue',type=Path,required=True); ap.add_argument('--state',type=Path,required=True); ap.add_argument('--max-jobs',type=int,default=0); a=ap.parse_args(); state={}
 if a.state.exists(): state=json.loads(a.state.read_text())
 jobs=[json.loads(x) for x in a.queue.read_text().splitlines() if x.strip()]; done=0
 for j in jobs:
  aid=j['id']; base_attempt=Path(j['attempt']); attempt=base_attempt
  completed=list(base_attempt.parent.glob('attempt*/attempt_complete.json')) if base_attempt.parent.exists() else []
  if completed:
   marker=max(completed, key=lambda p: p.stat().st_mtime); state[aid]={'status':'complete','attempt':str(marker.parent)}; continue
  if attempt.exists():
   nums=[int(p.name.replace('attempt','')) for p in base_attempt.parent.glob('attempt[0-9][0-9][0-9]') if p.name.replace('attempt','').isdigit()]
   attempt=base_attempt.parent/f'attempt{max(nums+[0])+1:03d}'
  cmd=[str(attempt) if x==str(base_attempt) else x for x in j['command']]
  if state.get(aid,{}).get('status')=='complete': continue
  state[aid]={'status':'running','attempt':str(attempt),'started':time.time()}; a.state.parent.mkdir(parents=True,exist_ok=True); a.state.write_text(json.dumps(state,indent=2)+'\n')
  print(json.dumps({'event':'start','id':aid}),flush=True)
  # A completed vLLM worker can keep its CUDA context visible briefly after
  # the wrapper returns.  Retry a preflight-only failure while no attempt
  # directory/receipt was created; real generation failures still stop.
  rc=None
  for retry in range(5):
   rc=subprocess.run(cmd,cwd=Path(__file__).resolve().parents[1]).returncode
   if rc==0 or attempt.exists(): break
   time.sleep(60)
  marker=attempt/'attempt_complete.json'
  state[aid]={'status':'complete' if rc==0 and marker.exists() else 'failed','returncode':rc,'attempt':str(attempt),'finished':time.time()}; a.state.write_text(json.dumps(state,indent=2)+'\n'); print(json.dumps({'event':'finish','id':aid,'status':state[aid]['status']}),flush=True)
  done+=1
  if rc!=0: raise SystemExit(rc)
  if a.max_jobs and done>=a.max_jobs: break
 print(json.dumps({'jobs_processed':done,'complete':sum(v.get('status')=='complete' for v in state.values()),'failed':sum(v.get('status')=='failed' for v in state.values())}))
if __name__=='__main__':main()
