"""Finite HE124 replay qualification with operational hang timeout."""
import json,os,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parent
checks=[]
for cmd in [["nvidia-smi","--query-gpu=name,memory.used,utilization.gpu","--format=csv,noheader"],["nvidia-smi","--query-compute-apps=pid,process_name,used_gpu_memory","--format=csv,noheader"]]:
 p=subprocess.run(cmd,text=True,capture_output=True);checks.append(dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
 if p.returncode:raise RuntimeError("GPU unavailable")
if checks[1]["stdout"].strip():raise RuntimeError("GPU occupied; no preemption")
started=time.time();env=os.environ.copy()
overrides=dict(PYTHONPATH=str(r.parents[2]/"src")+":"+str(r.parents[2]),HF_HUB_OFFLINE="1",TORCHINDUCTOR_DETERMINISTIC="1",VLLM_USE_V2_MODEL_RUNNER="1")
for var,leaf in [("VLLM_CACHE_ROOT","vllm"),("TRITON_CACHE_DIR","triton"),("TORCHINDUCTOR_CACHE_DIR","inductor"),("CUDA_CACHE_PATH","cuda")]:overrides[var]=str(r/"cache"/leaf)
env.update(overrides)
cmd=["timeout","--signal=TERM","--kill-after=15s","1800","/home/exouser/dyh/v31_env_20261005/bin/python",str(r/"qualification001/code/worker.py"),"--root",str(r)]
with (r/"qualification001/launch_receipt.json").open("x") as f:json.dump(dict(command=cmd,environment=overrides,source_checkpoint=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),gpu_checks=checks,started_unix=started,qualification_timeout_seconds=1800,campaign_budget_ceiling=None),f,indent=2)
with (r/"private/qualification_worker.log").open("x") as f:result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
resource=json.loads((r/"private/live_resource.json").read_text()) if (r/"private/live_resource.json").exists() else {}
receipt=dict(returncode=result.returncode,allocated_worker_seconds=time.time()-started,resource=resource)
with (r/"qualification001/exit_receipt.json").open("x") as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt),flush=True)
raise SystemExit(result.returncode)
