"""Wait for the observed GPU workload to exit; launch once GPU is free."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

def main():
    occupied_pid=int(sys.argv[1])
    root=Path('results/diffusion_gemma_aime26_blasst_aggressive')
    while True:
        proc=Path(f'/proc/{occupied_pid}/stat')
        try: alive=proc.read_text().rsplit(')',1)[1].split()[0]!='Z'
        except FileNotFoundError: alive=False
        gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
        state=dict(time=time.strftime('%FT%TZ',time.gmtime()),waiting_for_pid=occupied_pid,observed_alive=alive,gpu=gpu.stdout.strip(),gpu_query_ok=gpu.returncode==0)
        with (root/'queue.jsonl').open('a') as f: f.write(json.dumps(state)+'\n')
        print(json.dumps(state),flush=True)
        if not alive and gpu.returncode==0 and not gpu.stdout.strip(): break
        time.sleep(900)
    os.execv(sys.executable,[sys.executable,'-m','experiments.diffusion_gemma_aime26.run','run'])

if __name__=='__main__': main()
