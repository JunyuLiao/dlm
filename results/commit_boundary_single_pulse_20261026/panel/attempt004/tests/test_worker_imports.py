"""Exercise stdlib/Torch imports with the real worker's script directory first."""
from pathlib import Path
import os,subprocess,sys
CODE=Path(__file__).resolve().parents[1]/'code'
def test_worker_directory_has_working_stdlib_selectors():
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['PYTHONDONTWRITEBYTECODE']='1'
 script="import sys;sys.path.insert(0,sys.argv[1]);import select,selectors,subprocess;assert callable(select.select);import torch;import runtime,bank,measure;print(torch.__version__)"
 p=subprocess.run([sys.executable,'-c',script,str(CODE)],env=env,capture_output=True,text=True)
 assert p.returncode==0,p.stderr
