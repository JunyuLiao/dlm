"""Prepare a small new source-frozen K128 native qualification only."""
from pathlib import Path
import os
import json,hashlib,shutil,datetime
from scripts.optimized_refresh_campaign import ROOT,write,sha,utc

def prepare(name,index=None):
 old=ROOT/'results/optimized_refresh_benchmarks_20261021/attempt004'
 study=ROOT/'results/sparse_execution_20261021'/name;study.mkdir(parents=True,exist_ok=False)
 ex=study/'private/execution';shutil.copytree(old/'private/execution',ex,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for rel in ['experiments/numerical_qk_reuse/resident_sparse.py','experiments/numerical_qk_reuse/resident_execution_runtime.py','experiments/numerical_qk_reuse/refresh_v31_fast.py','experiments/numerical_qk_reuse/optimized_refresh_runtime.py','scripts/optimized_refresh_worker.py','scripts/optimized_refresh_campaign.py']:
  shutil.copyfile(ROOT/rel,ex/rel)
 for label in ['pools','schedules','calibration_pools']:
  if label=='schedules' and index is not None:shutil.copytree(old/'private'/label,study/'private'/label)
  else:(study/'private'/label).symlink_to((old/'private'/label).resolve(),target_is_directory=True)
 if index is not None:
  cells=[]
  for path in (study/'private/schedules/humaneval').glob('shard*.json'):
   cells.extend([x for x in json.loads(path.read_text()) if x['index']==index])
  if len(cells)!=1:raise RuntimeError('one known diagnostic index required')
  write(study/'private/schedules/humaneval/smoke.json',cells)
 shutil.copytree(old/'private/kernel',study/'private/kernel');(study/'private/cache').symlink_to((old/'private/cache').resolve(),target_is_directory=True)
 shutil.copyfile(ROOT/'docs/SPARSE_EXECUTION_20261021.md',study/'spec.md')
 p=json.loads((old/'protocol.json').read_text());p.update(created_utc=utc(),execution_sha256={str(x.relative_to(ex)):sha(x) for x in ex.rglob('*') if x.is_file()},spec_sha256=sha(study/'spec.md'),purpose='bounded native resident-consumer qualification only; no benchmark relaunch',resident_sparse=True,resident_execution_options={k:os.environ.get(k,'0') for k in ['RESIDENT_SPARSE_EXECUTION','RESIDENT_SPARSE_GRAPH_BUCKETS','RESIDENT_V31_PAGED_REFRESH']})
 for suite in ['aime26','math500','humaneval']:p['suites'][suite]['mage_k']=128
 if index is not None:
  p['input_sha256']['private/schedules/humaneval/smoke.json']=sha(study/'private/schedules/humaneval/smoke.json')
  p['diagnostic_qualification']=dict(index=index,previously_examined=True,accuracy_panel=False,workload_calibration=False)
 write(study/'protocol.json',p);write(study/'state.json',dict(status='prepared',updated_utc=utc()))
 print(study)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--name',default='native002');p.add_argument('--humaneval-index',type=int);a=p.parse_args();prepare(a.name,a.humaneval_index)
