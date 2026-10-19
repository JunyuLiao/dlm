"""Freeze and run the bounded compiler-compatibility regression."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from scripts import v31_humaneval_cfgdet_study as campaign

ROOT=campaign.ROOT
read,write,sha=campaign.read,campaign.write,campaign.sha
DEFAULT=ROOT/'results/humaneval_index_put_fallback_20261019/attempt001'


def prepare(root):
    parent=campaign.verify(ROOT/'results/humaneval_value_cfgdet_k256_20261019/attempt001')
    assert not root.exists()
    execution=root/'private/execution'
    (execution/'scripts').mkdir(parents=True)
    (execution/'scripts/__init__.py').write_text('')
    for source,target in [
        ('scripts/v31_humaneval_index_put_dtype.py','scripts/v31_humaneval_index_put_dtype.py'),
        ('tests/test_humaneval_index_put_dtype.py','test_humaneval_index_put_dtype.py')]:
        shutil.copyfile(ROOT/source,execution/target)
    shutil.copyfile(ROOT/'docs/HUMANEVAL_INDEX_PUT_FALLBACK_PLAN_20261019.md',root/'spec.md')
    shutil.copyfile(Path(__file__),root/'private/controller.py')
    p=dict(status='frozen_before_execution',source_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),runtime=parent['runtime'],
        parent_protocol_sha256=sha(ROOT/'results/humaneval_value_cfgdet_k256_20261019/attempt001/protocol.json'),
        phase='k256',maximum_job_reservation_s=180,native_model_forwards=0,
        native_sampler_calls=0,cublas_workspace_config=':4096:8',
        files_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    write(root/'protocol.json',p,True)


def run(root):
    p=read(root/'protocol.json')
    assert campaign.base.runtime_identity()==p['runtime']
    for name,digest in p['files_sha256'].items():assert sha(root/name)==digest
    assert not (root/'receipt.json').exists() and not (root/'failed_attempt.json').exists()
    campaign.base.wait_idle()
    budget,reservation=campaign.reserve(root,p,root)
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=p['cublas_workspace_config'],
        PYTHONPATH=str(root/'private/execution'),
        TORCHINDUCTOR_CACHE_DIR=str(root/'private/cache/inductor'),
        TRITON_CACHE_DIR=str(root/'private/cache/triton'))
    cmd=['/home/exouser/dyh/v31_env_20261005/bin/python','-m','pytest','-q',
        str(root/'private/execution/test_humaneval_index_put_dtype.py'),
        '--junitxml='+str(root/'tests.xml')]
    write(root/'private/launch.json',dict(command=cmd,environment={k:env[k] for k in
        ('CUBLAS_WORKSPACE_CONFIG','PYTHONPATH','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR')}),True)
    start=time.monotonic();code=1;failure=None
    try:
        with (root/'private/worker.log').open('w') as log:
            result=subprocess.run(cmd,cwd=root/'private/execution',env=env,stdout=log,
                stderr=subprocess.STDOUT,timeout=reservation['reserved_s'])
            code=result.returncode
    except Exception as error:
        failure=repr(error)
    finally:
        reservation.update(elapsed_s=time.monotonic()-start,returncode=code)
        write(campaign.INVESTIGATION/'gpu_budget.json',budget)
    suites=ET.parse(root/'tests.xml').getroot() if (root/'tests.xml').exists() else None
    cases=[] if suites is None else list(suites.iter('testcase'))
    passed=code==0 and len(cases)==1 and all(not any(c.tag in ('error','failure','skipped') for c in t) for t in cases)
    receipt=dict(status='complete' if passed else 'failed',returncode=code,error=failure,
        tests=len(cases),gpu_reservation_s=reservation['elapsed_s'],protocol_sha256=sha(root/'protocol.json'),
        files_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    write(root/('receipt.json' if passed else 'failed_attempt.json'),receipt,True)
    print(json.dumps(receipt),flush=True)
    if not passed:raise RuntimeError('preserved failed dtype regression; native probe remains gated')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--study',type=Path,default=DEFAULT)
    args=parser.parse_args()
    (prepare if args.action=='prepare' else run)(args.study.absolute())
