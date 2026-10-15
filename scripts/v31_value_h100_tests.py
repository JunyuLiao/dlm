"""Run and bind the entire refinement correctness suite after diagnostic work."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from scripts import v31_value_decision_replay as io
from scripts import v31_value_native_study as native


def run(root,attempt):
    if io.read(root/'decision_batch_complete.json')['status']!='complete':
        raise RuntimeError('finish the diagnostic GPU batch before running correctness tests')
    dest=root/'receipts/h100_qualification'/attempt
    if dest.exists():raise RuntimeError('preserve the previous test attempt; choose a new attempt')
    native.base.wait_idle()
    device=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip()
    if device!='NVIDIA H100 80GB HBM3':raise RuntimeError('the pinned H100 is required')
    dest.mkdir(parents=True)
    sources={rel:io.sha(io.ROOT/rel) for rel in native.NATIVE_CODE+native.TEST_CODE}
    command=[native.base.PYTHON,'-m','pytest','-q',*native.TEST_CODE,'--junitxml='+str(dest/'tests.xml')]
    env=dict(os.environ,PYTHONPATH=f'{io.ROOT}/src:{io.ROOT}',PYTHONNOUSERSITE='1',
        CUDA_VISIBLE_DEVICES='0',TRITON_CACHE_DIR=str(root/'private/test_cache/triton'))
    with (dest/'pytest.log').open('w') as log:
        proc=subprocess.run(command,cwd=io.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    totals=dict(tests=0,failures=0,errors=0,skipped=0)
    if (dest/'tests.xml').exists():
        for suite in ET.parse(dest/'tests.xml').iter('testsuite'):
            for key in totals:totals[key]+=int(suite.attrib.get(key,0))
    unchanged=all(io.sha(io.ROOT/p)==digest for p,digest in sources.items())
    passed=proc.returncode==0 and totals['tests']>0 and not any(totals[k] for k in ('failures','errors','skipped')) and unchanged
    receipt=dict(status='passed' if passed else 'failed',device=device,returncode=proc.returncode,
        tests_passed=totals['tests']-totals['failures']-totals['errors']-totals['skipped'],tests_skipped=totals['skipped'],
        test_totals=totals,source_sha256=sources,source_unchanged_during_tests=unchanged,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=io.ROOT,text=True).strip(),
        pytest_log_sha256=io.sha(dest/'pytest.log'),timing='paused; correctness tests only')
    io.write(dest/'receipt.json',receipt,True)
    print(dict(status=receipt['status'],test_totals=totals,receipt=str((dest/'receipt.json').relative_to(io.ROOT))),flush=True)
    if not passed:raise RuntimeError('preserved failed H100 test attempt; inspect pytest.log')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,default=io.DEFAULT)
    p.add_argument('--attempt',required=True)
    a=p.parse_args()
    if not (len(a.attempt)==10 and a.attempt.startswith('attempt') and a.attempt[7:].isdigit()):
        p.error('use a zero-padded attempt, for example attempt001')
    run(a.study.resolve(),a.attempt)
