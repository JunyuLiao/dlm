"""Reuse the validated GPU/native generation backend; change only benchmark scoring."""
import json
import os
import time
from unittest.mock import patch

from experiments.diffusion_gemma_value_aware_gpu import runner as backend
from experiments.diffusion_gemma_value_aware_followup.engine import check_result
from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write,_append
from . import nemo


def generate(adapter,row,config=None,thresholds=None,validate=False,reference=False):
    with patch.object(backend,'score',nemo.score):
        return backend.generate(adapter,row,config,thresholds,validate,reference)


def cached(adapter,root,row,stage,name,config,thresholds,contract,validate=False):
    path=shard_path(root,stage,name,row['id'])
    if path.exists():
        result=json.loads(path.read_text())
        check_result(result,row,contract['fingerprint'],config,thresholds)
        if validate and not result.get('kernel_validation'):
            raise ValueError('Cache lacks required reference mask validation')
        return result
    if adapter is None:raise FileNotFoundError(path)
    _write(root/'progress.json',dict(pid=os.getpid(),stage=stage,condition=name,id=row['id'],started=time.time()))
    result=generate(adapter,row,config,thresholds,validate=validate)
    result['fingerprint']=contract['fingerprint']
    check_result(result,row,contract['fingerprint'],config,thresholds)
    _write(path,result)
    _append(root/'completed.jsonl',dict(stage=stage,condition=name,id=row['id'],path=str(path),finished=time.time()))
    print(time.strftime('%FT%TZ',time.gmtime()),stage,name,row['id'],'complete',flush=True)
    return result
