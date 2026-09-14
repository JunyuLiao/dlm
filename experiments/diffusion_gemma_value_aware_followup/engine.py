"""Thin benchmark/cache adapter over the existing attention execution engine."""
import json
from pathlib import Path
from unittest.mock import patch

from experiments.diffusion_gemma_value_aware import run as original
from experiments.diffusion_gemma_value_aware.execution import execution_cache, provenance
from experiments.diffusion_gemma_value_aware.protocol import fingerprint as previous_fingerprint, frozen_write, sha
from experiments.diffusion_gemma_value_aware.report import inspect_output
from experiments.diffusion_gemma_value_aware.ranking_guards import cache_joint_screen, source_sha as guard_sha
from experiments.diffusion_gemma_value_aware.refinements import source_sha as refinement_sha
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write, _fingerprint
from .protocol import PREVIOUS, prepare, score


def contract(root):
    import torch
    import transformers
    setup=prepare(root)
    versions=dict(torch=torch.__version__,transformers=transformers.__version__)
    previous_runtime=json.loads((PREVIOUS/'model_runtime.json').read_text())
    if any(previous_runtime[k]!=v for k,v in versions.items()):
        raise ValueError('runtime versions changed; cached AIME reuse requires an explicit new parity audit')
    files=[Path(__file__),Path(__file__).with_name('protocol.py'),Path(__file__).with_name('run.py')]
    files += [Path('experiments/diffusion_gemma_value_aware')/name for name in
        ('operators.py','routing.py','run.py','refinements.py','ranking_guards.py','execution.py')]
    data=dict(schema='followup_execution_v1',setup_sha256=sha((root/'setup.json').read_bytes()),
        previous_fingerprint=previous_fingerprint(PREVIOUS),runtime_versions=versions,
        sources={str(p):sha(p.read_bytes()) for p in files},
        scientific_settings={k:setup[k] for k in ('model','revision','precision','tile_size','regions','decoding')})
    data['fingerprint']=_fingerprint(data)
    frozen_write(root/'execution_contract.json',data)
    return data


def check_result(data,row,fp,config,thresholds,screen=False):
    if data['screen'] != screen:
        raise ValueError('screen versus final cache mismatch')
    checked=dict(data)
    if screen and not config:
        # Dense observers have explicit extra code provenance but never change
        # the operator. Audit those hashes before applying the shared dense audit.
        for key,expected in [('refinement_sha256',refinement_sha()),('ranking_guard_sha256',guard_sha())]:
            if checked.get(key)!=expected:
                raise ValueError('dense screen observer source mismatch')
            checked.pop(key)
    condition=dict(config=config,thresholds={row['benchmark']:thresholds},**provenance(config))
    _,_,errors=inspect_output(checked,row,fp,condition)
    if errors:
        raise ValueError(f'raw generation audit: {errors}')


def import_source(root,row,stage,name,config,thresholds,screen,execution):
    """Only unchanged AIME generations may cross benchmark bundle boundaries."""
    if row['benchmark']!='aime26':return None
    old_setup=json.loads((PREVIOUS/'setup.json').read_text())
    matching=[r for r in old_setup['final'] if r['id']==row['id']]
    if len(matching)!=1:return None
    for key in ('prompt','prompt_hash','prompt_tokens','seed','generation_budget'):
        if matching[0][key]!=row[key]:return None
    candidates=[]
    if screen and not config:
        candidates.append(original.shard_path(PREVIOUS,'refinement_screen','dense',row['id']))
    elif not screen:
        candidates.append(original.shard_path(PREVIOUS,'final',name,row['id']))
    for source in candidates:
        if not source.exists():continue
        data=json.loads(source.read_text())
        try:check_result(data,row,execution['previous_fingerprint'],config,thresholds,screen)
        except (ValueError,KeyError):continue
        if screen and not source.with_suffix('.npz').exists():continue
        index_path=root/'imported_sources.json'
        index=json.loads(index_path.read_text()) if index_path.exists() else {}
        key=f'{stage}/{name}/{row["id"]}'
        entry=dict(path=str(source),sha256=sha(source.read_bytes()),fingerprint=data['fingerprint'])
        if screen:
            entry['arrays_path']=str(source.with_suffix('.npz'))
            entry['arrays_sha256']=sha(source.with_suffix('.npz').read_bytes())
        if key in index and index[key]!=entry:raise ValueError('import source changed')
        index[key]=entry;_write(index_path,index)
        return data
    return None


def cached(adapter,root,row,stage,name,config=None,thresholds=None,screen=False,execution=None):
    config=config or {};execution=execution or contract(root)
    fp=execution['fingerprint'];path=original.shard_path(root,stage,name,row['id'])
    if path.exists():
        data=json.loads(path.read_text());check_result(data,row,fp,config,thresholds,screen)
        if screen and not path.with_suffix('.npz').exists():raise ValueError('missing screen margins')
        return data
    imported=import_source(root,row,stage,name,config,thresholds,screen,execution)
    if imported is not None:return imported
    if adapter is None:raise FileNotFoundError(path)
    # The entire scientific generator and cache remain unchanged; only scoring
    # dispatch is extended to the new benchmark. The scoped patch restores even
    # on failure and this worker is deliberately single-threaded.
    with patch.object(original,'score',score):
        if screen:
            if config or thresholds is not None:raise ValueError('shared screen must be dense')
            data=cache_joint_screen(adapter,row,path,fp,root)
        else:
            data=execution_cache(config)(adapter,row,path,fp,config,thresholds,root=root)
    check_result(data,row,fp,config,thresholds,screen)
    return data
