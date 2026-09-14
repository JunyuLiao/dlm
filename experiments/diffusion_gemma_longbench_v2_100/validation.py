"""New long-context native parity and all requested operator/reference checks."""
import json
from pathlib import Path
from unittest.mock import patch

import torch
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
from experiments.diffusion_gemma_value_aware.protocol import sha
from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_value_aware_followup.calibrate import smoke_dense
from experiments.diffusion_gemma_value_aware_followup.evidence import DECODING_FIELDS,check_sources
from .calibration import starting_policy,Observer
from .protocol import METHODS,TARGETS,CONFIGS
from .runner import cached,generate,backend


def smoke(adapter,root,setup,contract):
    path=root/'validation/smoke.json'
    if path.exists():
        result=json.loads(path.read_text())
        if result['fingerprint']!=contract['fingerprint'] or not result['passed']:raise ValueError('Incompatible smoke')
        check_sources(result['sources']);return result
    dev=sorted(setup['development'],key=lambda r:len(r['prompt_tokens']))
    cases=[];observer_cases=[];sources={str(Path(__file__)):sha(Path(__file__).read_bytes())}
    for example in (dev[0],dev[-1]):
        row=dict(example,generation_budget=16)
        native=smoke_dense(adapter,root,row,contract)
        native_path=shard_path(root,'sparse_smoke','native_dense16',row['id']);sources[str(native_path)]=sha(native_path.read_bytes())
        observer_path=shard_path(root,'validation','dense_calibration_observer',row['id'])
        if observer_path.exists():observed=json.loads(observer_path.read_text())
        else:
            with patch.object(backend,'Attention',Observer):observed=generate(adapter,row)
            observed['fingerprint']=contract['fingerprint'];_write(observer_path,observed)
        from experiments.diffusion_gemma_value_aware_followup.engine import check_result
        check_result(observed,row,contract['fingerprint'],{},None)
        if observed['completion_tokens']!=native['completion_tokens'] or any(r['skipped'] for r in observed['records']):
            raise AssertionError('Dense calibration observer changed native output')
        sources[str(observer_path)]=sha(observer_path.read_bytes())
        observer_cases.append(dict(id=row['id'],dense_parity=True,calibration_data_used=False))
        del observed
        conditions=[('dense',{},None)]+[(f'{n}_s{int(t*100)}',CONFIGS[n],starting_policy(None,'longbench_v2',n,CONFIGS[n],t)) for t in TARGETS for n in METHODS]
        for name,config,threshold in conditions:
            out=cached(adapter,root,row,'validation',name,config,threshold,contract,validate=True)
            parity=name=='dense'
            if parity and out['completion_tokens']!=native['completion_tokens']:raise AssertionError('Native dense token parity failed')
            for key in DECODING_FIELDS:
                if out['generation_metadata'].get(key)!=native['generation_metadata'].get(key):raise AssertionError('Decoding changed')
            if name in ('mass_s50','risk_s50'):
                ref_path=shard_path(root,'validation',name+'_reference',row['id'])
                if ref_path.exists():ref=json.loads(ref_path.read_text())
                else:
                    ref=generate(adapter,row,config,threshold,reference=True);ref['fingerprint']=contract['fingerprint'];_write(ref_path,ref)
                from experiments.diffusion_gemma_value_aware_followup.engine import check_result
                check_result(ref,row,contract['fingerprint'],config,threshold)
                if ref['completion_tokens']!=out['completion_tokens']:raise AssertionError('Sparse GPU/reference token parity failed')
                # Physical counters must be identical, not merely approximately close.
                fields=('eligible','skipped','prefix_eligible','prefix_skipped','canvas_eligible','canvas_skipped')
                def counts(x):return {(r['layer'],r['head'],r['step']):tuple(r[k] for k in fields) for r in x['records']}
                if counts(ref)!=counts(out):raise AssertionError('Sparse physical counter parity failed')
                sources[str(ref_path)]=sha(ref_path.read_bytes());parity=True
            raw=shard_path(root,'validation',name,row['id']);sources[str(raw)]=sha(raw.read_bytes())
            cases.append(dict(id=row['id'],condition=name,passed=True,exact_generation_parity=parity,
                checked_calls=len(out['kernel_validation']),finite_calls=out['finite_calls'],
                prefix_eligible=sum(r['prefix_eligible'] for r in out['records']),
                canvas_eligible=sum(r['canvas_eligible'] for r in out['records']),
                prefix_skipped=sum(r['prefix_skipped'] for r in out['records']),
                canvas_skipped=sum(r['canvas_skipped'] for r in out['records'])))
            _write(root/'validation/progress.json',dict(cases=cases,expected=18))
            torch.cuda.empty_cache()
    if len(cases)!=18 or any(not c['checked_calls'] or not c['prefix_eligible'] or not c['canvas_eligible'] for c in cases):
        raise AssertionError('Incomplete layer/head/step/region smoke coverage')
    if any(not any(c[f'{region}_skipped'] for c in cases) for region in ('prefix','canvas')):
        raise AssertionError('Missing actual prefix/canvas skipping')
    result=dict(passed=True,fingerprint=contract['fingerprint'],cases=cases,sources=sources,expected=18,observer_cases=observer_cases,
        sparse_generated_sequence_reference_replays=4,thresholds='Historical proposals; final thresholds reverified on new calibration set')
    _write(path,result);return result
