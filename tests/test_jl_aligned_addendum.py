import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.diffusion_gemma_jl_output_aware import aligned_addendum as addon
from experiments.diffusion_gemma_jl_output_aware import calibration, protocol
from experiments.diffusion_gemma_value_aware.operators import block_state, value_summaries, screen_risks, Config


def test_aligned_screen_reuses_native_token_norm_semantics(tmp_path):
    torch.manual_seed(17)
    sources=[];expected={}
    for kind in ('local','global'):
        path=tmp_path/f'{kind}.pt'
        scores=torch.randn(1,1,7,193).bfloat16()
        valid=torch.rand(1,1,7,193)>.1
        valid[...,0,:]=False
        values=torch.randn(1,1,193,16).bfloat16()
        data=dict(scores=scores,valid=valid,values=values,kv_valid=valid.any(-2))
        torch.save(data,path)
        sources.append(dict(path=str(path),sha256=addon.sha(path.read_bytes()),split='calibration',
            id='aime26/test',attention_type=kind))
        state=block_state(scores,valid,values)
        risk,_=screen_risks(state,value_summaries(values,data['kv_valid']),Config(method='aligned'))
        expected[kind]=risk[0,0][state['eligible'][0,0]].numpy()
    original=torch.load
    def load(*args,**kwargs):
        kwargs['map_location']='cpu';return original(*args,**kwargs)
    with patch.object(addon.torch,'load',load):addon.screen(tmp_path,sources,{'fingerprint':'test'})
    values,proof=calibration.distributions(tmp_path,'aligned','aime26')
    for kind in expected:np.testing.assert_array_equal(values[kind],np.sort(expected[kind]))
    assert len(proof)==2
    # Resume must only read completed proposal arrays, not recompute PV.
    with patch.object(addon.torch,'load',side_effect=AssertionError('unexpected replay')):
        addon.screen(tmp_path,sources,{'fingerprint':'test'})


def test_aligned_import_uses_unchanged_historical_aime_policy(tmp_path):
    setup=json.loads((protocol.ROOT/'setup.json').read_text())
    with patch.object(calibration,'BASELINES',dict(aligned=addon.CONFIG)):
        for target in (.5,.75):
            p=calibration.import_aime_baseline(tmp_path,'aligned',target,addon.CONFIG,setup,{'fingerprint':'test'})
            assert p['imported_aime'] and p['config']=={'method':'aligned'}
            calibration.audit_policy(tmp_path,p,setup,{'fingerprint':'test'})


def test_addendum_identity_does_not_change_primary_contract(tmp_path):
    before=(protocol.ROOT/'execution_contract.json').read_bytes()
    setup,c=addon.identity(tmp_path)
    assert c['sample_ids']==[r['id'] for r in setup['final']]
    assert c['parent_fingerprint']==json.loads(before)['fingerprint']
    assert before==(protocol.ROOT/'execution_contract.json').read_bytes()
    assert c==addon.identity(tmp_path)[1]
