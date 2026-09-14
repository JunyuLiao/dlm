import pytest
import numpy as np
import json

from experiments.diffusion_gemma_value_aware_followup.analysis import screen_source,select_pooling


def test_screen_analysis_refuses_final_inputs_before_reading_files(tmp_path):
    with pytest.raises(ValueError,match='final example'):
        screen_source(tmp_path,dict(id='heldout',split='final'),{}, {})


def test_v1_screen_cannot_be_imported_for_v2_question(tmp_path):
    row=dict(id='v2/example',split='calibration',benchmark='longbench_v2')
    imports={'screen/dense/v2/example':dict(fingerprint='previous',path='unread')}
    with pytest.raises(ValueError,match='only matching AIME'):
        screen_source(tmp_path,row,dict(fingerprint='new',previous_fingerprint='previous'),imports)


def test_pooling_selection_reuses_both_benchmarks_but_not_development_scores():
    rows=[]
    for family in ('value','mass_value','risk'):
        for pool in ('max','mean','rms','p95','vector_mean'):
            for benchmark in ('aime26','longbench_v2'):
                for kind in ('local','global'):
                    for target in (50,75):
                        rows.append(dict(benchmark=benchmark,split='calibration',attention_type=kind,
                            probe=f'screen/{family}_{pool}/s{target}',relative_error=.1 if pool=='vector_mean' else .2))
                        rows.append(dict(benchmark=benchmark,split='development',attention_type=kind,
                            probe=f'screen/{family}_{pool}/s{target}',relative_error=100 if pool=='vector_mean' else 0))
    selected=select_pooling(rows)
    assert all(v['pooling']=='vector_mean' for v in selected.values())
    assert all(v['mean_error']==pytest.approx(.1) for v in selected.values())


def test_joined_screen_supplies_pooling_and_counts_dense_execution_once(tmp_path,monkeypatch):
    from experiments.diffusion_gemma_value_aware_followup import analysis
    setup=dict(calibration=[dict(id=b,benchmark=b,split='calibration') for b in ('aime26','longbench_v2')],development=[])
    monkeypatch.setattr(analysis,'prepare',lambda root:setup)
    monkeypatch.setattr(analysis,'contract',lambda root:dict(fingerprint='fp'))
    joint_path=tmp_path/'joint.npz';base_path=tmp_path/'base.npz'
    np.savez(joint_path,**{f'mass_exact__{k}':np.arange(4,dtype=float) for k in ('local','global')})
    np.savez(base_path,**{f'value_rms__{k}':np.arange(4,dtype=float) for k in ('local','global')})
    execution=[dict(probe='execution',attention_type=k,eligible=10,skipped=0,rows=1,mass_sum=1)
        for k in ('local','global')]
    pooling=[]
    for family in ('value','mass_value','risk'):
        for pool in ('max','mean','rms','p95','vector_mean'):
            for kind in ('local','global'):
                for target in (50,75):
                    pooling.append(dict(probe=f'screen/{family}_{pool}/s{target}',attention_type=kind,
                        eligible=10,skipped=5,error_sq=1 if pool=='rms' else 4,dense_sq=100))
    def bundle(*args):
        return ((dict(records=execution,distributions=[]),joint_path,dict(split='calibration')),
            (dict(records=execution+pooling,distributions=[]),base_path,dict(split='calibration')))
    monkeypatch.setattr(analysis,'screen_bundle',bundle)
    rows=analysis.summarize(tmp_path)
    dense=[r for r in rows if r['probe']=='execution' and r['attention_type']=='overall']
    assert all(r['eligible']==20 for r in dense)
    selected=json.loads((tmp_path/'pooling_selection.json').read_text())
    assert all(r['pooling']=='rms' for r in selected['selected'].values())
    proposals=json.loads((tmp_path/'threshold_proposals.json').read_text())
    assert set(proposals['policies']['longbench_v2'])=={'mass_exact','value_rms'}
