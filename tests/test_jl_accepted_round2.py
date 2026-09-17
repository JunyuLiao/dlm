"""The user exception names one point; all other tolerance checks stay strict."""
from copy import deepcopy
import pytest
from experiments import diffusion_gemma_jl_accepted_round2 as accepted


def test_exception_scope_and_frozen_numerical_sources():
    p=dict(benchmark='longbench_v2',name='full_centered',config={'family':'identity'},target=.75,
        selected_round=1,measured=deepcopy(accepted.EXPECTED),user_accepted_tolerance_exception=True)
    assert accepted.is_exception(p)
    for key,value in [('benchmark','aime26'),('target',.5),('selected_round',0),('name','jl_gaussian_r32'),
                      ('config',{'family':'gaussian','rank':32}),('user_accepted_tolerance_exception',False)]:
        assert not accepted.is_exception(dict(p,**{key:value}))
    assert not accepted.BASE_WITHIN(accepted.EXPECTED,.75)
    old=accepted.READ(accepted.ROOT/'execution_contract.json');accepted.fb.check_sources(old['sources'])


def test_raw_audit_and_tolerance_exception_are_both_required(tmp_path,monkeypatch):
    p=dict(benchmark='longbench_v2',name='full_centered',config={'family':'identity'},target=.75,selected_round=1,
        measured=deepcopy(accepted.EXPECTED),user_accepted_tolerance_exception=True,policy={'local':1,'global':2})
    trace=tmp_path/'trace.json';trace.write_text('[]')
    selected=dict(policy=p['policy'],achieved=accepted.EXPECTED)
    monkeypatch.setattr(accepted,'point',lambda root:([],selected,trace))
    approval=accepted.folder(tmp_path)/'acceptance.json';approval.parent.mkdir()
    accepted.fb.frozen_write(approval,dict(selected_trace_sha256=accepted.fb.sha(trace.read_bytes())))
    p['acceptance_source']=dict(path=str(approval),sha256=accepted.fb.sha(approval.read_bytes()))
    checks=[]
    def raw_audit(root,policy,setup,contract):
        checks.append(True)
        assert accepted.fb.within(accepted.EXPECTED,.75)
        assert not accepted.fb.within({'overall':.75,'local':.75,'global':.70},.75)
    monkeypatch.setattr(accepted,'BASE_AUDIT',raw_audit)
    accepted.audit(tmp_path,p,{},{});assert checks==[True]
    assert not accepted.fb.within(accepted.EXPECTED,.75)
    with pytest.raises(ValueError):accepted.audit(tmp_path,dict(p,selected_round=0),{},{})
    p['policy']={'local':3,'global':4}
    with pytest.raises(ValueError):accepted.audit(tmp_path,p,{},{})
