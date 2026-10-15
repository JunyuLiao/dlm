from scripts.v31_value_native_study import paired
import pytest


def test_paired_accuracy_clusters_repeated_seeds_by_problem():
    control={'aime|0|42|0':True,'aime|0|43|0':False,'aime|1|42|0':True,'aime|1|43|0':True}
    candidate={**control,'aime|0|43|0':True}
    result=paired('aime26',control,candidate,{})
    assert result['clusters']==2
    assert result['delta_score_points']==25
    assert result['clustering']=='problem'


def test_ruler_fractional_scores_are_not_binary_and_cluster_by_task():
    control={'ruler|0|42|0':50.,'ruler|1|42|0':100.,'ruler|2|42|0':100.}
    candidate={**control,'ruler|0|42|0':40.}
    result=paired('ruler128k',control,candidate,{0:'task_a',1:'task_a',2:'task_b'})
    assert result['clusters']==2
    assert result['delta_score_points']==-2.5
    assert result['clustering']=='task'


def test_h100_promotion_rejects_source_edits_after_passed_tests(tmp_path,monkeypatch):
    from scripts import v31_value_native_study as study
    monkeypatch.setattr(study.io,'ROOT',tmp_path)
    for rel in study.NATIVE_CODE+study.TEST_CODE:
        p=tmp_path/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('tested source')
    receipt=tmp_path/'h100.json'
    study.io.write(receipt,dict(status='passed',device='NVIDIA H100 80GB HBM3',
        tests_passed=1,tests_skipped=0,source_sha256={p:study.io.sha(tmp_path/p) for p in study.NATIVE_CODE+study.TEST_CODE}))
    gate=dict(h100_tests_receipt='h100.json',h100_tests_receipt_sha256=study.io.sha(receipt))
    assert study.verify_h100_gate(gate)==receipt
    (tmp_path/study.NATIVE_CODE[0]).write_text('changed source')
    with pytest.raises(RuntimeError,match='current native source'):
        study.verify_h100_gate(gate)
