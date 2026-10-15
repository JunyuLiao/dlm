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


def test_trajectory_decomposition_separates_canvas_length_and_denoising_rate():
    from scripts.v31_value_native_study import paired_trajectory
    common=dict(dataset='aime',index=1,panel_seed=42,repeat=0,output_hash='a')
    c=[dict(**common,denoise_forwards=20,canvases=2,output_tokens=400)]
    v=[dict(**common,denoise_forwards=24,canvases=3,output_tokens=600)]
    r=paired_trajectory('aime26',c,v,{})
    # More canvases add nine calls; faster convergence per canvas removes five.
    assert r['N_change_decomposition']['canvas_count_component']==9
    assert r['N_change_decomposition']['calls_per_canvas_component']==-5
    assert r['denoise_forwards']['delta']==4


def test_native_contract_reads_real_named_model_settings(tmp_path):
    from scripts import v31_value_native_study as study
    study.io.write(tmp_path/'config.json',dict(canvas_length=256))
    study.io.write(tmp_path/'generation_config.json',dict(confidence_threshold=.005,
        sampler_config=dict(entropy_bound=.1),stability_threshold=1,max_denoising_steps=48,t_max=.8,t_min=.4))
    contract=study.native_contract(tmp_path)
    assert contract['confidence_threshold']==.005 and contract['entropy_bound']==.1
    assert contract['denoise_steps']==48 and contract['canvas']==256
    assert 'thinking' not in contract and 'projection_rank' not in contract
