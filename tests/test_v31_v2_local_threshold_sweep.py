import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]

def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

sweep=module('v31_v2_local_threshold_sweep')


def test_canvas_ids_are_ranked_not_zero_filled():
    a=dict(global_kept_tiles=12,value_canvas_phase_tiles={
        '4':{'dense':{'eligible_tiles':8,'kept_tiles':8}},
        '5':{'held':{'eligible_tiles':20,'kept_tiles':4}}})
    assert sweep.ordinal_canvas(a)=={1:{'eligible_tiles':8,'kept_tiles':8},2:{'eligible_tiles':20,'kept_tiles':4}}
    a['global_kept_tiles']=13
    with pytest.raises(RuntimeError,match='canvas accounting'):sweep.ordinal_canvas(a)


def test_equal_weight_workloads_accuracy_first():
    xs=[dict(threshold=.021,macro_accuracy=.6,mean_N_ratio=1.2,mean_kept_ratio=1.3),
        dict(threshold=.0195,macro_accuracy=.599,mean_N_ratio=.7,mean_kept_ratio=.5),
        dict(threshold=.0225,macro_accuracy=.6,mean_N_ratio=1.1,mean_kept_ratio=1.4)]
    assert sweep.rank_candidates(xs)[0]['threshold']==.0225
    assert sweep.stage2_thresholds(.021)==[.02025,.02175]


def test_native_and_variant_environment_isolation(tmp_path,monkeypatch):
    monkeypatch.setenv('STABLE_REFRESH_CONFIG','unexpected')
    monkeypatch.setenv('VALUE_MASS_ANCHOR','64')
    monkeypatch.setenv('CACHED_REFRESH_POLICY','hybrid')
    monkeypatch.setenv('CANVAS_RESEED','1')
    e=sweep.env_for(tmp_path,'aime26','value_v2_online_preserve_mass',.021)
    assert not any(k.startswith(('CACHED_REFRESH_','STABLE_')) for k in e)
    assert 'CANVAS_RESEED' not in e
    assert e['VALUE_MASS_ANCHOR']=='0' and e['VALUE_THRESHOLD']=='0.021'
    assert e['MAGE_K']=='1728' and e['MAX_MODEL_LEN']=='9216'
    assert e['VALUE_CLEAN_TIMING']=='0' and 'VALUE_DIAGNOSTIC_DIR' not in e


def test_clustered_bootstrap_treats_repeated_seeds_as_one_problem():
    c={};v={};cr=[];vr=[]
    for index in range(2):
        for seed in [42,43,44]:
            key=f'aime26|{index}|{seed}|0';c[key]=False;v[key]=(index==0)
            r=dict(dataset='aime26',index=index,panel_seed=seed,repeat=0,output_hash='a',
                output_tokens=10,denoise_forwards=5,canvases=1,
                receipts={'adapter':{'global_kept_tiles':20}})
            cr.append(r);vr.append(r)
    p=sweep.paired(c,v,cr,vr)
    assert p['problem_clusters']==2 and p['delta_pp']==50.
    assert p['problem_cluster_bootstrap_ci95_pp']==[0.,100.]


def test_math_symbolic_and_unclosed_answer_handling():
    scorer=module('v31_score_math500_bound');v=scorer.make_verifier()
    assert scorer.judge(v,'1/2',r'Final answer: \boxed{0.5}')[0]
    assert not scorer.judge(v,'1/2',r'Final answer: \boxed{2}')[0]
    assert scorer.judge(v,'1/2','')==(False,False,False)


def test_legacy_count_and_quick_token_schedules_share_binding(tmp_path):
    pool=tmp_path/'pool';pool.mkdir()
    sweep.write(pool/'sample_generation_manifest.json',[{'id':'one','prompt_tokens':[1,2,3]}])
    cells=tmp_path/'cells.json'
    for tokens in [3,[1,2,3]]:
        sweep.write(cells,[{'dataset':'sample','index':0,'id':'one','prompt_tokens':tokens}])
        assert len(sweep.check_inputs(pool,cells))==1
    sweep.write(cells,[{'dataset':'sample','index':0,'id':'one','prompt_tokens':4}])
    with pytest.raises(RuntimeError,match='input mismatch'):sweep.check_inputs(pool,cells)
