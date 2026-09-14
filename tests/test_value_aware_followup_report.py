import copy
import json
import pytest

from experiments.diffusion_gemma_value_aware_followup import report,final,evidence
from experiments.diffusion_gemma_value_aware.report_metrics import aggregate


def row(identity,task,accuracy,calibration=False):
    a=aggregate([dict(eligible=10,skipped=5,rows=2,mass_sum=1.5,error_sq=1,dense_sq=4)])
    return dict(id=identity,benchmark='longbench_v2',task=task,condition='value_s50',calibration=calibration,
        accuracy=accuracy,dense_accuracy=1.,matching=2,compared=4,exact_match=False,
        output_length=4,termination_reason='eos',unparsed_answer=False,
        aggregates={k:dict(a) for k in ('overall','global','local')})


def test_report_macro_and_attention_counts_reuse_existing_aggregators():
    raw=[row('a','qa',1.),row('b','code',0.)]
    raw[0]['aggregates']['overall']=aggregate([dict(eligible=100,skipped=90,rows=3,mass_sum=2)])
    conditions={'value_s50':dict(config=dict(method='value',pooling='vector_mean'),target=.5,target_metric='physical_sparsity')}
    rows,tasks=report.grouped_summaries(raw,conditions)
    assert len(rows)==1 and len(tasks)==2
    assert rows[0]['accuracy']==.5 and rows[0]['token_agreement']==.5
    assert rows[0]['overall_physical_sparsity']==pytest.approx(95/110)
    assert rows[0]['overall_mass']==pytest.approx(3.5/5)


def test_aime_full30_includes_calibration_but_heldout24_is_separate():
    raw=[dict(row(str(i),'aime26',1.,i<6),benchmark='aime26') for i in range(30)]
    conditions={'value_s50':dict(config=dict(method='value'),target=.5,target_metric='physical_sparsity')}
    summaries,_=report.grouped_summaries(raw,conditions)
    assert {r['split']:r['count'] for r in summaries}=={'full':30,'calibration6':6,'heldout24':24}


def test_matched_comparison_intervals_refuse_unpaired_examples():
    comparisons=[dict(comparable_within_three_points=True,benchmark='longbench_v2',split='full',
        candidate='value_s50',reference='blasst_s50')]
    raw=[row('a','qa',1.),dict(row('b','qa',0.),condition='blasst_s50')]
    with pytest.raises(ValueError,match='sample mismatch'):report.comparison_intervals(comparisons,raw)


def test_identical_settings_reuse_final_source_but_not_different_thresholds(tmp_path,monkeypatch):
    conditions={'x':dict(config=dict(method='blasst'),thresholds={'longbench_v2':{'local':1}}),
        'y':dict(config=dict(method='blasst'),thresholds={'longbench_v2':{'local':1}}),
        'z':dict(config=dict(method='blasst'),thresholds={'longbench_v2':{'local':2}})}
    r=dict(id='v2/a',benchmark='longbench_v2')
    calls=[]
    def read(*args,**kwargs):
        calls.append(args[3]);return {'completion_tokens':[1]},dict(path='raw.json',sha256='hash',fingerprint='fp')
    monkeypatch.setattr(final,'read_result',read)
    result=final.reuse_equivalent(tmp_path,r,'x',conditions,dict(fingerprint='fp'))
    assert result['completion_tokens']==[1] and calls==['y']
    aliases=json.loads((tmp_path/'final_aliases.json').read_text())
    assert aliases['x/v2/a']['source_stage']=='final'
    with pytest.raises(FileNotFoundError):final.reuse_equivalent(tmp_path,r,'z',conditions,dict(fingerprint='fp'))


def test_final_alias_is_audited_against_requested_thresholds(tmp_path,monkeypatch):
    path=tmp_path/'source.json';path.write_text(json.dumps(dict(config={'method':'mass'},thresholds={'local':1})))
    from experiments.diffusion_gemma_value_aware_followup.protocol import sha
    alias=dict(path=str(path),sha256=sha(path.read_bytes()),fingerprint='fp',source_stage='final')
    (tmp_path/'final_aliases.json').write_text(json.dumps({'mass_s50/a':alias}))
    def check(data,row,fp,config,thresholds,**kwargs):
        if data['config']!=config or data['thresholds']!=thresholds:raise ValueError('wrong alias configuration')
    monkeypatch.setattr(evidence,'check_result',check)
    row=dict(id='a',benchmark='longbench_v2')
    data,_=evidence.read_result(tmp_path,row,'final','mass_s50',dict(method='mass'),{'local':1},dict(fingerprint='fp'))
    assert data['thresholds']=={'local':1}
    with pytest.raises(ValueError,match='wrong alias'):
        evidence.read_result(tmp_path,row,'final','mass_s50',dict(method='mass'),{'local':2},dict(fingerprint='fp'))


def test_report_regeneration_is_deterministic_and_refuses_one_missing_final(tmp_path,monkeypatch):
    from experiments.diffusion_gemma_value_aware_followup.policies import candidate_configs
    configs=candidate_configs(dict(selected={n:dict(pooling='rms') for n in ('value','mass_value','risk')}))
    conditions={'dense':dict(config={},target=0.,target_metric='physical_sparsity')}
    conditions.update({f'{n}_s50':dict(config=c,target=.5,
        target_metric='pv_omission' if n in ('compensate','zero_pv') else 'physical_sparsity') for n,c in configs.items()})
    rows=[]
    for benchmark in ('aime26','longbench_v2'):
        for i in range(30):
            rows.append(dict(id=f'{benchmark}/{i}',benchmark=benchmark,task='aime26' if benchmark=='aime26' else f'domain{i//10}',
                seed=42,prompt_hash=f'hash{i}',generation_budget=128,calibration=benchmark=='aime26' and i<6,expected='B'))
    setup=dict(final=rows,revision='test');frozen=dict(conditions=conditions,expected_shards=780,scope='unit test')
    (tmp_path/'final_contracts').mkdir();(tmp_path/'final_contracts'/'broad50.json').write_text('{}')
    monkeypatch.setattr(report,'prepare',lambda root:setup)
    monkeypatch.setattr(report,'contract',lambda root:dict(fingerprint='fp'))
    monkeypatch.setattr(report,'load_contract',lambda *a:frozen)
    monkeypatch.setattr(evidence,'score',lambda r,p:float(p=='yes'))
    monkeypatch.setattr(evidence,'official_extractor',lambda:lambda p:'B')
    missing=set()
    def read(root,row,stage,name,config,policy,execution):
        if (row['id'],name) in missing:raise FileNotFoundError('test missing final')
        prediction='yes' if name=='dense' or int(row['id'].split('/')[-1])%2==0 else 'no'
        out={k:row[k] for k in ('id','seed','prompt_hash','generation_budget')}
        out.update(config=config,thresholds=policy,generation_metadata={},score=float(prediction=='yes'),
            prediction=prediction,completion_tokens=[1,2] if prediction=='yes' else [1,3],termination_reason='eos',
            records=[dict(layer=layer,head=0,step=0,probe='execution',attention_type=kind,
                eligible=10,skipped=0 if name=='dense' else 5,pv_omitted=0 if name=='dense' else 5,
                rows=1,mass_sum=1.,denominator_mass_sum=1.,error_sq=0.,dense_sq=1.)
                for layer,kind in ((0,'local'),(5,'global'))])
        return out,dict(path=f'test/{name}/{row["id"]}',sha256='test')
    monkeypatch.setattr(report,'read_result',read)
    def figures(dest,data):
        (dest/'figures').mkdir(exist_ok=True)
        for name in report.expected_figures(data):(dest/'figures'/name).write_bytes(b'test fixture')
    monkeypatch.setattr(report,'plots',figures)
    first=report.regenerate(tmp_path,'broad50')
    assert first['complete'] and first['completed']==780 and len(first['artifacts'])==21
    second=report.regenerate(tmp_path,'broad50')
    assert second['artifacts']==first['artifacts']
    missing.add(('aime26/0','value_s50'))
    partial=report.regenerate(tmp_path,'broad50')
    assert not partial['complete'] and not partial['raw_complete'] and partial['completed']==779
