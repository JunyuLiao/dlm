import copy
import json
from collections import Counter
import pytest

from experiments.diffusion_gemma_value_aware_followup import quick_review as quick,quick_report as report,evidence
from experiments.diffusion_gemma_value_aware_followup.policies import candidate_configs
from experiments.diffusion_gemma_value_aware.report_metrics import aggregate


def manifest():
    rows=[]
    for i in range(1,31):
        rows.append(dict(id=f'aime26/{i}',source_id=i,benchmark='aime26',task='AIME26',
            calibration=i in (2,8,14,20,23,30),seed=42,split='final',generation_budget=2048,prompt_hash=f'a{i}'))
    for domain in quick.DOMAINS:
        for difficulty in ('easy','hard'):
            for i in range(5):
                identity=f'{domain}/{difficulty}/{i}'
                rows.append(dict(id=identity,source_id=identity,benchmark='longbench_v2',task=domain,
                    difficulty=difficulty,calibration=False,seed=42,split='final',generation_budget=128,prompt_hash=identity))
    return rows


def test_fixed_selection_is_order_independent_and_excludes_calibration():
    rows=manifest();chosen=quick.select_rows(rows)
    assert chosen==quick.select_rows(list(reversed(rows)))
    assert len(chosen)==25 and Counter(r['benchmark'] for r in chosen)==Counter(quick.EXPECTED)
    assert not any(r['calibration'] for r in chosen)
    assert Counter((int(r['source_id'])-1)//15 for r in chosen if r['benchmark']=='aime26')=={0:5,1:5}
    assert Counter(r['difficulty'] for r in chosen if r['benchmark']=='longbench_v2')=={'easy':8,'hard':7}
    # Scores and availability do not enter selection.
    modified=[dict(r,prediction='changed',score=i,cache_available=i%2==0) for i,r in enumerate(rows)]
    assert [r['id'] for r in quick.select_rows(modified)]==[r['id'] for r in chosen]


@pytest.mark.parametrize('mutation',['missing','duplicate','calibration','seed','domain'])
def test_review_scope_guard(mutation):
    rows=copy.deepcopy(quick.select_rows(manifest()))
    if mutation=='missing':rows.pop()
    if mutation=='duplicate':rows[0]=rows[1]
    if mutation=='calibration':rows[0]['calibration']=True
    if mutation=='seed':rows[0]['seed']=7
    if mutation=='domain':rows[0]['task']='unrequested'
    with pytest.raises(ValueError):quick.validate_rows(rows)


def metrics_row(identity,benchmark,condition,accuracy,eligible=10,skipped=5):
    a=aggregate([dict(eligible=eligible,skipped=skipped,rows=2,mass_sum=1.5,error_sq=1,dense_sq=4)])
    return dict(id=identity,benchmark=benchmark,task='AIME26' if benchmark=='aime26' else 'domain',
        condition=condition,accuracy=accuracy,dense_accuracy=1.,calibration=False,matching=2,compared=4,
        exact_match=False,output_length=4,termination_reason='eos',unparsed_answer=False,
        aggregates={k:dict(a) for k in ('overall','global','local')})


def test_quick_metrics_are_count_weighted_not_sample_averaged():
    raw=[metrics_row('a','aime26','value_s50',1.,100,90),metrics_row('b','aime26','value_s50',0.)]
    rows,_=report.summarize(raw,{'value_s50':dict(config={'method':'value'},target=.5,target_metric='physical_sparsity')})
    assert rows[0]['correct']==1 and rows[0]['accuracy']==.5
    assert rows[0]['overall_physical_sparsity']==pytest.approx(95/110)
    assert rows[0]['token_agreement']==.5
    assert rows[0]['split']=='quick_review'  # Never mislabel this heldout24/full30.


def test_provisional_keep_compares_both_references_and_does_not_claim_matched_budget():
    conditions={n:dict(config={'method':m},target=.5,target_metric='physical_sparsity') for n,m in
        [('blasst_original_s50','blasst'),('blasst_aggressive_s50','blasst'),('value_s50','value')]}
    raw=[]
    for b,count in quick.EXPECTED.items():
        for n,right,skip in [('blasst_original_s50',5,2),('blasst_aggressive_s50',6,5),('value_s50',5,5)]:
            for i in range(count):raw.append(metrics_row(f'{b}/{i}',b,n,float(i<right),skipped=skip))
    rows,_=report.summarize(raw,conditions);comp=report.comparisons(rows,raw)
    assert all(r['extra_wrong_vs_better']==1 and r['provisional_keep_on_both_benchmarks'] for r in comp)
    assert all(not r['matched_physical_budget_vs_original'] and r['matched_physical_budget_vs_aggressive'] for r in comp)
    assert comp[0]['delta_vs_aggressive_pp']==pytest.approx(-10)
    broken=raw[:-1]
    with pytest.raises(ValueError,match='unpaired'):report.comparisons(rows,broken)


def test_full_quick_report_audits_325_and_refuses_one_missing_sample(tmp_path,monkeypatch):
    rows=quick.select_rows(manifest())
    configs=candidate_configs(dict(selected={n:dict(pooling='rms') for n in ('value','mass_value','risk')}))
    conditions={'dense':dict(config={},target=0.,target_metric='physical_sparsity')}
    conditions.update({f'{n}_s50':dict(config=c,target=.5,thresholds={'aime26':{},'longbench_v2':{}},
        target_metric='pv_omission' if n in ('compensate','zero_pv') else 'physical_sparsity') for n,c in configs.items()})
    frozen=dict(conditions=conditions,selection=dict(rows=rows,rule='fixture',truncated_v2=0))
    dest=quick.destination(tmp_path);dest.mkdir();(dest/'contract.json').write_text('{}')
    monkeypatch.setattr(report,'load',lambda root:(frozen,dict(fingerprint='fp')))
    monkeypatch.setattr(evidence,'score',lambda row,p:float(p=='yes'))
    monkeypatch.setattr(evidence,'official_extractor',lambda:lambda p:'B')
    missing=set()
    def read(root,row,stage,name,config,policy,execution):
        if (row['id'],name) in missing:raise FileNotFoundError('missing fixture')
        out={k:row[k] for k in ('id','seed','prompt_hash','generation_budget')}
        out.update(config=config,thresholds=policy,generation_metadata={},score=1.,prediction='yes',
            completion_tokens=[1,2],termination_reason='eos',records=[dict(layer=l,head=0,step=0,
                probe='execution',attention_type=k,eligible=10,skipped=0 if name=='dense' else 5,
                pv_omitted=0 if name=='dense' else 5,rows=1,mass_sum=1.,denominator_mass_sum=1.,error_sq=0.,dense_sq=1.)
                for l,k in ((0,'local'),(5,'global'))])
        return out,dict(path=f'fixture/{name}/{row["id"]}',sha256='fixture')
    monkeypatch.setattr(report,'read_result',read)
    def plot(dest,rows):
        folder=dest/'figures';folder.mkdir(exist_ok=True);(folder/'test.png').write_bytes(b'fixture');return ['test.png']
    monkeypatch.setattr(report,'plot',plot)
    first=report.regenerate(tmp_path);second=report.regenerate(tmp_path)
    assert first['complete'] and first['completed']==325 and set(first['per_condition'].values())=={25}
    assert first['artifacts']==second['artifacts']
    text=(dest/'report.md').read_text()
    assert 'PAUSED for user review' in text and 'No later targets/full study are launched' in text
    assert json.loads((dest/'review_ready.json').read_text())['later_targets_authorized'] is False
    missing.add((rows[0]['id'],'value_s50'))
    partial=report.regenerate(tmp_path)
    assert not partial['complete'] and partial['completed']==324 and not partial['raw_complete']
    assert not json.loads((dest/'review_ready.json').read_text())['complete']


@pytest.mark.parametrize('fail_one',[False,True])
def test_runner_executes_only_selected25_reuses_dense_and_continues_independent_failures(tmp_path,monkeypatch,fail_one):
    import dllm.models
    rows=quick.select_rows(manifest())
    configs=candidate_configs(dict(selected={n:dict(pooling='rms') for n in ('value','mass_value','risk')}))
    conditions={'dense':dict(config={})}
    conditions.update({f'{n}_s50':dict(config=c,thresholds={'aime26':{},'longbench_v2':{}}) for n,c in configs.items()})
    dest=quick.destination(tmp_path);dest.mkdir()
    monkeypatch.setattr(quick,'load',lambda root:({'conditions':conditions,'selection':{'rows':rows}},{}))
    def read(root,row,stage,*args,**kwargs):
        if stage!='dense':raise FileNotFoundError('uncached')
        return {},{}
    monkeypatch.setattr(quick,'read_result',read)
    def missing(*args,**kwargs):raise FileNotFoundError('no equivalent')
    monkeypatch.setattr(quick,'reuse_equivalent',missing)
    monkeypatch.setattr(quick,'pair',lambda *args:{})
    class Adapter:
        def load(self):return self
    monkeypatch.setattr(dllm.models,'create_adapter',lambda *a,**k:Adapter())
    calls=[]
    def cached(adapter,root,row,stage,name,config,policy,execution):
        if adapter is None:raise FileNotFoundError('uncached')
        calls.append((name,row['id']))
        assert stage=='final' and name!='dense' and row in rows
        assert config==conditions[name]['config'] and policy==conditions[name]['thresholds'][row['benchmark']]
        if fail_one and name=='value_s50' and row['id']==rows[0]['id']:raise RuntimeError('injected generation failure')
        return {}
    monkeypatch.setattr(quick,'cached',cached)
    monkeypatch.setattr(report,'regenerate',lambda root:dict(complete=not fail_one))
    if fail_one:
        with pytest.raises(RuntimeError,match='incomplete'):quick.run(tmp_path)
    else:quick.run(tmp_path)
    assert len(calls)==300 and len(set(calls))==300
    assert {i for n,i in calls}=={r['id'] for r in rows}
    terminal=json.loads((dest/'run_terminal.json').read_text())
    assert terminal['failed_attempts']==int(fail_one) and 'PAUSE' in terminal['next_action']
