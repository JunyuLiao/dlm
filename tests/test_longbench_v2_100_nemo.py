from collections import Counter
from copy import deepcopy
import json
from unittest.mock import patch

import pytest

from experiments.diffusion_gemma_longbench_v2_100 import protocol,nemo,calibration,workflow,report
from experiments.diffusion_gemma_value_aware.report_metrics import SUM_FIELDS


def test_proportional_selection_reproducible_and_fresh():
    data=protocol.items();excluded={r['_id'] for r in data[:20]}
    first,quotas,sub=protocol.select(data,excluded)
    assert protocol.select(list(reversed(data)),excluded)==(first,quotas,sub)
    assert sorted(quotas.values())==[6,8,10,16,25,35]
    assert {k:len(v) for k,v in first.items()}==dict(development=6,calibration=12,final=100)
    ids=[r['_id'] for g in first.values() for r in g]
    assert len(set(ids))==118 and not set(ids)&excluded
    assert Counter(r['domain'] for r in first['final'])==quotas
    for domain,q in sub.items():assert Counter(r['sub_domain'] for r in first['final'] if r['domain']==domain)==q
    assert Counter(r['difficulty'] for r in first['calibration'])==dict(easy=6,hard=6)


def test_sampling_does_not_use_answers():
    data=protocol.items();changed=[dict(r,answer='A') for r in data]
    a,_,_=protocol.select(data,set());b,_,_=protocol.select(changed,set())
    assert {k:[r['_id'] for r in v] for k,v in a.items()}=={k:[r['_id'] for r in v] for k,v in b.items()}


def test_hamilton_ties_and_total():
    assert protocol.apportion(dict(z=1,a=1,b=1),2)==dict(z=0,a=1,b=1)
    assert sum(protocol.apportion(dict(a=17,b=33,c=50),37).values())==37


@pytest.mark.parametrize('prediction,answer,parsed,correct',[
    (r'Answer: \boxed{B}','B','B',True),
    (r'Reasoning. The final answer is C','C','C',True),
    ('Answer: A','B','A',False),
    ('This is unfinished reasoning','D',None,False),
    (r'Answer: \boxed{A}','D','A',False),
])
def test_actual_nemo_evaluator(prediction,answer,parsed,correct):
    out,=nemo.evaluate([dict(generation=prediction,expected_answer=answer)])
    assert (out['predicted_answer'] if out['predicted_answer'] in tuple('ABCD') else None)==parsed
    assert out['symbolic_correct']==correct


class Tokenizer:
    all_special_tokens=['<|image|>','<end_of_turn>']
    def encode(self,text,**unused):return list(map(ord,text))
    def decode(self,ids,**unused):return ''.join(map(chr,ids))


class Adapter:
    tokenizer=Tokenizer()
    def encode_prompt(self,text,settings):
        assert settings=={'thinking':False}
        return [1]+list(map(ord,text))+[2]


def test_context_only_truncation_retains_question_and_nemo_instruction():
    item=dict(protocol.items()[0],context='HEAD '+'x'*10000+' TAIL',question='DISTINCT QUESTION',choice_A='CHOICE_A')
    row=protocol.render(Adapter(),item,1600,4096,'final')
    assert row['truncated'] and len(row['prompt_tokens'])<=1600
    for text in ('HEAD','TAIL','DISTINCT QUESTION','CHOICE_A',r'\boxed'):
        assert text in row['prompt']
    assert row['generation_budget']==4096 and row['seed']==42
    assert row['prompt_hash']==protocol.sha(row['prompt'])
    assert row==protocol.render(Adapter(),item,1600,4096,'final')


def test_literal_special_tokens_are_escaped_not_dropped_or_interpreted():
    item=dict(protocol.items()[0],context='Literal <|image|> and <end_of_turn> in repository documentation.')
    row=protocol.render(Adapter(),item,10000,4096,'final')
    assert row['literal_special_token_escapes']=={'<|image|>':1,'<end_of_turn>':1}
    assert '&lt;|image|&gt;' in row['prompt'] and '<|image|>' not in row['prompt']
    assert row['source_id']==item['_id'] and row['expected']==item['answer']


def test_parallel_preparation_uses_isolated_real_tokenizers(tmp_path):
    from dllm.models import create_adapter
    adapter=create_adapter('diffusion_gemma',protocol.MODEL,device='cpu',precision='float32',revision=protocol.REVISION).load_tokenizer()
    data=[dict(protocol.items()[0],_id=f'fixture{i}',context=f'Example {i}: literal <|image|>, '+('short document. '*20)) for i in range(12)]
    selection=dict(groups={'final':[dict(_id=r['_id']) for r in data]},dataset_sha256='fixture')
    with patch.object(protocol,'selection',lambda root:selection),patch.object(protocol,'items',lambda:data):
        rows=protocol.rows_for(adapter,'final',1600,4096,tmp_path)
        repeated=protocol.rows_for(adapter,'final',1600,4096,tmp_path)
    assert rows==repeated
    assert rows==[protocol.render(adapter,r,1600,4096,'final') for r in data]
    assert len(list((tmp_path/'prepared_rows').glob('*.json')))==12


def small_setup():
    rows=[dict(id=f'x/{i}',source_id=str(i),split='calibration',benchmark='longbench_v2') for i in range(12)]
    return dict(calibration=rows,final=[dict(id='final')],development=[dict(id='dev')])


def test_calibration_rejects_final_contamination_and_counts():
    setup=small_setup();assert len(calibration.calibration_rows(setup))==12
    setup['calibration'][0]['split']='final'
    with pytest.raises(ValueError):calibration.calibration_rows(setup)
    setup=small_setup();setup['final']=[setup['calibration'][0]]
    with pytest.raises(ValueError):calibration.calibration_rows(setup)
    with pytest.raises(ValueError):calibration.collect(None,None,dict(split='final'),{})


def test_old_boundary_is_reverified_not_imported(tmp_path):
    folder=tmp_path/'final_configs';folder.mkdir()
    policy=dict(config={'method':'blasst'},thresholds={'longbench_v2':{
        'local':dict(log_threshold=0.,cap_one=True,unattainable=True),
        'global':dict(log_scale=7.,cap_one=True)}})
    (folder/'blasst_original_s75.json').write_text(json.dumps(policy))
    with patch.object(calibration,'PARENT',tmp_path):
        result=calibration.starting_policy(None,'longbench_v2','blasst_original',dict(method='blasst'),.75)
    assert result['local']['log_threshold']==0 and result['local']['cap_one']
    assert 'unattainable' not in result['local'] and result['global']['log_scale']==7


@pytest.mark.parametrize('n,limited,parsed,passed',[(100,5,90,True),(100,6,99,False),(100,0,89,False),(99,0,99,False)])
def test_dense_gate_is_completion_not_correctness(n,limited,parsed,passed):
    outputs=[dict(termination_reason='length' if i<limited else 'eos',prediction=str(i)) for i in range(n)]
    def evaluator(unused):return [dict(predicted_answer='D' if i<parsed else None,symbolic_correct=False) for i in range(n)]
    with patch.object(nemo,'evaluate',evaluator):assert workflow.dense_gate(outputs)['passed']==passed


def record(eligible,skipped,rows,mass,kind='global'):
    r=dict.fromkeys(SUM_FIELDS,0.)
    r.update(eligible=eligible,skipped=skipped,pv_omitted=skipped,softmax_skipped=skipped,rows=rows,mass_sum=mass,
        denominator_mass_sum=mass,dense_sq=1.,prefix_eligible=eligible,prefix_skipped=skipped,
        attention_type=kind,probe='execution',layer=5 if kind=='global' else 0,head=0,step=0)
    return r


def test_nemo_pairs_positional_agreement_and_count_weighted_sparsity():
    row=dict(id='longbench_v2/x',source_id='x',benchmark='longbench_v2',task='t',prompt_hash='h',seed=42,generation_budget=4096,expected='B')
    common={k:row[k] for k in ('id','prompt_hash','seed','generation_budget')}
    dense=dict(**common,config={},thresholds=None,prediction=r'Answer: \boxed{B}',score=1.,generation_metadata={},
        completion_tokens=[1,2,3],termination_reason='eos',records=[record(100,0,10,10)])
    sparse=dict(dense,config={'method':'mass'},thresholds={},completion_tokens=[1,9,3,4],
        records=[record(90,90,9,1),record(10,0,1,1,'local')])
    pairs,_=report.pairs(row,dict(dense=dense,mass_s50=sparse))
    p=pairs[1];assert (p['matching'],p['compared'])==(2,4)
    assert p['aggregates']['overall']['physical_sparsity']==.9
    assert p['aggregates']['overall']['mass']==.2
    assert p['parsed_answer']=='B' and p['accuracy']==1.
    assert not p['exact_match']
    bad=dict(sparse,score=0.)
    with pytest.raises(ValueError):report.pairs(row,dict(dense=dense,mass_s50=bad))


def test_summary_micro_is_not_equal_domain_macro():
    base=dict(task='a',accuracy=1.,dense_accuracy=1.,matching=1,compared=1,exact_match=True,
        unparsed_answer=False,termination_reason='eos',output_length=1,
        aggregates={k:report.aggregate([record(10,5,1,1)]) for k in ('overall','global','local')})
    group=[base]+[dict(base,task='b',accuracy=0.) for _ in range(3)]
    out=report.summarize(group)
    assert out['accuracy']==.25 and out['equal_task_macro']==.5
    assert out['delta']==-.75 and out['relative_accuracy']==.25


def test_raw_report_requires900_and_regenerates_identically(tmp_path):
    """Exercise the complete reporting path with synthetic CPU-only raw evidence."""
    labels=['dense']+[f'{n}_s{int(t*100)}' for t in protocol.TARGETS for n in protocol.METHODS]
    manifest=[dict(id=f'longbench_v2/{i}',source_id=str(i),benchmark='longbench_v2',task=f'd{i%6}',
        sub_domain=f't{i%20}',difficulty='hard',length_band='medium',prompt_tokens=[1,2],prompt_hash=str(i),
        seed=42,generation_budget=4096,expected='B',truncated=False) for i in range(100)]
    counts=Counter(r['task'] for r in manifest)
    setup=dict(final=manifest,conditions=labels,input_budget=32768,output_budget=4096,revision='fixture',
        selection=dict(domain_population=dict(counts),domain_quotas=dict(counts)))
    configs={};raw={}
    for label in labels:
        name=label.rsplit('_s',1)[0] if label!='dense' else label
        target=int(label.rsplit('_s',1)[1])/100 if label!='dense' else 0.
        c=dict(config=protocol.CONFIGS.get(name,{}),target=target,thresholds={} if name=='dense' else
            {'longbench_v2':{k:dict(log_threshold=-1.) for k in ('local','global')}},
            policy_sources={},target_metric='physical_sparsity')
        configs[label]=c
        path=tmp_path/'final_configs'/f'{label}.json';path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(dict(c,fingerprint='test',sources={})))
        if name!='dense':
            p=tmp_path/'verified_policies/longbench_v2'/f'{label}.json';p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(json.dumps(dict(name=name,target=target,measured={'local':target,'global':target},within_two_points=True)))
        for row in manifest:
            identity={k:row[k] for k in ('id','prompt_hash','seed','generation_budget')}
            output=dict(**identity,config=c['config'],thresholds=c['thresholds'].get('longbench_v2'),
                prediction=r'Answer: \boxed{B}',score=1.,completion_tokens=[1,2,3],generation_metadata={},
                termination_reason='eos',records=[record(80,80*target,8,8),record(20,20*target,2,2,'local')])
            raw[row['id'],label]=output
            path=report.shard_path(tmp_path,'dense' if name=='dense' else 'final',label,row['id'])
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(output))
    def cached(adapter,root,row,stage,label,config,threshold,execution):return raw[row['id'],label]
    def selected(root,setup,execution,name,target):return configs[f'{name}_s{int(target*100)}']
    with patch.object(report,'prepare',lambda root:setup),patch.object(report,'execution',lambda root:dict(fingerprint='test')),\
         patch.object(report,'cached',cached),patch.object(report,'condition',selected):
        first=report.regenerate(tmp_path)
        assert first['complete'] and first['completed']==900
        second=report.regenerate(tmp_path)
        assert first['artifacts']==second['artifacts']
        raw.pop((manifest[-1]['id'],labels[-1]))
        partial=report.regenerate(tmp_path)
        assert not partial['complete'] and partial['completed']<900
