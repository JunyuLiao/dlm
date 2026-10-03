"""CPU-only metric checks, no model or GPU execution."""
from experiments.value_direction_hopper.report import agreement,paired_ci


def test_positional_agreement_continues_after_divergence():
    assert agreement([1,2,3],[1,9,3,4])==dict(matches=2,compared=4,exact=False)
    assert agreement([],[])==dict(matches=0,compared=0,exact=True)


def test_bootstrap_is_paired_and_deterministic():
    pairs=[({'task':'a'},{'score':.5},{'score':1.}),({'task':'b'},{'score':1.},{'score':.5})]
    a=paired_ci(pairs,'a','b');b=paired_ci(pairs,'a','b')
    assert a==b
    assert a['mean']==a['lower']==a['upper']==0.


def test_independent_audit_rejects_rescored_accuracy_and_missing_coverage(tmp_path,monkeypatch):
    import hashlib,json
    from experiments.value_direction_hopper import systems_report
    from experiments.value_direction_hopper.experiment import shard_path
    from experiments import diffusion_gemma_ruler8k_jl as scoring
    monkeypatch.setattr(scoring,'score',lambda row,text:1.)
    row=dict(id='example',prompt_hash='same',seed=42,generation_budget=128,task='niah_single_1')
    contract=dict(fingerprint='frozen',methods=['native_dense'])
    (tmp_path/'configuration.json').write_text(json.dumps(contract))
    (tmp_path/'manifest.json').write_text(json.dumps([row]))
    path=shard_path(tmp_path,'native_dense',row);path.parent.mkdir(parents=True)
    record=dict(row,method='native_dense',fingerprint='frozen',status='complete',prediction='answer',
                score=0.,wall_seconds=1.,completion_tokens=[1],metadata=dict(denoising_configuration={},sampling={},thinking=False,native_canvas_length=256))
    path.write_text(json.dumps(record))
    audit,_=systems_report.verify_final(tmp_path)
    assert not audit['passed'] and 'rescoring mismatch' in audit['violations'][0]
    record['score']=1.;path.write_text(json.dumps(record))
    audit,_=systems_report.verify_final(tmp_path)
    assert audit['passed'] and audit['completed']==1
    record['generation_budget']=129;path.write_text(json.dumps(record))
    audit,_=systems_report.verify_final(tmp_path)
    assert not audit['passed'] and 'generation_budget' in audit['violations'][0]
    import gzip
    contract['methods']=['kernel_gaussian32']
    (tmp_path/'configuration.json').write_text(json.dumps(contract))
    path=shard_path(tmp_path,'kernel_gaussian32',row);path.parent.mkdir(parents=True)
    routing=path.with_suffix('.routing.json.gz')
    with gzip.open(routing,'wt') as f:
        json.dump([dict(layer=0,head=0,step=0,attention_type='local',eligible=1,skipped=0,
                        prefix_skipped=0,canvas_skipped=0,boundary_skipped=0)],f)
    record.update(method='kernel_gaussian32',generation_budget=128,routing_path=str(routing),
                  routing_sha256=hashlib.sha256(routing.read_bytes()).hexdigest())
    path.write_text(json.dumps(record))
    audit,_=systems_report.verify_final(tmp_path)
    assert not audit['passed'] and 'Missing layer/head coverage' in audit['violations'][0]
