from experiments import diffusion_gemma_ruler8k_jl_v14 as study


def test_fresh_result_has_identical_metadata_to_resume(tmp_path,monkeypatch):
    row=dict(id='ruler8k/test');contract=dict(fingerprint='test');config={};threshold=None
    path=study.base.shard_path(tmp_path,'smoke','dense',row['id'])
    result=dict(records=[dict(layer=0,eligible=10,skipped=0)],completion_tokens=[1,2],fingerprint='test')
    calls=[]
    def raw(adapter,root,row,stage,label,name,config,thresholds,contract,**kwargs):
        calls.append(adapter)
        if path.exists():return study.base.runner.load_output(path)
        study.base.runner.write_output(path,result)
        return result
    monkeypatch.setattr(study.base.runner,'cached',raw)
    monkeypatch.setattr(study.base.runner,'check_result',lambda *args:None)
    with study.complete_metadata():
        fresh=study.base.runner.cached('model',tmp_path,row,'smoke','dense','dense',config,threshold,contract)
        resumed=study.base.runner.cached(None,tmp_path,row,'smoke','dense','dense',config,threshold,contract)
    assert fresh==resumed and fresh['records']==result['records']
    assert fresh['records_source']['path']==str(path.with_suffix('.records.json.gz'))
    assert 'records_source' not in result and calls==['model',None]


def test_normalization_scope_restores_frozen_runner(monkeypatch):
    original=study.base.runner.cached
    with study.complete_metadata():assert study.base.runner.cached is not original
    assert study.base.runner.cached is original
