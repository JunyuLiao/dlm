import json

from scripts import v31_humaneval_cfgdet_report as report


def test_audit_equivalence_allows_method_differences_but_rejects_interval_changes(tmp_path,monkeypatch):
    (tmp_path/'protocol.json').write_text(json.dumps(dict(cohort=[0])))
    def records(root,mode,part,label,stage='runs',private=False):
        identity='control_before' if label=='control_after' else label
        values={}
        for seed in (42,43):
            if private:
                row=dict(completion='program-'+identity)
            else:
                row=dict(output_hash=identity,denoise_forwards=5,canvases=2,output_tokens=300,
                         sampler_calls=7,native_interval_steps=[2,3])
            values[(seed,0)]=row
        return values
    monkeypatch.setattr(report,'maps',records)
    assert all(x['passed'] for x in report.qualification_equivalence(tmp_path))
    def changed(*args,**kwargs):
        values=records(*args,**kwargs)
        if args[2]=='audit' and args[3]==report.campaign.V2 and not (len(args)>5 and args[5]):
            for row in values.values():row['native_interval_steps']=[1,4]
        return values
    monkeypatch.setattr(report,'maps',changed)
    result=report.qualification_equivalence(tmp_path)
    affected=[x for x in result if x['label']==report.campaign.V2]
    assert all(not x['passed'] and x['differences']['native_interval_steps']==2 for x in affected)
    assert all(x['differences']['denoise_forwards']==0 for x in affected)


def test_selected_population_does_not_silently_overwrite_duplicate_questions(tmp_path,monkeypatch):
    leaf=tmp_path/'export';leaf.mkdir()
    record=dict(index=0,panel_seed=42)
    (leaf/'records.jsonl').write_text(json.dumps(record)+'\n'+json.dumps(record)+'\n')
    monkeypatch.setattr(report,'selected_paths',lambda *args:{'control_before':[leaf]})
    import pytest
    with pytest.raises(RuntimeError,match='duplicate question'):
        report.maps(tmp_path,'thinking_on','clean','control_before')
