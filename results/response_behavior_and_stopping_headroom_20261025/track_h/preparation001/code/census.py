"""Existing dense-only later-canvas census; executable only after R readout."""
from collections import Counter
import hashlib
import json
from pathlib import Path


def stopping_class(row):
    if row['cap_reached']:
        return 'adaptive_and_cap' if row['native_history_ready'] and row['stable'] and row['confidence'] else 'cap_only_termination'
    if not row['native_history_ready']:return 'history_not_ready'
    if row['stable'] and not row['confidence']:return 'confidence_only_delay'
    if row['confidence'] and not row['stable']:return 'stability_only_delay'
    if not row['confidence'] and not row['stable']:return 'both_fail'
    return 'adaptive_predicates_satisfied'


def census(trace):
    canvas=0;consecutive=0;anchor=None;counts=Counter();canvases=[];current=[]
    for row in trace:
        if row['commit']:
            canvases.append(dict(canvas=canvas,calls=len(current),reached48=bool(current and current[-1]['native_step']>=48)))
            canvas+=1;consecutive=0;current=[];continue
        current.append(row);category=stopping_class(row);counts[category]+=1
        if row['cap_reached']:
            counts['cap_reached_calls']+=1
        if canvas>=1:
            counts['later_'+category]+=1
            consecutive=consecutive+1 if category=='confidence_only_delay' and not row['converged'] else 0
            if anchor is None and consecutive>=2 and row['native_step']<=46:
                anchor=dict(canvas=canvas,after_completed_calls=row['native_step'])
    return dict(counts=dict(counts),first_eligible_anchor=anchor,native_canvases=canvases,uncommitted_tail=len(current))


def main():
    root=Path(__file__).resolve().parents[1];repo=root.parents[3];study=root.parents[1]
    r=study/'track_r/attempt001/readout_frozen.json'
    assert r.exists(),'Track R readout must be frozen before Track H begins'
    old=repo/'results/denoising_e2e_research_20261024/attempt003'
    protocol=json.loads((old/'e2e_protocol.json').read_text());assert hashlib.sha256((old/'e2e_protocol.json').read_bytes()).hexdigest()=='88e46d4088a094968ebbeaed05b0ce4b9cf6e824a643675eba257ac9920d35dd'
    leaf=old/'runs/panel/long/dense_full_fix51994';marker=json.loads((leaf/'final_complete.json').read_text());assert marker['passed']
    binding=old.parent/'attempt006/execution_binding.json';expected=hashlib.sha256(binding.read_bytes()).hexdigest()
    assert marker['execution_binding_sha256']==expected
    rows=[];inputs={}
    for i,identity in enumerate(protocol['panel']['long']):
        path=leaf/f'audit{i:03d}.json';v=json.loads(path.read_text());assert(v['id'],v['seed'])==(identity['id'],identity['seed'])
        assert v['execution_binding_sha256']==expected
        clean=path.with_name(path.name.replace('audit','clean'));cv=json.loads(clean.read_text())
        assert all(v[k]==cv[k] for k in ['id','seed','output_sha256','N','C','T']) and cv['execution_binding_sha256']==expected
        for consumed in [path,clean,binding,leaf/'final_complete.json']:
            inputs[str(consumed.relative_to(repo))]=hashlib.sha256(consumed.read_bytes()).hexdigest()
        result=census(v['native_trace'])
        assert len(result['native_canvases'])==v['C'] and sum(x['calls'] for x in result['native_canvases'])+result['uncommitted_tail']==v['N']
        rows.append(dict(**identity,**result))
    out=root/'dense_census001';out.mkdir(exist_ok=False)
    for name,value in [('summary.json',dict(passed=True,rows=rows,scope='Existing corrected FULL long traces only; later canvases separated; no sparse outcomes used',Track_R_readout_sha256=hashlib.sha256(r.read_bytes()).hexdigest())),('input_hashes.json',inputs)]:
        with (out/name).open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps(dict(cells=len(rows),anchors=sum(x['first_eligible_anchor'] is not None for x in rows))))


if __name__=='__main__':main()
