"""Independent official CPU rescoring of complete canonical requests, no generation."""
from pathlib import Path
import sys,json,hashlib
S=Path(__file__).resolve().parent;REPO=S.parents[1];sys.path[:0]=[str(REPO),str(REPO/'scripts')]
from scripts import v31_official as official,v27_humaneval as he
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert json.loads((S/'attempt004/runs/qualification/final_complete.json').read_text())['passed'];paths=json.loads((S/'private/runtime_paths.json').read_text());manifest=json.loads((S/'private/manifest.json').read_text());byid={r['id']:r for r in manifest};gold={r['task_id']:r for r in json.loads(Path(paths['gold_path']).read_text()).values()};final,_=official.load_final_response();numeric,_=official.aime_numeric_score();he.sandbox_preflight();inputs={};rows=[];cache={}
    for attempt,jobs in [('attempt001',['dense_full','piecewise','references']),('attempt002',['treatments']),('attempt004',['qualification'])]:
        root=S/attempt
        for job in jobs:
            for p in sorted((root/'runs'/job).rglob('*.json')):
                r=json.loads(p.read_text())
                if 'model_forwards' not in r:continue
                source=byid[r['id']];private=(root/'private/generations'/r['label']).with_suffix('.json');assert sha(private)==r['private_generation_sha256'];inputs[str(p.relative_to(S))]=sha(p);inputs[str(private.relative_to(S))]=sha(private);text=json.loads(private.read_text())['completion'];key=(r['id'],hashlib.sha256(text.encode()).hexdigest())
                if key not in cache:
                    answer=final(text,True);task=r['benchmark']
                    if task=='aime26':v=numeric(answer,source['expected']);correct=v['correct'];parsed=v['extracted'] is not None
                    elif task=='longbench_v2':v=official.lb_extract_answer(answer);correct=v==source['expected'];parsed=v is not None
                    else:
                        problem=gold[source['source_id']];code,_=he.extract_code(text,problem['entry_point']);correct=he.run_test(he.program_for(problem,code))[0] if code is not None else False;parsed=code is not None
                    cache[key]=dict(correct=bool(correct),parsed=bool(parsed),missing_final=not answer)
                result=cache[key];checks=dict(correct=result['correct']==r['correct'],parsed=result['parsed']==r['answer_parsed'],missing_final=result['missing_final']==r['no_final_response']);rows.append(dict(id=r['id'],seed=r['seed'],benchmark=r['benchmark'],receipt=str(p.relative_to(S)),checks=checks,all_match=all(checks.values())))
    out=S/'official_rescore001';out.mkdir(exist_ok=False)
    for n,v in [('records.json',rows),('input_hashes.json',inputs),('final_complete.json',dict(passed=all(r['all_match'] for r in rows),records=len(rows),unique_question_outputs=len(cache),mismatches=[r for r in rows if not r['all_match']],official_gold_sha256=sha(Path(paths['gold_path'])),model_forwards=0))]:
        with (out/n).open('x') as f:json.dump(v,f,indent=2,sort_keys=True)
    assert all(r['all_match'] for r in rows),'Official score reproduction mismatch';print(json.dumps(dict(passed=True,records=len(rows),unique_question_outputs=len(cache))))
if __name__=='__main__':main()
