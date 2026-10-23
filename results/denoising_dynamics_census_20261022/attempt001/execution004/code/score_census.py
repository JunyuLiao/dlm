"""Official task scoring, with private text and public numeric outcomes only."""
import hashlib
import json
from pathlib import Path
import sys

REPO=Path('/home/exouser/ljy/dlm')
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'scripts'))
from scripts import v31_official as official
from scripts import v27_humaneval as he

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'scoring001'
DEST.mkdir(exist_ok=False)
rows={r['id']:r for r in json.loads((ROOT/'private/tokenized_manifest.json').read_text())}
gold_path=REPO/'results/humaneval_value_observe_no_overlap_k256_20261019/attempt001/private/pools/thinking_on/gold.json'
gold={r['task_id']:r for r in json.loads(gold_path.read_text()).values()}
final,final_source=official.load_final_response()
numeric,numeric_source=official.aime_numeric_score()
he.sandbox_preflight()
outcomes=[]
bindings={}
for group in ('short','long'):
    directory=ROOT/('census_'+group)
    marker=directory/'final_complete.json'
    if not marker.is_file():continue
    if not json.loads(marker.read_text())['passed']:raise RuntimeError('census gate failed')
    for record_path in sorted(directory.glob('request*/record.json')):
        record=json.loads(record_path.read_text())
        if record['label']=='warm':continue
        row=rows[record['id']]
        token_hash=hashlib.sha256(json.dumps(row['prompt_tokens']).encode()).hexdigest()
        if token_hash!=record['prompt_tokenization_sha256'] or row['prompt_hash']!=record['prompt_hash']:
            raise RuntimeError('scoring input identity drift')
        raw=json.loads((record_path.parent/'private/generation.json').read_text())['completion']
        response=final(raw,True)
        if row['benchmark']=='aime26':
            result=numeric(response,row['expected']);passed=bool(result['correct'])
            detail=dict(parsed=result['extracted'] is not None)
        elif row['benchmark']=='humaneval':
            problem=gold[row['source_id']]
            code,reason=he.extract_code(raw,problem['entry_point'])
            if code is None:passed,outcome=False,'not_run'
            else:passed,outcome=he.run_test(he.program_for(problem,code))
            if outcome=='timeout':passed,outcome=he.run_test(he.program_for(problem,code))
            detail=dict(extraction=reason,outcome=outcome)
        else:
            prediction=official.lb_extract_answer(response);passed=prediction==row['expected']
            detail=dict(parsed=prediction is not None)
        outcomes.append(dict(id=row['id'],benchmark=row['benchmark'],correct=bool(passed),
            finish_reason=record['finish_reason'],no_final_response=not response,**detail))
        bindings[row['id']]=dict(record_sha256=official.sha_file(record_path),
            prompt_hash=row['prompt_hash'],prompt_tokenization_sha256=token_hash,
            output_sha256=record['output_sha256'],generation_budget=row['generation_budget'],seed=row['seed'])
summary={}
for benchmark in sorted({r['benchmark'] for r in outcomes}):
    selected=[r for r in outcomes if r['benchmark']==benchmark]
    summary[benchmark]=dict(items=len(selected),correct=sum(r['correct'] for r in selected),
        accuracy=sum(r['correct'] for r in selected)/len(selected),
        capped=sum(r['finish_reason']=='length' for r in selected),
        no_final_response=sum(r['no_final_response'] for r in selected))
result=dict(schema='dense_census_official_scoring_v1',protocol_sha256=official.sha_file(ROOT/'protocol.json'),
    summary=summary,outcomes=outcomes,bindings=bindings,
    scorer_sources={str(p):official.sha_file(p) for p in [Path(he.__file__),Path(official.__file__),gold_path]},
    final_response_source=final_source,numeric_score_source=numeric_source,
    human_eval_sandbox_preflight='passed',paired_uncertainty=None,
    limitation='Dense-only seed42 descriptive screen; no matched sparse contrast or accuracy-preservation claim')
(DEST/'official_scores.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
(DEST/'scoring_complete.json').write_text(json.dumps(dict(complete=True,items=len(outcomes),
    score_sha256=official.sha_file(DEST/'official_scores.json')),indent=2)+'\n')
print(json.dumps(summary,indent=2))
