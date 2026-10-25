"""Independent official rescoring of historical LongBench outputs; no text export."""
from pathlib import Path
import hashlib,json,sys
R=Path(__file__).resolve().parents[1];REPO=R.parents[1];H=R.parent/'response_behavior_and_stopping_headroom_20261025/track_h/attempt001';sys.path.insert(0,str(REPO))
from scripts import v31_official as official

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 manifest=H/'private/manifest.json';protocol=json.loads((H/'protocol.json').read_text());assert sha(manifest)==protocol['private_manifest_sha256'];byid={r['id']:r for r in json.loads(manifest.read_text())};final,_=official.load_final_response();inputs={str(manifest.relative_to(REPO)):sha(manifest)};rows=[]
 paths=list((H/'executions/full').glob('cell*/receipt.json'))+list((H/'executions/references').glob('cell*_piecewise/receipt.json'))+list((H/'executions/references').glob('cell*_A_observed/receipt.json'))
 paths += [p for p in (H/'executions/development').glob('cell*/receipt.json') if json.loads(p.read_text())['runtime']['pulse_calls']]
 for p in sorted(paths):
  value=json.loads(p.read_text());job=p.parents[1].name;private=H/'private/generations'/job/(value['label']+'.json');raw=json.loads(private.read_text());text=raw['completion'];answer=final(text,True);parsed=official.lb_extract_answer(answer);row=byid[value['id']]
  checks=dict(correct=(parsed==row['expected'])==value['correct'],parsed=(parsed is not None)==value['answer_parsed'],final_available=(not answer)==value['no_final_response']);assert all(checks.values())
  rows.append(dict(id=value['id'],seed=value['seed'],label=value['label'],job=job,correct=parsed==row['expected'],answer_parsed=parsed is not None,no_final_response=not answer,checks=checks,receipt_sha256=sha(p),private_completion_sha256=sha(private)))
  inputs[str(p.relative_to(REPO))]=sha(p);inputs[str(private.relative_to(REPO))]=sha(private)
 assert len(rows)==34
 out=Path(__file__).parent
 with (out/'official_rescore_verification.json').open('x') as f:json.dump(dict(passed=True,requests=len(rows),deleting_requests=10,FULL_PIECEWISE_A_controls=24,rows=rows,input_hashes=inputs,model_forwards=0,answers_and_text_not_published=True),f,indent=2)
 print('Official independent rescore passed:',len(rows),'historical requests')
if __name__=='__main__':main()
