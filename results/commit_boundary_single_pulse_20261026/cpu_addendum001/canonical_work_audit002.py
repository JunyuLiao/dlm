from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[1];OLD=R.parent/'response_behavior_and_stopping_headroom_20261025/track_r/attempt001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 correction=OLD/'closeout002/physical_work_corrections.json';d=json.loads(correction.read_text());raw={}
 for p in (OLD/'runs/panel').glob('*/*/audit*.json'):
  h=sha(p);assert h not in raw;raw[h]=(p,json.loads(p.read_text()))
 assert len(raw)==700
 checked=[]
 for row in d['rows']:
  p,x=raw[row['input_sha256']];w=x['work']
  for scope,prefix in [('GLOBAL','global'),('LOCAL','local')]:
   eligible=w['global_eligible_tiles'] if prefix=='global' else w['native_local_eligible_rectangles']
   kept=w['global_kept_tiles'] if prefix=='global' else eligible-w['native_local_skipped_tiles']
   assert row[scope]['eligible']==eligible and row[scope]['kept']==kept
   assert row[scope]['skipped']==row[scope]['eligible']-row[scope]['kept']
  assert row['overall']['eligible']==row['GLOBAL']['eligible']+row['LOCAL']['eligible']
  assert row['overall']['skipped']==row['GLOBAL']['skipped']+row['LOCAL']['skipped']
  assert row['overall_rectangle_sparsity']==row['overall']['skipped']/row['overall']['eligible']
  assert sum(v['eligible'] for v in row['phases'].values())==row['GLOBAL']['eligible']
  assert sum(v['kept'] for v in row['phases'].values())==row['GLOBAL']['kept']
  checked.append(dict(id=row['id'],seed=row['seed'],arm=row['arm'],source=str(p.relative_to(R.parents[1])),sha256=sha(p)))
 with (Path(__file__).parent/'canonical_work_verification.json').open('x') as f:json.dump(dict(passed=True,raw_receipts=len(checked),stale_receipts=d['corrected_rows'],canonical_source=str(correction.relative_to(R.parents[1])),canonical_sha256=sha(correction),raw_inputs=checked,originals_unchanged=True),f,indent=2)
 print('Canonical work independently verified:',len(checked))
if __name__=='__main__':main()
