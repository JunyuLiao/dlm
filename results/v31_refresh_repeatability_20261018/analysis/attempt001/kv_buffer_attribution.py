"""CPU supplement: isolate buffer upkeep on held/carried steps without overlap."""
import hashlib,json,sqlite3
from pathlib import Path
from scripts.v31_refresh_followup import attribute
ROOT=Path(__file__).resolve().parents[4]
P=ROOT/'results/v31_refresh_repeatability_20261018/attempt001/nvtx'
read=lambda p:json.loads(p.read_text())
metadata={r['id']:dict(r) for r in read(P/'probe.json')['scopes']}
for row in metadata.values():
 if row['label']=='denoise/kv_buffer':
  parent=row['parent'];label=None
  while parent is not None:
   candidate=metadata[parent]
   if candidate['label'].startswith('denoise/global_'):
    label=candidate['label'].split('global_',1)[1];break
   parent=candidate['parent']
  assert label is not None
  row['label']='kv_buffer_under_'+label

db=sqlite3.connect((P/'private/trace.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
names=dict(db.execute('SELECT id,value FROM StringIds'))
columns=[r[1] for r in db.execute('PRAGMA table_info(NVTX_EVENTS)')]
ranges=[]
for values in db.execute('SELECT * FROM NVTX_EVENTS'):
 r=dict(zip(columns,values));label=r.get('text') or names.get(r.get('textId'),'')
 if str(label).startswith('refresh_probe:') and r.get('end') is not None:
  ranges.append(dict(id=int(label.split(':')[1]),start=r['start'],end=r['end'],thread=r['globalTid']))
assert {r['id'] for r in ranges}==set(metadata)
tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
apis=[];activities=[]
for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
 if table in tables:apis += [dict(start=a,end=b,thread=t,correlation=c) for a,b,t,c in db.execute(f'SELECT start,end,globalTid,correlationId FROM {table}')]
for table in ('CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
 if table in tables:activities += [dict(start=a,end=b,correlation=c) for a,b,c in db.execute(f'SELECT start,end,correlationId FROM {table}')]
report=attribute(ranges,apis,activities,metadata);db.close()
old=read(P/'attribution.json')['attribution']['rows']
kv=[r for r in report['rows'] if r['label'].startswith('kv_buffer_under_')]
for r in old:
 if r['label']!='denoise/kv_buffer':continue
 for key in ('cpu_ns','gpu_sum_ns','calls','activities'):
  assert r[key]==sum(x[key] for x in kv if x['request']==r['request']),(r['request'],key)
report=dict(status='complete_cpu_supplement',buffer_rows=kv,
 interpretation='Only held/carried buffer rows can be added to dense-parent deletion screens without double counting. This still does not measure runtime savings or certify removal of the buffer path.',
 input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'probe.json',P/'attribution.json')},
 analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
out=Path(__file__).with_name('kv_buffer_attribution.json')
with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(dict(status=report['status'],rows=len(kv),reconciliation='exact CPU/GPU/call/activity sums')))
