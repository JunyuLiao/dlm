"""Independent CPU parent-range attribution; no deepest-child helper reuse."""
import bisect,hashlib,json,sqlite3
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
P=ROOT/'results/v31_refresh_repeatability_20261018/attempt001/nvtx'
OUT=Path(__file__).parent
read=lambda p:json.loads(p.read_text())
meta={r['id']:r for r in read(P/'probe.json')['scopes']}
db=sqlite3.connect((P/'private/trace.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
strings=dict(db.execute('SELECT id,value FROM StringIds'));columns=[r[1] for r in db.execute('PRAGMA table_info(NVTX_EVENTS)')]
scopes={}
for value in db.execute('SELECT * FROM NVTX_EVENTS'):
 r=dict(zip(columns,value));label=r.get('text') or strings.get(r.get('textId'),'')
 if str(label).startswith('refresh_probe:') and r.get('end') is not None:scopes[int(label.split(':')[1])]=r
assert set(scopes)==set(meta)
families={name:defaultdict(list) for name in ['attention','buffer']}
labels={}
for ident,m in meta.items():
 r=scopes[ident]
 if '/global_' in m['label'] or m['label'].endswith('/local_attention'):
  families['attention'][r['globalTid']].append((r['start'],r['end'],ident));labels[ident]=m['label']
 elif m['label']=='denoise/kv_buffer':
  families['buffer'][r['globalTid']].append((r['start'],r['end'],ident))
  parent=m['parent']
  while not meta[parent]['label'].startswith('denoise/global_'):parent=meta[parent]['parent']
  labels[ident]='kv_buffer_under_'+meta[parent]['label'].split('global_',1)[1]
starts={};stats=defaultdict(lambda:dict(cpu_ns=0,gpu_sum_ns=0,calls=0,activities=0))
for name,by_thread in families.items():
 starts[name]={}
 for thread,v in by_thread.items():
  v.sort();assert all(a[1]<=b[0] for a,b in zip(v,v[1:])), 'primary intervals overlap'
  starts[name][thread]=[r[0] for r in v]
  for begin,end,ident in v:
   s=stats[(meta[ident]['request'],labels[ident])];s['cpu_ns']+=end-begin;s['calls']+=1
links={name:{} for name in families};conflicts=[]
tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
 if table not in tables:continue
 for a,b,thread,corr in db.execute(f'SELECT start,end,globalTid,correlationId FROM {table}'):
  for name,by_thread in families.items():
   v=by_thread.get(thread,[]);idx=bisect.bisect_right(starts[name].get(thread,[]),a)-1
   if idx>=0 and v[idx][0]<=a and b<=v[idx][1]:
    ident=v[idx][2]
    if corr in links[name] and links[name][corr]!=ident:conflicts.append((name,corr))
    links[name][corr]=ident
for table in ('CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
 if table not in tables:continue
 for corr,duration,count in db.execute(f'SELECT correlationId,SUM(end-start),COUNT(*) FROM {table} GROUP BY correlationId'):
  for name,link in links.items():
   if corr in link:
    ident=link[corr];s=stats[(meta[ident]['request'],labels[ident])];s['gpu_sum_ns']+=duration;s['activities']+=count
db.close();assert not conflicts
expected=read(P/'attribution.json')['attribution']['rows']+read(OUT/'kv_buffer_attribution.json')['buffer_rows']
compared=0
for r in expected:
 key=r['request'],r['label']
 if key not in stats:continue
 for field,value in stats[key].items():assert r[field]==value,(key,field,r[field],value)
 compared+=1
assert compared==len(stats)
report=dict(status='independent_parent_and_buffer_sums_match',groups_checked=compared,scope_count=len(scopes),
 correlation_conflicts=0,algorithm='Direct nonoverlapping parent intervals and separately nonoverlapping KV-buffer intervals; no bounded-depth search or child-ancestry activity propagation.',
 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P/'probe.json',P/'attribution.json',OUT/'kv_buffer_attribution.json']})
with (OUT/'independent_attribution_verification.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({k:report[k] for k in ['status','groups_checked','scope_count']}))
