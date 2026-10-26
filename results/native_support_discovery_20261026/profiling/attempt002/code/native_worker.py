"""Fixed observer qualification or complete finite baseline profiling."""
from pathlib import Path
import argparse,json,time
from native_harness import NativeHarness,save,sha
LIVE={}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',choices=['qualify','profile'],required=True);ap.add_argument('--arm',required=True);a=ap.parse_args()
    root=Path(__file__).resolve().parents[1];study=root.parents[1]
    assert sha(root/'private/roster.json')==json.loads((root/'protocol.json').read_text())['private_roster_sha256']
    rows=json.loads((root/'private/roster.json').read_text());leaf=root/'runs'/a.job/a.arm;leaf.mkdir(parents=True,exist_ok=False)
    LIVE['root']=root;LIVE['leaf']=leaf
    h=NativeHarness.__new__(NativeHarness);LIVE['harness']=h;h.__init__(root,a.arm)
    fixtures=[next(x for x in rows if x['benchmark']==task) for task in ['aime26','humaneval','longbench_v2']]
    for i,row in enumerate(fixtures):
        v=h.request(row,42,f'{a.job}/{a.arm}/warm{i:03d}',audit=False);save(leaf/f'warm{i:03d}.json',v)
    if a.job=='qualify':
        for i,row in enumerate(fixtures):
            records=[]
            for mode,audit in [('off',False),('observed',True),('repeat',True)]:
                v=h.request(row,42,f'{a.job}/{a.arm}/{mode}{i:03d}',audit=audit,GQA=mode=='observed');records.append(v);save(leaf/f'{mode}{i:03d}.json',v)
            keys=['output_sha256','N','C','T','final_cpu_rng','final_cuda_rng','model_forwards','prompt_tokenization_sha256','startup_model_calls_in_request','extra_inside_execute_Python_forwards']
            checks={k:len({json.dumps(v[k],sort_keys=True) for v in records})==1 for k in keys}
            assert records[1]['native_trace']==records[2]['native_trace']
            save(leaf/f'qualification{i:03d}.json',dict(passed=all(checks.values()),checks=checks,input_hashes={f'{mode}{i:03d}.json':sha(leaf/f'{mode}{i:03d}.json') for mode in ['off','observed','repeat']}))
            assert all(checks.values()),'observer qualification failed'
    else:
        assert json.loads((root/'qualification_complete.json').read_text())['passed']
        for i,(row,seed) in enumerate((r,s) for r in rows for s in [42,43]):
            v=h.request(row,seed,f'{a.job}/{a.arm}/audit{i:03d}',audit=True);save(leaf/f'audit{i:03d}.json',v)
    h.verify_sources();save(leaf/'final_complete.json',dict(passed=True,job=a.job,arm=a.arm,protocol_sha256=sha(root/'protocol.json'),**h.totals()))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        h=LIVE.get('harness');s=getattr(h,'state',{})
        if LIVE:save(LIVE['leaf']/'failure.json',dict(passed=False,error_class=type(e).__name__,reason='Worker failed; exact traceback retained only in the ignored job log.',engine_execute_forwards=s.get('executions',0),startup_model_calls_outside_execute=s.get('startup',0),extra_model_calls_inside_execute=s.get('extra_inside_execute',0)))
        raise
