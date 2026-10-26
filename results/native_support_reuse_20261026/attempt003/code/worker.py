"""Serial fresh complete native requests; qualification before treatment use."""
from pathlib import Path
import argparse,hashlib,json,sys
from reuse_runtime import ROOT,PROFILE,NativeHarness,ReuseStudy,save,sha
from qualification import exact_request,core_trace_equal

LIVE={}
def load(p):return json.loads(p.read_text())
def cid(row,seed):return hashlib.sha256(row['id'].encode()).hexdigest()[:12]+f'_seed{seed}'
def baseline(row,seed,arm):
    hits=[]
    for p in (PROFILE/'runs/profile'/arm).glob('audit*.json'):
        r=load(p)
        if (r['id'],r['seed'])==(row['id'],seed):hits.append((p,r))
    assert len(hits)==1;return hits[0]
def compare(a,b,trace=True):
    c=exact_request(a,b);assert all(c.values()),c
    if trace:assert core_trace_equal(a['native_trace'],b['native_trace']),'active trace mismatch'
    return c
def run(h,ext,row,seed,label,mode,config=None,audit=True,GQA=False):
    ext.mode=mode;ext.config=config or {};r=h.request(row,seed,label,audit=audit,GQA=GQA);save(ROOT/'runs'/(label+'.json'),r);return r
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',choices=['references','treatments','dense_full','piecewise'],required=True);args=ap.parse_args()
    rows=[r for r in load(ROOT.parent/'private/manifest.json') if r['partition']=='development'];leaf=ROOT/'runs'/args.job;leaf.mkdir(parents=True,exist_ok=False);LIVE['leaf']=leaf
    arm={'dense_full':'corrected_dense_full','piecewise':'dense_piecewise'}.get(args.job,'matched_allkept_l5_l17_l29')
    ext=ReuseStudy(ROOT);h=NativeHarness.__new__(NativeHarness);LIVE['h']=h;h.__init__(ROOT,arm,ext)
    fixtures=[next(r for r in rows if r['benchmark']==t) for t in ['aime26','humaneval','longbench_v2']]
    for i,row in enumerate(fixtures):run(h,ext,row,42,f'{args.job}/warm{i:03d}','warm',audit=False)
    if args.job in ['dense_full','piecewise']:
        for row in rows:
            for seed in [42,43]:
                label=f'{args.job}/{cid(row,seed)}';r=run(h,ext,row,seed,label,'control');p,b=baseline(row,seed,arm);c=compare(r,b);save(ROOT/'checks'/(label+'.json'),dict(passed=True,historical_source_receipt_sha256=sha(p),checks=c))
                if row in fixtures and seed==42:
                    off=run(h,ext,row,seed,label+'_off','control',audit=False);rep=run(h,ext,row,seed,label+'_repeat','control');compare(r,off,False);compare(r,rep)
    elif args.job=='references':
        for row in rows:
            for seed in [42,43]:
                p,b=baseline(row,seed,arm);anchor=b['anchors'].get('instability');label=f'references/{cid(row,seed)}'
                r=run(h,ext,row,seed,label,'reference',dict(anchor=anchor),GQA=row in fixtures and seed==42);c=compare(r,b)
                save(ROOT/'checks'/(label+'.json'),dict(passed=True,historical_source_receipt_sha256=sha(p),checks=c))
                if row in fixtures and seed==42:
                    cfg=dict(anchor=anchor,reference_before=r['pre_intervention'])
                    off=run(h,ext,row,seed,label+'_off','control',cfg,audit=False);rep=run(h,ext,row,seed,label+'_repeat','control',cfg);compare(r,off,False);compare(r,rep)
    else:
        manifest=load(ROOT/'branch_manifest.json');qualified=set()
        for cell in manifest['cells']:
            row=next(r for r in rows if r['id']==cell['id']);seed=cell['seed'];ref=load(ROOT/f'runs/references/{cid(row,seed)}.json');anchor=cell['anchor']
            if not anchor:continue
            for j,b in enumerate(cell['branches']):
                cfg=dict(b,anchor=anchor,reference_before=ref['pre_intervention'],reference_label=ref['label']);label=f'treatments/{cid(row,seed)}/b{j:03d}'
                r=run(h,ext,row,seed,label,'branch',cfg);r['branch_identity']=b['identity'];save(ROOT/'branch_records'/f'{b["identity"]}.json',r)
                qualification_key=(row['benchmark'],b['lifecycle'])
                if b['lifecycle'] in ['held4','refresh4'] and qualification_key not in qualified:
                    off=run(h,ext,row,seed,label+'_off','branch',cfg,audit=False);rep=run(h,ext,row,seed,label+'_repeat','branch',cfg);checks=[compare(r,off,False),compare(r,rep)]
                    support=lambda z:[e['actual_support_sha256'] for e in z['events'] if e.get('intervention')]
                    assert support(r)==support(off)==support(rep)
                    save(ROOT/'qualification'/f'{row["benchmark"]}_{b["lifecycle"]}.json',dict(passed=True,policy=b['identity'],checks=checks,actual_support_sequence_equal=True));qualified.add(qualification_key)
                if j==0:bracket(h,ext,row,seed,ref,'after_perturbation')
            bracket(h,ext,row,seed,ref,'after_group')
    h.verify_sources();save(leaf/'final_complete.json',dict(passed=True,job=args.job,protocol_sha256=sha(ROOT/'protocol.json'),**h.totals()))
def bracket(h,ext,row,seed,ref,kind):
    cfg=dict(anchor=ref.get('anchors',{}).get('instability'),reference_before=ref['pre_intervention'],reference_label=ref['label']);label=f'treatments/{cid(row,seed)}/{kind}';r=run(h,ext,row,seed,label,'control',cfg);c=compare(r,ref);save(ROOT/'checks'/(label+'.json'),dict(passed=True,checks=c))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        h=LIVE.get('h');s=getattr(h,'state',{})
        if LIVE.get('leaf'):save(LIVE['leaf']/'failure.json',dict(passed=False,error_class=type(e).__name__,reason='Private worker log preserves traceback',engine_execute_forwards=s.get('executions',0),startup_model_calls_outside_execute=s.get('startup',0),extra_model_calls_inside_execute=s.get('extra_inside_execute',0)))
        raise
