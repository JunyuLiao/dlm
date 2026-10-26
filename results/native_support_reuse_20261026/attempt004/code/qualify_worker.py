"""Full optional-observer-off versus observed parity for unchanged policies."""
from pathlib import Path
import json
from reuse_runtime import ROOT,NativeHarness,ReuseStudy,save,sha
from worker import cid,compare,load

LIVE={}
def main():
    plan=load(ROOT/'qualification_manifest.json');allrows=load(ROOT.parent/'private/manifest.json');leaf=ROOT/'runs/qualification';leaf.mkdir(parents=True,exist_ok=False);LIVE['leaf']=leaf
    ext=ReuseStudy(ROOT);h=NativeHarness.__new__(NativeHarness);LIVE['h']=h;h.__init__(ROOT,'matched_allkept_l5_l17_l29',ext)
    def run(row,seed,label,mode,cfg,audit):
        ext.mode=mode;ext.config=cfg;r=h.request(row,seed,label,audit=audit);save(ROOT/'runs'/(label+'.json'),r);return r
    for j,cell in enumerate(plan['cells']):
        row=next(r for r in allrows if r['id']==cell['id']);seed=42;ref=load(ROOT/f'runs/references/{cid(row,seed)}.json')
        run(row,seed,f'qualification/warm{j:03d}','warm',{},False)
        cfg=dict(anchor=cell['anchor'],reference_before=ref['pre_intervention'],reference_label=ref['label'])
        a=run(row,seed,f'qualification/{j:03d}_A','control',cfg,True);off=run(row,seed,f'qualification/{j:03d}_A_off','control',cfg,False);rep=run(row,seed,f'qualification/{j:03d}_A_repeat','control',cfg,True)
        compare(a,ref);compare(a,off,False);compare(a,rep)
        for b in cell['branches']:
            cfg=dict(b,anchor=cell['anchor'],reference_before=ref['pre_intervention'],reference_label=ref['label']);stem=f'qualification/{j:03d}_{b["lifecycle"]}'
            a=run(row,seed,stem,'branch',cfg,True);off=run(row,seed,stem+'_off','branch',cfg,False);rep=run(row,seed,stem+'_repeat','branch',cfg,True)
            compare(a,off,False);compare(a,rep)
            supports=lambda r:[e['actual_support_sha256'] for e in r['events'] if e.get('intervention')]
            assert supports(a)==supports(off)==supports(rep)
            original=load(ROOT.parent/f'attempt002/branch_records/{b["identity"]}.json');compare(a,original)
            save(ROOT/'qualification'/f'{j:03d}_{b["lifecycle"]}.json',dict(passed=True,optional_state_certificate_and_validation_replays_disabled_off=True,held_off_no_prior_logit_observer=b['lifecycle']=='held4',full_output_rng_counts_support_equal=True,observed002_compatible=True,branch_identity=b['identity']))
    save(leaf/'final_complete.json',dict(passed=True,protocol_sha256=sha(ROOT/'protocol.json'),**h.totals()))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        h=LIVE.get('h');s=getattr(h,'state',{})
        if LIVE.get('leaf'):save(LIVE['leaf']/'failure.json',dict(passed=False,error_class=type(e).__name__,engine_execute_forwards=s.get('executions',0),startup_model_calls_outside_execute=s.get('startup',0),reason='private log retains traceback'))
        raise
