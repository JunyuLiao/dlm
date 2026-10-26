"""CPU-only choice/unique branch binding; never reads sparse outcomes."""
from pathlib import Path
from collections import defaultdict
import argparse,hashlib,json,sys
from pulse_runtime import ROOT,PROFILE,save,sha,site_key
from study_worker import cell_id
def load(p):return json.loads(p.read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['early_maps','branches'],required=True);arg=ap.parse_args();rows=[r for r in load(PROFILE/'private/manifest.json') if r['partition']=='development'];fixtures=[next(r for r in rows if r['benchmark']==task) for task in ['aime26','humaneval','longbench_v2']]
    assert load(ROOT/'runs/references/final_complete.json')['passed']
    cells=[];inputs={};missing=[]
    if arg.stage=='early_maps':
        for row in fixtures:
            refpath=ROOT/'runs/references'/(cell_id(row,42)+'.json');ref=load(refpath);inputs[str(refpath.relative_to(ROOT))]=sha(refpath);anchors=ref['anchors'];name='instability' if 'instability' in anchors else 'confidence' if 'confidence' in anchors else None
            if name is None:missing.append(dict(id=row['id'],reason='no_prespecified_anchor'));continue
            a=anchors[name];sk=site_key(a,29);paths=[ROOT/p for p in ref['sites'] if Path(p).stem==sk];assert len(paths)==1;site=load(paths[0]);maps=[]
            for b in site['banks']:
                if 'alias_of' in b:continue
                bysupport={}
                for family in ['current_error','support_sham','readout_margin']:
                    if family not in b['selections']:continue
                    rec=b['candidates'][b['selections'][family]];key=rec['support_sha256'];fractions=[z['fraction'] for z in site['banks'] if z.get('drop_count')==b['drop_count']];bysupport.setdefault(key,dict(units=rec['units'],drop_count=b['drop_count'],fraction=b['fraction'],fractions=fractions,families=[]))['families'].append(family)
                maps.extend(bysupport.values())
            cells.append(dict(id=row['id'],seed=42,anchors={name:a},maps=maps,operand_anchor=sk))
        save(ROOT/'early_maps.json',dict(cells=cells,missing=missing,input_hashes=inputs,late_layer_maps_frozen_before_early_operator_measurement=True));return
    assert load(ROOT/'runs/early/final_complete.json')['passed']
    for row in rows:
        for seed in [42,43]:
            refpath=ROOT/'runs/references'/(cell_id(row,seed)+'.json');ref=load(refpath);inputs[str(refpath.relative_to(ROOT))]=sha(refpath);branches=[];seen={}
            for path in ref['sites']:
                p=ROOT/path;inputs[path]=sha(p);site=load(p)
                for bank in site['banks']:
                    if 'alias_of' in bank:continue
                    for index in bank['screen']:
                        rec=bank['candidates'][index];families=[f for f,i in bank['selections'].items() if i==index]
                        add_branch(branches,seen,row,seed,ref,site,bank,rec,families,path,index)
            early=ROOT/'runs/early'/(cell_id(row,seed)+'.json')
            if early.exists():
                e=load(early);inputs[str(early.relative_to(ROOT))]=sha(early)
                for path in e['sites']:
                    p=ROOT/path;inputs[path]=sha(p);site=load(p)
                    for index,rec in enumerate(site['banks']):
                        if rec['feasible']:add_branch(branches,seen,row,seed,ref,site,rec,rec,rec['families'],path,index)
                        else:missing.append(dict(id=row['id'],seed=seed,site=path,index=index,reason='transferred_map_failed_local_guards'))
            # Prospective perturbative first branch, then fixed geometric order.
            perturb=[b for b in branches if b['layer']==29 and .10 in b['fractions']]
            first=min(perturb,key=lambda b:(-b['current_error'],b['support_sha256'],b['stratum'])) if perturb else None
            ordered=sorted(branches,key=lambda b:(b['canvas'],b['step'],b['layer'],b['drop_count'],b['support_sha256']))
            if first:ordered.remove(first);ordered.insert(0,first)
            cells.append(dict(id=row['id'],seed=seed,benchmark=row['benchmark'],stratum=row['stratum'],branches=ordered,reference_label=ref['label']))
    save(ROOT/'branch_manifest.json',dict(cells=cells,excluded=missing,input_hashes=inputs,frozen_before_sparse_outcomes=True,unique_branches=sum(len(c['branches']) for c in cells),max_primary=384,max_additional_layer=36,source_protocol_sha256=sha(ROOT/'protocol.json')))

def add_branch(branches,seen,row,seed,ref,site,bank,rec,families,path,index):
    g=site['geometry'];key=(g['canvas'],g['step'],g['layer'],bank['drop_count'],rec['support_sha256'])
    if key in seen:
        b=seen[key];b['families']=sorted(set(b['families']+families));b['fractions']=sorted(set(b['fractions']+[bank['fraction']]));return
    identity=hashlib.sha256(json.dumps([row['id'],seed,*key],separators=(',',':')).encode()).hexdigest()
    fractions=bank.get('fractions',[bank['fraction']])
    if g['layer']==29:
        for other in site['banks']:
            if other.get('drop_count')==bank['drop_count']:fractions.append(other['fraction'])
    b=dict(identity=identity,layer=g['layer'],canvas=g['canvas'],step=g['step'],stratum=site['strata'][0],strata=site['strata'],units=rec['units'],support_sha256=rec['support_sha256'],drop_count=bank['drop_count'],fractions=sorted(set(fractions)),families=families,current_error=rec['current_error'],deleted_mass=rec['deleted_mass'],site={'geometry':g},dense_sha256=bank['dense_sha256'],sparse_sha256=rec['sparse_sha256'],bank_relative=path,candidate_index=index)
    seen[key]=b;branches.append(b)

if __name__=='__main__':main()
