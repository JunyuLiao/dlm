"""Frozen output-preserving V2 optimization and user-requested benchmark follow-up."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from scripts import optimized_refresh_campaign as base
from scripts import optimized_refresh_followup as previous

OLD=base.ROOT/'results/optimized_refresh_benchmarks_20261022/attempt009'
FOLLOW=base.ROOT/'results/optimized_refresh_followup_20261023/attempt003'
LIBRARY=base.ROOT/'results/v2_execution_20261023/attempt007/build/380b43b3784cdce8/v2_online_sm90.so'
PANELS={'aime26_k1920':('aime26',1920,['v31_optimized'],OLD),
        'math500_k384':('math500',384,['v31_optimized'],FOLLOW/'math500_lower'),
        'ruler128k_s45':('ruler128k',8192,['v2_online_optimized'],FOLLOW/'ruler128k'),
        'longbench_v2':('longbench_v2',8192,['v31_optimized','v2_online_optimized'],OLD)}
OVERLAYS=['experiments/numerical_qk_reuse/optimized_refresh_runtime.py','experiments/numerical_qk_reuse/resident_execution_runtime.py',
          'experiments/numerical_qk_reuse/v2_online_warpspec.py','experiments/numerical_qk_reuse/v2_paged_refresh.py',
          'scripts/v2_execution_followup.py']

def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    junit=base.ROOT/'results/v2_execution_20261023/attempt009/regression/junit.xml'
    import xml.etree.ElementTree as ET
    tree=ET.parse(junit).getroot()
    if len(list(tree.iter('testcase')))!=35 or any(x.tag in ['failure','error','skipped'] for x in tree.iter()):raise RuntimeError('operator gate')
    for panel,(suite,k,new_arms,source) in PANELS.items():
        study=root/panel;old=base.verify(source);ex=study/'private/execution'
        shutil.copytree(source/'private/execution',ex,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for rel in OVERLAYS:shutil.copyfile(base.ROOT/rel,ex/rel)
        settings=copy.deepcopy(old['suites'][suite]);settings['mage_k']=k
        pool=study/f'private/pools/{suite}';pool.parent.mkdir(parents=True);pool.symlink_to((source/f'private/pools/{suite}').resolve(),target_is_directory=True)
        schedules=study/f'private/schedules/{suite}';shutil.copytree(source/f'private/schedules/{suite}',schedules)
        for p in schedules.glob('shard???.json'):p.unlink()
        cells=base.read(schedules/'cells.json')
        # Smaller long-context shards bound recovery while amortizing engine load.
        shard_size=64 if suite=='longbench_v2' else 128
        for start in range(0,len(cells),shard_size):base.write(schedules/f'shard{start//shard_size+1:03d}.json',cells[start:start+shard_size])
        if suite=='ruler128k':
            shutil.copyfile(source/'private/scorer_location.json',study/'private/scorer_location.json')
        if suite in ['ruler128k','longbench_v2']:
            ordered=sorted(cells,key=lambda x:(len(x['prompt_tokens']) if isinstance(x['prompt_tokens'],list) else x['prompt_tokens'],x['index']))
            base.write(schedules/'smoke.json',[ordered[0],ordered[len(ordered)//2],ordered[-1]])
        kernel=study/'private/kernel';shutil.copytree(LIBRARY.parent,kernel,ignore=shutil.ignore_patterns('tmp'))
        shutil.copyfile(OLD/'private/kernel/v2_online_sm90.so',kernel/'previous_v2_online_sm90.so')
        (study/'private/cache').mkdir()
        shutil.copyfile(source/'model_source_fingerprints.json',study/'model_source_fingerprints.json')
        arms=['v2_online_optimized','v31_optimized'] if suite=='longbench_v2' else ['v2_online_optimized','v31_optimized','dense_full_fix51994','dense_piecewise','allkept_fa4']
        references={};reference_hashes={}
        for arm in arms:
            if arm in new_arms:continue
            found=list((source/f'runs/{suite}').glob(arm+'*'))
            if len(found)!=1:raise RuntimeError('reference arm ambiguity')
            target=study/f'runs/{suite}/{found[0].name}';target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(found[0].resolve(),target_is_directory=True)
            references[arm]=str(found[0].relative_to(base.ROOT))
            for f in found[0].rglob('*'):
                if f.is_file():reference_hashes[str(f.relative_to(base.ROOT))]=base.sha(f)
        spec=(base.ROOT/'docs/V2_EXECUTION_FOLLOWUP_20261023.md').read_text()+f'\nPanel {panel}: suite={suite}, V31 K{k}, new arms={new_arms}. Historical references retain their original execution/source identities; their clean timings are not fresh matched execution controls.\n'
        (study/'spec.md').write_text(spec)
        protocol=dict(old,created_utc=base.utc(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            suites={suite:settings},arms=arms,new_arms=new_arms,reference_arms=references,reference_sha256=reference_hashes,
            source_identity_note='base commit plus exact frozen execution hashes; changed source frozen separately from historical references',
            execution_sha256={str(p.relative_to(ex)):base.sha(p) for p in ex.rglob('*') if p.is_file()},
            input_sha256={str(p.relative_to(study)):base.sha(p) for folder in [pool,schedules] for p in folder.rglob('*') if p.is_file()},
            kernel_sha256=base.sha(kernel/'v2_online_sm90.so'),previous_kernel_sha256=base.sha(kernel/'previous_v2_online_sm90.so'),
            spec_sha256=base.sha(study/'spec.md'),operator_gate_sha256=base.sha(junit),purpose=panel,
            fixed_v2_threshold=.021 if suite=='longbench_v2' else .059460355750136064 if suite=='aime26' else .2 if suite=='math500' else None,
            execution_options={'V2_CANVAS_PROJECTION_REUSE':'1','V2_PAGED_REFRESH_LIBRARY':None},
            threshold_policy=dict(objective='40-50% all-step GLOBAL physical sparsity; report phases separately',target=[.4,.5],
                candidates=[.00525,.002625,.0013125,.00065625,.000328125,.0001640625,.0105,.021,0.],refinements=12,accuracy_used=False),
            user_authorization='2026-10-23 optimize execution without algorithm/output changes; AIME K1920 MATH K384 RULER all-step calibration; LongBench V31/V2 only')
        # No inherited qualification/calibration-reuse claims across changed code.
        for key in ['qualification_reuse','calibration_reuse','relaunch_source','relaunch_protocol_sha256']:protocol.pop(key,None)
        base.write(study/'protocol.json',protocol);base.write(study/'state.json',dict(status='prepared',utc=base.utc()))
    base.write(root/'protocol.json',dict(created_utc=base.utc(),panels={p:base.sha(root/p/'protocol.json') for p in PANELS},order=list(PANELS)))
    print(root,flush=True)

def configure(study):
    p=base.verify(study);base.SUITES.clear();base.SUITES.update(p['suites']);base.ARMS[:]=p['arms']
    for rel,digest in p['reference_sha256'].items():
        if base.sha(base.ROOT/rel)!=digest:raise RuntimeError('reference drift '+rel)
    original=base.environment
    def environment(*args,**kwargs):
        env=original(*args,**kwargs);env.pop('V2_PAGED_REFRESH_LIBRARY',None)
        env['V2_CANVAS_PROJECTION_REUSE']='1'
        if 'qualification_previous_execution' in str(args[-1]):
            env['V2_CANVAS_PROJECTION_REUSE']='0'
            env['OPTIMIZED_REFRESH_LIBRARY']=str(study/'private/kernel/previous_v2_online_sm90.so')
        return env
    base.environment=environment
    return p

def trajectory(a,b):
    ar=base.read_lines(a/'records.jsonl');br=base.read_lines(b/'records.jsonl')
    return [(x['output_hash'],x['denoise_forwards'],x.get('native_interval_steps')) for x in ar]==[(x['output_hash'],x['denoise_forwards'],x.get('native_interval_steps')) for x in br]

def qualify(study,suite,tau,stage='qualification'):
    dest=study/stage/'gates.json'
    if dest.exists():return
    protocol=base.read(study/'protocol.json');cells=study/f'private/schedules/{suite}/smoke.json';rows=[]
    for arm in protocol['new_arms']:
        threshold=tau if arm.startswith('v2') else None
        audit=base.run_job(study,suite,arm,'audit',cells,stage,threshold)
        clean=base.run_job(study,suite,arm,'clean',cells,stage,threshold)
        if not trajectory(audit,clean):raise RuntimeError('audit/clean trajectory mismatch')
        if arm.startswith('v2'):
            for row in base.read_lines(audit/'records.jsonl'):
                receipt=row['receipts']['adapter']
                if not receipt.get('optimized_refresh_prefix_projection_reuse'):raise RuntimeError('prefix reuse not exercised')
            previous_execution=base.run_job(study,suite,arm,'audit',cells,'qualification_previous_execution_'+stage,threshold)
            if not trajectory(audit,previous_execution):raise RuntimeError('execution optimization changes native output or trajectory')
        rows.append(dict(arm=arm,audit=str(audit.relative_to(study)),clean=str(clean.relative_to(study)),trajectory_match=True))
    base.write(dest,dict(status='pass',protocol_sha256=base.sha(study/'protocol.json'),rows=rows))

def calibrate(study,suite):
    dest=study/f'calibration_held/{suite}/selected_threshold.json';p=base.read(study/'protocol.json')
    if dest.exists():return base.read(dest)['threshold']
    if suite!='ruler128k':
        tau=p['fixed_v2_threshold'];base.write(dest,dict(status='fixed_threshold_by_user',threshold=tau,provenance={'source':'user fixed LongBench0.021 or preserved short-benchmark V2'},new_calibration=False));return tau
    cells=study/f'private/schedules/{suite}/calibration.json';points={};history=study/'calibration_points.json'
    if history.exists():points=base.read(history)
    def measure(t):
        key=str(float(t))
        if key not in points:
            leaf=base.run_job(study,suite,'v2_online_optimized','audit',cells,'ruler_allstep_calibration',t)
            points[key]=previous.physical(leaf);base.write(history,points)
    def acceptable(k):return .4<=points[k]['global_sparsity']<=.5
    for t in p['threshold_policy']['candidates']:
        if any(acceptable(k) for k in points):break
        measure(t)
    for _ in range(p['threshold_policy']['refinements']):
        if any(acceptable(k) for k in points):break
        pairs=[(float(x),float(y)) for x in points for y in points if float(x)<float(y) and points[x]['global_sparsity']<.45<=points[y]['global_sparsity']]
        if not pairs:break
        lo,hi=min(pairs,key=lambda z:z[1]-z[0]);measure(math.sqrt(lo*hi) if lo else hi/2)
    winner=min(points,key=lambda k:(not acceptable(k),abs(points[k]['global_sparsity']-.45)))
    result=dict(status='allstep_physical_calibration',threshold=float(winner),in_target_range=acceptable(winner),global_sparsity=points[winner]['global_sparsity'],
        held_sparsity=points[winner]['held_sparsity'],held_eligible_tiles=points[winner]['held_eligible_tiles'],held_kept_tiles=points[winner]['held_kept_tiles'],
        calibration_cells=len(base.read(cells)),transductive=True,accuracy_used=False,points=points,protocol_sha256=base.sha(study/'protocol.json'))
    base.write(dest,result)
    if not acceptable(winner):raise RuntimeError('RULER all-step calibration target missed; preserve attempt')
    return float(winner)

def run_panel(study):
    p=configure(study);suite=next(iter(p['suites']));initial=p['fixed_v2_threshold'] if suite!='ruler128k' else .00525
    qualify(study,suite,initial)
    tau=calibrate(study,suite)
    if tau!=initial:qualify(study,suite,tau,'qualification_selected_threshold')
    for cells in sorted((study/f'private/schedules/{suite}').glob('shard???.json')):
        for arm in p['new_arms']:
            for pass_name in ['audit','clean']:base.run_job(study,suite,arm,pass_name,cells,'runs',tau if arm.startswith('v2') else None)
    base.score_suite(study,suite)
    base.write(study/'final_complete.json',dict(status='complete',utc=base.utc(),protocol_sha256=base.sha(study/'protocol.json'),suite=suite,threshold=tau,reference_arms=p['reference_arms']))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','panel']);parser.add_argument('--study',type=Path,required=True);a=parser.parse_args();root=a.study.resolve()
    if a.action=='prepare':prepare(root);return
    if a.action=='panel':run_panel(root);return
    for panel in PANELS:
        study=root/panel
        if (study/'final_complete.json').exists():continue
        # Each suite runs in a fresh coordinator so mutable imported suite/arm
        # registries and environment wrappers cannot leak between panels.
        command=[base.PYTHON,str(study/'private/execution/scripts/v2_execution_followup.py'),'panel','--study',str(study)]
        with (study/'coordinator.log').open('a') as log:
            result=subprocess.run(command,cwd=study/'private/execution',env=dict(os.environ,PYTHONPATH=str(study/'private/execution')+'/src:'+str(study/'private/execution')),stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('panel failed '+panel)
    base.write(root/'final_complete.json',dict(status='complete',utc=base.utc(),panels=list(PANELS),protocol_sha256=base.sha(root/'protocol.json')))
if __name__=='__main__':main()
