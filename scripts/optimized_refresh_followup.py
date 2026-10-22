"""Source-frozen MATH recalibration, AIME K256 and RULER128K follow-up.

Original HumanEval/AIME panels and the paused LongBench campaign stay intact.
Reference generations are reused only when worker, input and runtime identities
match; every reference path and file fingerprint is explicit in the protocol.
"""
from pathlib import Path
import argparse
import collections
import json
import math
import shutil
import subprocess

from scripts import optimized_refresh_campaign as base

ROOT=base.ROOT
PARENT=ROOT/'results/optimized_refresh_benchmarks_20261022/attempt009'
RULER_PARENT=ROOT/'results/v31_v2_ruler128k_seed42_tau021_20261015'
ORCHESTRATION={'scripts/optimized_refresh_campaign.py','scripts/optimized_refresh_report.py',
               'scripts/optimized_refresh_followup.py','scripts/optimized_refresh_native_campaign.py'}
PANELS=['math500_lower','aime26_k256','ruler128k']


def configure(study):
    protocol=base.verify(study)
    base.SUITES.clear();base.SUITES.update(protocol['suites'])
    return protocol


def evidence(study,source):
    """Reject changed model/worker/runtime sources before any reference reuse."""
    new=base.read(study/'protocol.json');old=base.verify(source)
    filtered=lambda p:{k:v for k,v in p['execution_sha256'].items() if k not in ORCHESTRATION}
    if filtered(new)!=filtered(old) or any(new[k]!=old[k] for k in ['runtime','kernel_sha256','model_inventory_sha256']):
        raise RuntimeError('follow-up reference execution identity mismatch')
    for rel,digest in new['reference_sha256'].items():
        if base.sha(ROOT/rel)!=digest:raise RuntimeError('follow-up reference evidence drift: '+rel)


def add_reference(protocol,path):
    for p in path.rglob('*'):
        if p.is_file():protocol['reference_sha256'][str(p.relative_to(ROOT))]=base.sha(p)


def prepare(root):
    # The old coordinator must finish and score MATH, then honor the pause.
    if not (PARENT/'PAUSED_USER_REQUEST.json').exists() or not (PARENT/'analysis/math500/attempt001/analysis_complete.json').exists():
        raise RuntimeError('finish and pause the current MATH500 campaign first')
    if (PARENT/'runs/longbench_v2').exists():raise RuntimeError('unexpected LongBench generation before stop')
    old=base.verify(PARENT)
    root.mkdir(parents=True,exist_ok=False)
    for panel in PANELS:
        suite='math500' if panel=='math500_lower' else 'aime26' if panel=='aime26_k256' else 'ruler128k'
        study=root/panel;ex=study/'private/execution'
        shutil.copytree(PARENT/'private/execution',ex,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for rel in ['scripts/optimized_refresh_campaign.py','scripts/optimized_refresh_report.py','scripts/optimized_refresh_followup.py']:
            shutil.copyfile(ROOT/rel,ex/rel)
        shutil.copytree(PARENT/'private/kernel',study/'private/kernel')
        shutil.copyfile(PARENT/'model_source_fingerprints.json',study/'model_source_fingerprints.json')
        (study/'private/cache').mkdir()
        settings=dict(old['suites'][suite]) if suite!='ruler128k' else dict(
            dataset='ruler128k_v33ofc_subset',count=104,seeds=[42],mage_k=8192,
            max_len=136192,block=32,chunk=16384,memory=.85)
        if panel=='aime26_k256':settings['mage_k']=256
        source=PARENT if suite!='ruler128k' else RULER_PARENT
        pool=study/f'private/pools/{suite}';pool.parent.mkdir(parents=True)
        pool.symlink_to((source/f'private/pools/{suite}').resolve(),target_is_directory=True)
        schedules=study/f'private/schedules/{suite}'
        shutil.copytree(source/f'private/schedules/{suite}',schedules)
        for p in schedules.glob('shard???.json'):p.unlink()
        cells=base.read(schedules/'cells.json')
        for i in range(0,len(cells),128):base.write(schedules/f'shard{i//128+1:03d}.json',cells[i:i+128])
        if suite=='math500':
            # Four short-prompt strata severely underrepresented long reasoning.
            # Use 32 equal-count strata of observed full-request physical work.
            records=[]
            for p in (PARENT/'runs/math500').glob('v2*/audit/shard???/attempt???/records.jsonl'):
                if (p.parent/'attempt_complete.json').exists():records+=base.read_lines(p)
            assert len(records)==500 and len({x['index'] for x in records})==500
            ordered=sorted(records,key=lambda x:(x['receipts']['adapter']['global_eligible_tiles'],x['index']))
            chosen={ordered[(2*j+1)*len(ordered)//64]['index'] for j in range(32)}
            base.write(schedules/'calibration.json',[x for x in cells if x['index'] in chosen])
            base.write(schedules/'smoke.json',[x for x in cells if x['index'] in chosen][:1])
        elif suite=='ruler128k':
            info=base.read(pool/(settings['dataset']+'_rowinfo.json'))['rows']
            assert collections.Counter(x['task'] for x in info)=={t:8 for t in __import__('scripts.v31_official',fromlist=['RULER_TASKS']).RULER_TASKS}
            task_first={}
            for x in cells:task_first.setdefault(info[x['index']]['task'],x)
            base.write(schedules/'calibration.json',list(task_first.values()))
            base.write(schedules/'smoke.json',cells[:1])
        protocol=dict(old,created_utc=base.utc(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            suites={suite:settings},arms=base.ARMS,purpose=panel,reference_sha256={},reference_arms={},
            native=dict(old['native'],thinking=suite!='ruler128k'),parent_protocol_sha256=base.sha(PARENT/'protocol.json'),
            threshold_policy=dict(objective='held and whole-request GLOBAL physical tile sparsity40-50%' if suite=='math500' else 'match V31K8192 first-canvas GLOBAL absolute kept work within5% and kept fraction within2pp' if suite=='ruler128k' else 'retain prior V2; change only V31 budget toK256',
                candidates=[.05,.1,.2,.4,.8,.025,.0125,.00625,0.] if suite=='math500' else [.021,.042,.084,.168,.336,.672,1.344,2.688],refinements=10,accuracy_used=False),
            qualification='fresh audit/clean for new arms and changed workload; exact source-matched reference evidence otherwise')
        if suite!='ruler128k':
            for arm in base.ARMS:
                if arm==('v2_online_optimized' if suite=='math500' else 'v31_optimized'):continue
                sources=sorted((PARENT/f'runs/{suite}').glob(arm+'*'))
                assert len(sources)==1
                dst=study/f'runs/{suite}/{sources[0].name}';dst.parent.mkdir(parents=True,exist_ok=True)
                dst.symlink_to(sources[0].resolve(),target_is_directory=True)
                protocol['reference_arms'][arm]=str(sources[0].relative_to(ROOT));add_reference(protocol,sources[0])
            if suite=='aime26':protocol['fixed_v2_threshold']=.059460355750136064
        else:
            from scripts import v31_official as op
            loc=base.read(RULER_PARENT/'private/scorer_location.json');base.write(study/'private/scorer_location.json',loc)
            protocol['ruler_scorer']=op.ruler_verify_checkout(Path(loc['ruler']))
            protocol['ruler_subset']=dict(problems=104,tasks=13,per_task=8,panel_seed=42,subset_seed=6464,previously_examined=True,thinking=False,prompt_prefill='original official RULER answer prefix',generation_budgets_unchanged=True)
            add_reference(protocol,RULER_PARENT/'private/pools/ruler128k')
        text=f'''# Optimized refresh follow-up: {panel}

User requests retaining achieved HumanEval/AIME results and completing old MATH
before stopping ahead of LongBench. This is a new frozen attempt. Runtime/model,
selected attention kernels and native sampler are unchanged. V31 budget is
{settings['mage_k']}; scope is GLOBAL only and LOCAL/prefill stay dense.

MATH recalibration uses32 full-request physical-work strata of the examined500
cohort, lowers V2 threshold and targets actual held/whole GLOBAL sparsity40-50%.
This is transductive computation-only calibration. AIMEK256 changes only V31;
existing V2/dense/all-kept controls are explicitly referenced, never overwritten.
RULER uses the same104 sampled prompts (8 per13 tasks), seed42, original
thinking-off/prefilled prompts and budgets. Calibrate V2 to approximately matched
K8192 first-canvas physical kept work; disclose any remaining mismatch and actual
sparsity (expected above85% on sparse phases, not assumed for whole request).

Every new arm/profile needs fresh audit/clean trajectory agreement, exact receipt
and zero captures/JIT/extra forwards/LOCAL skips before full coverage. Warm every
input before clean timing. Charge all copies/index construction/launches. Score
officially; use paired problem-clustered intervals and13-task clusters for RULER.
Keep prior full panels and all failed/calibration attempts immutable and private
raw inputs/generations out of Git. Process-status checks are every15minutes.
'''
        (study/'spec.md').write_text(text)
        protocol.update(spec_sha256=base.sha(study/'spec.md'),execution_sha256={str(p.relative_to(ex)):base.sha(p) for p in ex.rglob('*') if p.is_file()},input_sha256={str(p.relative_to(study)):base.sha(p) for folder in [pool,schedules] for p in folder.rglob('*') if p.is_file()})
        base.write(study/'protocol.json',protocol);base.write(study/'state.json',dict(status='prepared',utc=base.utc()))
        evidence(study,PARENT)
    base.write(root/'protocol.json',dict(created_utc=base.utc(),panels={p:base.sha(root/p/'protocol.json') for p in PANELS},order=PANELS))
    print(root,flush=True)


def physical(leaf):
    result=dict(global_eligible_tiles=0,global_kept_tiles=0,held_eligible_tiles=0,held_kept_tiles=0,first_eligible_tiles=0,first_kept_tiles=0)
    for row in base.read_lines(leaf/'records.jsonl'):
        a=row['receipts']['adapter']
        for key in ['global_eligible_tiles','global_kept_tiles']:result[key]+=a[key]
        held=a['value_phase_tiles'].get('held',dict(eligible_tiles=0,kept_tiles=0))
        result['held_eligible_tiles']+=held['eligible_tiles'];result['held_kept_tiles']+=held['kept_tiles']
        phases=a['value_canvas_phase_tiles'];first=phases[min(phases,key=int)]
        result['first_eligible_tiles']+=sum(p['eligible_tiles'] for p in first.values())
        result['first_kept_tiles']+=sum(p['kept_tiles'] for p in first.values())
    result['global_sparsity']=1-result['global_kept_tiles']/result['global_eligible_tiles']
    if not result['held_eligible_tiles']:raise RuntimeError('no held physical work observed in calibration cohort')
    result['held_sparsity']=1-result['held_kept_tiles']/result['held_eligible_tiles']
    return result


def calibrate(study,suite):
    dest=study/f'calibration_held/{suite}/selected_threshold.json'
    if dest.exists():return base.read(dest)['threshold']
    protocol=base.read(study/'protocol.json');cells=study/f'private/schedules/{suite}/calibration.json'
    if suite=='aime26':
        result=dict(status='fixed_threshold_by_user',threshold=protocol['fixed_v2_threshold'],provenance={'source':'preserved achieved AIME26 panel'},calibration_cells=0)
        base.write(dest,result);return result['threshold']
    points={};reference=None
    if suite=='ruler128k':reference=physical(base.run_job(study,suite,'v31_optimized','audit',cells,'followup_calibration_control'))
    def measure(tau):
        key=str(float(tau))
        if key not in points:points[key]=physical(base.run_job(study,suite,'v2_online_optimized','audit',cells,'followup_calibration_candidates',float(tau)))
    def error(key):
        row=points[key]
        if reference:return abs(row['first_kept_tiles']/reference['first_kept_tiles']-1)
        return max(abs(row['held_sparsity']-.45),abs(row['global_sparsity']-.45))
    def acceptable(key):
        row=points[key]
        if reference:return error(key)<=.05 and abs(row['first_kept_tiles']/row['first_eligible_tiles']-reference['first_kept_tiles']/reference['first_eligible_tiles'])<=.02
        return .4<=row['held_sparsity']<=.5 and .4<=row['global_sparsity']<=.5
    for tau in protocol['threshold_policy']['candidates']:
        measure(tau)
        if any(acceptable(k) for k in points):break
    for _ in range(protocol['threshold_policy']['refinements']):
        if any(acceptable(k) for k in points):break
        order=sorted(map(float,points))
        center=lambda t:(1-points[str(t)]['first_kept_tiles']/reference['first_kept_tiles']) if reference else (points[str(t)]['global_sparsity']+points[str(t)]['held_sparsity'])/2-.45
        neighbors=[(lo,hi) for lo,hi in zip(order,order[1:]) if center(lo)<=0<=center(hi)]
        if not neighbors:break
        lo,hi=min(neighbors,key=lambda p:p[1]-p[0]);measure(math.sqrt(lo*hi) if lo else hi/2)
    winner=min(points,key=lambda k:(not acceptable(k),error(k),float(k)))
    result=dict(status='held_physical_tile_calibration',threshold=float(winner),held_sparsity=points[winner]['held_sparsity'],held_eligible_tiles=points[winner]['held_eligible_tiles'],held_kept_tiles=points[winner]['held_kept_tiles'],calibration_cells=len(base.read(cells)),in_target_range=acceptable(winner),points=points,reference=reference,accuracy_used=False,transductive=True,protocol_sha256=base.sha(study/'protocol.json'),objective=protocol['threshold_policy']['objective'])
    base.write(dest,result)
    if not result['in_target_range']:raise RuntimeError('follow-up calibration failed; preserve points and freeze refinement')
    return result['threshold']


def qualify(study,suite,threshold):
    from scripts import optimized_refresh_native_campaign as gate
    dest=study/'qualification/followup_gates.json'
    if dest.exists():return
    protocol=base.read(study/'protocol.json')
    arms=[a for a in base.ARMS if a not in protocol['reference_arms']]
    cells=study/f'private/schedules/{suite}/smoke.json';results=[]
    for arm in arms:
        tau=threshold if arm.startswith('v2') else None
        audit=base.run_job(study,suite,arm,'audit',cells,'qualification',tau)
        clean=base.run_job(study,suite,arm,'clean',cells,'qualification',tau)
        ar=base.read_lines(audit/'records.jsonl');cr=base.read_lines(clean/'records.jsonl')
        if [(x['output_hash'],x['denoise_forwards']) for x in ar]!=[(x['output_hash'],x['denoise_forwards']) for x in cr]:raise RuntimeError('follow-up audit/clean trajectory mismatch')
        for row in ar+cr:
            a=row['receipts']['adapter'] or {}
            if arm.startswith(('v2','v31')) and a['optimized_refresh_extra_model_forwards']!=0:raise RuntimeError('extra forward')
        results.append(dict(arm=arm,audit=str(audit.relative_to(study)),clean=str(clean.relative_to(study)),trajectory_match=True))
    base.write(dest,dict(status='pass',protocol_sha256=base.sha(study/'protocol.json'),rows=results,reference_arms=protocol['reference_arms']))


def run_panel(study):
    p=configure(study);evidence(study,PARENT);suite=next(iter(p['suites']))
    initial=p.get('fixed_v2_threshold',.05 if suite=='math500' else .168)
    qualify(study,suite,initial)
    threshold=calibrate(study,suite)
    # Final threshold receives its own paired native qualification.
    if threshold!=initial:
        cells=study/f'private/schedules/{suite}/smoke.json'
        a=base.run_job(study,suite,'v2_online_optimized','audit',cells,'qualification_final_threshold',threshold)
        c=base.run_job(study,suite,'v2_online_optimized','clean',cells,'qualification_final_threshold',threshold)
        if [(x['output_hash'],x['denoise_forwards']) for x in base.read_lines(a/'records.jsonl')]!=[(x['output_hash'],x['denoise_forwards']) for x in base.read_lines(c/'records.jsonl')]:raise RuntimeError('final-threshold trajectory mismatch')
    for cells in sorted((study/f'private/schedules/{suite}').glob('shard???.json')):
        for arm in base.ARMS:
            if arm in p['reference_arms']:continue
            for pass_name in ['audit','clean']:base.run_job(study,suite,arm,pass_name,cells,'runs',threshold if arm.startswith('v2') else None)
    base.score_suite(study,suite)
    if suite=='aime26':
        import numpy as np
        from scripts.v31_official import cell_key
        out=study/f'analysis/{suite}/attempt001'
        current=base.read(out/'official.json')['v31_optimized']
        previous=base.read(PARENT/f'analysis/{suite}/attempt001/official.json')['v31_optimized']
        if current.keys()!=previous.keys():raise RuntimeError('AIME K128/K256 coverage mismatch')
        groups=collections.defaultdict(list)
        for key in current:groups[key.split('|')[1]].append(key)
        delta=np.asarray([sum(float(current[k])-float(previous[k]) for k in keys)/len(keys) for keys in groups.values()])
        rng=np.random.default_rng(20261022);boot=rng.integers(0,len(delta),size=(10000,len(delta)))
        clean=lambda root:[x for path in (root/f'runs/{suite}/v31_optimized/clean').glob('shard???/attempt???/records.jsonl') if (path.parent/'attempt_complete.json').exists() for x in base.read_lines(path)]
        before={cell_key(x):x for x in clean(PARENT)};after={cell_key(x):x for x in clean(study)}
        paired=dict(cells=len(current),problem_clusters=len(groups),k128_correct=sum(previous.values()),k256_correct=sum(current.values()),accuracy_difference_pp=float(delta.mean()*100),accuracy_difference_ci95_pp=list(map(float,np.percentile(delta[boot].mean(1)*100,[2.5,97.5]))),reference='preserved source-matched AIME V31K128 panel')
        for field in ['wall_s','decode_s','denoise_forwards']:
            values=np.asarray([sum(math.log(after[k][field]/before[k][field]) for k in keys)/len(keys) for keys in groups.values()])
            paired[field]=dict(geometric_ratio=float(np.exp(values.mean())),ci95=list(map(float,np.exp(np.percentile(values[boot].mean(1),[2.5,97.5])))))
        base.write(out/'paired_k256_vs_k128.json',paired)
    base.write(study/'final_complete.json',dict(status='complete',suite=suite,protocol_sha256=base.sha(study/'protocol.json'),threshold=threshold,reference_arms=p['reference_arms'],utc=base.utc()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);parser.add_argument('--study',type=Path,required=True);args=parser.parse_args();root=args.study.resolve()
    if args.action=='prepare':prepare(root);return
    for panel in PANELS:
        study=root/panel
        if (study/'final_complete.json').exists():continue
        try:run_panel(study)
        except Exception as error:
            old=base.read(study/'state.json') if (study/'state.json').exists() else {}
            base.write(study/'state.json',dict(old,status='failed',error=str(error),utc=base.utc()));raise
    base.write(root/'final_complete.json',dict(status='complete',panels=PANELS,protocol_sha256=base.sha(root/'protocol.json'),utc=base.utc()))


if __name__=='__main__':main()
