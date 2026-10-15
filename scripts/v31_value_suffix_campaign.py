"""Source-frozen same-state suffix qualification, with resumable attempts.

One H100, no clean timing and no automatic native accuracy bypass. This stage
uses the real native sampler only to supply states; extra suffix calls are
offline interventions. A disabled/active pair must reproduce native outputs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from scripts import v31_value_decision_replay as io

PY='/home/exouser/dyh/v31_env_20261005/bin/python'
DEFAULT=io.DEFAULT/'suffix_qualification'
BENCH_SOURCE=('scripts/v31_same_state_suffix_worker.py', 'experiments/numerical_qk_reuse/v31_same_state_suffix.py')


def prepare(study, suite='longbench_v2', panel='qualification'):
    if (study/'protocol.json').exists():
        verify(study)
        return
    from scripts.v31_v2_local_threshold_sweep import SUITES
    settings=dict(SUITES.get(suite,dict(dataset='ruler128k_v33ofc_subset',max_len=136192,
        memory=.85,block=32,chunk=16384,mage_k=8192)))
    source=io.ROOT/'results/v31_cached_refresh_stable_20261013'
    cells=io.read(source/'private/cells_trajectory4.json')
    selected=[c for c in cells if c['dataset'].startswith('longbench')]
    # Choose the shorter of the two previously captured long-context items
    # using prompt length alone, before any new model decision measurement.
    selected=sorted(selected,key=lambda c: c['prompt_tokens'] if isinstance(c['prompt_tokens'],int) else len(c['prompt_tokens']))[:1]
    if len(selected)!=1:
        raise RuntimeError('frozen input panel unavailable')
    pool=source/'private/manifests'
    if panel=='decisions':
        qualification=study.parent/'suffix_qualification_attempt006'
        marker=io.read(qualification/'qualification_complete.json')
        if marker['reproducibility_sha256']!=io.sha(qualification/'reproducibility.json'):
            raise RuntimeError('parent diagnostic qualification changed')
        if io.read(qualification/'reproducibility.json')['status']!='passed':
            raise RuntimeError('instrumentation must qualify before a decision panel')
        source=io.ROOT/'results/v31_v2_local_threshold_sweep_20261014_attempt002'
        if suite=='ruler128k':source=io.ROOT/'results/v31_v2_ruler128k_seed42_tau021_20261015'
        pool=source/f'private/pools/{suite}'
        cells=io.read(source/f'private/schedules/{suite}/cells.json')
        if suite=='ruler128k':
            info=io.read(pool/(settings['dataset']+'_rowinfo.json'))['rows']
            first={}
            for row in sorted(info,key=lambda r:r['index']):first.setdefault(row['task'],row['index'])
            selected=[c for c in cells if c['index'] in set(first.values())]
            assert len(selected)==13
        else:
            count={'longbench_v2':8,'aime26':6,'math500':8}[suite]
            seed=min(c['seed'] for c in cells)
            selected=[c for c in cells if c['seed']==seed][:count]
            assert len(selected)==count
    elif suite!='longbench_v2':raise ValueError('initial instrumentation qualification uses LongBench')
    execution=study/'private/execution'; execution.mkdir(parents=True)
    files=subprocess.check_output(['git','ls-files'],cwd=io.ROOT,text=True).splitlines()
    files=sorted(set([p for p in files if p.endswith(('.py','.json','.yaml','.yml','.toml'))
        and p.split('/')[0] in ('scripts','src','experiments','tests')] + list(BENCH_SOURCE)))
    fingerprints={}
    for rel in files:
        src=io.ROOT/rel
        if not src.is_file():
            continue
        dst=execution/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dst)
        fingerprints[rel]=io.sha(dst)
    manifests=study/'private/manifests'; manifests.mkdir()
    dataset=selected[0]['dataset']
    # Local private copy; never publish tokens or raw generations.
    for f in pool.glob(dataset+'*'):
        if f.is_file(): shutil.copyfile(f,manifests/f.name)
    if not (manifests/(dataset+'_generation_manifest.json')).exists():
        raise RuntimeError('manifest copy incomplete')
    io.write(study/'private/cells.json',selected,True)
    protocol=dict(schema='value_same_state_suffix_eager_qualification_v2',source_sha256=fingerprints,
        input_sha256={str(p.relative_to(study)):io.sha(p) for p in [study/'private/cells.json',*manifests.iterdir()] if p.is_file()},
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        layer=29,suite=suite,panel=panel,panel_cells=len(selected),repeats=2,arms=['uninstrumented_control','diagnostics_disabled','diagnostics_active'],
        parent='whole-layer replay must be complete first',budget=settings['mage_k']//64,scope='GLOBAL layer29 interventions only',
        runtime=dict(max_model_len=settings['max_len'],memory=.80 if suite=='longbench_v2' else settings['memory'],
            block=settings['block'],chunk=settings['chunk'],graph='NONE',enforce_eager=True,native_steps=48),
        input_selection='shorter of two archived LongBench traces' if panel=='qualification' else 'first indices at minimum native seed; RULER first index per task, independent of correctness',
        qualification='uninstrumented/disabled/active native output hashes, N, C and T must agree in both repeats; exact same-state suffix logits; unchanged RNG and adapter counters',
        decision_metrics=['top1','acceptance','common_Gumbel','KL','entropy','stopping'],
        timing='paused; instrumented generation is diagnostic only; suffix calls excluded from native N',
        substrate_limit='Eager model diagnostics use common identical eager states, not production compiled V31 states. Production accuracy requires fresh compiled controls.',
        cli_mode='PIECEWISE is required by inherited adapter CLI; diagnostic wrapper explicitly sets enforce_eager, compilation mode0 and actual graph NONE',
        full_accuracy='not automatically launched by this qualifier')
    if panel=='decisions':
        protocol['instrumentation_parent_sha256']=io.sha(qualification/'qualification_complete.json')
    shutil.copyfile(Path(__file__),study/'coordinator_source.py')
    protocol['coordinator_sha256']=io.sha(study/'coordinator_source.py')
    io.write(study/'protocol.json',protocol,True)


def verify(study):
    protocol=io.read(study/'protocol.json')
    if 'coordinator_sha256' in protocol and io.sha(study/'coordinator_source.py')!=protocol['coordinator_sha256']:
        raise RuntimeError('frozen coordinator drift')
    for rel,digest in protocol['source_sha256'].items():
        if io.sha(study/'private/execution'/rel)!=digest:
            raise RuntimeError('frozen executable source drift')
    for rel,digest in protocol['input_sha256'].items():
        if io.sha(study/rel)!=digest:
            raise RuntimeError('frozen input drift')
    return protocol


def env_for(study,leaf,active):
    from scripts.v31_v2_local_threshold_sweep import env_for as base_env, SUITES
    p=io.read(study/'protocol.json');suite=p.get('suite','longbench_v2');r=p['runtime']
    if suite not in SUITES:SUITES[suite]=dict(block=r['block'],chunk=r['chunk'],memory=r['memory'],max_len=r['max_model_len'],mage_k=p['budget']*64)
    env=base_env(study,suite,'current_v31_control')
    env['MEM']=str(r['memory'])
    env['STUDY_ARM']='same_state_suffix_diagnostic'
    env['SAME_STATE_SUFFIX_DIR']=str(leaf/'decisions')
    env['SUFFIX_DIAGNOSTICS']='1' if active else '0'
    env['SUFFIX_INSTRUMENT']='0' if active is None else '1'
    env['SUFFIX_EAGER']='1'
    return env


def idle():
    return not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()


def run(study):
    protocol=verify(study)
    last=time.monotonic(); free_since=None
    while True:
        parent_done=(study.parent/'whole_layer/operator_complete.json').exists()
        if parent_done and idle():
            free_since=free_since or time.monotonic()
            if time.monotonic()-free_since>=120: break
        else:
            free_since=None
        if time.monotonic()-last>=1800:
            print(json.dumps(dict(event='waiting_for_whole_layer_and_idle_h100')),flush=True);last=time.monotonic()
        time.sleep(30)
    receipts=[]
    for repeat in (1,2):
        for active in (None,False,True):
            arm='uninstrumented_control' if active is None else ('diagnostics_active' if active else 'diagnostics_disabled')
            family=study/f'repeat{repeat:02d}'/arm
            done=sorted(family.glob('attempt*/complete.json'))
            if done:
                receipt=io.read(done[-1]);assert receipt['protocol_sha256']==io.sha(study/'protocol.json')
                assert receipt['records_sha256']==io.sha(done[-1].parent/'records.jsonl')
                receipts.append((repeat,active,done[-1].parent));continue
            if not idle():raise RuntimeError('H100 occupied; resume later')
            prior=[int(x.name[7:]) for x in family.glob('attempt*') if x.is_dir()]
            leaf=family/f'attempt{max(prior,default=0)+1:03d}'
            (leaf/'private').mkdir(parents=True);(leaf/'decisions').mkdir()
            io.write(study/'state.json',dict(stage='running',repeat=repeat,arm=arm,attempt=str(leaf.relative_to(study)),timing='paused'))
            execution=study/'private/execution'
            worker='v31_same_state_suffix_worker.py'
            command=[PY,str(execution/'scripts'/worker),str(io.MODEL),str(study/'private/manifests'),
                str(study/'private/cells.json'),str(leaf/'records.jsonl'),str(leaf/'private/run_diagnostic.private.jsonl'),'mage','PIECEWISE']
            io.write(leaf/'config.json',dict(protocol_sha256=io.sha(study/'protocol.json'),repeat=repeat,active=active),True)
            start=time.monotonic()
            with (leaf/'private/worker.log').open('w') as stream:
                p=subprocess.Popen(command,cwd=execution,env=env_for(study,leaf,active),stdout=stream,stderr=subprocess.STDOUT)
                last=time.monotonic()
                while p.poll() is None:
                    if time.monotonic()-last>=1800:
                        print(json.dumps(dict(event='running',repeat=repeat,arm=arm)),flush=True);last=time.monotonic()
                    time.sleep(5)
            if p.returncode:
                io.write(leaf/'failure.json',dict(returncode=p.returncode,gpu_reservation_s=time.monotonic()-start),True)
                io.write(study/'state.json',dict(stage='failed',attempt=str(leaf.relative_to(study)),resumable=True,timing='paused'))
                raise RuntimeError('diagnostic worker failed; preserve attempt and inspect private log')
            rows=[json.loads(x) for x in (leaf/'records.jsonl').read_text().splitlines()]
            if len(rows)!=protocol['panel_cells']:
                raise RuntimeError('incomplete native trajectory')
            if any(r.get('cuda_graph_captures')!=0 or not str(r.get('resolved_cudagraph_mode','')).endswith('NONE') for r in rows):
                raise RuntimeError('diagnostic did not execute the declared eager/no-capture substrate')
            if active and not (leaf/'decisions/decisions.jsonl').exists():
                raise RuntimeError('no same-state decisions collected')
            verify(study)
            io.write(leaf/'complete.json',dict(status='complete',protocol_sha256=io.sha(study/'protocol.json'),
                records_sha256=io.sha(leaf/'records.jsonl'),gpu_reservation_s=time.monotonic()-start,
                decisions_sha256=io.sha(leaf/'decisions/decisions.jsonl') if active else None),True)
            receipts.append((repeat,active,leaf))
    rows={}
    for repeat,active,leaf in receipts:
        rr=[json.loads(x) for x in (leaf/'records.jsonl').read_text().splitlines()]
        rows[repeat,active]={(r['dataset'],r['index'],r['panel_seed']):r for r in rr}
    baseline=rows[1,None]
    compare_keys=['output_hash','denoise_forwards','canvases','output_tokens']
    errors=[]
    for (repeat,active),rr in rows.items():
        if set(rr)!=set(baseline):raise RuntimeError('incomplete paired diagnostic cohort')
        errors.extend(dict(repeat=repeat,active=active,item_index=key[1],key=k)
            for key,row in rr.items() for k in compare_keys if row[k]!=baseline[key][k])
    io.write(study/'reproducibility.json',dict(status='passed' if not errors else 'failed',mismatches=errors,
        checks=compare_keys,native_trajectories=len(rows)*len(baseline),scope=f"{len(baseline)} previously examined {protocol['suite']} items; layer29 suffix only"),True)
    if errors:raise RuntimeError('diagnostic changed native trajectory or repetition floor')
    io.write(study/'qualification_complete.json',dict(status='complete',protocol_sha256=io.sha(study/'protocol.json'),
        reproducibility_sha256=io.sha(study/'reproducibility.json'),native_accuracy_launched=False,timing='paused'),True)
    io.write(study/'state.json',dict(stage='qualification_complete',timing='paused',full_accuracy_launched=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','run'],required=True)
    p.add_argument('--study',type=Path,default=DEFAULT)
    p.add_argument('--suite',choices=['longbench_v2','aime26','math500','ruler128k'],default='longbench_v2')
    p.add_argument('--panel',choices=['qualification','decisions'],default='qualification')
    a=p.parse_args()
    if a.stage=='prepare':prepare(a.study.resolve(),a.suite,a.panel)
    else:run(a.study.resolve())
