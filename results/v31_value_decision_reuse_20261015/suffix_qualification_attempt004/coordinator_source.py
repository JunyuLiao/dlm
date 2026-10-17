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


def prepare(study):
    if (study/'protocol.json').exists():
        verify(study)
        return
    source=io.ROOT/'results/v31_cached_refresh_stable_20261013'
    cells=io.read(source/'private/cells_trajectory4.json')
    selected=[c for c in cells if c['dataset'].startswith('longbench')]
    # Choose the shorter of the two previously captured long-context items
    # using prompt length alone, before any new model decision measurement.
    selected=sorted(selected,key=lambda c: c['prompt_tokens'] if isinstance(c['prompt_tokens'],int) else len(c['prompt_tokens']))[:1]
    if len(selected)!=1:
        raise RuntimeError('frozen input panel unavailable')
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
    for f in (source/'private/manifests').glob(dataset+'*'):
        if f.is_file(): shutil.copyfile(f,manifests/f.name)
    if not (manifests/(dataset+'_generation_manifest.json')).exists():
        raise RuntimeError('manifest copy incomplete')
    io.write(study/'private/cells.json',selected,True)
    protocol=dict(schema='value_same_state_suffix_qualification_v1',source_sha256=fingerprints,
        input_sha256={str(p.relative_to(study)):io.sha(p) for p in [study/'private/cells.json',*manifests.iterdir()] if p.is_file()},
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        layer=29,panel_cells=1,repeats=2,arms=['uninstrumented_control','diagnostics_disabled','diagnostics_active'],
        parent='whole-layer replay must be complete first',budget=128,scope='GLOBAL layer29 interventions only',
        runtime=dict(max_model_len=141312,memory=.80,block=32,chunk=16384,graph='PIECEWISE',native_steps=48),
        qualification='uninstrumented/disabled/active native output hashes, N, C and T must agree in both repeats; exact same-state suffix logits; unchanged RNG and adapter counters',
        decision_metrics=['top1','acceptance','common_Gumbel','KL','entropy','stopping'],
        timing='paused; instrumented generation is diagnostic only; suffix calls excluded from native N',
        full_accuracy='not automatically launched by this qualifier')
    io.write(study/'protocol.json',protocol,True)
    shutil.copyfile(Path(__file__),study/'coordinator_source.py')


def verify(study):
    protocol=io.read(study/'protocol.json')
    for rel,digest in protocol['source_sha256'].items():
        if io.sha(study/'private/execution'/rel)!=digest:
            raise RuntimeError('frozen executable source drift')
    for rel,digest in protocol['input_sha256'].items():
        if io.sha(study/rel)!=digest:
            raise RuntimeError('frozen input drift')
    return protocol


def env_for(study,leaf,active):
    from scripts.v31_v2_local_threshold_sweep import env_for as base_env
    env=base_env(study,'longbench_v2','current_v31_control')
    env['MEM']='.80'
    env['STUDY_ARM']='same_state_suffix_diagnostic'
    env['SAME_STATE_SUFFIX_DIR']=str(leaf/'decisions')
    env['SUFFIX_DIAGNOSTICS']='1' if active else '0'
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
            worker='v31_vllm_paired_bench.py' if active is None else 'v31_same_state_suffix_worker.py'
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
            if active and not (leaf/'decisions/decisions.jsonl').exists():
                raise RuntimeError('no same-state decisions collected')
            verify(study)
            io.write(leaf/'complete.json',dict(status='complete',protocol_sha256=io.sha(study/'protocol.json'),
                records_sha256=io.sha(leaf/'records.jsonl'),gpu_reservation_s=time.monotonic()-start),True)
            receipts.append((repeat,active,leaf))
    rows={}
    for repeat,active,leaf in receipts:
        rows[repeat,active]=json.loads((leaf/'records.jsonl').read_text().splitlines()[0])
    baseline=rows[1,None]
    compare_keys=['output_hash','denoise_forwards','canvases','output_tokens']
    errors=[dict(repeat=r,active=a,key=k) for (r,a),row in rows.items() for k in compare_keys if row[k]!=baseline[k]]
    io.write(study/'reproducibility.json',dict(status='passed' if not errors else 'failed',mismatches=errors,
        checks=compare_keys,native_trajectories=len(rows),scope='one previously examined LongBench item; layer29 suffix only'),True)
    if errors:raise RuntimeError('diagnostic changed native trajectory or repetition floor')
    io.write(study/'qualification_complete.json',dict(status='complete',protocol_sha256=io.sha(study/'protocol.json'),
        reproducibility_sha256=io.sha(study/'reproducibility.json'),native_accuracy_launched=False,timing='paused'),True)
    io.write(study/'state.json',dict(stage='qualification_complete',timing='paused',full_accuracy_launched=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','run'],required=True)
    p.add_argument('--study',type=Path,default=DEFAULT)
    a=p.parse_args();{'prepare':prepare,'run':run}[a.stage](a.study.resolve())
