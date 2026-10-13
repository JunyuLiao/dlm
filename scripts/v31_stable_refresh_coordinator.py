"""Resumable one-H100 correctness -> operator -> qualification -> development queue.

Every relaunch gets a new attempt. No partial shard is reused; completed
leaves are validated against current source/config identity. No clean timing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PY=Path('/home/exouser/dyh/v31_env_20261005/bin/python')
MODEL='/home/exouser/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b'
SOURCES=['experiments/numerical_qk_reuse/'+n for n in ('vllm_adapter.py','v31_stable_refresh.py','v31_stable_observe.py','v31_stable_trajectory.py','v31_fa4_observe.py','v27_fa4.py')]+['scripts/'+n for n in ('v31_vllm_paired_bench.py','v31_stable_refresh_campaign.py','v31_stable_snapshot_replay.py','v31_stable_refresh_coordinator.py')]


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,data):
    path.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
def idle():
    return not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
def external_runner_active():
    # A model runner can exist before its CUDA context appears in nvidia-smi.
    listing=subprocess.check_output(['ps','-eo','comm=,args='],text=True)
    return any(line.lstrip().startswith('python') and 'run_aime_eval.py' in line for line in listing.splitlines())
def wait_gpu(study):
    last=0;free_since=None
    while True:
        free=idle() and not external_runner_active()
        if free:
            if free_since is None:free_since=time.monotonic()
            if time.monotonic()-free_since>=120:return
        else:free_since=None
        if time.monotonic()-last>=1800:
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
            print(json.dumps({'status':'waiting_other_gpu_worker','gpu':gpu}),flush=True); last=time.monotonic()
        time.sleep(30)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--study',type=Path,required=True); args=p.parse_args(); study=args.study.resolve()
    os.chdir(ROOT)
    runtime_env=dict(os.environ,PYTHONPATH=str(ROOT)+':'+str(ROOT/'src'),PYTHONNOUSERSITE='1')
    identity={p:digest(ROOT/p) for p in SOURCES}
    parameter_file=study/'specs/parameters.json'
    failure=study/'coordinator_failure.json'
    if failure.exists():
        history=study/'coordinator_failures';history.mkdir(exist_ok=True)
        existing=list(history.glob('attempt*.json'))
        failure.rename(history/f'attempt{len(existing)+1:03d}.json')
    def run_logged(cmd,log):
        with log.open('w') as stream:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=runtime_env,stdout=stream,stderr=subprocess.STDOUT)
            last=time.monotonic()
            while proc.poll() is None:
                if time.monotonic()-last>=1800:
                    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
                    print(json.dumps({'status':'running','log':str(log.relative_to(study)),'gpu':gpu}),flush=True);last=time.monotonic()
                time.sleep(5)
        if proc.returncode: raise RuntimeError(f'failed command ({proc.returncode}); preserved log {log}')
    def attempt(base,arm,cells,trajectory=False,parameters=False):
        family=study/base/arm; family.mkdir(parents=True,exist_ok=True)
        done=sorted(family.glob('attempt*/attempt_complete.json'))
        for marker in done:
            cfg=json.loads((marker.parent/'config.json').read_text())
            if cfg['cells_sha256']!=digest(cells): continue
            if any(cfg['source_sha256'].get(p)!=h for p,h in identity.items() if p in cfg['source_sha256']): continue
            if json.loads(marker.read_text()).get('validation_errors'): continue
            return marker.parent
        used=[int(x.name[7:]) for x in family.glob('attempt*') if x.is_dir()]
        leaf=family/f'attempt{max(used,default=0)+1:03d}'
        wait_gpu(study)
        cmd=[str(PY),str(ROOT/'scripts/v31_stable_refresh_campaign.py'),'--python',str(PY),'--model',MODEL,
            '--manifests',str(study/'private/manifests'),'--cells',str(cells),'--attempt',str(leaf),'--arm',arm,
            '--purpose','audit','--result-root',str(study),'--max-model-len','141312','--gpu-memory-utilization','0.80']
        if trajectory: cmd+=['--trajectory-dir',str(leaf/'private/trajectories')]
        if parameters: cmd+=['--stable-parameters',str(parameter_file)]
        print(json.dumps({'status':'starting','leaf':str(leaf.relative_to(study))}),flush=True)
        run_logged(cmd,study/'private'/f'launch_{base.replace("/","_")}_{arm}_{leaf.name}.log')
        return leaf
    try:
        # Independent tensor GPU gate always precedes real model work.
        gate=study/'qualification/gpu_tests_complete.json'
        previous=json.loads(gate.read_text()).get('sources',{}) if gate.exists() else {}
        # Resource-wait orchestration does not invalidate completed selector/FA4 tests.
        gpu_inputs={p:h for p,h in identity.items() if not p.endswith(('v31_stable_refresh_coordinator.py','v31_stable_snapshot_replay.py'))}
        if any(previous.get(p)!=h for p,h in gpu_inputs.items()):
            wait_gpu(study)
            tests_logs=list((study/'qualification').glob('gpu_tests_attempt*.log'))
            run_logged([str(PY),'-m','pytest','-q','tests/test_v31_stable_refresh.py','tests/test_v31_stable_lifecycle.py','tests/test_v31_fa4_observe.py'],study/'qualification'/f'gpu_tests_attempt{len(tests_logs)+1:03d}.log')
            write(gate,dict(status='complete',sources=identity))
        trace=attempt('qualification/trajectory','v31_progress_dense_control',study/'private/cells_trajectory4.json',trajectory=True)
        write(study/'operator/trajectory_binding.json',dict(completed_leaf=str(trace.relative_to(study)),
            receipt_sha256=digest(trace/'receipt.json'),config_sha256=digest(trace/'config.json')))
        replay=study/'operator/real_replay_complete.json'
        if not replay.exists():
            wait_gpu(study)
            used=[int(x.name[7:]) for x in (study/'operator').glob('attempt*') if x.is_dir()]
            name=f'attempt{max(used,default=0)+1:03d}'
            replay_command=[str(PY),str(ROOT/'scripts/v31_stable_snapshot_replay.py'),'--study',str(study),'--attempt',name]
            frequency_binding=study/'specs/native_frequency_replay_binding.json'
            if frequency_binding.exists():
                binding=json.loads(frequency_binding.read_text());frozen=study/binding['parameter_file']
                if digest(frozen)!=binding['parameters_sha256']:raise RuntimeError('frequency replay parameters changed')
                replay_command+=['--frozen-parameters',str(frozen)]
            run_logged(replay_command,study/'operator'/f'{name}.log')
        replay_receipt=json.loads(replay.read_text())
        replay_summary=json.loads((study/replay_receipt['leaf']/'summary.json').read_text())
        if not replay_summary.get('native_frequency_configs_match_selected'):
            raise RuntimeError('no-progress frequency replay does not match frozen native parameters')
        parameter_file=study/json.loads(replay.read_text()).get('parameter_file','specs/parameters.json')
        # Repetition floor and observation-only native qualifier. Never score before parameter selection.
        quals=[]
        for rep in (1,2):
            for arm in ('v31_progress_dense_control','observation_only_progress_dense_s2'):
                quals.append(attempt(f'qualification/repeat{rep:02d}',arm,study/'private/cells_qualification3.json',parameters=True))
        qualrows={}
        for leaf in quals:
            arm=json.loads((leaf/'config.json').read_text())['arm']
            qualrows.setdefault(arm,[]).append([json.loads(x) for x in (leaf/'records.jsonl').read_text().splitlines()])
        control=qualrows['v31_progress_dense_control']; obs=qualrows['observation_only_progress_dense_s2']
        floor=sum(a['output_hash']!=b['output_hash'] for a,b in zip(*control))
        mismatches=[sum(a['output_hash']!=b['output_hash'] for a,b in zip(control[i],obs[i])) for i in (0,1)]
        write(study/'qualification/reproducibility.json',dict(control_repeat_output_mismatches=floor,
            observation_only_output_mismatches=mismatches,cells_per_run=3,numerical_floor_detected=bool(floor)))
        if floor==0 and any(mismatches): raise RuntimeError('observation-only native invariance failed beyond reproducibility floor')
        # Freeze the full inventory before the development panel.
        if any(digest(ROOT/p)!=h for p,h in identity.items()): raise RuntimeError('source changed before development freeze')
        write(study/'specs/execution_identity.json',dict(source_sha256=identity,parameter_file=str(parameter_file.relative_to(study)),parameters_sha256=digest(parameter_file),
            protocol_sha256=digest(study/'specs/protocol.json'),runtime_python=str(PY),model_revision=MODEL.split('/')[-1]))
        allcells=json.loads((study/'private/cells_native_panel154.json').read_text())
        protocol=json.loads((study/'specs/protocol.json').read_text()); arms=protocol['controls']+protocol['arms']
        if json.loads(parameter_file.read_text()).get('conditional_probe_promoted'):
            arms.append('stable_global_probe4_progress_dense_s2')
        queue=[]
        for shard in range(4):
            cells=[c for c in allcells if int(c['index'])%4==shard]
            path=study/'private'/f'cells_shard{shard+1:02d}.json'
            if not path.exists():write(path,cells)
            rotated=arms[shard:]+arms[:shard]
            for arm in rotated:
                leaf=attempt(f'native/shard{shard+1:02d}',arm,path,parameters=True)
                queue.append(dict(arm=arm,shard=shard+1,path=str(leaf.relative_to(study)),
                    artifacts_sha256={name:digest(leaf/name) for name in ('config.json','records.jsonl','receipt.json','attempt_complete.json')}))
                write(study/'queue_state.json',dict(status='running',completed=queue,total=4*len(arms)))
        write(study/'native_generation_complete.json',dict(status='complete',shards=queue,arms=arms,cells_per_arm=154))
        run_logged([str(PY),str(ROOT/'scripts/v31_stable_refresh_analysis.py'),'--study',str(study)],study/'scoring/analysis.log')
        write(study/'coordinator_complete.json',dict(status='complete',native_arms=len(arms),native_cells=len(arms)*154))
    except Exception as error:
        write(study/'coordinator_failure.json',dict(status='failed',error=str(error)))
        raise


if __name__=='__main__': main()
