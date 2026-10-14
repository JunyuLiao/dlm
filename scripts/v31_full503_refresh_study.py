"""Frozen one-H100 full-LongBench refresh study; resumable completed shards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
PY='/home/exouser/dyh/v31_env_20261005/bin/python'
MODEL=Path('/home/exouser/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b')
ARMS=['dense_full_fix51994','dense_piecewise','allkept_fa4','v31_progress_dense_control',
      'v31_no_progress_dense_hold','v31_every_step_dense_select_sparse_output',
      'raw_global_progress_dense_s2','stable_global_progress_dense_s2',
      'raw_global_no_progress_dense_s2','stable_global_no_progress_dense_s2',
      'stable_global_no_progress_dense_s1','stable_global_no_progress_dense_s4','stable_global_no_progress_dense_s8']


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()
def write(p,obj,immutable=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if immutable and p.exists():
        if p.read_text()!=text:raise RuntimeError('frozen artifact differs')
        return
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(text);tmp.replace(p)


def verify(study,protocol):
    for rel,h in protocol['execution_sha256'].items():
        if sha(study/'private/execution'/rel)!=h:raise RuntimeError('frozen execution source drift')
    for rel,h in protocol['input_sha256'].items():
        if sha(study/rel)!=h:raise RuntimeError('frozen native input drift')


def prepare(study):
    if study.exists():raise RuntimeError('new study root required')
    study.mkdir(parents=True)
    execution=study/'private/execution'
    for name in ('src','scripts','experiments','tests'):
        shutil.copytree(ROOT/name,execution/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pt','*.npz','*.png','*.pdf','*.jsonl','*.safetensors','*.so'))
    source=ROOT/'results/v31_cached_refresh_stable_20261013'
    shutil.copyfile(source/'model_source_fingerprints.json',study/'model_source_fingerprints.json')
    inv=read(study/'model_source_fingerprints.json')
    for f in inv['files']:
        target=MODEL/f['file']
        if target.stat().st_size!=f['bytes'] or sha(target)!=f['sha256']:raise RuntimeError('model source mismatch')
    manifests=study/'private/manifests';manifests.mkdir()
    original=Path('/home/exouser/dyh/ljy_value_20261007/pools/longbench_v2_ofc')
    ds='longbench_v2_0shot_think'
    for suffix in ('generation_manifest.json','rowinfo.json','gold.json'):
        shutil.copyfile(original/f'{ds}_{suffix}',manifests/f'{ds}_{suffix}')
    cells=read(ROOT/'results/v31_value_v2_equal_work_20261011/private/cells_target_full503.json')
    rows=read(manifests/f'{ds}_generation_manifest.json')
    if len(cells)!=503 or {c['index'] for c in cells}!=set(range(503)):raise RuntimeError('full 503 coverage required')
    for c in cells:
        row=rows[c['index']]
        if c['seed']!=1 or c['id']!=row['id'] or c['prompt_tokens']!=len(row['prompt_tokens']):raise RuntimeError('native cell binding mismatch')
    write(study/'private/cells_full503.json',cells,True)
    write(study/'private/cells_qualification02.json',cells[:2],True)
    for shard in range(4):write(study/f'private/cells_shard{shard+1:02d}.json',[c for c in cells if c['index']%4==shard],True)
    (study/'specs').mkdir(exist_ok=True)
    shutil.copyfile(source/'specs/parameters_attempt004.json',study/'specs/parameters.json')
    execution_hashes={str(p.relative_to(execution)):sha(p) for p in execution.rglob('*') if p.is_file()}
    inputs={str(p.relative_to(study)):sha(p) for p in (study/'private').rglob('*.json') if not p.is_relative_to(execution)}
    inputs['specs/parameters.json']=sha(study/'specs/parameters.json')
    from scripts.v31_v2_local_threshold_sweep import runtime_identity
    protocol=dict(schema='v31_full503_refresh_v1',study_date='2026-10-14',arms=ARMS,cells_per_arm=503,total_native_cells=6539,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        execution_sha256=execution_hashes,input_sha256=inputs,runtime=runtime_identity(),model=str(MODEL),
        model_inventory_sha256=sha(study/'model_source_fingerprints.json'),K_prefix_tiles=128,K_tokens=8192,
        scope='5 GLOBAL layers; LOCAL, prefill and commits dense',seed_base=31,panel_seed=1,
        native=dict(canvas=256,max_steps=48,confidence=.1,stability=1,entropy=.005,temperature=[.8,.4],thinking=True,output_limits='official manifests'),
        runtime_settings=dict(block=32,chunk=16384,max_model_len=141312,gpu_memory_utilization=.80),
        hold='Existing per-canvas selection and carry; disable mid-canvas progress refresh; no cheap updates.',
        every_step='Full current V31 qblock-max mass; no sticky/carry/cache/warm dense output; discard dense observer output and return newly selected native sparse FA4 output on same call.',
        every_step_dense_cost='Full QK and PV observer work counted separately from sparse output work; no extra model denoising forward.',
        stable_parameters=read(study/'specs/parameters.json')['selector'],
        sampling_status='Previously examined full benchmark; no held-out or post-answer parameter tuning.',
        analysis='Official accuracy, paired problem bootstrap, N/N-C/output length/upper tail, physical GLOBAL/LOCAL, selector cost separately.',
        schedule='4 deterministic modulo-index shards; rotate arm order by shard; completed leaves reusable only after source/input/artifact validation.',
        monitor_minutes=15,clean_timing='paused',operator_source_inventory_sha256=sha(source/'specs/real_snapshot_inventory.json'))
    write(study/'specs/native_panel.json',dict(arms=ARMS,cells=503,seed=1,cell_hashes=[hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest() for c in cells],manifest_sha256=sha(manifests/f'{ds}_generation_manifest.json')),True)
    write(study/'specs/protocol.json',protocol,True)
    (study/'README.md').write_text('# Full LongBench refresh comparison\n\n13 arms x 503 seed-1 problems (6,539 cells). Seven selected raw/stable/progress/frequency variants, both held-map controls, a fresh same-call V31 selection plus sparse-output baseline, and matched FULL/PIECEWISE/all-kept controls. Native sampler and K=128 remain fixed. No-progress hold keeps existing per-canvas initialization/carry; it is not a once-per-problem frozen mask. Every-step selection uses no previous-map bonus or cache; its dense observer output is discarded. The observer full QK/PV cost is separate from output-consumer physical work. All inputs/completions and frozen executable sources remain private.\n\nCorrectness, identical-Q/K/V operator checks and repeated observation-only native qualification precede target generation. Rotated deterministic shards; a failed leaf remains intact and resumes get a new attempt leaf. Monitor the H100 every 15 minutes. Official scoring and fairness/trajectory audits gate final_complete.json. Clean timing stays paused. The separate V2 campaign remains user-paused.\n')
    (study/'specs/implementation_plan.md').write_text('# Execution plan\n\n1. Freeze copied sources, all model hashes, official full503 input bindings, native sampler, budgets and arm definitions.\n2. Run focused CPU/CUDA tests and current-Q/K/V full-information CPU-map and sparse-output checks for every-step baseline.\n3. Repeat V31 control and compare updates-disabled observations; smoke every-step output receipts on the same two cells.\n4. Run 13 arms in four rotated shards with one H100 worker, source checks and immutable attempts.\n5. Officially score all 503 cells per arm, audit matching sampler/runtime/source, publish accuracy/trajectory/physical work/selector overhead separately.\n6. Create final_complete only when coverage, receipts, official scorer identity and all hashes agree. Timing remains paused.\n')
    (study/'.gitignore').write_text('private/\n*.log\n*.pid\nmonitor.jsonl\n')
    print(json.dumps(dict(status='prepared',source_files=len(execution_hashes),arms=len(ARMS),native_cells=6539)))


def gpu_idle():
    return not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()


def run(study):
    protocol=read(study/'specs/protocol.json');verify(study,protocol)
    execution=study/'private/execution'
    env=dict(os.environ,PYTHONPATH=f'{execution}/src:{execution}',PYTHONNOUSERSITE='1',
             FROZEN_SOURCE_COMMIT=protocol['source_commit'])
    cache=study/'private/cache'
    env.update(TRITON_CACHE_DIR=str(cache/'triton'),CUDA_CACHE_PATH=str(cache/'cuda'))
    def logged(command,path):
        if path.exists():raise RuntimeError('new log required for resumed operation')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w') as stream:
            proc=subprocess.Popen(command,cwd=execution,env=env,stdout=stream,stderr=subprocess.STDOUT)
            last=0
            while proc.poll() is None:
                if time.monotonic()-last>=900:
                    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
                    write(study/'progress.json',dict(status='running',worker_pid=proc.pid,log=str(path.relative_to(study)),gpu=gpu,unix=time.time()))
                    with (study/'monitor.jsonl').open('a') as monitor:monitor.write(json.dumps(dict(unix=time.time(),gpu=gpu,log=str(path.relative_to(study))))+'\n')
                    print(json.dumps(dict(event='progress',gpu=gpu,log=str(path.relative_to(study)))),flush=True);last=time.monotonic()
                time.sleep(5)
        if proc.returncode:raise RuntimeError('worker failed; attempt and private log preserved')
    def idle():
        while not gpu_idle():
            write(study/'progress.json',dict(status='waiting_for_idle_H100',unix=time.time()))
            time.sleep(30)
    def launch(arm,cells,base,observation_only=False):
        verify(study,protocol)
        matches=sorted(study.glob(f'attempt*/{base}/{arm}/attempt_complete.json')) if base.startswith('qualification') else sorted(study.glob(f'attempt*/{arm}/{base}/attempt_complete.json'))
        for marker in matches:
            leaf=marker.parent;cfg=read(leaf/'config.json');receipt=read(marker)
            if receipt['status']=='complete' and not receipt['validation_errors'] and cfg['cells_sha256']==sha(cells):
                fingerprint=leaf/'completed_artifacts.json'
                if (fingerprint.exists() and all(sha(leaf/name)==h for name,h in read(fingerprint).items())
                    and all(sha(execution/rel)==h for rel,h in cfg['source_sha256'].items())):return leaf
        for attempt in range(1,1000):
            leaf=study/f'attempt{attempt:03d}'/base/arm if base.startswith('qualification') else study/f'attempt{attempt:03d}'/arm/base
            if not leaf.exists():break
        idle()
        cmd=[PY,str(execution/'scripts/v31_full503_refresh_campaign.py'),'--python',PY,'--model',str(MODEL),
             '--manifests',str(study/'private/manifests'),'--cells',str(cells),'--attempt',str(leaf),
             '--arm',arm,'--purpose','audit','--result-root',str(study),'--gpu-memory-utilization','.80',
             '--stable-parameters',str(study/'specs/parameters.json')]
        if observation_only:cmd.append('--observation-only')
        name=f'{leaf.parts[-3]}_{arm}_{base.replace("/","_")}'
        log=study/'private/launch_logs'/f'{name}.log'
        if log.exists():log=study/'private/launch_logs'/f'{name}_{time.time_ns()}.log'
        print(json.dumps(dict(event='launch',arm=arm,leaf=str(leaf.relative_to(study)))),flush=True)
        logged(cmd,log);verify(study,protocol)
        if read(leaf/'receipt.json')['status']!='complete':raise RuntimeError('unclean native receipt')
        write(leaf/'completed_artifacts.json',{name:sha(leaf/name) for name in ('config.json','records.jsonl','receipt.json','attempt_complete.json')},True)
        return leaf
    try:
        tests=study/'qualification/tests_complete.json'
        if not tests.exists():
            idle();n=len(list((study/'qualification').glob('tests_attempt*.log')))+1 if (study/'qualification').exists() else 1
            logged([PY,'-m','pytest','-q','tests/test_v31_every_step_refresh.py','tests/test_v31_stable_refresh.py',
                    'tests/test_v31_stable_lifecycle.py','tests/test_v31_fa4_observe.py','tests/test_v31_stable_legacy_parity.py',
                    'tests/test_v31_refresh_factorial.py'],study/f'qualification/tests_attempt{n:03d}.log')
            write(tests,dict(status='complete',log=f'tests_attempt{n:03d}.log'),True)
        operator=study/'qualification/operator_complete.json'
        if not operator.exists():
            idle();n=len(list((study/'qualification').glob('operator_attempt*.log')))+1
            logged([PY,str(execution/'scripts/v31_full503_operator_qualification.py'),'--source-study',
                    str(ROOT/'results/v31_cached_refresh_stable_20261013'),'--output',str(study/f'qualification/operator_attempt{n:03d}.json')],
                   study/f'qualification/operator_attempt{n:03d}.log')
            write(operator,dict(status='complete',receipt=f'operator_attempt{n:03d}.json'),True)
        native=study/'qualification/native_complete.json'
        if not native.exists():
            cells=study/'private/cells_qualification02.json'
            controls=[launch('v31_progress_dense_control',cells,f'qualification/control_repeat{r:02d}') for r in (1,2)]
            obs=launch('raw_global_progress_dense_s2',cells,'qualification/observation_only',True)
            extreme=launch('v31_every_step_dense_select_sparse_output',cells,'qualification/every_step')
            def rows(leaf):return [json.loads(x) for x in (leaf/'records.jsonl').read_text().splitlines()]
            a,b,c=rows(controls[0]),rows(controls[1]),rows(obs)
            for other in (b,c):
                if any(any(x[field]!=y[field] for field in ('output_hash','denoise_forwards','output_tokens','canvases')) for x,y in zip(a,other)):
                    raise RuntimeError('native control repeat or observation-only invariance failed')
            write(native,dict(status='complete',matched_cells=2,output_N_T_C_mismatches=0,every_step_receipt=str(extreme.relative_to(study))),True)
        complete=[]
        for shard in range(4):
            order=ARMS[shard:]+ARMS[:shard]
            for arm in order:
                cells=study/f'private/cells_shard{shard+1:02d}.json'
                leaf=launch(arm,cells,f'shard{shard+1:02d}')
                entry=dict(arm=arm,shard=shard+1,path=str(leaf.relative_to(study)),
                    artifacts_sha256={name:sha(leaf/name) for name in ('config.json','records.jsonl','receipt.json','attempt_complete.json')})
                complete.append(entry);write(study/'queue_state.json',dict(status='running',completed=complete,total=52))
        write(study/'native_generation_complete.json',dict(status='complete',arms=ARMS,shards=complete,cells_per_arm=503),True)
        logged([PY,str(execution/'scripts/v31_full503_refresh_report.py'),'--study',str(study)],study/f'report_attempt{len(list(study.glob("report_attempt*.log")))+1:03d}.log')
        write(study/'coordinator_complete.json',dict(status='complete',native_cells=6539,clean_timing='paused'),True)
        write(study/'progress.json',dict(status='complete',native_cells=6539,unix=time.time()))
        print(json.dumps(dict(status='complete',native_cells=6539)),flush=True)
    except BaseException as exc:
        write(study/f'failure_attempt{len(list(study.glob("failure_attempt*.json")))+1:03d}.json',dict(status='failed',reason=str(exc),unix=time.time()))
        write(study/'progress.json',dict(status='failed',reason=str(exc),unix=time.time()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    args=p.parse_args();prepare(args.study.resolve()) if args.prepare else run(args.study.resolve())
