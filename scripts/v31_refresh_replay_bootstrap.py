"""Source-matched bootstrap for the preserved standalone operator replay."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
REL='scripts/v31_refresh_replay_bootstrap.py'
REQUIRED=(REL,'docs/REFRESH_REPLAY_BOOTSTRAP_20261018.md','tests/test_v31_refresh_replay_bootstrap.py')


def boot_observer(apply_patch,load_observer):
    """Ordering matters: never retain an observer reference to the old module."""
    patched_path=apply_patch()
    return patched_path,load_observer()


def child(parent,output):
    import hashlib
    from scripts.v31_vllm_paired_bench import apply_fa4_local_fix
    def load():
        from experiments.numerical_qk_reuse import v31_fa4_observe
        return v31_fa4_observe
    if os.environ.get('FA4_LOCAL_FIX')!='1':raise RuntimeError('native local patch flag missing')
    path,observe=boot_observer(apply_fa4_local_fix,load)
    from vllm.vllm_flash_attn.cute import interface,flash_fwd_sm90
    assert observe._sm90 is flash_fwd_sm90
    assert getattr(flash_fwd_sm90,'_v31_fa4_local_fix',False)
    assert interface.FlashAttentionForwardSm90 is flash_fwd_sm90.FlashAttentionForwardSm90
    receipt=dict(native_patch_applied=True,observer_module_matches=True,interface_class_matches=True,
                 patched_sm90_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    (output/'bootstrap_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    from scripts.v31_refresh_operator_replay import run
    run(parent/'capture/private',output/'analysis')


def prepare(parent,output):
    from scripts import v31_refresh_followup as follow
    from scripts import v31_refresh_external_profile as ext
    from scripts import v31_v2_local_threshold_sweep as base
    if output.exists():raise RuntimeError('preserve prior corrected replay')
    tracked=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
    changed=set(subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).splitlines())
    if not set(REQUIRED)<=tracked or set(REQUIRED)&changed:raise RuntimeError('commit correction before prepare')
    p,original,_=follow.verify(parent)
    if base.read(parent/'capture/receipt.json')['status']!='complete_diagnostic':raise RuntimeError('completed capture required')
    snapshots=sorted((parent/'capture/private').glob('snapshot*.pt'))
    if len(snapshots)!=6:raise RuntimeError('six snapshots required')
    output.mkdir(parents=True);source=output/'private/bootstrap.py';source.parent.mkdir();shutil.copy2(ROOT/REL,source)
    for i in (1,2):shutil.copytree(ext.ORIGINAL/'private/cache',output/f'engine{i:03d}/private/cache')
    protocol=dict(schema='native_replay_bootstrap_v1',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_fingerprints={r:base.sha(ROOT/r) for r in REQUIRED},bootstrap_sha256=base.sha(source),
        parent_protocol_sha256=base.sha(parent/'protocol.json'),capture_receipt_sha256=base.sha(parent/'capture/receipt.json'),
        native_patch_expected_sha256=base.jsonl(parent/'capture/records.jsonl')[0]['fa4_local_fix_sha256'],
        snapshot_sha256={x.name:base.sha(x) for x in snapshots},engines=2,per_engine_seconds=300.,
        phase='replay',phase_seconds=600.,campaign_seconds=2700.,original_replay_leaves='skipped_unstarted')
    base.write(output/'protocol.json',protocol,True);print(json.dumps(dict(event='prepared_corrected_replay',source=protocol['source_commit'])),flush=True)


def run(parent,output):
    from scripts import v31_refresh_followup as follow
    from scripts import v31_refresh_external_profile as ext
    from scripts import v31_refresh_diagnostic_study as study
    from scripts import v31_v2_local_threshold_sweep as base
    with (ROOT/'results/.v31_refresh_h100.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=base.read(output/'protocol.json');_,original,_=follow.verify(parent)
        assert base.sha(parent/'protocol.json')==protocol['parent_protocol_sha256']
        assert base.sha(parent/'capture/receipt.json')==protocol['capture_receipt_sha256']
        for rel,h in protocol['source_fingerprints'].items():assert base.sha(ROOT/rel)==h
        for name,h in protocol['snapshot_sha256'].items():assert base.sha(parent/'capture/private'/name)==h
        assert base.sha(output/'private/bootstrap.py')==protocol['bootstrap_sha256']
        for i in (1,2):
            leaf=output/f'engine{i:03d}'
            if (leaf/'receipt.json').exists():continue
            if (leaf/'private/worker.log').exists():raise RuntimeError('preserve previous replay')
            ledger=base.read(follow.CAMPAIGN/'gpu_budget.json')
            if any(r.get('elapsed_s') is None for r in ledger['reservations']):raise RuntimeError('unresolved reservation')
            seconds=min(300.,follow.available('replay',ledger['reservations']))
            if seconds<120:raise RuntimeError('replay phase exhausted')
            study.require_idle();base.SUITES.update(original['suites'])
            env=base.env_for(leaf,'longbench_v2','current_v31_control')
            execution=parent/'private/execution';env.update(PYTHONPATH=f'{execution}/src:{execution}',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0')
            command=[base.PYTHON,str(output/'private/bootstrap.py'),'child','--parent',str(parent),'--output',str(leaf)]
            reservation=dict(attempt=str(output.relative_to(follow.CAMPAIGN)),job=f'engine{i:03d}',phase='replay',reserved_s=seconds,elapsed_s=None,status='started')
            ledger['reservations'].append(reservation);base.write(follow.CAMPAIGN/'gpu_budget.json',ledger)
            start=time.monotonic();error=None;process=None
            print(json.dumps(dict(event='launch_corrected_replay',engine=i,limit_s=seconds)),flush=True)
            try:process=ext.bounded_process(command,cwd=execution,env=env,log_path=leaf/'private/worker.log',deadline=start+seconds)
            except BaseException as exc:error=type(exc).__name__+': '+str(exc)
            finally:
                reservation.update(elapsed_s=time.monotonic()-start,status='finished' if error is None else 'failed')
                base.write(follow.CAMPAIGN/'gpu_budget.json',ledger)
            valid=False
            if (leaf/'bootstrap_receipt.json').exists():valid=base.read(leaf/'bootstrap_receipt.json')['patched_sm90_sha256']==protocol['native_patch_expected_sha256']
            receipt=dict(status='complete_diagnostic' if not error and process and process['returncode']==0 and valid and (leaf/'analysis/summary.json').exists() else 'failed_or_partial',
                process=process,error=error,native_patch_hash_matches_capture=valid,reservation_seconds=reservation['elapsed_s'],protocol_sha256=base.sha(output/'protocol.json'))
            base.write(leaf/'receipt.json',receipt,True);print(json.dumps(dict(engine=i,**receipt)),flush=True)
            if receipt['status']!='complete_diagnostic':break


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run','child'))
    p.add_argument('--parent',type=Path,default=ROOT/'results/v31_refresh_repeatability_20261018/attempt001')
    p.add_argument('--output',type=Path,default=ROOT/'results/v31_refresh_repeatability_20261018/replay_corrected/attempt001')
    a=p.parse_args();globals()[a.action](a.parent,a.output)
