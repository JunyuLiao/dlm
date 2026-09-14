"""Audited handoff from completed calibration to the frozen primary sweep.

Uses existing freeze/run/report gates; does not select candidates or fit policies.
CPU-only subprocesses cannot accidentally occupy the H100 before inference.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from .protocol import ROOT,TARGETS
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write


def missing_policy_recovery(root,methods):
    """One bounded resume; existing completed policies are never refitted."""
    decision=json.loads((root/'candidate_decision.json').read_text())
    if decision.get('heldout_used') is not False or set(decision['selected_methods'])!=set(methods):
        raise ValueError('handoff must preserve the pre-heldout candidate decision')
    names=list(dict.fromkeys(['blasst_original','blasst_aggressive',*methods]))
    missing=[(name,target) for name in names for target in TARGETS
        if any(not (root/'verified_policies'/benchmark/f'{name}_s{int(target*100)}.json').exists()
            for benchmark in ('aime26','longbench'))]
    return ([name for name in names if any(n==name for n,_ in missing)],
        [target for target in TARGETS if any(t==target for _,t in missing)])


def primary_handoff(methods,root=ROOT):
    from .efficient_iteration import guard_legacy_run
    guard_legacy_run(root,'finalize')
    if not methods:raise ValueError('explicit previously selected candidates required')
    cpu_env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2')
    prefix=[sys.executable,'-u','-m']
    recovery_names,recovery_targets=missing_policy_recovery(root,methods)
    status=dict(methods=methods,recovery_methods=recovery_names,recovery_targets=recovery_targets)
    if recovery_names:
        # A transient OOM may leave holes despite the preceding worker finishing.
        # Calibrate reuses completed policies and raw checkpoints; independent
        # missing configurations continue even if one fails again. No loop and
        # no heldout inference/accuracy is involved in this bounded recovery.
        recovery=subprocess.run(prefix+['experiments.diffusion_gemma_value_aware.launch','calibrate',
            '--methods',*recovery_names,'--targets',*[str(t) for t in recovery_targets],'--max-rounds','3'])
        status['recovery_exit_code']=recovery.returncode
        if recovery.returncode:
            status.update(complete=False,reason='calibration recovery failed; no final inference launched',finished=time.time())
            _write(root/'primary_handoff_terminal.json',status)
            return 1
    freeze=subprocess.run(prefix+['experiments.diffusion_gemma_value_aware.evaluate',
        'freeze','--output',str(root),'--methods',*methods],env=cpu_env)
    status['freeze_exit_code']=freeze.returncode
    if freeze.returncode:
        status.update(complete=False,reason='freeze audit failed; no final inference launched',finished=time.time())
        _write(root/'primary_handoff_terminal.json',status)
        return 1
    # The launcher owns the one GPU worker and its900-second health monitor.
    run=subprocess.run(prefix+['experiments.diffusion_gemma_value_aware.launch','run-final'])
    status['run_exit_code']=run.returncode
    report=subprocess.run(prefix+['experiments.diffusion_gemma_value_aware.report',
        '--output',str(root)],env=cpu_env)
    status['report_exit_code']=report.returncode
    audit_path=root/'audit.json'
    audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    # report CLI also returns normally for incomplete shards: its exit code alone
    # is deliberately insufficient to claim that the sweep is complete.
    status.update(complete=not run.returncode and not report.returncode and audit.get('complete') is True,
        expected=audit.get('expected'),completed=audit.get('completed'),finished=time.time())
    _write(root/'primary_handoff_terminal.json',status)
    return 0 if status['complete'] else 1


def main():
    p=argparse.ArgumentParser();p.add_argument('--methods',nargs='+',required=True)
    args=p.parse_args();sys.exit(primary_handoff(args.methods))


if __name__=='__main__':main()
