"""Finite frozen end-to-end schedule; no campaign resource ceiling."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', choices=['short', 'long'], required=True)
    ap.add_argument('--arms', nargs='*')
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    protocol = json.loads((root / 'e2e_protocol.json').read_text())
    arms = args.arms or protocol['arms']
    if set(arms) - set(protocol['arms']):
        raise RuntimeError('unplanned arm')
    for arm in arms:
        parent = root / 'launches' / args.suite / arm
        parent.mkdir(parents=True, exist_ok=False)
        for f, h in protocol['sources'].items():
            if hashlib.sha256(Path(f).read_bytes()).hexdigest() != h:
                raise RuntimeError('source drift ' + f)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
        if apps.strip():
            raise RuntimeError('GPU occupied; no preemption: ' + apps)
        env = os.environ.copy()
        for key in list(env):
            if key.startswith(('MAGE_', 'VALUE_', 'FORCE_', 'RESIDENT_', 'OPTIMIZED_')):
                env.pop(key)
        overrides = dict(PYTHONPATH='/home/exouser/ljy/dlm/src:/home/exouser/ljy/dlm',
            HF_HUB_OFFLINE='1', TORCHINDUCTOR_DETERMINISTIC='1',
            VLLM_USE_V2_MODEL_RUNNER='1', VLLM_ENABLE_V1_MULTIPROCESSING='0',
            MAX_MODEL_LEN=str(protocol['runtime_settings'][args.suite]['max_model_len']),
            RESIDENT_SPARSE_EXECUTION='1', RESIDENT_V31_PAGED_REFRESH='1')
        for variable, leaf in [('VLLM_CACHE_ROOT', 'vllm'), ('TRITON_CACHE_DIR', 'triton'),
                               ('TORCHINDUCTOR_CACHE_DIR', 'inductor'), ('CUDA_CACHE_PATH', 'cuda')]:
            overrides[variable] = str(root / 'cache' / args.suite / leaf)
        env.update(overrides)
        command = ['timeout', '--signal=TERM', '--kill-after=15s', '14400',
            '/home/exouser/dyh/v31_env_20261005/bin/python', str(root / 'code/e2e_worker.py'),
            '--root', str(root), '--arm', arm, '--suite', args.suite, '--stage', 'panel']
        started = time.time()
        with (parent / 'launch_receipt.json').open('x') as f:
            json.dump(dict(command=command, environment_overrides=overrides,
                source_checkpoint=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                protocol_sha256=hashlib.sha256((root / 'e2e_protocol.json').read_bytes()).hexdigest(),
                gpu_idle_before=True, started_unix=started, campaign_budget=None), f, indent=2)
        log = root / 'private/logs' / args.suite / (arm + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('x') as f:
            result = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT)
        receipt = dict(returncode=result.returncode, allocated_worker_seconds=time.time()-started,
            status='worker_completed' if result.returncode == 0 else 'failed_preserved', arm=arm, suite=args.suite)
        with (parent / 'exit_receipt.json').open('x') as f:
            json.dump(receipt, f, indent=2)
        print(json.dumps(receipt), flush=True)
        # A failure is preserved and requires a new attempt; never auto-retry it.
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
