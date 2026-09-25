"""v10 scheduled natural-request executions (extends the v9 clean driver).

One process runs an explicit schedule of (arm, id) executions through
``runner._one`` (production adapter call, own seed_everything(42), native
adaptive stopping, thinking ON, 8192, EOS, diagnostic off). Arms:

  D   native_dense (unbound)          T   fresh Junyu T (verified extension)
  <name>  M1 preqk current output with {selector, kernel_variant, telemetry}

The first execution of an (arm, id) in a ledger is attempt 0 (quality; raw
receipt kept privately); later ones are timing repeats checked against it.
Triton in-memory misses are logged with durations via cache_hook /
compiled_hook (they fire only on a miss, so the warm path is untouched).
A per-execution SIGALRM timeout records a failure row and continues.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.v9_clean_request_timing import append, disk_cache_entries, redacted, token_hash

LIBRARY = '/home/exouser/dyh/numerical_qk_reuse_native_20260924/build_cp1/value_direction_db080045f7a5fbce.so'
TORCH_LIBRARY = ('/home/exouser/dyh/numerical_qk_reuse_native_20260924/build_cp1/torch_4c65c048754f9fb7/'
                 'value_direction_torch_4c65c048754f9fb7.so')
PLUGIN = 'experiments.numerical_qk_reuse.integration:install'


class Timeout(Exception):
    pass


def arm_config(args, arm: dict) -> dict:
    from experiments.numerical_qk_reuse.runner import _config
    condition = arm['condition']
    ns = SimpleNamespace(
        condition=condition, phase=args.phase, ids=args.ids, seeds=[42], manifest=args.manifest,
        policy=args.policy, policy_name='T_s50', model=args.model, revision=args.revision,
        decision_interval=arm.get('decision_interval', 1), score_refresh_period=8, support='legacy_junyu_mask',
        output_mode='historical_route_preqk_current_output',
        selector=arm.get('selector', 'legacy_recompute'), selector_layers='local',
        kernel_variant=arm.get('kernel_variant', 'static'),
        library=Path(LIBRARY) if condition == 'fresh_junyu_T' else None,
        torch_library=Path(TORCH_LIBRARY) if condition == 'fresh_junyu_T' else None,
        plugin=PLUGIN if condition in ('M1', 'M3') else None, diagnostic=False, timing_events=False,
        extra_source=[])
    config = _config(ns)
    config['telemetry'] = arm.get('telemetry', 'full')
    for key in ('guard_mode', 'consumer', 'support_build', 'collect'):   # v11 arm keys
        if key in arm:
            config[key] = arm[key]
    if arm.get('support_build'):
        identity = json.loads(Path(arm['support_build']).read_text())
        config['support_binary'] = dict(key=identity['key'], kernel_sha256=identity['kernel_sha256'],
                                        bridge_sha256=identity['bridge_sha256'], sources=identity['sources'])
    config['v10_arm'] = arm['name']
    config['fingerprint'] = hashlib.sha256(json.dumps({k: v for k, v in config.items() if k != 'fingerprint'},
                                                      sort_keys=True).encode()).hexdigest()
    return config


def import_identity() -> dict:
    import transformers
    import triton
    return dict(executable=sys.executable, python=sys.version.split()[0],
                torch=torch.__version__, torch_file=torch.__file__, triton=triton.__version__,
                triton_file=triton.__file__, transformers=transformers.__version__,
                transformers_file=transformers.__file__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
                triton_cache_dir=os.environ.get('TRITON_CACHE_DIR', '~/.triton/cache (default)'),
                pythonnousersite=os.environ.get('PYTHONNOUSERSITE'), sys_path_head=sys.path[:6])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', required=True)
    parser.add_argument('--arms', required=True, help='JSON list of arm dicts {name, condition, ...}')
    parser.add_argument('--schedule', required=True, help='JSON list of [arm_name, id]')
    parser.add_argument('--phase', default='v10')
    parser.add_argument('--warmup-kernels', action='store_true', help='generic startup warmup before requests')
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    args = parser.parse_args()
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse.runner import GOLD_FIELDS, _atomic, _one, _rows, _selected
    import triton
    from triton.runtime.jit import JITFunction
    arms = {arm['name']: arm for arm in json.loads(args.arms)}
    schedule = json.loads(args.schedule)
    rows = {r['id']: r for r in _selected([{k: v for k, v in row.items() if k not in GOLD_FIELDS}
                                           for row in _rows(args.manifest)], args.ids, args.phase)}
    configs = {name: arm_config(args, arm) for name, arm in arms.items()}
    for name, config in configs.items():
        _atomic(args.private / 'configs' / f'{args.phase}.{name}.json', config)
    seen = {}
    if args.ledger.exists():
        for line in args.ledger.read_text().splitlines():
            event = json.loads(line)
            if event.get('event') == 'run' and event.get('attempt0') and event.get('ok'):
                seen[(event['label'], event['id'])] = event
    compiles = []
    t0 = time.perf_counter()

    def before(**kw):
        compiles.append(dict(fn=kw['fn'].name, t=time.perf_counter() - t0))
        return False

    def after(**kw):
        compiles[-1]['seconds'] = time.perf_counter() - t0 - compiles[-1]['t']
        return False

    JITFunction.cache_hook, JITFunction.compiled_hook = before, after
    append(args.ledger, dict(event='start', pid=os.getpid(), pgid=os.getpgid(0), when=time.time(),
                             identity=import_identity(), disk_cache_entries=disk_cache_entries(),
                             fingerprints={k: c['fingerprint'] for k, c in configs.items()},
                             source_hashes=next(iter(configs.values()))['source_hashes'],
                             schedule=schedule))
    started = time.perf_counter()
    adapter = create_adapter('diffusion_gemma', str(args.model), device='cuda',
                             precision='bfloat16', revision=args.revision).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    append(args.ledger, dict(event='model_loaded', seconds=time.perf_counter() - started))
    if args.warmup_kernels:
        from experiments.numerical_qk_reuse.cached_executor import warmup_generic
        n = len(compiles)
        seconds, launches = warmup_generic(next(c for c in configs.values() if c['condition'] in ('M1', 'M3'))['policy'])
        append(args.ledger, dict(event='kernel_warmup', seconds=seconds, launches=launches,
                                 misses=len(compiles) - n,
                                 compile_s=sum(c.get('seconds', 0) for c in compiles[n:]),
                                 disk_cache_entries=disk_cache_entries()))

    def alarm(*_):
        raise Timeout()
    signal.signal(signal.SIGALRM, alarm)
    for index, (name, id_) in enumerate(schedule):
        config, row = configs[name], rows[id_]
        n, disk_before = len(compiles), disk_cache_entries()
        torch.cuda.reset_peak_memory_stats()
        outer = time.perf_counter()
        signal.alarm(args.timeout)
        try:
            receipt = _one(adapter, row, 42, config)
            error = None
        except Timeout:
            receipt, error = None, f'timeout>{args.timeout}s'
        except Exception as exc:                         # recorded, never hidden
            receipt, error = None, f'{type(exc).__name__}: {exc}'[:500]
        finally:
            signal.alarm(0)
        outer_wall = time.perf_counter() - outer
        misses = compiles[n:]
        record = dict(event='run', index=index, label=name, id=id_, ok=receipt is not None, error=error,
                      outer_wall_s=outer_wall, triton_misses=len(misses),
                      triton_miss_s=sum(c.get('seconds', 0) for c in misses),
                      triton_disk_entries_added=disk_cache_entries() - disk_before,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      kernel_variant=config.get('kernel_variant'), telemetry=config.get('telemetry'),
                      fingerprint=config['fingerprint'])
        if receipt is not None:
            record.update(redacted(receipt))
            record['label'] = name
            first = seen.get((name, id_))
            record['attempt0'] = first is None
            if first is None:
                seen[(name, id_)] = record
                destination = (args.private / 'attempt0' / name /
                               f"{hashlib.sha256(id_.encode()).hexdigest()[:20]}.json")
                _atomic(destination, receipt)
                record['private_receipt'] = str(destination)
            else:
                record['matches_own_attempt0'] = dict(
                    tokens=record['completion_token_hash'] == first['completion_token_hash'],
                    calls=record['per_canvas_calls'] == first['per_canvas_calls'],
                    termination=record['termination'] == first['termination'])
        else:
            record['attempt0'] = (name, id_) not in seen
        append(args.ledger, record)
        print(json.dumps({k: record.get(k) for k in ('label', 'id', 'ok', 'attempt0', 'api_wall_s',
                                                     'decoder_calls', 'triton_misses', 'error')}), flush=True)
    JITFunction.cache_hook = JITFunction.compiled_hook = None
    append(args.ledger, dict(event='done', when=time.time(), disk_cache_entries=disk_cache_entries(),
                             process_s=time.perf_counter() - started))
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
