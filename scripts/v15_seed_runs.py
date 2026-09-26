"""v15 driver: the v13 seed-safe scheduled request driver (identity, frozen schedule, validated
resume, exactly-once executions, first output immutable) with three v15 additions:
  * the generation manifest must contain NO gold field (the worker never loads answers);
  * every execution records shared objects newly mapped into the process (CUDA extension /
    library loads, compiled launcher modules) next to the Triton JIT compile hook;
  * warm acceptance additionally rejects a warm row that mapped a new shared object.
Everything else is imported unchanged from scripts/v13_seed_runs.py.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from scripts.v13_seed_runs import (arm_config_hash, execution_key, is_device_error, run_schedule, validate_resume,
                                   warm_acceptance as v13_warm_acceptance)


def mapped_shared_objects() -> set:
    try:
        with open('/proc/self/maps') as fh:
            return {line.split()[-1] for line in fh if '.so' in line.split()[-1]}
    except OSError:
        return set()


def warm_acceptance(attempt0, warm):
    out = v13_warm_acceptance(attempt0, warm)
    if warm.get('new_shared_objects'):
        out['reasons'].append('new_shared_object')
        out['accepted'] = False
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--max-executions', type=int, default=None)
    parser.add_argument('--deadline-epoch', type=float, default=None,
                        help='stop cleanly (after the current execution) once this UNIX time has passed; the ledger stays resumable')
    parser.add_argument('--only-keys', nargs='*', default=None, help='restrict to these execution keys (compat checks)')
    args = parser.parse_args()
    import torch
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse.runner import GOLD_FIELDS, _atomic, _one, _rows
    from scripts.v9_clean_request_timing import append, disk_cache_entries, redacted
    from scripts.v10_request_runs import arm_config, import_identity
    from triton.runtime.jit import JITFunction
    protocol = json.loads(args.protocol.read_text())
    raw_rows = _rows(args.manifest)
    leaked = sorted({k for r in raw_rows for k in r if k in GOLD_FIELDS})
    if leaked:
        raise RuntimeError(f'generation manifest contains gold fields {leaked}; the worker must not load answers')
    manifest_rows = {r['id']: r for r in raw_rows}
    ns = SimpleNamespace(phase=protocol['protocol_id'], ids=protocol['ids'], manifest=args.manifest,
                         policy=Path(protocol['policy_file']), model=Path(protocol['model_path']),
                         revision=protocol['model_revision'], seeds=protocol['seeds'])
    configs = {name: arm_config(ns, arm) for name, arm in protocol['arms'].items()}
    for name, config in configs.items():
        if arm_config_hash(config) != protocol['arm_hashes'][name]:
            raise RuntimeError(f'arm {name}: effective config hash differs from the frozen protocol')
    schedule = protocol['schedule']
    if args.only_keys is not None:
        schedule = [e for e in schedule if execution_key(e) in set(args.only_keys)]
    source_hashes = next(iter(configs.values()))['source_hashes']
    identity = dict(protocol_id=protocol['protocol_id'], model_revision=protocol['model_revision'],
                    arm_hashes=protocol['arm_hashes'], source_hashes=source_hashes, private_root=str(args.private.resolve()))
    events = [json.loads(l) for l in args.ledger.read_text().splitlines() if l.strip()] if args.ledger.exists() else []
    done = validate_resume(events, identity, protocol['schedule'])
    for name, config in configs.items():
        _atomic(args.private / 'configs' / f'{name}.json', config)
    compiles = []
    t0 = time.perf_counter()

    def before(**kw):
        compiles.append(dict(fn=kw['fn'].name, t=time.perf_counter() - t0))
        return False

    def after(**kw):
        compiles[-1]['seconds'] = time.perf_counter() - t0 - compiles[-1]['t']
        return False
    JITFunction.cache_hook, JITFunction.compiled_hook = before, after
    append(args.ledger, dict(event='start', pid=os.getpid(), when=time.time(), import_identity=import_identity(),
                             resumed_executions=len(done), disk_cache_entries=disk_cache_entries(), **identity))
    started = time.perf_counter()
    adapter = create_adapter('diffusion_gemma', protocol['model_path'], device='cuda', precision='bfloat16',
                             revision=protocol['model_revision']).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    append(args.ledger, dict(event='model_loaded', seconds=time.perf_counter() - started))

    class Timeout(Exception):
        pass

    def alarm(*_):
        raise Timeout()
    signal.signal(signal.SIGALRM, alarm)
    first_by_cell = {e['cell_id']: e for e in events if e.get('event') == 'run' and e.get('role') == 'attempt0'}

    def execute_one(row, seed, config, entry):
        n, disk_before, so_before = len(compiles), disk_cache_entries(), mapped_shared_objects()
        torch.cuda.reset_peak_memory_stats()
        outer = time.perf_counter()
        signal.alarm(args.timeout)
        fatal, receipt, error = False, None, None
        try:
            receipt = _one(adapter, row, seed, config)
        except Timeout:
            error = f'timeout>{args.timeout}s'
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'[:500]
            fatal = is_device_error(error)
        finally:
            signal.alarm(0)
        if error and not fatal:
            try:
                torch.cuda.synchronize()
            except Exception as exc:                         # context unreliable -> stop this worker
                error += f' | sync after failure: {exc}'[:200]
                fatal = True
        record = dict(ok=receipt is not None, error=error, outer_wall_s=time.perf_counter() - outer,
                      triton_misses=len(compiles) - n, triton_miss_s=sum(c.get('seconds', 0) for c in compiles[n:]),
                      triton_disk_entries_added=disk_cache_entries() - disk_before,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), arm_config_hash=protocol['arm_hashes'][entry['arm']],
                      generation_seed=seed, deploy=protocol.get('deploy'),
                      new_shared_objects=sorted(mapped_shared_objects() - so_before))
        if receipt is not None:
            if receipt['seed'] != seed:
                raise AssertionError('generation seed was not threaded into the request')
            record.update(redacted(receipt))
            c = receipt.get('counters') or {}
            if 'score_refresh_calls' in c:
                record['phases'] = dict(anchor=c['score_refresh_calls'],
                                        reselect=c['decision_refresh_calls'] - c['score_refresh_calls'],
                                        held=c['held_decision_calls'], routed_calls=c['attention_calls'])
            if entry['role'] == 'warm':
                record['acceptance'] = warm_acceptance(first_by_cell.get(entry['cell_id']), record)
            else:
                first_by_cell[entry['cell_id']] = dict(record, ok=True)
        elif entry['role'] == 'attempt0':
            first_by_cell[entry['cell_id']] = dict(record)
        print(json.dumps({k: record.get(k) for k in ('ok', 'api_wall_s', 'decoder_calls', 'termination', 'error')} |
                         {k: entry[k] for k in ('index', 'arm', 'id', 'seed', 'role')}), flush=True)
        return dict(record=record, receipt=receipt, fatal=fatal)

    status = run_schedule(schedule, done=done, execute_one=execute_one, rows=manifest_rows, configs=configs,
                          ledger_append=lambda r: append(args.ledger, r), private=args.private,
                          save_receipt=_atomic, max_executions=args.max_executions,
                          stop_flag=lambda: args.deadline_epoch is not None and time.time() > args.deadline_epoch)
    JITFunction.cache_hook = JITFunction.compiled_hook = None
    append(args.ledger, dict(event='worker_end', status=status, when=time.time(), process_s=time.perf_counter() - started))
    print('STATUS', status, flush=True)
    sys.exit({'complete': 0, 'stopped': 0, 'device_error': 3}[status])


if __name__ == '__main__':
    main()
