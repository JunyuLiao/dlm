"""v13 seed-safe scheduled natural-request driver (replaces the seed-42-only v10 driver
for multi-seed studies; v10/v12 files are left untouched).

Identity:
  arm_config_hash = sha256 of the effective arm config WITHOUT per-run volatile
                    fields (ids, seeds, phase, manifest, fingerprints)
  cell_id         = sha256(protocol_id, model_revision, arm_config_hash, question_id, generation_seed)[:24]
  execution key   = (cell_id, role, repeat)   role in {attempt0, warm}
The GENERATION seed of the cell is threaded into ``runner._one`` (which puts it in
``GenerationRequest.seed`` -> ``adapter.generate`` -> ``seed_everything``), the
ledger, the private receipt path and every public row. Seeds of the same arm and
question are different cells: never timing repeats of each other.

Resume: the ledger's start events must carry the same protocol_id / arm hashes /
model revision / source hashes / private root, and every recorded execution key
must belong to the frozen schedule (old seed-42 ledgers are refused). Every
scheduled execution runs exactly once; a failure is recorded and never silently
retried. A CUDA/device error stops the worker (exit 3) so a clean process resumes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

VOLATILE = ('ids', 'seeds', 'phase', 'manifest', 'manifest_sha256', 'fingerprint', 'v10_arm')


def sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


REPO = str(Path(__file__).resolve().parents[1])


def arm_config_hash(config: dict) -> str:
    """Source hash KEYS carry the checkout/deploy root; only the relative path and
    the content hash identify the code, so the root is normalized before hashing."""
    stable = {k: v for k, v in config.items() if k not in VOLATILE}
    if isinstance(stable.get('source_hashes'), dict):
        stable['source_hashes'] = {k.replace(REPO, '<repo>'): v for k, v in stable['source_hashes'].items()}
    return sha(stable)


def cell_id(protocol_id: str, model_revision: str, arm_hash: str, question_id: str, seed: int) -> str:
    return sha([protocol_id, model_revision, arm_hash, question_id, int(seed)])[:24]


def execution_key(entry: dict) -> str:
    return f"{entry['cell_id']}:{entry['role']}:{entry['repeat']}"


def plan_schedule(protocol_id, model_revision, arm_hashes: dict, ids: list, seeds: list, schedule_seed: int,
                  warm_repeats: int = 1) -> list[dict]:
    """Blocks (question, seed) in an order drawn from an independent RNG. Within a
    block: every arm's attempt 0 (cyclic Latin rotation by block position), then the
    warm repeats in the REVERSED arm order. Nothing depends on answers or timings."""
    rng = random.Random(schedule_seed)
    blocks = [(q, s) for q in ids for s in seeds]
    rng.shuffle(blocks)
    arms = sorted(arm_hashes)
    schedule = []
    for position, (q, s) in enumerate(blocks):
        k = position % len(arms)
        order = arms[k:] + arms[:k]
        for role, arm_order in (('attempt0', order),) + tuple(('warm', order[::-1]) for _ in range(warm_repeats)):
            for arm in arm_order:
                repeat = 0 if role == 'attempt0' else sum(1 for e in schedule if e['arm'] == arm and e['id'] == q
                                                           and e['seed'] == s and e['role'] == 'warm') + 1
                schedule.append(dict(index=len(schedule), block=position, arm=arm, id=q, seed=int(s), role=role,
                                     repeat=repeat, cell_id=cell_id(protocol_id, model_revision, arm_hashes[arm], q, s)))
    return schedule


def receipt_path(private: Path, entry: dict) -> Path:
    return Path(private) / 'cells' / entry['cell_id'] / f"{entry['role']}{entry['repeat']}.json"


def validate_resume(events: list[dict], identity: dict, schedule: list[dict]) -> set:
    """Returns completed execution keys; raises on ANY incompatibility."""
    keys = {execution_key(e) for e in schedule}
    done = set()
    for event in events:
        if event.get('event') == 'start':
            for field in ('protocol_id', 'model_revision', 'arm_hashes', 'source_hashes', 'private_root'):
                if event.get(field) != identity[field]:
                    raise RuntimeError(f'resume refused: {field} differs from the frozen protocol/run identity')
        elif event.get('event') == 'run':
            key = event.get('execution_key')
            if key not in keys:
                raise RuntimeError(f'resume refused: ledger contains an execution outside this schedule: {key}')
            if key in done:
                raise RuntimeError(f'duplicate execution in ledger: {key}')
            done.add(key)
    return done


def warm_acceptance(attempt0: dict | None, warm: dict) -> dict:
    """A warm row is accepted only if it ran, compiled nothing new, and reproduced
    its own attempt 0 (tokens, per-canvas calls, termination). Rejected rows stay raw."""
    reasons = []
    if not warm.get('ok'):
        reasons.append('execution_failed')
    if warm.get('triton_misses'):
        reasons.append('new_compilation')
    if attempt0 is None or not attempt0.get('ok'):
        reasons.append('attempt0_missing_or_failed')
    else:
        for field, name in (('completion_token_hash', 'tokens'), ('per_canvas_calls', 'calls'), ('termination', 'termination')):
            if warm.get(field) != attempt0.get(field):
                reasons.append(f'{name}_mismatch')
    return dict(accepted=not reasons, reasons=reasons)


def is_device_error(message: str) -> bool:
    text = message.lower()
    return any(s in text for s in ('cuda error', 'device-side', 'illegal memory', 'cudaerror', 'nccl', 'cublas_status'))


def run_schedule(schedule, *, done: set, execute_one: Callable, rows: dict, configs: dict, ledger_append: Callable,
                 private: Path, save_receipt: Callable, max_executions=None, stop_flag: Callable = lambda: False):
    """Executes every not-yet-done entry exactly once, in frozen order."""
    ran = 0
    first = {}
    for entry in schedule:
        key = execution_key(entry)
        if key in done:
            continue
        if stop_flag() or (max_executions is not None and ran >= max_executions):
            return 'stopped'
        outcome = execute_one(rows[entry['id']], entry['seed'], configs[entry['arm']], entry)
        record = dict(event='run', execution_key=key, **{k: entry[k] for k in ('index', 'block', 'arm', 'id', 'seed', 'role', 'repeat', 'cell_id')},
                      **outcome['record'])
        if outcome.get('receipt') is not None and entry['role'] == 'attempt0':
            path = receipt_path(private, entry)
            save_receipt(path, outcome['receipt'])
            record['private_receipt'] = str(path)
        ledger_append(record)
        done.add(key)
        ran += 1
        if outcome.get('fatal'):
            return 'device_error'
    return 'complete'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--max-executions', type=int, default=None)
    parser.add_argument('--warmup-kernels', action='store_true')
    parser.add_argument('--only-keys', nargs='*', default=None, help='restrict to these execution keys (compat checks)')
    args = parser.parse_args()
    import torch
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse.runner import GOLD_FIELDS, _atomic, _one, _rows
    from scripts.v9_clean_request_timing import append, disk_cache_entries, redacted
    from scripts.v10_request_runs import arm_config, import_identity
    from triton.runtime.jit import JITFunction
    protocol = json.loads(args.protocol.read_text())
    manifest_rows = {r['id']: {k: v for k, v in r.items() if k not in GOLD_FIELDS} for r in _rows(args.manifest)}
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
    if args.warmup_kernels:
        from experiments.numerical_qk_reuse.cached_executor import warmup_generic
        n = len(compiles)
        seconds, launches = warmup_generic(configs[next(a for a in configs if configs[a]['condition'] in ('global_M1', 'global_M3', 'global_B8'))]['policy'])
        append(args.ledger, dict(event='kernel_warmup', seconds=seconds, launches=launches, misses=len(compiles) - n))

    class Timeout(Exception):
        pass

    def alarm(*_):
        raise Timeout()
    signal.signal(signal.SIGALRM, alarm)
    first_by_cell = {e['cell_id']: e for e in events if e.get('event') == 'run' and e.get('role') == 'attempt0'}

    def execute_one(row, seed, config, entry):
        n, disk_before = len(compiles), disk_cache_entries()
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
                      generation_seed=seed, deploy=protocol.get('deploy'))
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
                          save_receipt=_atomic, max_executions=args.max_executions)
    JITFunction.cache_hook = JITFunction.compiled_hook = None
    append(args.ledger, dict(event='worker_end', status=status, when=time.time(), process_s=time.perf_counter() - started))
    print('STATUS', status, flush=True)
    sys.exit({'complete': 0, 'stopped': 0, 'device_error': 3}[status])


if __name__ == '__main__':
    main()
