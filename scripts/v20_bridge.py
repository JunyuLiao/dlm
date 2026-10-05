"""Bounded same-input v20 cross-host native-adaptive bridge (seven or eight calls).

``run`` uses block 0's frozen RULER question, seed 101, once per main arm.
The raw receipt remains private. A redacted JSONL row is flushed after each
execution; failure marks all remaining arms not_run and never retries them.
``compare`` is CPU-only and checks paired outputs/work/termination across hosts.
This bridge is separate from the 700 scheduled panel executions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from scripts.v20_bind import atomic_new, sha_bytes


ARMS = ('D_native', 'D_matched', 'T_scope', 'M1_R1_A8_current_output',
        'M3_R2_A8_current_output', 'M3_R3_A8_current_output', 'B_A8_matched')
HISTORICAL = 'G75L30_nativeQ128'


def bridge_target(protocol):
    entry = protocol['block_assignments']['0']
    if entry['dataset'] != 'ruler4k' or entry['seed'] != 101:
        raise ValueError('frozen bridge target is not first84 RULER seed101')
    if entry['id'] not in protocol['ids']['ruler4k']:
        raise ValueError('bridge target absent from frozen RULER IDs')
    return entry['id'], 101


def source_value_hash(config):
    """Path-independent digest; every source/binary hash value remains visible."""
    pairs = sorted((Path(path).name, digest) for path, digest in config['source_hashes'].items())
    return sha_bytes(json.dumps(pairs, separators=(',', ':')).encode())


def append_row(path, row):
    data = (json.dumps(row, sort_keys=True, default=str) + '\n').encode()
    with Path(path).open('ab') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def failure_tail(arms, failed_arm, common):
    found = False
    result = []
    for arm in arms:
        if arm == failed_arm:
            found = True
            continue
        if found:
            result.append(dict(common, arm=arm, status='not_run', reason='prior_bridge_failure'))
    return result


def compare_rows(left, right):
    """Exact paired bridge flags, with source identity compared separately."""
    required = ('protocol_sha256', 'binding_sha256', 'id', 'seed', 'model_revision')
    if any(left.get(k) != right.get(k) for k in required):
        raise ValueError('different bridge input/protocol/model identity')
    def selected(rows):
        out = {}
        for row in rows['rows']:
            if row['arm'] in out:
                raise ValueError('duplicate bridge arm row')
            out[row['arm']] = row
        return out
    a, b = selected(left), selected(right)
    if set(a) != set(b):
        raise ValueError('bridge arm coverage differs')
    fields = ('completion_token_hash', 'per_canvas_calls', 'termination',
              'router_phase_evidence')
    details = {}
    for arm in a:
        x, y = a[arm], b[arm]
        exact = {field: x.get(field) == y.get(field) for field in fields}
        details[arm] = dict(status_a=x['status'], status_b=y['status'],
                            exact=exact, all_exact=x['status'] == y['status'] == 'ok' and all(exact.values()),
                            source_value_hash_match=x.get('source_value_hash') == y.get('source_value_hash'),
                            config_fingerprint_match=x.get('config_fingerprint') == y.get('config_fingerprint'))
    return dict(schema='v20_bridge_compare_v1', id=left['id'], seed=left['seed'],
                hosts=[left['host'], right['host']], per_arm=details,
                exact_all=all(d['all_exact'] for d in details.values()),
                note='full token hash/calls/termination/router phases must pair; source/config identity reported separately')


def read_summary(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows or rows[0].get('event') != 'start':
        raise ValueError('bridge summary missing start')
    return {**rows[0], 'rows': [r for r in rows[1:] if r.get('event') == 'arm']}


def run(args):
    import fcntl
    from scripts.v20_run import router_phase_evidence, validate_inputs
    uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                   text=True).splitlines()[0].strip()
    protocol, binding, rows, configs = validate_inputs(args.protocol, args.binding,
                                                        args.manifests_dir, args.host, uuid,
                                                        stage='initial')
    id_, seed = bridge_target(protocol)
    arms = list(ARMS)
    if args.include_historical:
        _, _, _, historical = validate_inputs(args.protocol, args.binding,
                                               args.manifests_dir, args.host, uuid,
                                               stage='historical')
        configs['ruler4k'][HISTORICAL] = historical['ruler4k'][HISTORICAL]
        arms.append(HISTORICAL)
    if args.private.exists() or args.summary.exists() or args.lock.exists():
        raise FileExistsError('bridge private/summary/lock already exists; immutable first bridge')
    args.private.mkdir(parents=True, exist_ok=False)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    common = dict(event='arm', id=id_, seed=seed, host=args.host, gpu_uuid=uuid,
                  model_revision=protocol['model_revision'],
                  protocol_sha256=sha_bytes(args.protocol.read_bytes()),
                  binding_sha256=sha_bytes(args.binding.read_bytes()))
    with args.lock.open('x', encoding='utf-8') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        append_row(args.summary, dict(common, event='start', arms=arms,
                                      stage='bridge_only_outside_panel', target_block=0,
                                      bridge_source_sha256=sha_bytes(Path(__file__).read_bytes())))
        try:
            from scripts.v9_clean_request_timing import redacted
            from experiments.numerical_qk_reuse.runner import _one
            from dllm.models import create_adapter
            import torch
            adapter = create_adapter('diffusion_gemma', binding['host_models'][args.host],
                                     device='cuda', precision='bfloat16',
                                     revision=protocol['model_revision']).load()
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except BaseException as exc:
            append_row(args.summary, dict(common, arm=arms[0], status='failed',
                                          error_type=type(exc).__name__, reason='model_load'))
            for arm in arms[1:]:
                append_row(args.summary, dict(common, arm=arm, status='not_run',
                                              reason='model_load_failure'))
            return 'failed'
        class Timeout(Exception):
            pass
        def alarm(*_):
            raise Timeout()
        signal.signal(signal.SIGALRM, alarm)
        for arm in arms:
            config = configs['ruler4k'][arm]
            base = dict(common, arm=arm, condition=config['condition'],
                        config_fingerprint=config['fingerprint'],
                        source_value_hash=source_value_hash(config),
                        policy_sha256=config['policy_sha256'],
                        scope=config.get('v20_scope'),
                        model_metadata_hashes=config['model_metadata_hashes'])
            started = time.perf_counter()
            try:
                signal.alarm(args.timeout)
                receipt = _one(adapter, rows[id_], seed, config)
                signal.alarm(0)
                raw = args.private / f'{arm}.json'
                atomic_new(raw, receipt)
                public = redacted(receipt)
                evidence = router_phase_evidence(arm, receipt.get('counters'))
                if evidence is None:
                    raise RuntimeError('router phase evidence absent')
                row = dict(base, status='ok', raw_receipt=str(raw),
                           outer_wall_s=time.perf_counter()-started,
                           router_phase_evidence=evidence,
                           completion_token_hash=public['completion_token_hash'],
                           per_canvas_calls=public['per_canvas_calls'],
                           termination=public['termination'],
                           decoder_calls=public['decoder_calls'],
                           output_tokens=public['output_tokens'],
                           counters=public['counters'])
                append_row(args.summary, row)
            except BaseException as exc:
                signal.alarm(0)
                append_row(args.summary, dict(base, status='failed', error_type=type(exc).__name__,
                                              outer_wall_s=time.perf_counter()-started))
                for omitted in failure_tail(arms, arm, common):
                    append_row(args.summary, omitted)
                return 'failed'
    return 'complete'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    for name in ('protocol', 'binding', 'manifests-dir', 'private', 'summary', 'lock'):
        r.add_argument('--'+name, type=Path, required=True)
    r.add_argument('--host', required=True)
    r.add_argument('--include-historical', action='store_true')
    r.add_argument('--timeout', type=int, default=900)
    c = sub.add_parser('compare')
    c.add_argument('--summary-a', type=Path, required=True)
    c.add_argument('--summary-b', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    if a.command == 'run':
        print(run(a), flush=True)
    else:
        atomic_new(a.out, compare_rows(read_summary(a.summary_a), read_summary(a.summary_b)))


if __name__ == '__main__':
    main()
