"""Bounded two-host v18 bridge: two calibration prompts, seed101, three arms.

Freeze/run separately on each host; compare reads private receipts and writes only
redacted qualification evidence. Neither phase loads gold or scores answers.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from collections import Counter
from pathlib import Path

from scripts.v13_seed_runs import (arm_config_hash, execution_key, is_device_error, plan_schedule,
                                   receipt_path, run_schedule, validate_resume)
from scripts.v18_evaluate import eval_config, native_phase_evidence, write_immutable_json
from scripts.v18_protocol import sha, verify_prompt_tokens

ARMS = ('D_native', 'U50', 'T50')
SEED = 101


def inventory_identity(path):
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    tensors = value.get('tensors')
    if not isinstance(tensors, dict) or len(tensors) != 1047:
        raise ValueError('bridge requires complete 1047-tensor inventory')
    if any(not isinstance(t.get('sha256'), str) or len(t['sha256']) != 64 for t in tensors.values()):
        raise ValueError('tensor inventory lacks exact SHA-256 fields')
    return dict(path=str(Path(path).resolve()), file_sha256=sha(raw),
                index_sha256=value.get('index_sha256'), count=len(tensors),
                tensor_identity_sha256=sha(json.dumps(tensors, sort_keys=True)))


def math_identity(config):
    return {k: config.get(k) for k in ('frontier_arm', 'method', 'target', 'fast_t', 'support_geometry',
                                       'policy', 'm_ref', 'beta', 'gamma', 'thinking', 'max_new_tokens',
                                       'revision', 'dtype', 'timing_events')}


def trajectory_identity(receipt):
    """Only deterministic generation facts; reject absent stop/call evidence."""
    tokens = receipt.get('completion_tokens')
    canvases = receipt.get('per_canvas')
    if not isinstance(tokens, list) or not tokens or not isinstance(canvases, list) or not canvases:
        raise ValueError('missing completion tokens or canvas evidence')
    if not isinstance(receipt.get('prompt_token_hash'), str) or not receipt['prompt_token_hash']:
        raise ValueError('missing prompt-token hash')
    if not isinstance(receipt.get('termination_reason'), str) or not receipt['termination_reason']:
        raise ValueError('missing termination')
    normalized = []
    for canvas in canvases:
        steps = canvas.get('schedule_steps')
        if (not isinstance(steps, list) or not steps or any(type(s) is not int for s in steps)
                or any(x <= y for x, y in zip(steps, steps[1:]))
                or type(canvas.get('decoder_calls')) is not int
                or canvas['decoder_calls'] != len(steps)
                or type(canvas.get('native_stop_final_call')) is not bool
                or type(canvas.get('iteration_cap_final_call')) is not bool):
            raise ValueError('missing or invalid per-canvas calls/schedule/stop evidence')
        normalized.append(dict(decoder_calls=len(steps), schedule_steps=steps,
                               native_stop_final_call=canvas['native_stop_final_call'],
                               iteration_cap_final_call=canvas['iteration_cap_final_call']))
    if type(receipt.get('total_decoder_calls')) is not int or receipt['total_decoder_calls'] != sum(c['decoder_calls'] for c in normalized):
        raise ValueError('total decoder calls disagree with canvases')
    return dict(completion_tokens=tokens, prompt_token_hash=receipt['prompt_token_hash'],
                termination_reason=receipt['termination_reason'],
                total_decoder_calls=receipt['total_decoder_calls'], per_canvas=normalized)


def freeze_bridge(cal_manifest, calibration, policy_file, model, library, torch_library,
                  tensor_inventory, out, host, gpu_uuid):
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse.runner import _rows
    rows = _rows(cal_manifest, allow_task_budgets=True, allow_thinking_off=True)
    if len(rows) != 26 or any(r.get('thinking') is not False for r in rows):
        raise ValueError('corrected gold-free RULER calibration manifest required')
    if any(k in r for r in rows for k in ('outputs', 'answer', 'expected', 'expected_answer', 'gold')):
        raise ValueError('gold in bridge generation manifest')
    selected = rows[:2]
    tokenizer = create_adapter('diffusion_gemma', str(model), device='cpu', precision='float32',
                               revision='f7f5b7f5fa82ffc52addd066915886d497f5517b').load_tokenizer()
    verify_prompt_tokens(tokenizer, selected)
    frozen = json.loads(calibration.read_text())
    if not all(a in frozen for a in ('U50', 'T50')):
        raise ValueError('final U50/T50 calibration absent')
    inventory = inventory_identity(tensor_inventory)
    if not host or not gpu_uuid:
        raise ValueError('host/GPU identity required')
    phase = 'v18_bridge_' + sha(json.dumps([sha(cal_manifest.read_bytes()), sha(calibration.read_bytes()),
                                           host, gpu_uuid, inventory['tensor_identity_sha256']], sort_keys=True))[:12]
    draft = dict(ids=[r['id'] for r in selected], model_revision='f7f5b7f5fa82ffc52addd066915886d497f5517b',
                 generation={'thinking': False})
    configs = {arm: eval_config(arm, draft, cal_manifest, policy_file, model, library, torch_library,
                                frozen, phase) for arm in ARMS}
    hashes = {arm: arm_config_hash(config) for arm, config in configs.items()}
    schedule = plan_schedule(phase, draft['model_revision'], hashes, draft['ids'], [SEED], 2026092617,
                             warm_repeats=0)
    protocol = dict(schema='v18_bridge_protocol_v1', protocol_id=phase, status='frozen',
                    ids=draft['ids'], seeds=[SEED], arms=list(ARMS), schedule=schedule,
                    planned_executions=6, host=host, gpu_uuid=gpu_uuid,
                    model_path=str(model.resolve()), model_revision=draft['model_revision'],
                    tensor_inventory=inventory, manifest=str(cal_manifest.resolve()),
                    manifest_sha256=sha(cal_manifest.read_bytes()),
                    calibration_file=str(calibration.resolve()), calibration_sha256=sha(calibration.read_bytes()),
                    policy_file=str(policy_file.resolve()), policy_file_sha256=sha(policy_file.read_bytes()),
                    binary_sha256={'kernel': sha(library.read_bytes()), 'bridge': sha(torch_library.read_bytes())},
                    bridge_source_sha256=sha(Path(__file__).read_bytes()),
                    arm_hashes=hashes, math_identity={a: math_identity(c) for a, c in configs.items()},
                    nonindex_model_metadata_hashes={a: {k: v for k, v in c['model_metadata_hashes'].items()
                                                         if k != 'model.safetensors.index.json'}
                                                    for a, c in configs.items()},
                    source_hash_values={a: sorted(c['source_hashes'].values()) for a, c in configs.items()},
                    source_hashes=configs[ARMS[0]]['source_hashes'])
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'bridge_protocol.json'
    write_immutable_json(path, protocol)
    for arm, config in configs.items():
        write_immutable_json(out / 'configs' / f'{arm}.json', config)
    return protocol


def run_bridge(protocol_path, private, ledger, lock_path, *, deadline_epoch, timeout):
    import fcntl
    import torch
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse.runner import _atomic, _one, _rows
    from scripts.v15_seed_runs import mapped_shared_objects
    from scripts.v9_clean_request_timing import append, disk_cache_entries, redacted
    from scripts.v10_request_runs import import_identity
    from triton.runtime.jit import JITFunction

    protocol = json.loads(protocol_path.read_text())
    worker = (socket.gethostname(), subprocess.check_output(['nvidia-smi', '--query-gpu=uuid',
               '--format=csv,noheader'], text=True).splitlines()[0].strip())
    if worker != (protocol['host'], protocol['gpu_uuid']):
        raise ValueError('bridge host/GPU assignment mismatch')
    configs = {arm: json.loads((protocol_path.parent / 'configs' / f'{arm}.json').read_text()) for arm in ARMS}
    if {arm: arm_config_hash(c) for arm, c in configs.items()} != protocol['arm_hashes']:
        raise ValueError('bridge config hash drift')
    if sha(Path(__file__).read_bytes()) != protocol['bridge_source_sha256']:
        raise ValueError('bridge runner source drift')
    if inventory_identity(protocol['tensor_inventory']['path']) != protocol['tensor_inventory']:
        raise ValueError('model tensor identity drift')
    for filename, expected in ((protocol['manifest'], protocol['manifest_sha256']),
                               (protocol['calibration_file'], protocol['calibration_sha256']),
                               (protocol['policy_file'], protocol['policy_file_sha256'])):
        if sha(Path(filename).read_bytes()) != expected:
            raise ValueError('bridge input byte drift: ' + filename)
    for config in configs.values():
        for filename, expected in config['source_hashes'].items():
            if sha(Path(filename).read_bytes()) != expected:
                raise ValueError('bridge source/binary drift: ' + filename)
    rows = _rows(Path(protocol['manifest']), allow_task_budgets=True, allow_thinking_off=True)
    by_id = {r['id']: r for r in rows}
    identity = dict(protocol_id=protocol['protocol_id'], model_revision=protocol['model_revision'],
                    arm_hashes=protocol['arm_hashes'], source_hashes=configs[ARMS[0]]['source_hashes'],
                    private_root=str(private.resolve()))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        events = [json.loads(s) for s in ledger.read_text().splitlines() if s.strip()] if ledger.exists() else []
        done = validate_resume(events, identity, protocol['schedule'])
        for entry in protocol['schedule']:
            if execution_key(entry) not in done and receipt_path(private, entry).exists():
                raise ValueError('orphan bridge receipt without ledger; preserve first output')
        if len(done) == 6:
            return 'complete'
        started = time.perf_counter()
        append(ledger, dict(event='start', pid=os.getpid(), when=time.time(), host=worker[0], gpu_uuid=worker[1],
                            import_identity=import_identity(), **identity))
        adapter = create_adapter('diffusion_gemma', protocol['model_path'], device='cuda',
                                 precision='bfloat16', revision=protocol['model_revision']).load()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        compiles = []
        JITFunction.cache_hook = lambda **kw: compiles.append(time.perf_counter()) or False
        class Timeout(Exception):
            pass
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(Timeout()))
        def execute_one(row, seed, config, entry):
            n, cache_before, so_before = len(compiles), disk_cache_entries(), mapped_shared_objects()
            receipt, error = None, None
            start = time.perf_counter()
            signal.alarm(timeout)
            try:
                receipt = _one(adapter, row, seed, config)
            except Timeout:
                error = f'timeout>{timeout}s'
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'[:500]
            finally:
                signal.alarm(0)
            record = dict(ok=receipt is not None, error=error, generation_seed=seed,
                          arm_config_hash=protocol['arm_hashes'][entry['arm']],
                          triton_misses=len(compiles) - n,
                          triton_disk_entries_added=disk_cache_entries() - cache_before,
                          new_shared_objects=sorted(mapped_shared_objects() - so_before),
                          host=worker[0], gpu_uuid=worker[1], outer_wall_s=time.perf_counter() - start)
            if receipt is not None:
                record.update(redacted(receipt))
                record['phase_evidence'] = native_phase_evidence(receipt, sparse=entry['arm'] != 'D_native')
            return dict(record=record, receipt=receipt, fatal=bool(error and is_device_error(error)))
        try:
            status = run_schedule(protocol['schedule'], done=done, execute_one=execute_one,
                                  rows=by_id, configs=configs, ledger_append=lambda e: append(ledger, e),
                                  private=private, save_receipt=_atomic,
                                  stop_flag=lambda: deadline_epoch is not None and time.time() >= deadline_epoch)
        finally:
            JITFunction.cache_hook = None
        append(ledger, dict(event='worker_end', status=status, when=time.time(), host=worker[0], gpu_uuid=worker[1],
                            gpu_process_seconds=time.perf_counter() - started))
    return status


def compare_bridge(protocol_a, ledger_a, protocol_b, ledger_b, *, private_a=None, private_b=None,
                   inventory_a=None, inventory_b=None):
    protocols = [json.loads(Path(p).read_text()) for p in (protocol_a, protocol_b)]
    issues = []
    a, b = protocols
    hosts = [dict(host=p['host'], gpu_uuid=p['gpu_uuid']) for p in protocols]
    if hosts[0] == hosts[1]:
        issues.append('same_host_or_gpu')
    for field in ('ids', 'seeds', 'arms', 'manifest_sha256', 'calibration_sha256', 'policy_file_sha256',
                  'binary_sha256', 'bridge_source_sha256', 'math_identity',
                  'nonindex_model_metadata_hashes', 'source_hash_values'):
        if a[field] != b[field]:
            issues.append(field + '_mismatch')
    inventory_paths = [Path(inventory_a or a['tensor_inventory']['path']),
                       Path(inventory_b or b['tensor_inventory']['path'])]
    inventories = [json.loads(p.read_text()) for p in inventory_paths]
    for i, path, protocol in zip(inventories, inventory_paths, protocols):
        if sha(path.read_bytes()) != protocol['tensor_inventory']['file_sha256'] or i.get('index_sha256') != protocol['tensor_inventory']['index_sha256']:
            issues.append('host_inventory_receipt_mismatch')
        if sha(json.dumps(i.get('tensors'), sort_keys=True)) != protocol['tensor_inventory']['tensor_identity_sha256']:
            issues.append('inventory_receipt_mismatch')
    if any(len(i.get('tensors', {})) != 1047 for i in inventories) or inventories[0]['tensors'] != inventories[1]['tensors']:
        issues.append('model_tensor_identity_mismatch')
    index_hashes = [p['tensor_inventory']['index_sha256'] for p in protocols]
    records = []
    for protocol, ledger in zip(protocols, (ledger_a, ledger_b)):
        events = [json.loads(s) for s in Path(ledger).read_text().splitlines() if s.strip()]
        runs = [e for e in events if e.get('event') == 'run']
        expected = {(e['arm'], e['id'], e['seed']) for e in protocol['schedule']}
        if len(runs) != 6 or {(e.get('arm'), e.get('id'), e.get('seed')) for e in runs} != expected:
            issues.append('incomplete_or_duplicate_bridge_runs')
        records.append({(e['arm'], e['id'], e['seed']): e for e in runs})
    matched = 0
    for key in {(arm, rid, SEED) for arm in ARMS for rid in a['ids']}:
        x, y = (records[0].get(key), records[1].get(key))
        if not x or not y or not x.get('ok') or not y.get('ok'):
            issues.append('missing_or_failed_cell')
            continue
        paths = []
        for record, root in ((x, private_a), (y, private_b)):
            expected_name = f"{record['role']}{record['repeat']}.json"
            if Path(record['private_receipt']).name != expected_name:
                issues.append('receipt_path_identity_mismatch')
            paths.append((Path(root) / 'cells' / record['cell_id'] / expected_name) if root else Path(record['private_receipt']))
        rx, ry = (json.loads(path.read_text()) for path in paths)
        normalized = []
        for record, receipt in ((x, rx), (y, ry)):
            try:
                normalized.append(trajectory_identity(receipt))
            except ValueError:
                issues.append('missing_trajectory_evidence')
                normalized.append(None)
            if (receipt.get('id') != record['id'] or receipt.get('seed') != record['seed'] or
                    sha(json.dumps(receipt.get('completion_tokens'), separators=(',', ':'))) != record.get('completion_token_hash') or
                    receipt.get('termination_reason') != record.get('termination') or
                    [c['decoder_calls'] for c in receipt.get('per_canvas', [])] != record.get('per_canvas_calls')):
                issues.append('receipt_ledger_identity_mismatch')
        if normalized[0] is None or normalized[1] is None:
            continue
        if normalized[0] != normalized[1]:
            issues.append('trajectory_mismatch')
        elif x.get('completion_token_hash') != y.get('completion_token_hash') or x.get('termination') != y.get('termination'):
            issues.append('ledger_receipt_mismatch')
        else:
            matched += 1
    return dict(schema='v18_bridge_qualification_v1', qualified=not issues and matched == 6,
                hosts=hosts, matched_cells=matched, required_cells=6,
                model_index_sha256_by_host=index_hashes,
                repacked_index_differs=index_hashes[0] != index_hashes[1],
                model_tensor_identity_sha256=a['tensor_inventory']['tensor_identity_sha256'],
                protocol_sha256=[sha(Path(p).read_bytes()) for p in (protocol_a, protocol_b)],
                ledger_sha256=[sha(Path(p).read_bytes()) for p in (ledger_a, ledger_b)],
                issues=dict(Counter(issues)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    a = sub.add_parser('freeze')
    for name in ('cal-manifest', 'calibration', 'policy', 'model', 'library', 'torch-library',
                 'tensor-inventory', 'out', 'host', 'gpu-uuid'):
        a.add_argument('--' + name, required=True)
    b = sub.add_parser('run')
    for name in ('protocol', 'private', 'ledger', 'lock'):
        b.add_argument('--' + name, type=Path, required=True)
    b.add_argument('--deadline-epoch', type=float)
    b.add_argument('--timeout', type=int, default=900)
    c = sub.add_parser('compare')
    for name in ('protocol-a', 'ledger-a', 'protocol-b', 'ledger-b', 'out'):
        c.add_argument('--' + name, type=Path, required=True)
    for name in ('private-a', 'private-b', 'inventory-a', 'inventory-b'):
        c.add_argument('--' + name, type=Path)
    args = p.parse_args()
    if args.action == 'freeze':
        result = freeze_bridge(Path(args.cal_manifest), Path(args.calibration), Path(args.policy), Path(args.model),
                               Path(args.library), Path(args.torch_library), Path(args.tensor_inventory),
                               Path(args.out), args.host, args.gpu_uuid)
        print(json.dumps({'protocol_id': result['protocol_id'], 'executions': 6}))
    elif args.action == 'run':
        print(run_bridge(args.protocol, args.private, args.ledger, args.lock,
                         deadline_epoch=args.deadline_epoch, timeout=args.timeout))
    else:
        result = compare_bridge(args.protocol_a, args.ledger_a, args.protocol_b, args.ledger_b,
                                private_a=args.private_a, private_b=args.private_b,
                                inventory_a=args.inventory_a, inventory_b=args.inventory_b)
        write_immutable_json(args.out, result)
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(0 if result['qualified'] else 2)


if __name__ == '__main__':
    main()
