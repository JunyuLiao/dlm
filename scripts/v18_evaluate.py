"""Freeze and run primary v18 RULER/AIME paired blocks after density calibration.

Generation reads only gold-free manifests. Freeze is CPU-only; run is an explicit
single-GPU action, resumable by the v13 execution ledger and immutable receipts.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from scripts.v13_seed_runs import (arm_config_hash, execution_key, is_device_error, plan_schedule,
                                   receipt_path, run_schedule, validate_resume)
from scripts.v18_protocol import SEEDS, sha, verify_prompt_tokens

ARMS = ('D_native', 'D_matched', 'U50', 'U60', 'T50', 'T60')
PLUGIN = 'experiments.value_direction_hopper.frontier_scope:install'


def write_immutable_json(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != payload:
        raise ValueError('frozen file already exists with different bytes: ' + str(path))
    path.write_text(payload)


def aime_token_overlay(draft, rows, adapter, manifest_path):
    """Create a new gold-free AIME identity with exact pinned CPU token lists."""
    if draft.get('dataset') != 'aime26' or len(rows) != 30:
        raise ValueError('AIME all30 draft required')
    result = []
    for row in rows:
        if any(k in row for k in ('answer', 'expected', 'expected_answer', 'gold', 'outputs')):
            raise ValueError('gold in generation manifest')
        if row.get('thinking') is not True or sha(row['prompt']) != draft['prompt_hashes'].get(row['id']):
            raise ValueError('AIME prompt/thinking identity mismatch')
        tokens = adapter.encode_prompt(row['prompt'], {'thinking': True})
        if not isinstance(tokens, list) or not tokens:
            raise ValueError('AIME tokenizer did not return a token list')
        if 'prompt_tokens' in row and row['prompt_tokens'] != tokens:
            raise ValueError('existing AIME prompt tokens differ from pinned tokenizer')
        result.append(dict(row, prompt_tokens=tokens, prompt_token_count=len(tokens)))
    verify_prompt_tokens(adapter, result)
    write_immutable_json(manifest_path, result)
    updated = dict(draft, protocol_id=draft['protocol_id'] + '_token_' + sha(manifest_path.read_bytes())[:8],
                   generation_manifest_path=str(manifest_path.resolve()),
                   generation_manifest_sha256=sha(manifest_path.read_bytes()),
                   prompt_token_hashes={r['id']: sha(json.dumps(r['prompt_tokens'], sort_keys=True,
                                                           separators=(',', ':'))) for r in result},
                   token_identity='exact_pinned_CPU_tokenizer_thinking_ON')
    return updated


def native_phase_evidence(receipt, *, sparse=False):
    canvases = receipt.get('per_canvas')
    timeline = receipt.get('generation_gpu_timeline_seconds')
    if not isinstance(canvases, list) or not canvases or timeline is None or timeline <= 0:
        return None
    result = []
    for canvas in canvases:
        steps = canvas.get('schedule_steps')
        if (not isinstance(steps, list) or not steps or
                any(type(x) is not int for x in steps) or
                type(canvas.get('native_stop_final_call')) is not bool or
                type(canvas.get('iteration_cap_final_call')) is not bool or
                canvas.get('decoder_calls') != len(steps)):
            return None
        result.append(dict(decoder_calls=len(steps), schedule_steps=steps,
                           native_stop=canvas['native_stop_final_call'],
                           iteration_cap=canvas['iteration_cap_final_call']))
    if sum(c['decoder_calls'] for c in result) != receipt.get('total_decoder_calls'):
        return None
    return dict(phase='fresh_sparse_decoder' if sparse else 'native_fresh_decoder',
                fresh_decoder_calls=receipt['total_decoder_calls'],
                per_canvas=result, initial_prefill_end_observed=True,
                prefill_end_to_finish_gpu_s=timeline)


def strict_warm(first, warm):
    reasons = []
    if not first or not first.get('ok'):
        reasons.append('attempt0_missing_or_failed')
    if not warm.get('ok'):
        reasons.append('warm_failed')
    for label, row in (('attempt0', first), ('warm', warm)):
        if not row:
            continue
        for field in ('completion_token_hash', 'per_canvas_calls', 'termination', 'phase_evidence'):
            value = row.get(field)
            if value is None or value == '' or value == [] or value == {}:
                reasons.append(label + '_' + field + '_absent')
    if first and first.get('ok') and warm.get('ok'):
        for field in ('completion_token_hash', 'per_canvas_calls', 'termination'):
            if first.get(field) != warm.get(field):
                reasons.append(field + '_mismatch')
        a, b = first.get('phase_evidence'), warm.get('phase_evidence')
        if a and b:
            for field in ('phase', 'fresh_decoder_calls', 'per_canvas', 'initial_prefill_end_observed'):
                if a.get(field) != b.get(field):
                    reasons.append('phase_' + field + '_mismatch')
    for field in ('triton_misses', 'triton_disk_entries_added', 'new_shared_objects'):
        if field not in warm:
            reasons.append('warm_' + field + '_absent')
    if warm.get('triton_misses') or warm.get('triton_disk_entries_added') or warm.get('new_shared_objects'):
        reasons.append('new_compile_or_shared_object')
    return dict(accepted=not reasons, reasons=sorted(set(reasons)))


def canonical_arm_hash(config, binary_hashes, tensor_identity_sha256):
    """Mathematical arm identity independent of host-local deployment paths."""
    fields = ('condition', 'frontier_arm', 'method', 'target', 'fast_t', 'support_geometry',
              'policy', 'm_ref', 'beta', 'gamma', 'thinking', 'max_new_tokens',
              'revision', 'dtype', 'diagnostic', 'timing_events', 'collect')
    identity = dict(math={k: config.get(k) for k in fields},
                    source_sha256_values=sorted(config['source_hashes'].values()),
                    model_metadata_hashes={k: v for k, v in config['model_metadata_hashes'].items()
                                           if k != 'model.safetensors.index.json'},
                    model_tensor_identity_sha256=tensor_identity_sha256,
                    binary_sha256=binary_hashes,
                    manifest_sha256=config['manifest_sha256'],
                    policy_file_sha256=config['policy_sha256'])
    return sha(json.dumps(identity, sort_keys=True, allow_nan=True))


def logical_protocol_digest(protocol):
    fields = ('protocol_id', 'stage', 'ids', 'seeds', 'arm_hashes', 'manifest_sha256',
              'calibration_sha256', 'binary_hashes', 'model_tensor_identity_sha256',
              'assigned_hosts', 'bridge_receipt_sha256',
              'block_assignments', 'initial_prefix_blocks', 'schedule')
    return sha(json.dumps({k: protocol[k] for k in fields}, sort_keys=True, allow_nan=True))


def eval_config(arm, draft, manifest, policy_file, model, library, torch_library, frozen, phase):
    from experiments.numerical_qk_reuse.runner import _config, _fingerprint
    native = arm == 'D_native'
    dense = arm == 'D_matched'
    method = 'native_dense' if native else 'kernel_dense' if dense else 'unweighted' if arm.startswith('U') else 'T'
    target = None if native or dense else int(arm[1:])
    if dense:
        policy = {k: {'log_threshold': -float('inf')} for k in ('local', 'global')}
    elif native:
        policy = {k: {'log_threshold': -float('inf')} for k in ('local', 'global')}
    else:
        if arm not in frozen:
            raise ValueError(f'missing calibrated policy for {arm}')
        policy = frozen[arm]['policy']
        if set(policy) != {'local', 'global'} or any('log_threshold' not in policy[k] for k in policy):
            raise ValueError(f'bad calibrated policy for {arm}')
    args = SimpleNamespace(condition='native_dense' if native else 'native_legal_all_layers',
                           phase=phase, ids=draft['ids'], seeds=list(SEEDS), manifest=manifest,
                           policy=policy_file, policy_name='T_s50', model=model,
                           revision=draft['model_revision'], decision_interval=1,
                           score_refresh_period=1, support='native_legal',
                           output_mode='historical_route_preqk_current_output', selector='legacy_recompute',
                           selector_layers='all', kernel_variant='static',
                           library=None if native else library, torch_library=None if native else torch_library,
                           plugin=None if native else PLUGIN, diagnostic=False, timing_events=True,
                           extra_source=[Path(__file__), Path(__file__).with_name('v18_protocol.py'),
                                         Path(__file__).with_name('v18_frontier.py'),
                                         Path(__file__).with_name('v18_bridge.py')])
    config = _config(args)
    config.update(frontier_arm=arm, method=method, target=target, fast_t=method == 'T',
                  support_geometry='native_legal', policy=policy, collect=not native,
                  max_new_tokens=8192, thinking=draft['generation']['thinking'])
    config['fingerprint'] = _fingerprint({k: v for k, v in config.items() if k != 'fingerprint'})
    return config


def prioritized_schedule(phase, revision, hashes, ids, warm_ids, seed):
    planned = plan_schedule(phase, revision, hashes, ids, list(SEEDS), seed)
    planned = [e for e in planned if e['role'] == 'attempt0' or e['id'] in warm_ids]
    blocks = {}
    for e in planned:
        blocks.setdefault(e['block'], []).append(e)
    ordered_blocks = sorted(blocks, key=lambda b: (not any(e['id'] in warm_ids for e in blocks[b]), b))
    schedule = []
    for new_block, old_block in enumerate(ordered_blocks):
        for e in blocks[old_block]:
            schedule.append(dict(e, block=new_block, index=len(schedule)))
    return schedule


def run_complete_blocks(schedule, done, *, deadline_epoch, gpu_budget_s, remaining_requests,
                        block_guard_s, process_started, execute_block, wall_clock=time.time,
                        process_clock=time.perf_counter):
    """Check resource guards only between frozen logical blocks."""
    newly_run = 0
    for block in dict.fromkeys(e['block'] for e in schedule):
        entries = [e for e in schedule if e['block'] == block]
        missing = [e for e in entries if execution_key(e) not in done]
        if not missing:
            continue
        if (wall_clock() + block_guard_s > deadline_epoch or
                process_clock() - process_started + block_guard_s > gpu_budget_s or
                newly_run + len(missing) > remaining_requests):
            return 'stopped_before_block'
        outcome = execute_block(entries)
        newly_run += len(missing)
        if outcome != 'complete':
            return outcome
    return 'complete'


def freeze_eval(draft_path, calibration_path, policy_file, model, library, torch_library, out, host, gpu_uuid,
                *, tensor_inventory, secondary_host=None, secondary_gpu_uuid=None, bridge_receipt=None):
    from scripts.v18_bridge import inventory_identity
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse.runner import _rows
    draft = json.loads(draft_path.read_text())
    if draft.get('status') != 'calibration_pending' or draft['dataset'] not in ('ruler4k', 'aime26'):
        raise ValueError('unsupported draft protocol')
    manifest = Path(draft['generation_manifest_path'])
    if sha(manifest.read_bytes()) != draft['generation_manifest_sha256']:
        raise ValueError('generation manifest byte drift')
    rows = _rows(manifest, allow_task_budgets=True, allow_thinking_off=draft['dataset'] == 'ruler4k')
    if len(rows) != len(draft['ids']) or {r['id'] for r in rows} != set(draft['ids']):
        raise ValueError('manifest ID coverage mismatch')
    if any(k in r for r in rows for k in ('outputs', 'answer', 'expected', 'expected_answer', 'gold')):
        raise ValueError('generation manifest contains gold')
    if any(sha(r['prompt']) != draft['prompt_hashes'][r['id']] for r in rows):
        raise ValueError('prompt hash mismatch')
    if any(r.get('thinking') is not draft['generation']['thinking'] for r in rows):
        raise ValueError('manifest thinking differs from frozen dataset request')
    tokenizer = create_adapter('diffusion_gemma', str(model), device='cpu', precision='float32',
                               revision=draft['model_revision']).load_tokenizer()
    verify_prompt_tokens(tokenizer, rows)
    frozen = json.loads(calibration_path.read_text())
    if any(arm not in frozen for arm in ARMS if arm[0] in ('U', 'T')):
        raise ValueError('four calibrated method points required')
    if any(set(frozen[arm]['calibration_ids']) & set(draft['ids']) for arm in frozen if arm in ARMS):
        raise ValueError('calibration/evaluation overlap')
    if not host or not gpu_uuid:
        raise ValueError('host and GPU UUID must be frozen before evaluation')
    inventory = inventory_identity(tensor_inventory)
    hosts = [dict(host=host, gpu_uuid=gpu_uuid)]
    bridge_sha = None
    if secondary_host or secondary_gpu_uuid or bridge_receipt:
        if not (secondary_host and secondary_gpu_uuid and bridge_receipt):
            raise ValueError('secondary host requires UUID and bridge qualification receipt')
        bridge = json.loads(Path(bridge_receipt).read_text())
        expected_hosts = {(host, gpu_uuid), (secondary_host, secondary_gpu_uuid)}
        observed_hosts = {(x['host'], x['gpu_uuid']) for x in bridge.get('hosts', [])}
        if bridge.get('qualified') is not True or observed_hosts != expected_hosts:
            raise ValueError('cross-host bridge is not qualified for these host/GPU identities')
        if bridge.get('model_tensor_identity_sha256') != inventory['tensor_identity_sha256']:
            raise ValueError('local model tensor inventory differs from qualified bridge')
        hosts.append(dict(host=secondary_host, gpu_uuid=secondary_gpu_uuid))
        bridge_sha = sha(Path(bridge_receipt).read_bytes())
    phase = 'v18_' + draft['dataset'] + '_primary_' + sha(json.dumps([draft['generation_manifest_sha256'],
                                                                    sha(calibration_path.read_bytes()), hosts,
                                                                    bridge_sha], sort_keys=True))[:12]
    configs = {arm: eval_config(arm, draft, manifest, policy_file, model, library, torch_library,
                                frozen, phase) for arm in ARMS}
    binaries = {'kernel': sha(library.read_bytes()), 'bridge': sha(torch_library.read_bytes())}
    hashes = {arm: canonical_arm_hash(config, binaries, inventory['tensor_identity_sha256'])
              for arm, config in configs.items()}
    execution_hashes = {arm: arm_config_hash(config) for arm, config in configs.items()}
    schedule = prioritized_schedule(phase, draft['model_revision'], hashes, draft['ids'],
                                    draft['warm_ids'], draft['schedule_seed'])
    block_count = len({e['block'] for e in schedule})
    block_assignments = {str(block): hosts[block % len(hosts)] for block in range(block_count)}
    initial_prefix_blocks = min(block_count, len(draft['warm_ids']) * len(SEEDS)) if draft['dataset'] == 'ruler4k' else block_count
    protocol = dict(schema='v18_primary_eval_protocol_v1', status='frozen', protocol_id=phase,
                    stage=draft['dataset'] + '_primary', priority=1 if draft['dataset'] == 'ruler4k' else 2,
                    construction=draft['construction'], authorization=draft['authorization'],
                    ids=draft['ids'], seeds=list(SEEDS), arms=list(ARMS), arm_hashes=hashes,
                    execution_config_hashes=execution_hashes,
                    model_path=str(model.resolve()), model_revision=draft['model_revision'],
                    tensor_inventory=inventory,
                    model_tensor_identity_sha256=inventory['tensor_identity_sha256'],
                    manifest=str(manifest.resolve()), manifest_sha256=draft['generation_manifest_sha256'],
                    prompt_hashes=draft['prompt_hashes'], source_manifest_sha256=draft['source_manifest_sha256'],
                    calibration_file=str(calibration_path.resolve()),
                    calibration_sha256=sha(calibration_path.read_bytes()), calibrated_points={arm: frozen[arm] for arm in frozen if arm in ARMS},
                    policy_file=str(policy_file.resolve()), policy_file_sha256=sha(policy_file.read_bytes()),
                    source_hashes=configs[ARMS[0]]['source_hashes'], binary_hashes=binaries,
                    assigned_hosts=hosts, bridge_receipt_file=str(Path(bridge_receipt).resolve()) if bridge_receipt else None,
                    bridge_receipt_sha256=bridge_sha,
                    block_assignments=block_assignments,
                    block_assignment_rule='frozen round robin by schedule block index; every arm/repeat of a block on one GPU',
                    identity_note='arm_hashes are canonical math+content IDs; execution_config_hashes bind this host-local path/config copy',
                    initial_prefix_blocks=initial_prefix_blocks,
                    warm_ids=draft['warm_ids'], schedule_seed=draft['schedule_seed'], schedule=schedule,
                    planned_executions=len(schedule), planned_blocks=block_count,
                    stage_order='RULER initial timed subset, then AIME all30, then RULER remainder',
                    timing='strict warm: token/per-canvas calls/termination/native phase evidence, zero JIT/disk/new SO; routing counters read after request sync equally for matched dense/U/T')
    protocol['logical_protocol_sha256'] = logical_protocol_digest(protocol)
    out.mkdir(parents=True, exist_ok=True)
    destination = out / (protocol['stage'] + '_protocol.json')
    if destination.exists() and json.loads(destination.read_text()) != protocol:
        raise ValueError('frozen protocol would change')
    destination.write_text(json.dumps(protocol, indent=2, sort_keys=True) + '\n')
    for arm, config in configs.items():
        path = out / 'configs' / protocol['stage'] / f'{arm}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError('frozen arm config would change')
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
    return protocol


def run_eval(protocol_path, private, ledger, lock_path, *, stage, max_blocks, deadline_epoch,
             gpu_budget_s, remaining_requests, block_guard_s, timeout):
    import fcntl
    import torch
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse.runner import _atomic, _one, _rows
    from scripts.v15_seed_runs import mapped_shared_objects
    from scripts.v9_clean_request_timing import append, disk_cache_entries, redacted
    from scripts.v10_request_runs import import_identity
    from triton.runtime.jit import JITFunction

    protocol = json.loads(protocol_path.read_text())
    if protocol.get('status') != 'frozen' or tuple(protocol['arms']) != ARMS:
        raise ValueError('protocol is not the frozen six-arm primary stage')
    if logical_protocol_digest(protocol) != protocol['logical_protocol_sha256']:
        raise ValueError('logical protocol drift')
    uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()[0].strip()
    worker = dict(host=socket.gethostname(), gpu_uuid=uuid)
    if worker not in protocol['assigned_hosts']:
        raise ValueError('host/GPU assignment mismatch')
    configs = {arm: json.loads((protocol_path.parent / 'configs' / protocol['stage'] / f'{arm}.json').read_text()) for arm in ARMS}
    if {arm: arm_config_hash(config) for arm, config in configs.items()} != protocol['execution_config_hashes']:
        raise ValueError('host-local execution config hash drift')
    from scripts.v18_bridge import inventory_identity
    if inventory_identity(protocol['tensor_inventory']['path']) != protocol['tensor_inventory']:
        raise ValueError('local 1047-tensor inventory changed')
    if {arm: canonical_arm_hash(config, protocol['binary_hashes'],
                                protocol['tensor_inventory']['tensor_identity_sha256'])
            for arm, config in configs.items()} != protocol['arm_hashes']:
        raise ValueError('canonical arm identity drift')
    for filename, expected in ((protocol['calibration_file'], protocol['calibration_sha256']),
                               (protocol['policy_file'], protocol['policy_file_sha256'])):
        if sha(Path(filename).read_bytes()) != expected:
            raise ValueError('calibration/policy byte drift: ' + filename)
    if protocol['bridge_receipt_file'] and sha(Path(protocol['bridge_receipt_file']).read_bytes()) != protocol['bridge_receipt_sha256']:
        raise ValueError('bridge qualification receipt changed')
    for config in configs.values():
        for filename, expected in config['source_hashes'].items():
            path = Path(filename)
            if not path.is_file() or sha(path.read_bytes()) != expected:
                raise ValueError('source/binary drift: ' + filename)
        for filename, expected in config['model_metadata_hashes'].items():
            if sha((Path(config['model']) / filename).read_bytes()) != expected:
                raise ValueError('model metadata drift: ' + filename)
    manifest = Path(protocol['manifest'])
    if sha(manifest.read_bytes()) != protocol['manifest_sha256']:
        raise ValueError('manifest byte drift')
    rows = _rows(manifest, allow_task_budgets=True, allow_thinking_off=protocol['stage'] == 'ruler4k_primary')
    if len(rows) != len(protocol['ids']) or {r['id'] for r in rows} != set(protocol['ids']):
        raise ValueError('manifest ID coverage mismatch')
    if any(sha(r['prompt']) != protocol['prompt_hashes'][r['id']] for r in rows):
        raise ValueError('prompt hash drift')
    if any(k in r for r in rows for k in ('outputs', 'answer', 'expected', 'expected_answer', 'gold')):
        raise ValueError('generation manifest contains gold')
    identity = dict(protocol_id=protocol['protocol_id'], model_revision=protocol['model_revision'],
                    arm_hashes=protocol['arm_hashes'], source_hashes=configs[ARMS[0]]['source_hashes'],
                    private_root=str(private.resolve()))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        events = [json.loads(s) for s in ledger.read_text().splitlines() if s.strip()] if ledger.exists() else []
        assigned = [e for e in protocol['schedule'] if protocol['block_assignments'][str(e['block'])] == worker]
        done = validate_resume(events, identity, assigned)
        if stage == 'initial':
            eligible_blocks = list(range(protocol['initial_prefix_blocks']))
        elif stage == 'remainder':
            eligible_blocks = list(range(protocol['initial_prefix_blocks'], protocol['planned_blocks']))
        else:
            eligible_blocks = list(range(protocol['planned_blocks']))
        if max_blocks is not None:
            eligible_blocks = eligible_blocks[:max_blocks]
        eligible = set(eligible_blocks)
        schedule = [e for e in assigned if e['block'] in eligible]
        for entry in schedule:
            if execution_key(entry) not in done and receipt_path(private, entry).exists():
                raise ValueError('orphan private receipt without ledger row; preserve first output')
        if all(f"{e['cell_id']}:{e['role']}:{e['repeat']}" in done for e in schedule):
            return 'complete_prefix'
        started = time.perf_counter()
        append(ledger, dict(event='start', pid=os.getpid(), pgid=os.getpgid(0), when=time.time(),
                            host=socket.gethostname(), gpu_uuid=uuid, import_identity=import_identity(), **identity))
        adapter = create_adapter('diffusion_gemma', protocol['model_path'], device='cuda',
                                 precision='bfloat16', revision=protocol['model_revision']).load()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        compiles = []
        def before(**kw):
            compiles.append(time.perf_counter())
            return False
        JITFunction.cache_hook = before
        class Timeout(Exception):
            pass
        def alarm(*_):
            raise Timeout()
        signal.signal(signal.SIGALRM, alarm)
        first = {e['cell_id']: e for e in events if e.get('event') == 'run' and e.get('role') == 'attempt0'}
        def execute_one(row, seed, config, entry):
            n, cache_before, so_before = len(compiles), disk_cache_entries(), mapped_shared_objects()
            receipt, error = None, None
            outer = time.perf_counter()
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
                          outer_wall_s=time.perf_counter() - outer, host=socket.gethostname(), gpu_uuid=uuid)
            if receipt is not None:
                if receipt['seed'] != seed:
                    raise AssertionError('generation seed mismatch')
                record.update(redacted(receipt))
                record['phase_evidence'] = native_phase_evidence(receipt, sparse=entry['arm'] not in ('D_native', 'D_matched'))
            if entry['role'] == 'warm':
                record['acceptance'] = strict_warm(first.get(entry['cell_id']), record)
                if receipt is not None:
                    path = receipt_path(private, entry)
                    _atomic(path, receipt)
                    record['private_receipt'] = str(path)
            else:
                first[entry['cell_id']] = record
            return dict(record=record, receipt=receipt, fatal=bool(error and is_device_error(error)))
        try:
            def execute_block(entries):
                return run_schedule(entries, done=done, execute_one=execute_one,
                                    rows={r['id']: r for r in rows}, configs=configs,
                                    ledger_append=lambda r: append(ledger, r), private=private,
                                    save_receipt=_atomic, stop_flag=lambda: False)
            status = run_complete_blocks(schedule, done, deadline_epoch=deadline_epoch,
                                         gpu_budget_s=gpu_budget_s, remaining_requests=remaining_requests,
                                         block_guard_s=block_guard_s, process_started=started,
                                         execute_block=execute_block)
        finally:
            JITFunction.cache_hook = None
        append(ledger, dict(event='worker_end', status=status, when=time.time(), host=socket.gethostname(),
                            gpu_uuid=uuid, gpu_process_seconds=time.perf_counter() - started))
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    a = sub.add_parser('freeze')
    for name in ('draft', 'calibration', 'policy', 'model', 'library', 'torch-library',
                 'tensor-inventory', 'out', 'host', 'gpu-uuid'):
        a.add_argument('--' + name, required=True)
    a.add_argument('--secondary-host')
    a.add_argument('--secondary-gpu-uuid')
    a.add_argument('--bridge-receipt')
    c = sub.add_parser('aime-overlay')
    for name in ('draft', 'model', 'private-out', 'out'):
        c.add_argument('--' + name, required=True)
    b = sub.add_parser('run')
    for name in ('protocol', 'private', 'ledger', 'lock'):
        b.add_argument('--' + name, type=Path, required=True)
    b.add_argument('--max-blocks', type=int)
    b.add_argument('--stage', choices=('initial', 'remainder', 'all'), default='initial')
    b.add_argument('--deadline-epoch', type=float, required=True)
    b.add_argument('--gpu-budget-s', type=float, required=True)
    b.add_argument('--remaining-requests', type=int, required=True)
    b.add_argument('--block-guard-s', type=float, required=True)
    b.add_argument('--timeout', type=int, default=900)
    args = p.parse_args()
    if args.action == 'aime-overlay':
        from dllm.models import create_adapter
        draft = json.loads(Path(args.draft).read_text())
        rows = json.loads(Path(draft['generation_manifest_path']).read_text())
        adapter = create_adapter('diffusion_gemma', args.model, device='cpu', precision='float32',
                                 revision=draft['model_revision']).load_tokenizer()
        updated = aime_token_overlay(draft, rows, adapter, Path(args.private_out))
        write_immutable_json(Path(args.out), updated)
        print(json.dumps({'questions': len(rows), 'manifest_sha256': updated['generation_manifest_sha256']}))
    elif args.action == 'freeze':
        proto = freeze_eval(Path(args.draft), Path(args.calibration), Path(args.policy), Path(args.model),
                            Path(args.library), Path(args.torch_library), Path(args.out), args.host, args.gpu_uuid,
                            secondary_host=args.secondary_host, secondary_gpu_uuid=args.secondary_gpu_uuid,
                            bridge_receipt=args.bridge_receipt, tensor_inventory=Path(args.tensor_inventory))
        print(json.dumps({'protocol_id': proto['protocol_id'], 'blocks': proto['planned_blocks'],
                          'executions': proto['planned_executions']}))
    else:
        if args.max_blocks is not None and args.max_blocks < 1:
            p.error('--max-blocks must be positive')
        if args.gpu_budget_s <= 0 or args.remaining_requests < 1 or args.block_guard_s <= 0:
            p.error('budget and block guard must be positive')
        print(run_eval(args.protocol, args.private, args.ledger, args.lock,
                       stage=args.stage, max_blocks=args.max_blocks,
                       deadline_epoch=args.deadline_epoch, gpu_budget_s=args.gpu_budget_s,
                       remaining_requests=args.remaining_requests,
                       block_guard_s=args.block_guard_s, timeout=args.timeout))


if __name__ == '__main__':
    main()
