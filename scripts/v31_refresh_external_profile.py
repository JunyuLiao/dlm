"""Bounded external CUDA tracing of the unchanged, frozen original clean worker.

No model/attention/sampler hooks or annotations are added. CUDA API boundary
alignment is conditional and fails closed; GPU activity is not a critical-path
speedup bound. All original artifacts remain read-only and caches are copied.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time

from scripts import v31_refresh_diagnostic_study as study
from scripts import v31_v2_local_threshold_sweep as base
from scripts import v31_official as official

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'results/v31_refresh_diagnostic_20261018'
ORIGINAL = CAMPAIGN / 'attempt001'
FACTORIAL_ATTEMPT = CAMPAIGN / 'attempt003'
DEFAULT_OUTPUT = CAMPAIGN / 'external_headroom/attempt001'
NSYS = Path('/usr/local/bin/nsys')
SPEC = 'docs/REFRESH_EXTERNAL_PROFILE_20261018.md'
REQUIRED = (SPEC, 'scripts/v31_refresh_external_profile.py', 'tests/test_v31_refresh_external_profile.py')
HOST_CODE = REQUIRED + ('scripts/v31_refresh_diagnostic_study.py', 'scripts/v31_v2_local_threshold_sweep.py',
    'scripts/v31_official.py', 'scripts/v31_vllm_paired_bench.py', 'experiments/diffusion_gemma_aime26_modes/protocol.py')
PROFILE_FLAGS = ('--trace=cuda', '--sample=none', '--cpuctxsw=none', '--cuda-graph-trace=node',
                 '--cuda-memory-usage=false', '--force-overwrite=false', '--stats=false', '--export=none')
RESERVE_SECONDS = 600.
TOTAL_SECONDS = 7200.
TIMING_ABS_TOL_S = .003
TIMING_REL_TOL = .0001


def inventory(root):
    root = Path(root)
    return {str(p.relative_to(root)): dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns,
                sha256=base.sha(p)) for p in sorted(root.rglob('*')) if p.is_file()}


def reserve_available(reservations):
    used = study.charged_seconds(reservations)
    return max(0., min(RESERVE_SECONDS - used['reserve'], TOTAL_SECONDS - sum(used.values())))


def source_plan(protocol):
    matches = [p for p in protocol['plans'] if p['engine'] == 2]
    if len(matches) != 1:
        raise RuntimeError('original engine2 is not unique')
    plan = matches[0]
    expected = dict(phase='headroom', family='shared', suite='longbench_v2', pass_name='clean',
                    request_count=10, measured_count=8)
    if any(plan.get(k) != v for k, v in expected.items()):
        raise RuntimeError('original shared clean schedule changed')
    return plan


def require_committed():
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))
    changed = set(subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=ROOT, text=True).splitlines())
    if not set(HOST_CODE) <= tracked or changed & set(HOST_CODE):
        raise RuntimeError('commit the external plan, runner, tests and dependencies before prepare')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def verify_original():
    study.pause_unchanged()
    protocol = base.read(ORIGINAL / 'protocol.json'); plan = source_plan(protocol)
    leaf = ORIGINAL / plan['folder']
    study.validate_completion(leaf, plan, base.sha(ORIGINAL / 'protocol.json'), protocol['source_commit'])
    for rel, digest in protocol['execution_fingerprints'].items():
        if base.sha(ORIGINAL / 'private/execution' / rel) != digest:
            raise RuntimeError('original frozen source changed')
    for rel, digest in protocol['input_fingerprints'].items():
        if base.sha(ORIGINAL / rel) != digest:
            raise RuntimeError('original frozen input changed')
    if base.sha(ORIGINAL / 'model_source_fingerprints.json') != protocol['model_inventory_sha256']:
        raise RuntimeError('original model inventory fingerprint changed')
    for item in base.read(ORIGINAL / 'model_source_fingerprints.json')['files']:
        path = base.MODEL / item['file']
        if path.stat().st_size != item['bytes'] or (not item['file'].endswith('.safetensors') and base.sha(path) != item['sha256']):
            raise RuntimeError('model inventory changed')
    if base.runtime_identity() != protocol['runtime']:
        raise RuntimeError('original runtime/GPU identity changed')
    return protocol, plan, leaf


def prepare(output):
    if output.exists():
        raise RuntimeError('preserve previous external attempt')
    commit = require_committed(); original, plan, leaf = verify_original()
    factorial = base.read(FACTORIAL_ATTEMPT / 'protocol.json')
    output.mkdir(parents=True)
    cache_manifest = inventory(ORIGINAL / 'private/cache')
    shutil.copytree(ORIGINAL / 'private/cache', output / 'private/cache', symlinks=False)
    copied = inventory(output / 'private/cache')
    if copied != cache_manifest:
        raise RuntimeError('cache copy does not match the original snapshot')
    for name in ('cells', 'arms'):
        shutil.copy2(leaf / f'private/{name}.json', output / f'private/{name}.json')
    for rel in REQUIRED:
        dest = output / 'private/runner_source' / rel
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / rel, dest)
    protocol = dict(schema='v31_external_clean_nsys_v1', status='frozen_external_diagnostic',
        runner_commit=commit, runner_source_sha256={r: base.sha(ROOT / r) for r in HOST_CODE},
        original_protocol_sha256=base.sha(ORIGINAL / 'protocol.json'), original_source_commit=original['source_commit'],
        original_completion_sha256=base.sha(leaf / 'attempt_complete.json'), original_plan=plan,
        factorial_protocol_sha256=base.sha(FACTORIAL_ATTEMPT / 'protocol.json'),
        factorial_source_commit=factorial['source_commit'], runtime=original['runtime'],
        cache_copy_manifest=cache_manifest, cells_sha256=base.sha(output / 'private/cells.json'),
        arms_sha256=base.sha(output / 'private/arms.json'), nsys_flags=list(PROFILE_FLAGS),
        nsys_version=subprocess.check_output([str(NSYS), '--version'], text=True).strip(),
        nsys_sha256=base.sha(NSYS.resolve()), phase='reserve', engine=13, reserve_seconds=RESERVE_SECONDS,
        total_gpu_seconds=TOTAL_SECONDS, recorded_requests=10, measured_requests=8, implicit_warmups=1,
        explicit_warmups=2, warmup_and_generation_limits='identical original frozen schedule and native worker',
        segmentation=dict(boundaries='three cudaDeviceSynchronize calls per request: start, prefill-end, request-end',
            source='frozen clean benchmark, boundary_timing branch; 11 requests including implicit warmup',
            abs_tolerance_s=TIMING_ABS_TOL_S, relative_tolerance=TIMING_REL_TOL,
            qualification='unique same-thread consecutive33-sync alignment plus all10 recorded wall/prefill/decode durations; reject GPU crossings'),
        timing='external trace and wrapper wall time are diagnostic, not clean latency',
        limitation='no refresh/layer attribution from kernel names alone; unknown activity retained')
    base.write(output / 'protocol.json', protocol, True)
    print(json.dumps(dict(event='prepared_external', runner_commit=commit, original_source_commit=original['source_commit'])), flush=True)


def verify(output, protocol):
    original, plan, leaf = verify_original()
    if (base.sha(ORIGINAL / 'protocol.json') != protocol['original_protocol_sha256']
            or original['source_commit'] != protocol['original_source_commit'] or plan != protocol['original_plan']
            or base.sha(leaf / 'attempt_complete.json') != protocol['original_completion_sha256']):
        raise RuntimeError('original source/schedule/completion binding changed')
    for rel, digest in protocol['runner_source_sha256'].items():
        if base.sha(ROOT / rel) != digest:
            raise RuntimeError('external runner dependency changed')
    for name in ('cells', 'arms'):
        if base.sha(output / f'private/{name}.json') != protocol[name + '_sha256']:
            raise RuntimeError('copied schedule changed')
    if inventory(ORIGINAL / 'private/cache') != protocol['cache_copy_manifest']:
        raise RuntimeError('original cache changed after snapshot')
    if inventory(output / 'private/cache') != protocol['cache_copy_manifest']:
        raise RuntimeError('fresh cache copy changed before launch')
    if (base.sha(NSYS.resolve()) != protocol['nsys_sha256'] or list(PROFILE_FLAGS) != protocol['nsys_flags']):
        raise RuntimeError('nsys identity/options changed')
    if base.sha(FACTORIAL_ATTEMPT / 'protocol.json') != protocol['factorial_protocol_sha256']:
        raise RuntimeError('factorial protocol changed')
    return original, plan, leaf


def worker_command(output, original, seconds):
    execution = ORIGINAL / 'private/execution'
    base.SUITES.update(original['suites'])
    env = base.env_for(output, 'longbench_v2', 'current_v31_control')
    for key in list(env):
        if key.startswith(('REFRESH_', 'HEADROOM_')):
            env.pop(key)
    env.update(PYTHONPATH=f'{execution}/src:{execution}', PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1', STUDY_ARM='refresh_shared',
        FROZEN_SOURCE_SHA256=base.sha(ORIGINAL / 'receipts/source_fingerprints.json'),
        REFRESH_SOURCE_COMMIT=original['source_commit'], REFRESH_PROTOCOL_SHA256=base.sha(ORIGINAL / 'protocol.json'),
        REFRESH_SCHEDULE=str(output / 'private/arms.json'), REFRESH_BASE_WORKER=str(execution / 'scripts/v31_vllm_paired_bench.py'),
        REFRESH_AUDIT='0', HEADROOM_PROFILE='0', VALUE_AUDIT='0', VALUE_CLEAN_TIMING='1', VALUE_BOUNDARY_TIMING='1')
    command = [str(NSYS), 'profile', *PROFILE_FLAGS, '--wait=all', '--kill=sigterm',
        '--duration=' + str(max(1, int(seconds - 65))), '--output=' + str(output / 'private/nsys/trace'),
        base.PYTHON, str(execution / 'scripts/v31_refresh_diagnostic_worker.py'), str(base.MODEL),
        str(ORIGINAL / 'private/pools/longbench_v2'), str(output / 'private/cells.json'),
        str(output / 'records.jsonl'), str(output / 'private/generation.private.jsonl'), 'mage', 'PIECEWISE']
    return command, env


def parity_check(rows, reference, arms):
    if len(rows) != len(reference) or len(rows) != len(arms) - 1:
        return dict(status='incomplete', measured_matches=0, measured_expected=8, comparisons=[])
    fields = ('dataset', 'index', 'panel_seed', 'rng_seed', 'manifest_sha256', 'prompt_sha256', 'budget',
        'prompt_tokens', 'max_model_len', 'chunk', 'block_size', 'frozen_source_identity_sha256',
        'cudagraph_mode', 'resolved_cudagraph_mode', 'fix_51994', 'fa4_local_fix', 'output_hash', 'denoise_forwards',
        'canvases', 'output_tokens', 'finish_reason', 'sampler_calls', 'engine_steps', 'prefill_steps')
    comparisons = []
    for i, (row, old, arm) in enumerate(zip(rows, reference, arms[1:])):
        config_fields = tuple(key for key in old if key.startswith(('mage_', 'value_', 'cached_refresh_', 'local_')))
        runtime_fields = tuple(key for key in ('adapter_sha256', 'method_fingerprint', 'vllm', 'torch', 'gpu',
            'float32_matmul_precision', 'cuda_matmul_allow_tf32', 'gpu_memory_utilization',
            'seed_base', 'kv_copy_backend', 'merge_backend', 'fa4_local_fix_sha256') if key in old)
        checked = fields + config_fields + runtime_fields
        missing = [key for key in checked if key not in row or key not in old]
        different = [key for key in checked if row.get(key) != old.get(key)]
        adapter = (row.get('receipts') or {}).get('adapter') or {}
        reference_adapter = (old.get('receipts') or {}).get('adapter') or {}
        adapter_differences = sorted(key for key in set(adapter) | set(reference_adapter)
            if key not in adapter or key not in reference_adapter or adapter[key] != reference_adapter[key])
        if adapter_differences or not adapter or not reference_adapter:
            different += ['adapter.' + key for key in adapter_differences] or ['adapter receipt missing']
        valid = (not missing and not different and row.get('cuda_graph_captures') == 0
                 and row.get('timer_policy') == 'request_and_prefill_boundaries'
                 and row.get('value_clean_timing') is True and row.get('value_audit') is False
                 and adapter.get('refresh_label') == arm['label'] and adapter.get('refresh_policy') == arm['policy']
                 and adapter.get('refresh_request_index') == i + 1 and 'headroom' not in adapter)
        comparisons.append(dict(ordinal=i, label=arm['label'], warmup=arm['warmup'],
            index=row.get('index'), matched=valid, missing_fields=missing, differing_fields=different))
    measured = [row for row in comparisons if not row['warmup']]
    return dict(status='matched' if all(row['matched'] for row in measured) else 'mismatched',
                measured_matches=sum(row['matched'] for row in measured), measured_expected=len(measured), comparisons=comparisons)


def check_private_bind(rows, raw, pools):
    if len(rows) != len(raw):
        raise RuntimeError('public/private coverage mismatch')
    for row, private in zip(rows, raw):
        _, _, verified = official.bind(private, pools)
        if verified is not True:
            raise RuntimeError('private record was not verified against the frozen pool')
        fields = official.COMPARE_FIELDS + ('dataset', 'index', 'panel_seed', 'finish_reason')
        if any(key not in row or key not in private or row[key] != private[key] for key in fields):
            raise RuntimeError('public/private input binding mismatch')


def validate_output(output):
    """CPU child: potentially slow file reads/hashes remain under parent deadline."""
    errors = []
    protocol = base.read(output / 'protocol.json')
    original = base.read(ORIGINAL / 'protocol.json'); plan = source_plan(original)
    old_leaf = ORIGINAL / plan['folder']
    rows = study.complete_jsonl(output / 'records.jsonl', allow_truncated=True)
    raw = study.complete_jsonl(output / 'private/generation.private.jsonl', allow_truncated=True)
    arms = base.read(output / 'private/arms.json'); cells = base.read(output / 'private/cells.json')
    parity = parity_check(rows, base.jsonl(old_leaf / 'records.jsonl'), arms)
    base.write(output / 'parity.json', parity, True)
    if len(rows) != 10 or len(raw) != 10:
        errors.append('incomplete original public/private request coverage')
    errors += sorted({error for i, (r, c, a) in enumerate(zip(rows, cells, arms[1:]))
        for error in study.validate_row(r, c, a, i, plan, original['suites']['longbench_v2'])})
    try:
        check_private_bind(rows, raw, official.Pools(ORIGINAL / 'private/pools/longbench_v2', fields=('thinking',)))
    except Exception as exc:
        errors.append(type(exc).__name__ + ': ' + str(exc))
    cache_unchanged = inventory(ORIGINAL / 'private/cache') == protocol['cache_copy_manifest']
    if not cache_unchanged:
        errors.append('original cache mutated during external profiling')
    result = dict(status='complete' if not errors else 'failed_or_partial', errors=errors,
        records=len(rows), private_records=len(raw), measured_records=max(0, len(rows)-2),
        parity_status=parity['status'], original_cache_unchanged=cache_unchanged)
    base.write(output / 'validation.json', result, True)
    return result


def align_boundaries(syncs, records, *, abs_tolerance=TIMING_ABS_TOL_S, relative_tolerance=TIMING_REL_TOL):
    """Accept a unique complete same-thread triplet sequence, never duration alone.

    The first triple is the implicit warmup; the remaining triples correspond
    to all recorded requests including the two explicit warmups. Runtime API
    completion timestamps delimit the three verified clean-worker boundaries.
    """
    if len(records) != 10 or any(r.get('timer_policy') != 'request_and_prefill_boundaries' for r in records):
        return dict(status='unknown', reason='complete original boundary-timed schedule required', windows=[])
    by_thread = defaultdict(list)
    for row in syncs:
        if row['end'] < row['start']:
            raise ValueError('negative CUDA API interval')
        by_thread[row['thread']].append(row)
    candidates = []
    close = lambda observed, expected: abs(observed - expected) <= abs_tolerance + relative_tolerance * expected
    needed = 3 * (len(records) + 1)
    for thread, values in by_thread.items():
        values.sort(key=lambda r: (r['start'], r['end']))
        for offset in range(len(values) - needed + 1):
            selected = values[offset:offset + needed]
            if any(a['end'] > b['start'] for a, b in zip(selected, selected[1:])):
                continue
            windows = []; valid = True
            for ordinal in range(len(records) + 1):
                a, b, c = selected[3 * ordinal:3 * ordinal + 3]
                if not a['end'] < b['end'] < c['end']:
                    valid = False; break
                if ordinal:
                    r = records[ordinal - 1]
                    observed = ((c['end'] - a['end']) / 1e9, (b['end'] - a['end']) / 1e9, (c['end'] - b['end']) / 1e9)
                    if any(not close(x, float(r[key])) for x, key in zip(observed, ('wall_s', 'prefill_s', 'decode_s'))):
                        valid = False; break
                windows.append(dict(request_index=ordinal, start_ns=a['end'], prefill_end_ns=b['end'], end_ns=c['end']))
            if valid:
                candidates.append(dict(thread=thread, sync_offset=offset, windows=windows))
    if len(candidates) != 1:
        return dict(status='unknown', reason='CUDA API alignment absent or ambiguous', candidate_count=len(candidates), windows=[])
    return dict(status='candidate', candidate_count=1, **candidates[0],
                qualification='unique consecutive triplets + frozen source order + complete schedule + timing agreement; GPU boundary crossings still require validation')


def kernel_category(name, kind):
    name = name.lower()
    if kind != 'kernel': return 'copy_or_memset'
    if any(x in name for x in ('fused_moe', 'moe_', 'grouped_gemm', 'groupedgemm')): return 'name_moe_candidate'
    if any(x in name for x in ('flashattention', 'flash_attn', 'fmha')): return 'name_attention_candidate'
    if any(x in name for x in ('gemm', 'matmul', 'cublas')): return 'name_other_gemm'
    return 'unknown'


def activity_summary(activities, alignment, arms):
    """Clip activities to qualified request/phase windows; keep startup separate."""
    windows = alignment.get('windows', [])
    boundaries = sorted({w[key] for w in windows for key in ('start_ns', 'prefill_end_ns', 'end_ns')})
    import bisect
    totals = defaultdict(lambda: dict(activities=0, activity_sum_ms=0.))
    per_request = {w['request_index']: dict(request_index=w['request_index'], phases={}) for w in windows}
    busy = {}; crossings = 0
    starts = [w['start_ns'] for w in windows]
    for item in activities:
        start, end = item['start'], item['end']
        if end < start: raise ValueError('negative GPU activity')
        category = kernel_category(item.get('name', ''), item['kind'])
        totals[category]['activities'] += 1; totals[category]['activity_sum_ms'] += (end - start) / 1e6
        first = bisect.bisect_right(boundaries, start)
        if first < len(boundaries) and boundaries[first] < end:
            crossings += 1
        first_window = max(0, bisect.bisect_right(starts, start) - 1)
        for w in windows[first_window:]:
            if w['start_ns'] >= end: break
            if w['end_ns'] <= start: continue
            for phase, left, right in (('prefill', w['start_ns'], w['prefill_end_ns']), ('decode', w['prefill_end_ns'], w['end_ns'])):
                a, b = max(start, left), min(end, right)
                if b <= a: continue
                row = per_request[w['request_index']]['phases'].setdefault(phase, dict(categories={}, devices={}))
                c = row['categories'].setdefault(category, dict(activities=0, activity_sum_ms=0.))
                c['activities'] += 1; c['activity_sum_ms'] += (b - a) / 1e6
                key = (w['request_index'], phase, str(item['device']))
                # Input activities are sorted by start, so union needs O(1) state.
                previous = busy.get(key)
                if previous is None: busy[key] = [a, b, 0.]
                elif a > previous[1]: previous[2] += previous[1] - previous[0]; previous[:2] = [a, b]
                else: previous[1] = max(previous[1], b)
    for (index, phase, device), (start, end, prior) in busy.items():
        per_request[index]['phases'][phase]['devices'][device] = dict(gpu_busy_union_ms=(prior + end - start) / 1e6)
    qualified = alignment.get('status') == 'candidate' and crossings == 0
    for w in windows:
        r = per_request[w['request_index']]; arm = arms[w['request_index']]
        r.update(label=arm['label'], policy=arm['policy'], warmup=arm['warmup'])
    return dict(whole_process_categories=dict(totals), segmentation_status='qualified' if qualified else 'unknown',
        gpu_activities_crossing_boundaries=crossings,
        request_windows=list(per_request.values()) if qualified else [],
        interpretation='Whole-process totals include startup and all warmups. Names are heuristic; no layer, refresh or critical-path attribution. Phase GPU sums may overlap; union is per device. Trace timings are not clean latency.')


def sqlite_summary(path, records, arms):
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    try:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'StringIds' not in tables or 'CUPTI_ACTIVITY_KIND_RUNTIME' not in tables:
            return dict(status='unknown', reason='required CUDA runtime/string tables absent')
        names = dict(db.execute('SELECT id,value FROM StringIds'))
        syncs = [dict(start=a, end=b, thread=t) for a, b, t, n in db.execute(
            'SELECT start,end,globalTid,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME')
            if str(names.get(n, '')).startswith('cudaDeviceSynchronize')]
        alignment = align_boundaries(syncs, records)
        selects = []
        for table, kind in (('CUPTI_ACTIVITY_KIND_KERNEL', 'kernel'), ('CUPTI_ACTIVITY_KIND_MEMCPY', 'memcpy'), ('CUPTI_ACTIVITY_KIND_MEMSET', 'memset')):
            if table not in tables: continue
            columns = {r[1] for r in db.execute(f'PRAGMA table_info({table})')}
            if not {'start', 'end', 'deviceId'} <= columns: continue
            name = 'demangledName' if 'demangledName' in columns else 'shortName' if 'shortName' in columns else 'NULL'
            selects.append(f"SELECT start,end,deviceId,{name} AS nameId,'{kind}' AS kind FROM {table}")
        if not selects:
            return dict(status='unknown', reason='GPU activity tables absent', alignment=alignment)
        cursor = db.execute(' UNION ALL '.join(selects) + ' ORDER BY start')
        activities = (dict(start=a, end=b, device=d, name=str(names.get(n, '')), kind=k) for a, b, d, n, k in cursor)
        return dict(status='complete_cpu_summary', alignment=alignment, **activity_summary(activities, alignment, arms))
    finally:
        db.close()


def bounded_process(command, *, cwd, env, log_path, deadline):
    proc = None; timed_out = False
    with log_path.open('x') as log:
        try:
            proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while proc.poll() is None:
                left = deadline - time.monotonic() - 4.
                if left <= 0:
                    timed_out = True; study.stop_owned_process(proc); break
                try: proc.wait(timeout=min(30., left))
                except subprocess.TimeoutExpired: pass
        finally:
            if proc is not None and proc.poll() is None: study.stop_owned_process(proc)
    return dict(returncode=proc.returncode, timed_out=timed_out, pid=proc.pid)


def run(output):
    protocol = base.read(output / 'protocol.json')
    if (output / 'receipt.json').exists() or (output / 'private/worker.log').exists():
        raise RuntimeError('preserve prior external attempt; no automatic retry')
    with (ROOT / 'results/.v31_refresh_h100.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path = CAMPAIGN / 'gpu_budget.json'; ledger = base.read(ledger_path)
        if any(row.get('elapsed_s') is None for row in ledger['reservations']):
            raise RuntimeError('campaign worker is active or has an unresolved reservation')
        state = base.read(FACTORIAL_ATTEMPT / 'state.json')
        if state.get('status') not in ('complete_diagnostic', 'budget_limited_partial', 'causal_gate_unresolved'):
            raise RuntimeError('factorial must reach a preserved terminal state before external profiling')
        if state.get('protocol_sha256') != protocol['factorial_protocol_sha256']:
            raise RuntimeError('factorial terminal identity mismatch')
        original, plan, old_leaf = verify(output, protocol); study.require_idle()
        seconds = reserve_available(ledger['reservations'])
        if seconds < 180.: raise RuntimeError('insufficient external reserve; no GPU process launched')
        command, env = worker_command(output, original, seconds)
        (output / 'private/nsys').mkdir(parents=True)
        base.write(output / 'config.json', dict(protocol_sha256=base.sha(output / 'protocol.json'), max_reservation_s=seconds,
            original_source_commit=original['source_commit'], nsys_flags=list(PROFILE_FLAGS)), True)
        reservation = dict(attempt=str(output.relative_to(CAMPAIGN)), engine=13, phase='reserve', reserved_s=seconds,
                           elapsed_s=None, status='started')
        ledger['reservations'].append(reservation); base.write(ledger_path, ledger)
        start = time.monotonic(); deadline = start + seconds
        process, validation_process, export, summary, error, validation = None, None, None, None, None, None
        cpu_env = dict(os.environ, PYTHONPATH=f'{ROOT}/src:{ROOT}', PYTHONDONTWRITEBYTECODE='1')
        runner = str(ROOT / 'scripts/v31_refresh_external_profile.py')
        try:
            process = bounded_process(command, cwd=ORIGINAL / 'private/execution', env=env,
                                      log_path=output / 'private/worker.log', deadline=deadline - 50.)
            # No validation/export/summary begins until the entire nsys target
            # process group has exited. All potentially slow CPU work is bounded.
            if deadline - time.monotonic() > 5.:
                validation_process = bounded_process([base.PYTHON, runner, 'validate', '--output', str(output)],
                    cwd=ROOT, env=cpu_env, log_path=output / 'private/validation.log', deadline=deadline)
            if (output / 'validation.json').exists():
                validation = base.read(output / 'validation.json')
            report = output / 'private/nsys/trace.nsys-rep'
            if process['returncode'] == 0 and report.exists() and deadline - time.monotonic() > 10.:
                export = bounded_process([str(NSYS), 'export', '--type=sqlite', '--force-overwrite=false',
                    '--output=' + str(output / 'private/nsys/trace.sqlite'), str(report)], cwd=output, env=cpu_env,
                    log_path=output / 'private/export.log', deadline=deadline)
            if export and export['returncode'] == 0 and deadline - time.monotonic() > 5.:
                summary = bounded_process([base.PYTHON, runner, 'summarize', '--output', str(output)], cwd=ROOT,
                    env=cpu_env, log_path=output / 'private/summary.log', deadline=deadline)
        except BaseException as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        finally:
            reservation.update(elapsed_s=time.monotonic() - start, status='finished' if error is None else 'failed',
                               returncode=None if process is None else process['returncode'])
            base.write(ledger_path, ledger)
        receipt = dict(status='complete_diagnostic' if error is None and process and process['returncode'] == 0
                and validation and validation['status'] == 'complete' else 'failed_or_partial',
            error=error, process=process, validation_process=validation_process, validation=validation, sqlite_export=export, cpu_summary=summary,
            parity_status=None if validation is None else validation['parity_status'],
            trace_exists=(output / 'private/nsys/trace.nsys-rep').exists(),
            activity_summary_exists=(output / 'activity_summary.json').exists(),
            gpu_reservation_s=reservation['elapsed_s'], protocol_sha256=base.sha(output / 'protocol.json'),
            runner_commit=protocol['runner_commit'], native_source_commit=protocol['original_source_commit'],
            timing='external trace is not clean latency; parity and segmentation gates are separate')
        base.write(output / 'receipt.json', receipt, True)
        print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'validate', 'summarize'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); output = args.output.resolve()
    if CAMPAIGN.resolve() not in output.parents: raise RuntimeError('external output must remain inside this campaign')
    if args.action == 'prepare': prepare(output)
    elif args.action == 'run': run(output)
    elif args.action == 'validate': validate_output(output)
    else:
        summary = sqlite_summary(output / 'private/nsys/trace.sqlite', study.complete_jsonl(output / 'records.jsonl'),
                                 base.read(output / 'private/arms.json'))
        parity = base.read(output / 'parity.json') if (output / 'parity.json').exists() else {}
        summary['clean_reproduction_parity'] = parity.get('status', 'unknown')
        summary['original_clean_request_window_transfer_eligible'] = (parity.get('status') == 'matched'
            and summary.get('segmentation_status') == 'qualified')
        summary['critical_path_refresh_bound'] = 'unresolved: matching output and request windows do not identify refresh phases or causal latency savings'
        base.write(output / 'activity_summary.json', summary, True)


if __name__ == '__main__': main()
