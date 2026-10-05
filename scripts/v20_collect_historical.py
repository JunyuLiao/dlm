"""One-shot CPU scoring for frozen G75 historical extension beside core700.

The core scoring tree is read-only. Finalized historical partial archives are
scored honestly; absent/uncertain archives block without retrying generation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path, PurePosixPath
import statistics
import re
import subprocess
import sys
import tarfile
import time

try:
    from scripts import v20_collect_score as C
except ImportError:
    import importlib.util
    sibling = Path(__file__).with_name('v20_collect_score.py')
    spec = importlib.util.spec_from_file_location('v20_collect_score_private', sibling)
    if spec is None or spec.loader is None:
        raise
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)


SCHEMA = 'v20_collect_historical_v1'
STAGE = 'historical_001'
FROZEN_MAIN_SHA = '25f5483e6650c118744fd642ecd3ec62d3178514eb8e8b2a0d730f1227ee9d5f'


def validate(config, hosts):
    if config.get('schema') != SCHEMA or config.get('stage') != STAGE:
        raise ValueError('wrong historical collector schema/stage')
    C.validate_config(dict(config, schema=C.SCHEMA, stage='initial_001'), hosts)
    remote_dir = config.get('remote_stage_dir', 'historical')
    if (not isinstance(remote_dir, str) or not re.fullmatch(r'[A-Za-z0-9_/-]+', remote_dir) or
            remote_dir.startswith('/') or '..' in PurePosixPath(remote_dir).parts):
        raise ValueError('unsafe historical remote stage directory')
    main = config.get('main_score_root', '')
    if (not isinstance(main, str) or
            main != str(PurePosixPath(hosts['mpk']['root']) / 'offline_scoring' / 'remainder_001') or
            config.get('main_summary_sha256') != FROZEN_MAIN_SHA):
        raise ValueError('unfrozen core700 scoring root/summary')


def terminal_historical(config, hosts):
    outcome = json.loads((Path(config['watch_dir']) / f'{STAGE}.outcome.json').read_text(encoding='utf-8'))
    if outcome.get('schema') != 'v20_stage_watch_v1' or outcome.get('stage') != STAGE:
        raise ValueError('historical watcher identity mismatch')
    binding = json.loads(Path(config['binding_local']).read_text(encoding='utf-8'))
    results = {}
    for alias, host in hosts.items():
        row = outcome.get('hosts', {}).get(alias, {})
        if row.get('status') not in ('complete', 'failed'):
            return None, f'{alias} historical stage not terminal-finalized'
        artifact = row.get('artifacts', {}).get(f'{STAGE}.json')
        if not artifact:
            return None, f'{alias} final wrapper or archive absent'
        path = Path(artifact['path'])
        if C.sha_file(path) != artifact.get('sha256') or path.stat().st_size != artifact.get('bytes'):
            raise ValueError(f'{alias} historical watcher artifact hash drift')
        receipt = json.loads(path.read_text(encoding='utf-8'))
        public = receipt.get('public_identity', {})
        archive = receipt.get('private_archive') or {}
        expected_configs = {f'{dataset}/{arm}': item['sha256']
                            for dataset, arms in binding.get('host_configs', {}).get(host['host'], {}).items()
                            for arm, item in arms.items()}
        if (receipt.get('schema') != 'v20_request_stage_v1' or
                receipt.get('stage_name') != STAGE or receipt.get('mode') != 'request' or
                receipt.get('status') not in ('complete', 'failed') or
                public.get('host') != host['host'] or
                public.get('binding_sha256') != config['binding_sha256'] or
                public.get('protocol_sha256') != config['protocol_sha256'] or
                public.get('execution_config_sha256') != expected_configs or
                archive.get('path') != str(PurePosixPath(host['root']) /
                                           config.get('remote_stage_dir', 'historical') /
                                           f'{STAGE}.private.tar.gz') or
                not isinstance(archive.get('sha256'), str) or len(archive['sha256']) != 64 or
                not isinstance(archive.get('bytes'), int) or archive['bytes'] <= 0 or
                receipt.get('finalization_error_type')):
            return None, f'{alias} historical wrapper/archive identity incomplete'
        results[alias] = dict(stage_status=row['status'], wrapper_status=receipt['status'],
                              execution_status=receipt.get('execution', {}).get('status'),
                              stage_sha256=artifact['sha256'], archive=archive)
    return results, None


def historical_completion(extension, wrapper_status):
    datasets = extension.get('datasets', {})
    accepted = sum(row.get('warm_accepted', 0) for row in datasets.values())
    first_success = sum(row.get('first_success', 0) for row in datasets.values())
    planned = extension.get('planned_executions')
    recorded = extension.get('recorded_executions')
    all_rows = extension.get('execution_rows_complete') is True
    complete = (planned == 100 and recorded == 100 and all_rows and
                accepted == 50 and first_success == 50 and
                all(value == 'complete' for value in wrapper_status.values()))
    return dict(status='scored_complete' if complete else 'scored_partial',
                historical_planned=planned, historical_recorded=recorded,
                historical_all_rows=all_rows, historical_accepted_warm=accepted,
                historical_successful_first=first_success)


def _execution_key(spec):
    return f"{spec['cell_id']}:{spec['role']}:{spec['repeat']}"


def _ledger_records(paths, schedule):
    expected = {_execution_key(spec): spec for spec in schedule}
    if len(expected) != len(schedule):
        raise ValueError('duplicate frozen execution key')
    rows = {}
    for path in paths:
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get('event') != 'run':
                continue
            key = event.get('execution_key')
            if key not in expected or key in rows:
                raise ValueError('foreign or duplicate work ledger execution')
            spec = expected[key]
            if any(event.get(field) != spec[field] for field in
                   ('arm', 'id', 'seed', 'block', 'role', 'repeat', 'cell_id', 'host', 'gpu_uuid')):
                raise ValueError('work ledger differs from frozen schedule')
            rows[key] = event
    return rows


def historical_work(protocol, core_ledgers, historical_ledgers):
    """Gold-free work and same-GPU paired timing from redacted ledgers only."""
    hist_schedule = protocol['historical_extension']['schedule']
    core_schedule = protocol['schedule']
    hist = _ledger_records(historical_ledgers, hist_schedule)
    core = _ledger_records(core_ledgers, core_schedule)
    native = {(s['dataset'], s['id'], s['seed'], s['role']): s for s in core_schedule
              if s['arm'] == 'D_native'}
    pairs = defaultdict(dict)
    for spec in hist_schedule:
        comparator = native.get((spec['dataset'], spec['id'], spec['seed'], spec['role']))
        if comparator is None or (comparator['host'], comparator['gpu_uuid'], comparator['block']) != (
                spec['host'], spec['gpu_uuid'], spec['block']):
            raise ValueError('historical/native pair crossed frozen GPU or block')
        pairs[(spec['dataset'], spec['id'], spec['seed'])][spec['role']] = (spec, comparator)
    output = {}
    for dataset in protocol['ids']:
        firsts, warm_by_host, paired_by_host = [], defaultdict(list), defaultdict(list)
        for key, roles in pairs.items():
            if key[0] != dataset:
                continue
            first_spec, native_first_spec = roles['attempt0']
            warm_spec, native_warm_spec = roles['warm']
            first = hist.get(_execution_key(first_spec))
            warm = hist.get(_execution_key(warm_spec))
            native_first = core.get(_execution_key(native_first_spec))
            native_warm = core.get(_execution_key(native_warm_spec))
            if first and first.get('ok') is True:
                firsts.append(first)
            accepted = bool(first and first.get('ok') is True and warm and warm.get('ok') is True and
                            (warm.get('acceptance') or {}).get('accepted') is True)
            if not accepted:
                continue
            host = first_spec['host']
            wall = warm.get('api_wall_s')
            if not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0:
                raise ValueError('accepted historical warm lacks valid wall time')
            warm_by_host[host].append(warm)
            native_accepted = bool(native_first and native_first.get('ok') is True and
                                   native_warm and native_warm.get('ok') is True and
                                   (native_warm.get('acceptance') or {}).get('accepted') is True)
            if native_accepted:
                if (warm['host'], warm['gpu_uuid']) != (native_warm['host'], native_warm['gpu_uuid']):
                    raise ValueError('paired warm crossed GPU')
                paired_by_host[host].append((warm, native_warm))
        canvas_calls = [n for row in firsts for n in row.get('per_canvas_calls', [])]
        canvases = sum(row.get('canvases', 0) for row in firsts)
        calls = sum(row.get('decoder_calls', 0) for row in firsts)
        terms = Counter(row.get('termination', 'unknown') for row in firsts)
        cap_canvases = sum(sum(bool(x.get('iteration_cap')) for x in
                          (row.get('phase_evidence') or {}).get('per_canvas', [])) for row in firsts)
        phase = Counter()
        phase_measured = 0
        for row in firsts:
            evidence = row.get('router_phase_evidence') or {}
            if all(type(evidence.get(k)) is int for k in ('A', 'D', 'H')):
                phase.update({k: evidence[k] for k in ('A', 'D', 'H')})
                phase_measured += 1
        hosts = {}
        for host in sorted(set(warm_by_host) | set(paired_by_host)):
            warms = warm_by_host[host]
            spans = [(row.get('phase_evidence') or {}).get('prefill_end_to_finish_gpu_s') for row in warms]
            spans = [x for x in spans if isinstance(x, (int, float)) and math.isfinite(x) and x > 0]
            matched = paired_by_host[host]
            hist_wall = sum(a['api_wall_s'] for a, _ in matched)
            native_wall = sum(b['api_wall_s'] for _, b in matched)
            hist_calls = sum(a['decoder_calls'] for a, _ in matched)
            native_calls = sum(b['decoder_calls'] for _, b in matched)
            if matched and min(hist_wall, native_wall, hist_calls, native_calls) <= 0:
                raise ValueError('invalid accepted paired work/time')
            wall_ratio = hist_wall / native_wall if matched else None
            calls_ratio = hist_calls / native_calls if matched else None
            amortized_ratio = ((hist_wall / hist_calls) / (native_wall / native_calls)) if matched else None
            if matched and not math.isclose(wall_ratio, calls_ratio * amortized_ratio, rel_tol=1e-12):
                raise AssertionError('paired work/time ratio identity failed')
            hosts[host] = dict(accepted_warm_requests=len(warms),
                accepted_warm_request_s_mean=(statistics.mean(r['api_wall_s'] for r in warms) if warms else None),
                qualified_timeline_spans=len(spans),
                prefill_end_to_finish_gpu_s_mean=(statistics.mean(spans) if spans else None),
                paired_native_warm_cells=len(matched),
                paired_historical_request_s_total=hist_wall if matched else None,
                paired_native_request_s_total=native_wall if matched else None,
                paired_historical_decoder_calls_total=hist_calls if matched else None,
                paired_native_decoder_calls_total=native_calls if matched else None,
                paired_total_request_time_ratio=wall_ratio,
                paired_decoder_call_ratio=calls_ratio,
                paired_amortized_wall_per_call_ratio=amortized_ratio)
        all_ratios = [a['api_wall_s'] / b['api_wall_s'] for rows in paired_by_host.values()
                      for a, b in rows]
        output[dataset] = dict(first_success=len(firsts), decoder_calls_total=calls,
            canvases_total=canvases, decoder_calls_per_canvas_pooled=(calls / canvases if canvases else None),
            calls_per_canvas_distribution=dict(sorted(Counter(canvas_calls).items())),
            output_tokens_total=sum(row.get('output_tokens', 0) for row in firsts),
            termination=dict(sorted(terms.items())), request_output_cap=terms['length'],
            iteration_cap_canvases=cap_canvases,
            router_phase_layer_calls=(dict(A=phase['A'], D=phase['D'], H=phase['H'],
                                           measured_requests=phase_measured) if phase_measured else None),
            accepted_warm_by_host=hosts,
            paired_geometric_request_ratio_across_hosts=(math.exp(statistics.mean(math.log(x) for x in all_ratios))
                                                         if all_ratios else None),
            timing_identity='same-host paired total request ratio = decoder-call ratio x amortized wall/call ratio',
            timeline_note='CUDA event span after initial prefill when available; includes host launch gaps')
    return dict(schema='v20_historical_work_v1', source='redacted ledgers only; no private receipts or gold',
                datasets=output)


def replace_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.augmented.writing')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def core_paths(payload):
    """Identify four immutable core ledgers and their existing private roots."""
    root = Path(payload['main_score_root'])
    summary = root / 'remainder.redacted_summary.json'
    if C.sha_file(summary) != payload['main_summary_sha256']:
        raise ValueError('core700 summary byte drift')
    roots_path = root / 'private_roots.json'
    private_roots = json.loads(roots_path.read_text(encoding='utf-8'))
    if set(private_roots) != set(payload['host_ips'].values()):
        raise ValueError('core private-roots host map drift')
    for path in private_roots.values():
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(root.resolve()) or not resolved.is_dir():
            raise ValueError('core private root escaped frozen scoring tree')
    ledgers = []
    for alias in ('mpk', 'dllm'):
        for stage in ('initial', 'remainder'):
            folder = root / f'{alias}_{stage}' / 'ledger'
            found = list(folder.iterdir())
            if len(found) != 1 or not found[0].is_file() or found[0].is_symlink():
                raise ValueError('core ledger tree drift')
            ledgers.append(found[0])
    return summary, roots_path, ledgers


def remote_score(payload):
    root = Path(payload['score_root'])
    if root.exists():
        raise FileExistsError('historical scoring root exists; preserve uncertain attempt')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU process active; defer CPU scorer')
    protocol, binding = Path(payload['remote_protocol']), Path(payload['remote_binding'])
    if C.sha_file(protocol) != payload['protocol_sha256'] or C.sha_file(binding) != payload['binding_sha256']:
        raise ValueError('remote protocol/binding byte drift')
    main_summary, private_roots, core_ledgers = core_paths(payload)
    root.mkdir(parents=True, exist_ok=False)
    historical_roots, historical_ledgers = {}, []
    for alias in ('mpk', 'dllm'):
        item = payload['archives'][alias]
        archive = Path(item['path'])
        if C.sha_file(archive) != item['sha256'] or archive.stat().st_size != item['bytes']:
            raise ValueError(f'{alias} historical archive drift')
        ledger, private = C.extract_archive(archive, root / alias)
        historical_ledgers.append(ledger)
        historical_roots[item['host_ip']] = private
    historical_roots_path = root / 'historical_private_roots.json'
    C.write_new(historical_roots_path, {host: str(path) for host, path in historical_roots.items()})
    summary_path = root / 'historical.redacted_summary.json'
    command = [sys.executable, '-m', 'scripts.v20_score', '--protocol', str(protocol),
               '--binding', str(binding), '--ruler-gold', payload['ruler_gold'],
               '--aime-gold', payload['aime_gold'], '--longbench-gold', payload['longbench_gold'],
               '--ruler-root', payload['ruler_root'], '--private-roots', str(private_roots),
               '--historical-private-roots', str(historical_roots_path), '--out', str(summary_path)]
    for ledger in core_ledgers:
        command.extend(('--ledger', str(ledger)))
    for ledger in historical_ledgers:
        command.extend(('--historical-ledger', str(ledger)))
    scored = subprocess.run(command, capture_output=True, text=True, check=False)
    if scored.returncode:
        (root / 'scorer.stdout.private.txt').write_text(scored.stdout, encoding='utf-8')
        (root / 'scorer.stderr.private.txt').write_text(scored.stderr, encoding='utf-8')
        raise RuntimeError(f'offline historical scorer failed with exit {scored.returncode}')
    if C.sha_file(main_summary) != payload['main_summary_sha256']:
        raise ValueError('core700 summary changed during historical scoring')
    result = json.loads(summary_path.read_text(encoding='utf-8'))
    result['historical_work'] = historical_work(json.loads(protocol.read_text(encoding='utf-8')),
                                                core_ledgers, historical_ledgers)
    replace_json(summary_path, result)
    completion = historical_completion(result['historical_extension'], payload['wrapper_status'])
    print(json.dumps(dict(transport_status='scored', summary_path=str(summary_path),
                          summary_sha256=C.sha_file(summary_path),
                          summary_bytes=summary_path.stat().st_size, **completion)))


def collect(config, hosts, out_dir, *, transport, wait_until_epoch=None,
            clock=time.time, sleep=time.sleep):
    validate(config, hosts)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / f'{STAGE}.collect.status.json'
    if status_path.exists():
        raise FileExistsError('prior historical collection outcome exists')
    with (out_dir / f'{STAGE}.collect.lock').open('x', encoding='utf-8') as stream:
        stream.write(f'pid={os.getpid()}\n')
    status = dict(schema=SCHEMA, stage=STAGE, status='started')
    try:
        outcome_path = Path(config['watch_dir']) / f'{STAGE}.outcome.json'
        while not outcome_path.is_file():
            if wait_until_epoch is None or clock() >= wait_until_epoch:
                status.update(status='waiting_stage', reason='local historical watcher outcome absent at deadline')
                C.write_new(status_path, status)
                return status
            sleep(min(30, max(0, wait_until_epoch-clock())))
        receipts, reason = terminal_historical(config, hosts)
        if receipts is None:
            status.update(status='blocked', reason=reason)
            C.write_new(status_path, status)
            return status
        archives = {}
        for alias, host in hosts.items():
            item = receipts[alias]['archive']
            if transport.remote_sha(host, item['path']) != (item['sha256'], item['bytes']):
                raise ValueError(f'{alias} remote historical archive drift')
            local = out_dir / f'{alias}.{STAGE}.private.tar.gz'
            transport.copy_from(host, item['path'], local)
            if C.sha_file(local) != item['sha256'] or local.stat().st_size != item['bytes']:
                raise ValueError(f'{alias} local historical archive drift')
            with tarfile.open(local, 'r:gz') as stream:
                C.safe_archive_members(stream)
            archives[alias] = dict(path=str(local), sha256=item['sha256'],
                                   bytes=item['bytes'], host_ip=host['host'])
        mpk = hosts['mpk']
        wait_seconds, waited = int(config.get('gpu_wait_seconds', 0)), 0
        while transport.gpu_pids(mpk):
            if waited >= wait_seconds:
                status.update(status='waiting_gpu', reason='mpk GPU occupied; scorer not dispatched')
                C.write_new(status_path, status)
                return status
            span = min(30, wait_seconds - waited)
            sleep(span)
            waited += span
        offline = str(PurePosixPath(mpk['root']) / 'offline_scoring')
        score_root = str(PurePosixPath(offline) / STAGE)
        remote_dllm = str(PurePosixPath(offline) / f'dllm.{STAGE}.private.tar.gz')
        remote_payload = str(PurePosixPath(offline) / f'{STAGE}.remote_payload.json')
        remote_helper = str(PurePosixPath(offline) / 'v20_collect_historical.py')
        prior_helper = str(PurePosixPath(offline) / 'v20_collect_score.py')
        source = Path(C.__file__)
        if transport.remote_sha(mpk, prior_helper) != (C.sha_file(source), source.stat().st_size):
            raise ValueError('first84 collector helper source drift')
        prepare = ('import pathlib,sys; base=pathlib.Path(sys.argv[1]); '
                   'targets=[pathlib.Path(x) for x in sys.argv[2:]]; '
                   'assert not any(p.exists() for p in targets), "historical scoring output exists"; '
                   'base.mkdir(parents=True,exist_ok=True)')
        transport.remote(mpk, [mpk['python'], '-c', prepare, offline,
                               score_root, remote_dllm, remote_payload, remote_helper])
        transport.copy_to(mpk, archives['dllm']['path'], remote_dllm)
        if transport.remote_sha(mpk, remote_dllm) != (archives['dllm']['sha256'], archives['dllm']['bytes']):
            raise ValueError('dllm historical cross-host archive drift')
        payload = {key: config[key] for key in ('binding_sha256', 'protocol_sha256', 'remote_protocol',
                   'remote_binding', 'ruler_gold', 'aime_gold', 'longbench_gold', 'ruler_root',
                   'main_score_root', 'main_summary_sha256')}
        payload.update(score_root=score_root, host_ips={a: h['host'] for a, h in hosts.items()},
                       wrapper_status={a: x['wrapper_status'] for a, x in receipts.items()},
                       archives={'mpk': dict(archives['mpk'], path=receipts['mpk']['archive']['path']),
                                 'dllm': dict(archives['dllm'], path=remote_dllm)})
        payload_path = out_dir / f'{STAGE}.remote_payload.json'
        C.write_new(payload_path, payload)
        transport.copy_to(mpk, payload_path, remote_payload)
        transport.copy_to(mpk, Path(__file__), remote_helper)
        result = json.loads(transport.remote(mpk, [mpk['python'], remote_helper, 'remote-score',
                                              '--payload', remote_payload],
                                     cwd=config['remote_deploy'], env=mpk['env'], timeout=3600))
        if result.get('transport_status') != 'scored' or result.get('status') not in ('scored_complete', 'scored_partial'):
            raise ValueError('historical offline scorer did not finish')
        summary_local = out_dir / f'{STAGE}.redacted_summary.json'
        transport.copy_from(mpk, result['summary_path'], summary_local)
        if C.sha_file(summary_local) != result['summary_sha256'] or summary_local.stat().st_size != result['summary_bytes']:
            raise ValueError('historical scored summary transfer drift')
        status.update(status=result['status'], wrapper_status={a: x['wrapper_status'] for a, x in receipts.items()},
                      stage_sha256={a: x['stage_sha256'] for a, x in receipts.items()},
                      archive_sha256={a: x['sha256'] for a, x in archives.items()},
                      summary_sha256=result['summary_sha256'],
                      historical_recorded=result['historical_recorded'],
                      historical_accepted_warm=result['historical_accepted_warm'],
                      historical_successful_first=result['historical_successful_first'])
    except BaseException as exc:
        status.update(status='failed', error_type=type(exc).__name__)
        C.write_new(status_path, status)
        raise
    C.write_new(status_path, status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    local = sub.add_parser('collect')
    local.add_argument('--config', type=Path, required=True)
    local.add_argument('--out-dir', type=Path, required=True)
    local.add_argument('--wait-until-epoch', type=float)
    remote = sub.add_parser('remote-score')
    remote.add_argument('--payload', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == 'remote-score':
        remote_score(json.loads(args.payload.read_text(encoding='utf-8')))
        return 0
    config = json.loads(args.config.read_text(encoding='utf-8'))
    hosts = json.loads(Path(config['hosts_path']).read_text(encoding='utf-8'))
    result = collect(config, hosts, args.out_dir, transport=C.Transport(),
                     wait_until_epoch=args.wait_until_epoch)
    print(json.dumps(dict(status=result['status'], historical_recorded=result.get('historical_recorded'))))
    return 0 if result['status'] in ('scored_complete', 'scored_partial') else 2


if __name__ == '__main__':
    raise SystemExit(main())
