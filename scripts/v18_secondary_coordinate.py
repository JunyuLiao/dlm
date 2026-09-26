"""CPU-only secondary stage gate, quota planner, and optional bounded launcher.

`--execute` is an explicit future authorization: it never starts a GPU worker
until every primary RULER/AIME execution has a ledger row. All campaign ledgers
must be local and current; the coordinator does not fetch or mutate host state.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from scripts.v13_seed_runs import execution_key
from scripts.v18_secondary import (CORE, CONTROLS, advance_calibration_point,
                                   finalize_calibration, freeze_calibration_point,
                                   prerequisite_gate, stage_completion)
from scripts.v18_stage_driver import completed_keys
from scripts.v18_protocol import sha


def usage_by_host(ledger_by_host: dict[str, list[Path]], now: float) -> tuple[dict, dict]:
    """Charge complete workers and open/crashed workers, including model load."""
    gpu, requests = {}, {}
    seen_paths = set()
    for host, paths in ledger_by_host.items():
        gpu[host], requests[host] = 0., 0
        for path in paths:
            path = Path(path).resolve()
            if path in seen_paths:
                raise ValueError('campaign ledger path listed twice')
            seen_paths.add(path)
            if not path.exists():
                continue
            pending = None
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get('event') == 'run':
                    if event.get('host', host) != host:
                        raise ValueError('campaign run recorded under wrong host')
                    requests[host] += 1
                elif event.get('event') == 'start':
                    if pending is not None:
                        gpu[host] += max(0., event['when'] - pending)
                    pending = event['when']
                elif event.get('event') == 'worker_end':
                    if pending is None or event.get('host') != host:
                        raise ValueError('campaign worker_end has no matching host start')
                    seconds = event.get('gpu_process_seconds')
                    if type(seconds) not in (int, float) or seconds < 0:
                        raise ValueError('invalid campaign worker duration')
                    gpu[host] += max(float(seconds), max(0., event['when'] - pending))
                    pending = None
            if pending is not None:
                gpu[host] += max(0., now - pending)
    return gpu, requests


def primary_quiet_gate(prerequisites: dict, summaries: dict[str, Path],
                       ledger_by_host: dict[str, list[Path]], *, gpu_idle: bool) -> dict:
    """Require both finalized primary scores and closed worker intervals."""
    required = ('ruler4k_primary', 'aime26_primary')
    if set(summaries) != set(required) or not all(stage in prerequisites for stage in required):
        raise ValueError('both primary finalizer summaries required')
    for stage in required:
        protocol = prerequisites[stage][0]
        summary = json.loads(Path(summaries[stage]).read_text())
        if (summary.get('schema') != 'v18_redacted_summary_v1' or
                summary.get('stage') != stage or
                summary.get('protocol_id') != protocol['protocol_id'] or
                summary.get('complete') is not True or
                summary.get('recorded_executions') != len(protocol['schedule']) or
                summary.get('planned_executions') != len(protocol['schedule'])):
            raise ValueError('primary scorer completion/identity gate failed')
    primary_paths = [Path(path) for stage in required for path in prerequisites[stage][1]]
    all_inventory = {Path(path).resolve() for paths in ledger_by_host.values() for path in paths}
    if not all(Path(path).resolve() in all_inventory for path in primary_paths):
        raise ValueError('primary worker ledgers missing from campaign accounting')
    for path in primary_paths:
        if not path.exists():
            continue
        open_workers = 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get('event') == 'start':
                open_workers += 1
            elif event.get('event') == 'worker_end':
                open_workers -= 1
                if open_workers < 0:
                    raise ValueError('worker_end without start')
        if open_workers:
            raise ValueError('campaign GPU worker has no done marker')
    return dict(ready=bool(gpu_idle), primary_finalized=True, workers_closed=True,
                gpu_idle=bool(gpu_idle))


def local_gpu_idle() -> bool:
    """Read the local GPU process table immediately before a new stage launch."""
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True)
    return not any(line.strip() and line.strip() != '[Not Supported]'
                   for line in result.stdout.splitlines())


def budget_plan(secondary: dict, prerequisites: dict, stage_ledgers: list[Path],
                ledger_by_host: dict[str, list[Path]], budget: dict, host: str,
                *, now: float | None = None, primary_summaries: dict | None = None,
                gpu_idle: bool = False) -> dict:
    now = time.time() if now is None else now
    if host not in budget['gpu_cap_s_by_host'] or host not in budget['request_cap_by_host']:
        raise ValueError('host absent from frozen campaign cap')
    if set(ledger_by_host) != set(budget['gpu_cap_s_by_host']) or set(ledger_by_host) != set(budget['request_cap_by_host']):
        raise ValueError('all campaign host ledgers required')
    if (sum(budget['baseline_requests_by_host'].values()) != budget['baseline_requests'] or
            sum(budget['request_cap_by_host'].values()) > budget['request_cap']):
        raise ValueError('frozen per-host/global request budget inconsistent')
    if set(prerequisites) != set(secondary['prerequisite_protocol_ids']):
        raise ValueError('all stage prerequisites required')
    for name, pair in prerequisites.items():
        if pair[0]['protocol_id'] != secondary['prerequisite_protocol_ids'][name]:
            raise ValueError('prerequisite identity drift')
    gate = prerequisite_gate(secondary, prerequisites['ruler4k_primary'],
                             prerequisites['aime26_primary'], prerequisites.get('ruler_secondary70'))
    quiet = (primary_quiet_gate(prerequisites, primary_summaries, ledger_by_host,
                                gpu_idle=gpu_idle) if primary_summaries is not None else
             dict(ready=False, reason='primary finalizer summaries and GPU-idle check required'))
    gpu_used, request_used = usage_by_host(ledger_by_host, now)
    accounted_gpu = {h: budget['baseline_gpu_s_by_host'][h] + gpu_used[h] for h in ledger_by_host}
    accounted_requests = {h: budget['baseline_requests_by_host'][h] + request_used[h] for h in ledger_by_host}
    gpu_remaining = budget['gpu_cap_s_by_host'][host] - accounted_gpu[host]
    request_remaining = min(budget['request_cap_by_host'][host] - accounted_requests[host],
                            budget['request_cap'] - sum(accounted_requests.values()))
    # The frozen deadline_epoch is already the GPU cutoff with scoring reserved.
    deadline = budget['deadline_epoch']
    done = completed_keys(secondary, stage_ledgers)
    selected = [e for e in secondary['schedule']
                if secondary['block_assignments'][str(e['block'])]['host'] == host]
    outstanding = sorted({e['block'] for e in selected if execution_key(e) not in done})
    next_requests = (sum(execution_key(e) not in done for e in selected if e['block'] == outstanding[0])
                     if outstanding else 0)
    guard = budget['block_guard_s']
    allowed = bool(gate['ready'] and quiet['ready'] and outstanding and gpu_remaining > guard and
                   request_remaining >= next_requests and now + guard < deadline)
    return dict(stage=secondary['stage'], host=host, gate=gate, quiet=quiet,
                allowed=allowed, host_stage_complete=not outstanding,
                next_block_requests=next_requests, outstanding_blocks=len(outstanding),
                gpu_used_s_by_host=accounted_gpu, requests_used_by_host=accounted_requests,
                gpu_remaining_s=gpu_remaining, requests_remaining=request_remaining,
                deadline_epoch=deadline, block_guard_s=guard)


def calibration_quota(budget: dict, ledger_by_host: dict[str, list[Path]], host: str,
                      next_requests: int, *, now: float | None = None) -> dict:
    """Reserve one whole frozen density point using the campaign's existing cutoff."""
    now = time.time() if now is None else now
    hosts = set(budget['gpu_cap_s_by_host'])
    if (host not in hosts or set(ledger_by_host) != hosts or
            set(budget['request_cap_by_host']) != hosts or
            set(budget['baseline_gpu_s_by_host']) != hosts or
            set(budget['baseline_requests_by_host']) != hosts or
            sum(budget['request_cap_by_host'].values()) > budget['request_cap'] or
            sum(budget['baseline_requests_by_host'].values()) != budget['baseline_requests']):
        raise ValueError('complete consistent frozen campaign budget required')
    gpu, requests = usage_by_host(ledger_by_host, now)
    charged_gpu = {h: budget['baseline_gpu_s_by_host'][h] + gpu[h] for h in hosts}
    charged_requests = {h: budget['baseline_requests_by_host'][h] + requests[h] for h in hosts}
    remaining_gpu = budget['gpu_cap_s_by_host'][host] - charged_gpu[host]
    remaining_requests = min(budget['request_cap_by_host'][host] - charged_requests[host],
                             budget['request_cap'] - sum(charged_requests.values()))
    cutoff = budget['deadline_epoch']
    guard = budget['block_guard_s']
    return dict(allowed=bool(next_requests > 0 and next_requests <= remaining_requests and
                             remaining_gpu > guard and now + guard < cutoff),
                host=host, next_requests=next_requests, remaining_requests=remaining_requests,
                gpu_remaining_s=remaining_gpu, requests_used_by_host=charged_requests,
                gpu_used_s_by_host=charged_gpu, deadline_epoch=cutoff,
                block_guard_s=guard)


def calibrate_group(spec: dict, prerequisites: dict, ledger_by_host: dict[str, list[Path]],
                    budget: dict, *, execute: bool, timeout: int = 900,
                    gpu_idle: bool = False) -> dict:
    """Resume all <=5 measured density points for one independently frozen arm pair."""
    from scripts.v18_secondary import MAX_THRESHOLD_PAIRS, TARGET
    from scripts.v18_frontier import best_point, next_policy

    group = spec['group']
    arms = CORE if group == 'core70' else CONTROLS if group == 'controls' else None
    if arms is None or set(spec['initial_policies']) != set(arms):
        raise ValueError('one complete frozen secondary calibration pair required')
    if (len(spec['ids']) != 26 or len(set(spec['ids'])) != 26 or
            set(spec['prerequisite_protocol_ids']) !=
            ({'ruler4k_primary', 'aime26_primary'} |
             ({'ruler_secondary70'} if group == 'controls' else set()))):
        raise ValueError('independent 26 IDs and ordered frozen prerequisites required')
    for stage, expected_id in spec['prerequisite_protocol_ids'].items():
        if stage not in prerequisites or prerequisites[stage][0].get('stage') != stage or \
                prerequisites[stage][0].get('protocol_id') != expected_id:
            raise ValueError('calibration prerequisite identity drift')
    checks = [stage_completion(*prerequisites[stage]) for stage in spec['prerequisite_protocol_ids']]
    missing = [dict(stage=c['stage'], **row) for c in checks for row in c['missing']]
    if missing:
        return dict(status='prerequisite_incomplete', missing=missing)
    quiet = primary_quiet_gate(prerequisites,
                               {k: Path(v) for k, v in spec['primary_summaries'].items()},
                               ledger_by_host, gpu_idle=gpu_idle)
    if not quiet['ready']:
        return dict(status='primary_not_quiet', quiet=quiet)
    host = socket.gethostname()
    if host != spec['calibration_host']:
        raise ValueError('calibration host differs from frozen assignment')
    root = Path(spec['out'])
    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / 'calibration_state.json'
    identity = dict(schema='v18_secondary_calibration_state_v1', group=group,
                    spec_sha256=sha(json.dumps(spec, sort_keys=True)),
                    prerequisite_protocol_ids=spec['prerequisite_protocol_ids'],
                    ids_sha256=sha(json.dumps(spec['ids'])))
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('durable calibration state/spec identity drift')
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + '\n')
    history_path = root / 'history.json'
    frozen_path = root / 'calibrated.json'
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        if set(frozen) != set(arms) or any(frozen[a]['calibration_ids'] != spec['ids'] for a in arms):
            raise ValueError('frozen calibration identity drift')
        return dict(status='complete', frozen=str(frozen_path))
    history = json.loads(history_path.read_text()) if history_path.exists() else {}
    point = max((len(history.get(a, [])) for a in arms), default=0)
    if point > MAX_THRESHOLD_PAIRS or any(len(history.get(a, [])) > point for a in arms):
        raise ValueError('calibration histories must contain completed whole points')
    def pending_policies():
        output = {}
        for arm in arms:
            entries = history.get(arm, [])
            if not entries:
                if point != 0:
                    raise ValueError('missing earlier calibration history')
                output[arm] = spec['initial_policies'][arm]
                continue
            if any(set(e) != {'point', 'policy', 'actual'} or e['point'] != i
                   for i, e in enumerate(entries)):
                raise ValueError('calibration history contains non-density fields')
            chosen = best_point(entries, TARGET[arm])
            attained = all(abs(chosen['actual'][kind] - TARGET[arm]) <= .02
                           for kind in ('whole', 'local', 'global'))
            if not attained and len(entries) < MAX_THRESHOLD_PAIRS:
                if len(entries) != point:
                    raise ValueError('unfinished arm omitted from prior point')
                output[arm] = next_policy(entries, TARGET[arm])
            elif len(entries) == point - 1 and not attained:
                raise ValueError('nonattained arm stopped before five pairs')
        return output
    policies = pending_policies()
    while point < MAX_THRESHOLD_PAIRS and policies:
        protocol = freeze_calibration_point(
            Path(spec['manifest']), spec['ids'], Path(spec['policy_file']), Path(spec['model']),
            Path(spec['library']), Path(spec['torch_library']), root, policies, point,
            spec['authorization'],
            prerequisite_protocol_ids=spec['prerequisite_protocol_ids'],
            calibration_host=host, calibration_gpu_uuid=spec['calibration_gpu_uuid'])
        protocol_path = root / f'secondary_calibration_p{point}_protocol.json'
        ledger = root / f'secondary_calibration_p{point}_ledger.jsonl'
        inventory = {h: list(paths) for h, paths in ledger_by_host.items()}
        if ledger not in inventory[host]:
            inventory[host].append(ledger)
        completed = {e.get('execution_key') for e in
                     (json.loads(line) for line in ledger.read_text().splitlines() if line.strip())
                     if e.get('event') == 'run'} if ledger.exists() else set()
        from scripts.v13_seed_runs import execution_key
        expected = {execution_key(e) for e in protocol['schedule']}
        if not completed <= expected:
            raise ValueError('foreign calibration execution in point ledger')
        pending = len(expected - completed)
        quota = calibration_quota(budget, inventory, host, pending)
        if pending and not quota['allowed']:
            return dict(status='budget_stop', group=group, point=point, quota=quota)
        if pending and not execute:
            return dict(status='ready', group=group, point=point, quota=quota,
                        protocol=str(protocol_path))
        if pending:
            prior_path = root / 'prerequisites.json'
            prior = {stage: dict(protocol=str(Path(spec['prerequisites'][stage]['protocol'])),
                                 ledgers=spec['prerequisites'][stage]['ledgers'])
                     for stage in spec['prerequisite_protocol_ids']}
            prior_payload = json.dumps(prior, indent=2, sort_keys=True) + '\n'
            if prior_path.exists() and prior_path.read_text() != prior_payload:
                raise ValueError('calibration prerequisite file drift')
            prior_path.parent.mkdir(parents=True, exist_ok=True)
            prior_path.write_text(prior_payload)
            command = [sys.executable, '-m', 'scripts.v18_secondary', 'run-calibration',
                       '--protocol', str(protocol_path), '--prerequisites', str(prior_path),
                       '--private', str(root / 'private' / f'p{point}'), '--ledger', str(ledger),
                       '--lock', str(Path(spec['lock'])), '--max-executions', str(len(expected)),
                       '--deadline-epoch', str(quota['deadline_epoch']), '--timeout', str(timeout)]
            hard_timeout = max(1, int(min(quota['deadline_epoch'] - time.time(),
                                          quota['gpu_remaining_s'])))
            result = subprocess.run(command, timeout=hard_timeout, check=False, env=os.environ.copy())
            if result.returncode:
                return dict(status='worker_failed', group=group, point=point, returncode=result.returncode)
        advanced = advance_calibration_point(protocol_path, ledger, history_path)
        point += 1
        history = json.loads(history_path.read_text())
        policies = pending_policies()
    if policies:
        raise ValueError('secondary calibration exceeded five threshold pairs')
    frozen = finalize_calibration(history_path, spec['ids'], frozen_path, arms=arms)
    return dict(status='complete', group=group, frozen=str(frozen_path),
                attained={arm: frozen[arm]['attained'] for arm in arms})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'prerequisites', 'ledger-inventory', 'budget', 'private', 'ledger', 'lock'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--stage-ledger', type=Path, action='append', required=True,
                   help='One secondary stage ledger per assigned host, including this worker ledger')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--primary-summary', action='append', required=True,
                   help='Both STAGE=path primary redacted summary files')
    p.add_argument('--wait-for-prerequisites', action='store_true')
    p.add_argument('--poll-seconds', type=int, default=30)
    p.add_argument('--timeout', type=int, default=900)
    args = p.parse_args()
    secondary = json.loads(args.protocol.read_text())
    raw_prior = json.loads(args.prerequisites.read_text())
    prior = {stage: (json.loads(Path(value['protocol']).read_text()),
                     [Path(x) for x in value['ledgers']]) for stage, value in raw_prior.items()}
    inventory = json.loads(args.ledger_inventory.read_text())
    ledgers = {host: [Path(x) for x in paths] for host, paths in inventory.items()}
    budget = json.loads(args.budget.read_text())
    summaries = {}
    for item in args.primary_summary:
        stage, sep, path = item.partition('=')
        if not sep or stage in summaries:
            raise ValueError('primary summary must be unique STAGE=path')
        summaries[stage] = Path(path)
    stage_ledgers = args.stage_ledger
    if args.ledger not in stage_ledgers:
        raise ValueError('worker ledger must be among all secondary stage ledgers')
    host = socket.gethostname()
    while True:
        plan = budget_plan(secondary, prior, stage_ledgers, ledgers, budget, host,
                           primary_summaries=summaries,
                           gpu_idle=local_gpu_idle() if args.execute else False)
        print(json.dumps(plan, sort_keys=True), flush=True)
        if (plan['gate']['ready'] or not args.wait_for_prerequisites or
                plan['gpu_remaining_s'] <= plan['block_guard_s'] or
                plan['requests_remaining'] < plan['next_block_requests'] or
                time.time() + plan['block_guard_s'] >= plan['deadline_epoch']):
            break
        time.sleep(args.poll_seconds)
    if not args.execute or not plan['allowed']:
        return
    hard_timeout = max(1, int(min(plan['deadline_epoch'] - time.time(), plan['gpu_remaining_s'])))
    cmd = [sys.executable, '-m', 'scripts.v18_secondary', 'run-eval',
           '--protocol', str(args.protocol), '--prerequisites', str(args.prerequisites),
           '--private', str(args.private), '--ledger', str(args.ledger), '--lock', str(args.lock),
           '--max-blocks', str(secondary['planned_blocks']), '--deadline-epoch', str(plan['deadline_epoch']),
           '--gpu-budget-s', str(plan['gpu_remaining_s']),
           '--remaining-requests', str(plan['requests_remaining']),
           '--block-guard-s', str(plan['block_guard_s']), '--timeout', str(args.timeout)]
    result = subprocess.run(cmd, timeout=hard_timeout, env=os.environ.copy(), check=False)
    if result.returncode:
        raise SystemExit(result.returncode)


def calibration_main() -> None:
    """Calibrate one pair after its frozen prerequisites, with bounded resume."""
    p = argparse.ArgumentParser(description='Resume one secondary density calibration pair')
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--ledger-inventory', type=Path, required=True)
    p.add_argument('--budget', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--wait-for-prerequisites', action='store_true')
    p.add_argument('--poll-seconds', type=int, default=30)
    p.add_argument('--timeout', type=int, default=900)
    args = p.parse_args()
    spec = json.loads(args.spec.read_text())
    prior = {stage: (json.loads(Path(value['protocol']).read_text()),
                     [Path(x) for x in value['ledgers']])
             for stage, value in spec['prerequisites'].items()}
    inventory = {host: [Path(x) for x in paths]
                 for host, paths in json.loads(args.ledger_inventory.read_text()).items()}
    budget = json.loads(args.budget.read_text())
    while True:
        result = calibrate_group(spec, prior, inventory, budget,
                                 execute=args.execute, timeout=args.timeout,
                                 gpu_idle=local_gpu_idle() if args.execute else False)
        print(json.dumps(result, sort_keys=True), flush=True)
        if (result['status'] != 'prerequisite_incomplete' or
                not args.wait_for_prerequisites or
                time.time() + budget['block_guard_s'] >= budget['deadline_epoch']):
            break
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'calibrate-group':
        del sys.argv[1]
        calibration_main()
    else:
        main()
