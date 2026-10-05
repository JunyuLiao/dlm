"""Adopt and advance the two-host frozen v18 stages using status metadata only.

Run on the coordinator workstation. Reads stage status and SHA-256 hashes over
SSH every 60 seconds; never reads prompts, gold, completions, or private receipts.
Each stage launcher is immutable and handles its own GPU/request/deadline guards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

HOSTS = {
    'mpk': ('exouser@149.165.151.254', '/media/volume/dllm-1/dyh/junyu_frontier_v18_20260926'),
    'dllm': ('exouser@149.165.159.64', '/home/exouser/dyh/junyu_frontier_v18_20260926'),
}
DEPLOY = 'driver_5b14a21'
STAGES = ('initial', 'aime', 'remainder')
SSH_OPTIONS = ('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15')


class UncertainDispatch(RuntimeError):
    """Remote launch may exist despite a missing first status read."""


def marker_state(started, done):
    if done is not None:
        if started is None or done.get('rc') != 0 or done.get('start') != started.get('start'):
            return 'failed'
        return 'complete'
    return 'running' if started is not None else 'absent'


def remote(host, command):
    for attempt in range(3):
        try:
            return subprocess.check_output(['ssh', *SSH_OPTIONS, HOSTS[host][0], command],
                                           text=True, timeout=60).strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt == 2:
                raise
            time.sleep(60)


def marker(host, stage, suffix):
    root = HOSTS[host][1]
    path = f'{root}/evaluation/status/{host}_{stage}.{suffix}.json'
    value = remote(host, f'if test -f {shlex.quote(path)}; then cat {shlex.quote(path)}; fi')
    return json.loads(value) if value else None


def launch(host, stage, *, deploy=DEPLOY, segment=None, budget_name='campaign_budget.json'):
    target, root = HOSTS[host]
    marker_stage = segment or stage
    script = f'{root}/deploy/{deploy}/scripts/v18_run_stage.sh'
    log = f'{root}/logs/eval_{marker_stage}.log'
    command = (f'setsid -f bash {shlex.quote(script)} {shlex.quote(deploy)} {shlex.quote(stage)} '
               f'{shlex.quote(marker_stage)} {shlex.quote(budget_name)} '
               f'> {shlex.quote(log)} 2>&1 < /dev/null &')
    try:
        subprocess.run(['ssh', *SSH_OPTIONS, target, command], check=True, timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        # Unknown dispatch result: read a marker, never issue a second launch.
        if marker(host, marker_stage, 'started') is None:
            raise UncertainDispatch(f'{host}/{marker_stage} launch uncertain and no marker; manual inspection required')
    until = time.monotonic() + 15
    while marker(host, marker_stage, 'started') is None:
        if time.monotonic() >= until:
            raise UncertainDispatch(f'{host}/{marker_stage} dispatched but no started marker; no automatic retry')
        time.sleep(1)


def alive(host, pid):
    if type(pid) is not int or pid <= 0:
        return False
    result = remote(host, f'if kill -0 {pid} 2>/dev/null; then echo yes; else echo no; fi')
    return result == 'yes'


def wait_stage(stage, *, adopt, poll_seconds, deadline_epoch, deploy=DEPLOY, segment=None,
               budget_name='campaign_budget.json', hosts=None):
    marker_stage = segment or stage
    selected_hosts = tuple(HOSTS if hosts is None else hosts)
    if not selected_hosts or not set(selected_hosts) <= set(HOSTS):
        raise ValueError('continuation must name one or both assigned hosts')
    for host in selected_hosts:
        state = marker_state(marker(host, marker_stage, 'started'), marker(host, marker_stage, 'done'))
        if state == 'failed':
            raise RuntimeError(f'{host}/{marker_stage} failed; inspect immutable status and ledger, no automatic retry')
        if state == 'absent':
            if adopt:
                raise RuntimeError(f'{host}/{marker_stage} has no started marker to adopt')
            if deploy == DEPLOY and segment is None and budget_name == 'campaign_budget.json':
                launch(host, stage)
            else:
                launch(host, stage, deploy=deploy, segment=segment, budget_name=budget_name)
    while True:
        if time.time() >= deadline_epoch:
            raise RuntimeError('campaign hard deadline reached; inspect live workers and preserve ledgers')
        states = {host: marker_state(marker(host, marker_stage, 'started'), marker(host, marker_stage, 'done'))
                  for host in selected_hosts}
        print(json.dumps({'stage': marker_stage, 'states': states}), flush=True)
        if 'failed' in states.values() or 'absent' in states.values():
            raise RuntimeError(f'{marker_stage} failed or lost a started marker; no automatic retry')
        if all(state == 'complete' for state in states.values()):
            return
        for host, state in states.items():
            if state == 'running':
                started = marker(host, marker_stage, 'started')
                if not alive(host, started.get('pid')):
                    # The wrapper writes done immediately before exiting. A
                    # supervisor can disappear between the first done read and
                    # this liveness check; accept only the matching fresh done.
                    fresh_started = marker(host, marker_stage, 'started')
                    fresh_done = marker(host, marker_stage, 'done')
                    fresh_state = marker_state(fresh_started, fresh_done)
                    if fresh_started == started and fresh_state == 'complete':
                        states[host] = 'complete'
                        continue
                    raise RuntimeError(f'{host}/{marker_stage} started marker has no live supervisor; no retry')
        if all(state == 'complete' for state in states.values()):
            return
        time.sleep(poll_seconds)


def sync_ledger(source, dataset, staging):
    peer = 'dllm' if source == 'mpk' else 'mpk'
    src_host, src_root = HOSTS[source]
    dst_host, dst_root = HOSTS[peer]
    basename = f'{source}_{dataset}.jsonl'
    src = f'{src_root}/evaluation/ledgers/{basename}'
    dst = f'{dst_root}/evaluation/ledgers/{basename}'
    local = staging / basename
    subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                    f'{src_host}:{src}', str(local)], check=True, timeout=120)
    local_hash = hashlib.sha256(local.read_bytes()).hexdigest()
    source_hash = remote(source, f'sha256sum {shlex.quote(src)}').split()[0]
    if local_hash != source_hash:
        raise RuntimeError('source ledger changed during transfer; preserve and inspect stage')
    subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                    str(local), f'{dst_host}:{dst}'], check=True, timeout=120)
    dest_hash = remote(peer, f'sha256sum {shlex.quote(dst)}').split()[0]
    if dest_hash != local_hash:
        raise RuntimeError('peer ledger copy hash mismatch')
    return dict(source=source, dataset=dataset, sha256=local_hash)


def own_process_alive(host, pid, command_token):
    """Check both PID liveness and command identity; a reused PID is not ours."""
    if type(pid) is not int or pid <= 0:
        raise ValueError('known worker/supervisor PID missing')
    line = remote(host, f'ps -ww -p {pid} -o stat=,args= 2>/dev/null || true')
    if not line:
        return False
    status, _, command = line.partition(' ')
    return not status.startswith('Z') and command_token in command


def open_worker_pids(host):
    """Only these two redacted campaign ledgers can identify own GPU workers."""
    root = HOSTS[host][1]
    pending = []
    for dataset in ('ruler', 'aime'):
        path = f'{root}/evaluation/ledgers/{host}_{dataset}.jsonl'
        payload = remote(host, f'if test -f {shlex.quote(path)}; then cat {shlex.quote(path)}; fi')
        open_starts = []
        for line in payload.splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get('event') == 'start':
                open_starts.append(event)
            elif event.get('event') == 'worker_end':
                if not open_starts:
                    raise RuntimeError(f'{host}/{dataset} has worker_end without start')
                open_starts.pop(0)
        for start in open_starts:
            if type(start.get('pid')) is not int or start['pid'] <= 0:
                raise RuntimeError(f'{host}/{dataset} open worker start lacks a verifiable PID')
            pending.append(dict(dataset=dataset, pid=start['pid']))
    return pending


def wait_known_writers_quiet(stages, *, poll_seconds, deadline_epoch):
    """Never finalize while a known supervisor or GPU worker still owns a PID."""
    while True:
        snapshot, active = {}, []
        for host in HOSTS:
            host_stages = {}
            for stage in stages:
                started, done = marker(host, stage, 'started'), marker(host, stage, 'done')
                state = marker_state(started, done)
                if started is not None:
                    supervisor_live = own_process_alive(host, started.get('pid'), 'v18_run_stage.sh')
                    if supervisor_live:
                        active.append(dict(host=host, stage=stage, role='supervisor'))
                else:
                    supervisor_live = False
                host_stages[stage] = dict(state=state, supervisor_live=supervisor_live,
                                          done_marker=done is not None)
            workers = open_worker_pids(host)
            for worker in workers:
                worker['live'] = own_process_alive(host, worker['pid'], 'scripts.v18_evaluate')
                if worker['live']:
                    active.append(dict(host=host, dataset=worker['dataset'], role='worker'))
            snapshot[host] = dict(stages=host_stages, open_workers=workers)
        if not active:
            return snapshot
        if time.time() >= deadline_epoch:
            raise RuntimeError('known GPU writers remain live at CPU finalization cutoff: ' +
                               json.dumps(active, sort_keys=True))
        time.sleep(poll_seconds)


def sync_all_ledgers(staging):
    """Hash-check latest redacted ledgers for both hosts and both datasets."""
    return [sync_ledger(host, dataset, staging)
            for dataset in ('ruler', 'aime') for host in HOSTS]


def completion_detail(protocol, ledgers):
    """Never retry a partly written block or a non-clean budget stop."""
    from scripts.v13_seed_runs import execution_key
    from scripts.v18_stage_driver import completed_keys

    done = completed_keys(protocol, ledgers)
    blocks = {}
    for entry in protocol['schedule']:
        blocks.setdefault(entry['block'], []).append(entry)
    partial, missing = [], []
    state_by_block = {}
    for block, entries in blocks.items():
        count = sum(execution_key(entry) in done for entry in entries)
        if 0 < count < len(entries):
            partial.append(block)
            state_by_block[block] = 'partial'
        elif count == 0:
            missing.append(block)
            state_by_block[block] = 'missing'
        else:
            state_by_block[block] = 'full'
    prefix_violations = []
    for host in HOSTS:
        assigned_blocks = [block for block in sorted(blocks)
                           if protocol['block_assignments'][str(block)]['host'] == host]
        seen_missing = False
        for block in assigned_blocks:
            if state_by_block[block] == 'missing':
                seen_missing = True
            elif seen_missing and state_by_block[block] == 'full':
                prefix_violations.append(block)
    last_end = {}
    for host, path in zip(HOSTS, ledgers):
        pending = None
        for line in path.read_text().splitlines():
            if line.strip():
                event = json.loads(line)
                if event.get('event') == 'start':
                    if pending is not None or event.get('host') != host or \
                            event.get('protocol_id') != protocol['protocol_id'] or \
                            event.get('model_revision') != protocol['model_revision']:
                        raise ValueError('AIME start/host/protocol identity mismatch')
                    pending = event
                elif event.get('event') == 'worker_end':
                    if pending is None or event.get('host') != host:
                        raise ValueError('AIME worker_end without matching host start')
                    last_end[host] = event.get('status')
                    pending = None
        if pending is not None:
            raise ValueError('AIME worker start has no terminal event')
    missing_hosts = {protocol['block_assignments'][str(block)]['host'] for block in missing}
    return dict(complete=len(done) == len(protocol['schedule']),
                completed=len(done), planned=len(protocol['schedule']),
                protocol_id=protocol['protocol_id'], partial_blocks=sorted(partial),
                missing_blocks=sorted(missing), missing_hosts=sorted(missing_hosts),
                prefix_violations=sorted(prefix_violations),
                clean_stop=not partial and not prefix_violations and all(last_end.get(host) == 'stopped_before_block'
                                               for host in missing_hosts),
                latest_worker_end=last_end)


def panel_complete(staging, dataset):
    """Check the exact frozen panel against both latest host ledgers."""
    source, root = HOSTS['mpk']
    if dataset not in ('ruler', 'aime'):
        raise ValueError('unknown frozen panel')
    name = 'aime26_primary_protocol.json' if dataset == 'aime' else 'ruler4k_primary_protocol.json'
    path = f'{root}/evaluation/{name}'
    local = staging / name
    subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                    f'{source}:{path}', str(local)], check=True, timeout=120)
    local_hash = hashlib.sha256(local.read_bytes()).hexdigest()
    if local_hash != remote('mpk', f'sha256sum {shlex.quote(path)}').split()[0]:
        raise RuntimeError('frozen protocol changed during completion check')
    protocol = json.loads(local.read_text())
    ledgers = [staging / f'{host}_{dataset}.jsonl' for host in HOSTS]
    return completion_detail(protocol, ledgers)


def aime_complete(staging):
    return panel_complete(staging, 'aime')


def orphan_private_receipts(host):
    """Count missing-ledger receipts on the assigned host without reading them."""
    root = HOSTS[host][1]
    python = ('/home/exouser/miniconda3/envs/ljy_dlm/bin/python' if host == 'mpk' else
              root + '/bridge/runtime/miniconda3/envs/ljy_dlm/bin/python')
    code = ('import json,sys,pathlib; '
            'p=json.load(open(sys.argv[1])); '
            'done={e["execution_key"] for line in open(sys.argv[2]) if line.strip() '
            'for e in [json.loads(line)] if e.get("event")=="run"}; '
            'host=sys.argv[4]; base=pathlib.Path(sys.argv[3]); '
            'missing=[e for e in p["schedule"] if '
            'p["block_assignments"][str(e["block"])]["host"]==host and '
            'f"{e[\'cell_id\']}:{e[\'role\']}:{e[\'repeat\']}" not in done]; '
            'print(sum((base/"cells"/e["cell_id"]/(e["role"]+str(e["repeat"])+".json")).exists() '
            'for e in missing))')
    arguments = [python, '-c', code,
                 root + '/evaluation/aime26_primary_protocol.json',
                 root + f'/evaluation/ledgers/{host}_aime.jsonl',
                 root + '/private_eval/aime', host]
    return int(remote(host, ' '.join(shlex.quote(arg) for arg in arguments)))


def continuation_ready(completion, *, orphan_counts):
    if completion['complete']:
        return True
    if completion['partial_blocks'] or not completion['missing_blocks'] or not completion['clean_stop']:
        return False
    return all(count == 0 for count in orphan_counts.values())


def discovered_segments():
    """Read only numbered marker names; unknown launches require manual review."""
    found = {}
    for host, (_, root) in HOSTS.items():
        directory = root + '/evaluation/status'
        command = (f'if test -d {shlex.quote(directory)}; then '
                   f'find {shlex.quote(directory)} -maxdepth 1 -type f '
                   f'-name {shlex.quote(host + "_aime_c*.started.json")} -printf "%f\\n"; fi')
        names = remote(host, command).splitlines()
        segments = set()
        for name in names:
            match = re.fullmatch(re.escape(host) + r'_(aime_c\d{3})\.started\.json', name)
            if not match:
                raise ValueError('unexpected numbered continuation marker')
            segments.add(match.group(1))
        found[host] = segments
    return found


def persist_outcome(path, outcome):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(outcome, indent=2, sort_keys=True) + '\n')
    os.replace(temp, path)


def recovery_audit(previous, previous_bytes):
    """Keep the exact earlier failure visible when a clean status is adopted."""
    history = list(previous.get('recovery_audit', []))
    history.append(dict(previous_outcome_sha256=hashlib.sha256(previous_bytes).hexdigest(),
                        previous_started_epoch=previous.get('started_epoch'),
                        previous_finished_epoch=previous.get('finished_epoch'),
                        gpu_stop_reason=previous.get('gpu_stop_reason'),
                        stage_error=previous.get('stage_error'),
                        finalization=previous.get('finalization'),
                        finalization_error=previous.get('finalization_error')))
    return history


def completion_main(args, budget):
    """Adopt old AIME, then append only wholly unstarted frozen blocks."""
    if args.start_at != 'aime' or not args.continuation_deploy or '/' in args.continuation_deploy:
        raise ValueError('completion coordinator must adopt AIME with one immutable deploy')
    if (budget.get('deadline_is_gpu_cutoff') is not True or
            budget.get('request_cap') != 7000 or
            budget.get('request_cap_by_host') != {'mpk': 3500, 'dllm': 3500} or
            budget.get('gpu_cap_s_by_host') != {'mpk': 86400, 'dllm': 86400} or
            budget.get('deadline_epoch') != 1790573520 or
            args.budget.name != 'campaign_budget_extension_20260926.json'):
        raise ValueError('approved bounded completion budget required')
    deadline = budget['deadline_epoch']
    cpu_deadline = deadline + 60 * budget['scoring_minutes_reserved']
    budget_sha = hashlib.sha256(args.budget.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    if args.outcome.exists():
        previous_bytes = args.outcome.read_bytes()
        outcome = json.loads(previous_bytes)
        if (outcome.get('schema') != 'v18_completion_coordinator_outcome_v1' or
                outcome.get('original_deploy') != DEPLOY or
                outcome.get('continuation_deploy') != args.continuation_deploy or
                outcome.get('bounded_budget_sha256') != budget_sha or
                not isinstance(outcome.get('segments'), list)):
            raise ValueError('existing continuation outcome has a different frozen identity')
        outcome['recovery_audit'] = recovery_audit(outcome, previous_bytes)
        outcome['resumed_epoch'] = time.time()
        outcome['gpu_stop_reason'] = 'pending'
        outcome['finalization'] = 'pending'
        for terminal in ('stage_error', 'finalization_error', 'finished_epoch'):
            outcome.pop(terminal, None)
    else:
        outcome = dict(schema='v18_completion_coordinator_outcome_v1',
                       original_deploy=DEPLOY, continuation_deploy=args.continuation_deploy,
                       bounded_budget_sha256=budget_sha, started_epoch=time.time(),
                       start_at='aime', segments=[], gpu_stop_reason='pending', finalization='pending')
    persist_outcome(args.outcome, outcome)
    with tempfile.TemporaryDirectory(prefix='v18_completion_') as folder:
        staging = Path(folder)
        known_stages = ['aime'] + [item['segment'] for item in outcome['segments']]
        if outcome.get('remainder_intended'):
            known_stages.append('remainder')
        safe_to_finalize = False
        try:
            wait_stage('aime', adopt=True, poll_seconds=args.poll_seconds,
                       deadline_epoch=deadline)
            expected = {host: {item['segment'] for item in outcome['segments']
                               if host in item['hosts']} for host in HOSTS}
            if discovered_segments() != expected:
                raise UncertainDispatch('numbered marker without matching durable launch intent')
            for item in outcome['segments']:
                if (not re.fullmatch(r'aime_c\d{3}', item['segment']) or
                        not item['hosts'] or not set(item['hosts']) <= set(HOSTS)):
                    raise ValueError('invalid persisted numbered segment assignment')
                wait_stage('aime', adopt=True, poll_seconds=args.poll_seconds,
                           deadline_epoch=deadline, deploy=args.continuation_deploy,
                           segment=item['segment'], budget_name=args.budget.name,
                           hosts=item['hosts'])
            safe_to_finalize = True
            sync_all_ledgers(staging)
            completion = aime_complete(staging)
            outcome['aime_completion'] = completion
            persist_outcome(args.outcome, outcome)
            if not completion['complete']:
                wait_known_writers_quiet(known_stages, poll_seconds=args.poll_seconds,
                                         deadline_epoch=cpu_deadline)
                for host in HOSTS:
                    root = HOSTS[host][1]
                    remote_budget = (f'{root}/deploy/{args.continuation_deploy}/results/'
                                     'junyu_frontier_v18_20260926/' + args.budget.name)
                    if remote(host, f'sha256sum {shlex.quote(remote_budget)}').split()[0] != budget_sha:
                        raise RuntimeError('remote bounded budget byte drift')
                for index in range(len(outcome['segments']) + 1, 33):
                    orphan_counts = {host: orphan_private_receipts(host) for host in HOSTS}
                    if not continuation_ready(completion, orphan_counts=orphan_counts):
                        raise RuntimeError('AIME has partial/uncertain block or orphan receipt; no retry')
                    segment = f'aime_c{index:03d}'
                    known_stages.append(segment)
                    hosts = completion['missing_hosts']
                    outcome['segments'].append(dict(segment=segment, hosts=hosts,
                                                    status='planned'))
                    persist_outcome(args.outcome, outcome)
                    safe_to_finalize = False
                    wait_stage('aime', adopt=False, poll_seconds=args.poll_seconds,
                               deadline_epoch=deadline, deploy=args.continuation_deploy,
                               segment=segment, budget_name=args.budget.name,
                               hosts=hosts)
                    safe_to_finalize = True
                    sync_all_ledgers(staging)
                    completion = aime_complete(staging)
                    outcome['segments'][-1].update(status='closed', completion=completion)
                    outcome['aime_completion'] = completion
                    persist_outcome(args.outcome, outcome)
                    if completion['complete']:
                        break
                    wait_known_writers_quiet(known_stages, poll_seconds=args.poll_seconds,
                                             deadline_epoch=cpu_deadline)
                else:
                    raise RuntimeError('32 clean continuation segments exhausted')
            remainder_adopt = bool(outcome.get('remainder_intended'))
            outcome['remainder_intended'] = True
            persist_outcome(args.outcome, outcome)
            if 'remainder' not in known_stages:
                known_stages.append('remainder')
            safe_to_finalize = False
            wait_stage('remainder', adopt=remainder_adopt, poll_seconds=args.poll_seconds,
                       deadline_epoch=deadline, deploy=args.continuation_deploy,
                       budget_name=args.budget.name)
            safe_to_finalize = True
            sync_all_ledgers(staging)
            ruler = panel_complete(staging, 'ruler')
            outcome['ruler_completion'] = ruler
            if not ruler['complete']:
                raise RuntimeError('RULER remainder incomplete under bounded extension')
            outcome['gpu_stop_reason'] = 'all_frozen_panels_closed'
            persist_outcome(args.outcome, outcome)
        except (Exception, KeyboardInterrupt) as error:
            outcome['gpu_stop_reason'] = 'stage_failure'
            outcome['stage_error'] = dict(type=type(error).__name__, message=str(error)[:1000])
            persist_outcome(args.outcome, outcome)
        if not safe_to_finalize:
            outcome['finalization'] = 'deferred_uncertain_writer'
            outcome['finished_epoch'] = time.time()
            persist_outcome(args.outcome, outcome)
            raise SystemExit(1)
        try:
            outcome['known_writers'] = wait_known_writers_quiet(
                known_stages, poll_seconds=args.poll_seconds, deadline_epoch=cpu_deadline)
            outcome['ledger_copies_verified'] = sync_all_ledgers(staging)
            argv = json.loads(args.after_stages_command_file.read_text())
            if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or not x for x in argv):
                raise ValueError('finalization hook must be a JSON argv array')
            remaining = cpu_deadline - time.time()
            if remaining <= 0:
                raise TimeoutError('bounded CPU finalization window exhausted')
            subprocess.run(argv, check=True, timeout=remaining)
            outcome['finalization'] = 'complete'
        except (Exception, KeyboardInterrupt) as error:
            outcome['finalization'] = 'failed'
            outcome['finalization_error'] = dict(type=type(error).__name__, message=str(error)[:1000])
        outcome['finished_epoch'] = time.time()
        persist_outcome(args.outcome, outcome)
    if outcome['gpu_stop_reason'] != 'all_frozen_panels_closed' or outcome['finalization'] != 'complete':
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--start-at', choices=STAGES, default='initial',
                        help='Adopt an already started stage when resuming a coordinator')
    parser.add_argument('--budget', type=Path, default=Path('results/junyu_frontier_v18_20260926/campaign_budget.json'))
    parser.add_argument('--continuation-deploy',
                        help='Use a new immutable driver only after adopted AIME cleanly stops')
    parser.add_argument('--after-stages-command-file', type=Path, required=True,
                        help='JSON argv for the vetted CPU-only finalization hook')
    parser.add_argument('--outcome', type=Path,
                        default=Path('results/junyu_frontier_v18_20260926/coordinator_outcome.json'),
                        help='Persistent CPU coordinator result, including partial recovery errors')
    args = parser.parse_args()
    if args.poll_seconds < 60:
        parser.error('status polling must be at least 60 seconds')
    budget = json.loads(args.budget.read_text())
    if args.continuation_deploy:
        completion_main(args, budget)
        return
    deadline_epoch = budget['deadline_epoch']
    cpu_deadline = deadline_epoch + 60 * budget['scoring_minutes_reserved']
    outcome = dict(schema='v18_coordinator_outcome_v1', deploy=DEPLOY,
                   started_epoch=time.time(), gpu_cutoff_epoch=deadline_epoch,
                   cpu_finalization_deadline_epoch=cpu_deadline,
                   start_at=args.start_at, stage_results=[], finalization='pending')
    if args.outcome.exists():
        previous_bytes = args.outcome.read_bytes()
        previous = json.loads(previous_bytes)
        if (previous.get('schema') != outcome['schema'] or previous.get('deploy') != DEPLOY or
                previous.get('start_at') != args.start_at or
                previous.get('gpu_cutoff_epoch') != deadline_epoch):
            raise ValueError('existing coordinator outcome has a different frozen identity')
        outcome['recovery_audit'] = recovery_audit(previous, previous_bytes)
    persist_outcome(args.outcome, outcome)
    with tempfile.TemporaryDirectory(prefix='v18_ledgers_') as folder:
        staging = Path(folder)
        known_stages = []
        try:
            for stage in STAGES[STAGES.index(args.start_at):]:
                known_stages.append(stage)
                wait_stage(stage, adopt=stage == args.start_at, poll_seconds=args.poll_seconds,
                           deadline_epoch=deadline_epoch)
                dataset = 'aime' if stage == 'aime' else 'ruler'
                synced = [sync_ledger(host, dataset, staging) for host in HOSTS]
                outcome['stage_results'].append(dict(stage=stage, status='closed',
                                                     ledger_copies_verified=synced))
                persist_outcome(args.outcome, outcome)
                print(json.dumps({'stage': stage, 'ledger_copies_verified': synced}), flush=True)
                if stage == 'aime':
                    completion = aime_complete(staging)
                    outcome['aime_completion'] = completion
                    persist_outcome(args.outcome, outcome)
                    if not completion['complete']:
                        outcome['gpu_stop_reason'] = 'clean_partial_aime'
                        break
        except (Exception, KeyboardInterrupt) as exc:
            outcome['gpu_stop_reason'] = 'stage_failure'
            outcome['stage_error'] = dict(type=type(exc).__name__, message=str(exc)[:1000],
                                          stage=known_stages[-1] if known_stages else None)
            persist_outcome(args.outcome, outcome)
        if 'gpu_stop_reason' not in outcome:
            outcome['gpu_stop_reason'] = 'all_stages_closed'
        try:
            outcome['known_writers'] = wait_known_writers_quiet(
                STAGES, poll_seconds=args.poll_seconds, deadline_epoch=cpu_deadline)
            outcome['writers_quiet'] = True
            persist_outcome(args.outcome, outcome)
            outcome['ledger_copies_verified'] = sync_all_ledgers(staging)
            persist_outcome(args.outcome, outcome)
            argv = json.loads(args.after_stages_command_file.read_text())
            if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or not x for x in argv):
                raise ValueError('--after-stages-command-file must be a JSON argv array')
            remaining = cpu_deadline - time.time()
            if remaining <= 0:
                raise TimeoutError('CPU finalization reservation exhausted before final hook')
            subprocess.run(argv, check=True, timeout=remaining)
            outcome['finalization'] = 'complete'
        except (Exception, KeyboardInterrupt) as exc:
            outcome['finalization'] = 'failed'
            outcome['finalization_error'] = dict(type=type(exc).__name__, message=str(exc)[:1000])
        outcome['finished_epoch'] = time.time()
        persist_outcome(args.outcome, outcome)
        print(json.dumps({'gpu_stop_reason': outcome['gpu_stop_reason'],
                          'finalization': outcome['finalization'],
                          'outcome': str(args.outcome)}), flush=True)
    if outcome['gpu_stop_reason'] != 'all_stages_closed' or outcome['finalization'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
