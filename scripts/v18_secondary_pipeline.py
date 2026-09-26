"""Windows-side, resumable CP3 coordinator. It never reads prompt or answer data.

The frozen JSON configuration names the already qualified two hosts, their local
paths, and the immutable deploy. Remote workers do their own quota checks; this
coordinator only advances after both ledgers and stage markers close.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shlex
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from pathlib import PurePosixPath

SSH = ('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15')
STAGES = (('core70', 'ruler_secondary70'), ('controls', 'ruler_secondary_allocation'))
EXPECTED = {'mpk': 'exouser@149.165.151.254', 'dllm': 'exouser@149.165.159.64'}
GOLD_SHA = 'c0d87ef6f81e07d04e13b34ab723a8680ac4965e650ed298c6bb0f839ef7cbfb'
ROOTS = {'mpk': '/media/volume/dllm-1/dyh/junyu_frontier_v18_20260926',
         'dllm': '/home/exouser/dyh/junyu_frontier_v18_20260926'}
SDPA_SHA = '87f933d1a2d8508df572da5c0748c6b24c22ff2b625796949957dcd86cc57564'


class PrimaryPendingDeadline(RuntimeError):
    """Primary never closed; no CP3 generation or receipt scoring may have begun."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_config(config: dict, deploy: str, budget: dict, now: float) -> None:
    if set(config['hosts']) != set(EXPECTED):
        raise ValueError('exactly the two qualified hosts required')
    for host, target in EXPECTED.items():
        item = config['hosts'][host]
        if item['ssh'] != target or item['hostname'] != host or not item['gpu_uuid']:
            raise ValueError('host, SSH endpoint, or GPU UUID drift')
        for name in ('root', 'python', 'model', 'library', 'torch_library', 'draft',
                     'calibration_manifest', 'policy_file', 'primary_protocol',
                     'aime_protocol', 'primary_calibration', 'old_scope_policies',
                     'ledger_inventory'):
            if not str(item[name]).startswith('/'):
                raise ValueError(f'{host}/{name} must be an absolute host path')
    if not deploy or '/' in deploy or '\\' in deploy or config['deploy'] != deploy:
        raise ValueError('immutable deploy identity mismatch')
    if config['budget_sha256'] != digest(Path(config['budget_local']).read_bytes()):
        raise ValueError('frozen campaign budget byte drift')
    if budget.get('schema') != 'v18_frozen_campaign_budget_v1' or not budget.get('hard_timeout_required'):
        raise ValueError('frozen hard budget required')
    if (not Path(config['cpu_qualification']).is_absolute() or
            set(config['qualification_sources']) !=
            {'scripts.v18_secondary', 'scripts.v18_secondary_coordinate',
             'experiments.value_direction_hopper.frontier_scope',
             'transformers.integrations.sdpa_attention'} or
            any(len(value) != 64 for value in config['qualification_sources'].values())):
        raise ValueError('complete frozen CPU qualification plan required')
    if config.get('torch_version') != '2.6.0+cu124' or config.get('torch_cuda') != '12.4':
        raise ValueError('pinned Torch CPU import identity required')


class Transport:
    def __init__(self, config: dict):
        self.config = config

    def path(self, host: str, *parts: str) -> str:
        return '/'.join((self.config['hosts'][host]['root'].rstrip('/'), *parts))

    def ssh(self, host: str, command: str, timeout: int = 60) -> str:
        return subprocess.check_output(['ssh', *SSH, self.config['hosts'][host]['ssh'], command],
                                       text=True, timeout=timeout).strip()

    def scp(self, source: str, destination: str, timeout: int = 300) -> None:
        subprocess.run(['scp', *SSH, source, destination], check=True, timeout=timeout)

    def read(self, host: str, path: str) -> bytes | None:
        command = f'if test -f {shlex.quote(path)}; then base64 -w0 {shlex.quote(path)}; fi'
        for attempt in range(3):
            cutoff = getattr(self, 'read_deadline', None)
            remaining = cutoff - time.time() if cutoff is not None else None
            if remaining is not None and remaining <= 0:
                raise PrimaryPendingDeadline('primary read reached original GPU cutoff')
            timeout = min(60, max(1, int(remaining))) if remaining is not None else 60
            try:
                encoded = self.ssh(host, command, timeout=timeout)
                break
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                if attempt == 2:
                    raise
                delay = 2 * (attempt + 1)
                if cutoff is not None:
                    delay = min(delay, max(0, cutoff - time.time()))
                    if delay <= 0:
                        raise PrimaryPendingDeadline('primary read reached original GPU cutoff')
                time.sleep(delay)
        return base64.b64decode(encoded, validate=True) if encoded else None

    def write_immutable(self, host: str, path: str, data: bytes) -> None:
        old = self.read(host, path)
        if old is not None:
            if old != data:
                raise ValueError(f'immutable remote file changed: {host}/{Path(path).name}')
            return
        encoded = base64.b64encode(data).decode()
        folder = path.rsplit('/', 1)[0]
        self.ssh(host, f'mkdir -p {shlex.quote(folder)} && '
                       f'printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(path)}')
        if self.read(host, path) != data:
            raise ValueError('remote immutable write verification failed')

    def python(self, host: str, args: list[str], timeout: int = 3600) -> str:
        item = self.config['hosts'][host]
        code = self.path(host, 'deploy', self.config['deploy'])
        env = item['env']
        prefix = f'cd {shlex.quote(code)} && ' + ' '.join(
            f'{key}={shlex.quote(value)}' for key, value in sorted(env.items())) + ' '
        return self.ssh(host, prefix + ' '.join(shlex.quote(x) for x in [item['python'], *args]), timeout)

    def copy_verified(self, source: str, src_path: str, dest: str, dst_path: str,
                      folder: Path) -> str:
        """Relay through Windows, verifying source and destination bytes."""
        local = folder / (source + '_' + Path(src_path).name)
        self.scp(f"{self.config['hosts'][source]['ssh']}:{src_path}", str(local))
        expected = digest(local.read_bytes())
        current = self.read(source, src_path)
        if current is None or digest(current) != expected:
            raise ValueError('source changed during relay')
        existing = self.read(dest, dst_path)
        if existing not in (None, b'') and digest(existing) != expected:
            raise ValueError('destination identity differs; no overwrite')
        if existing is None or existing == b'':
            self.ssh(dest, f'mkdir -p {shlex.quote(dst_path.rsplit("/", 1)[0])}')
            self.scp(str(local), f"{self.config['hosts'][dest]['ssh']}:{dst_path}")
        if digest(self.read(dest, dst_path) or b'') != expected:
            raise ValueError('destination relay hash mismatch')
        return expected


def ledger_inventory(root: str) -> dict[str, list[str]]:
    """Charge every possible CP3 worker exactly once, including future points."""
    return {
        host: ([f'{root}/evaluation/ledgers/{host}_{dataset}.jsonl'
                for dataset in ('ruler', 'aime')] +
               ([f'{root}/secondary/calibration_{group}/secondary_calibration_p{point}_ledger.jsonl'
                 for group in ('core70', 'controls') for point in range(5)] if host == 'mpk' else []) +
               [f'{root}/secondary/ledgers/{host}_{stage}.jsonl'
                for _, stage in STAGES])
        for host in EXPECTED}


def freeze_config(inputs: dict, deploy: str, out: Path) -> dict:
    """Freeze the small Windows controller config from actual host protocols."""
    if set(inputs['hosts']) != set(EXPECTED):
        raise ValueError('both host input paths required')
    hosts = {}
    source_root = Path(__file__).resolve().parent.parent
    sources = {
        'scripts.v18_secondary': digest((source_root / 'scripts/v18_secondary.py').read_bytes()),
        'scripts.v18_secondary_coordinate': digest((source_root / 'scripts/v18_secondary_coordinate.py').read_bytes()),
        'experiments.value_direction_hopper.frontier_scope': digest(
            (source_root / 'experiments/value_direction_hopper/frontier_scope.py').read_bytes()),
        'transformers.integrations.sdpa_attention': SDPA_SHA}
    for host in EXPECTED:
        root = ROOTS[host]
        target = EXPECTED[host]
        proto_path = root + '/evaluation/ruler4k_primary_protocol.json'
        aime_path = root + '/evaluation/aime26_primary_protocol.json'
        def remote_json(path):
            raw = subprocess.check_output(['ssh', *SSH, target,
                'cat ' + shlex.quote(path)], text=True, timeout=60)
            return json.loads(raw)
        primary, aime = remote_json(proto_path), remote_json(aime_path)
        if (primary.get('stage') != 'ruler4k_primary' or aime.get('stage') != 'aime26_primary' or
                not any(item.get('host') == host and item.get('gpu_uuid')
                        for item in primary.get('assigned_hosts', []))):
            raise ValueError('actual frozen host protocol/assignment missing')
        uuid = next(item['gpu_uuid'] for item in primary['assigned_hosts'] if item['host'] == host)
        if host == 'mpk':
            python = '/home/exouser/miniconda3/envs/ljy_dlm/bin/python'
            library = '/home/exouser/dyh/numerical_qk_reuse_native_20260924/build_cp1/value_direction_db080045f7a5fbce.so'
            torch_library = '/home/exouser/dyh/numerical_qk_reuse_native_20260924/build_cp1/torch_4c65c048754f9fb7/value_direction_torch_4c65c048754f9fb7.so'
            env = {'PYTHONPATH': 'src:.', 'HF_HUB_OFFLINE': '1', 'OMP_NUM_THREADS': '4',
                   'TRITON_CACHE_DIR': '/media/volume/dllm-1/dyh/numerical_qk_overnight_runtime_20260925/tc_warmup'}
        else:
            runtime = root + '/bridge/runtime'
            python = runtime + '/miniconda3/envs/ljy_dlm/bin/python'
            library = root + '/bridge/binaries/value_direction_db080045f7a5fbce.so'
            torch_library = root + '/bridge/binaries/value_direction_torch_4c65c048754f9fb7.so'
            env = {'PYTHONPATH': runtime + '/.local/lib/python3.10/site-packages:src:.',
                   'PYTHONNOUSERSITE': '1', 'HF_HUB_OFFLINE': '1', 'OMP_NUM_THREADS': '4',
                   'LD_LIBRARY_PATH': root + '/bridge/binaries:' + runtime +
                   '/.local/lib/python3.10/site-packages/torch/lib:' +
                   runtime + '/miniconda3/envs/ljy_dlm/lib',
                   'TRITON_CACHE_DIR': root + '/triton_cache'}
        host_input = inputs['hosts'][host]
        hosts[host] = dict(ssh=target, hostname=host, gpu_uuid=uuid, root=root, python=python,
                           model=primary['model_path'], library=library, torch_library=torch_library,
                           draft=host_input['draft'],
                           calibration_manifest=host_input['calibration_manifest'],
                           policy_file=primary['policy_file'],
                           old_scope_policies=primary['policy_file'],
                           primary_protocol=proto_path, aime_protocol=aime_path,
                           primary_calibration=primary['calibration_file'],
                           ledger_inventory=root + '/secondary/campaign_ledger_inventory.json', env=env)
    budget_path = Path(inputs['budget_local']).resolve()
    config = dict(deploy=deploy, budget_local=str(budget_path),
                  budget_sha256=digest(budget_path.read_bytes()), hosts=hosts,
                  cpu_qualification=str(Path(inputs['cpu_qualification']).resolve()),
                  qualification_sources=sources, torch_version='2.6.0+cu124',
                  torch_cuda='12.4', ruler_gold=inputs['ruler_gold'],
                  ruler_root=inputs['ruler_root'])
    validate_config(config, deploy, json.loads(budget_path.read_text()), time.time())
    transport = Transport(config)
    for host in EXPECTED:
        path = hosts[host]['ledger_inventory']
        transport.write_immutable(host, path,
                                  (json.dumps(ledger_inventory(hosts[host]['root']),
                                              sort_keys=True, indent=2) + '\n').encode())
    payload = json.dumps(config, sort_keys=True, indent=2) + '\n'
    if out.exists() and out.read_text() != payload:
        raise ValueError('frozen coordinator config differs; no overwrite')
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload)
    return config


def primary_ready(t: Transport) -> bool:
    for host in EXPECTED:
        root = t.config['hosts'][host]['root']
        for stage in ('initial', 'aime', 'remainder'):
            prefix = f'{root}/evaluation/status/{host}_{stage}'
            started, done = (t.read(host, prefix + '.' + suffix + '.json')
                             for suffix in ('started', 'done'))
            if not started or not done:
                return False
            a, b = json.loads(started), json.loads(done)
            if a.get('start') != b.get('start') or b.get('rc') != 0:
                raise RuntimeError('failed primary stage; secondary cannot start')
    for dataset, stage in (('ruler', 'ruler4k_primary'), ('aime', 'aime26_primary')):
        payload = t.read('mpk', t.path('mpk', 'scoring', dataset + '_redacted_summary.json'))
        if not payload:
            return False
        item = json.loads(payload)
        protocol_file = t.config['hosts']['mpk'].get(
            'primary_protocol' if dataset == 'ruler' else 'aime_protocol')
        protocol_bytes = t.read('mpk', protocol_file) if protocol_file else None
        if not protocol_bytes:
            return False
        protocol = json.loads(protocol_bytes)
        if (item.get('schema') != 'v18_redacted_summary_v1' or item.get('stage') != stage or
                protocol.get('stage') != stage or item.get('protocol_id') != protocol.get('protocol_id') or
                item.get('complete') is not True or
                item.get('recorded_executions') != len(protocol.get('schedule', [])) or
                item.get('planned_executions') != len(protocol.get('schedule', []))):
            raise RuntimeError('primary offline summary incomplete or mismatched')
    return True


def qualify_cpu(t: Transport) -> dict:
    """Run only CPU tests and loaded-source checks after primary GPU work is quiet."""
    config = t.config
    receipt_path = Path(config['cpu_qualification'])
    plan = dict(deploy=config['deploy'], sources=config['qualification_sources'],
                torch=config['torch_version'], torch_cuda=config['torch_cuda'],
                tests=['tests/test_v18_secondary.py', 'tests/test_v18_native_scope.py'],
                hosts={host: dict(ssh=config['hosts'][host]['ssh'],
                                  gpu_uuid=config['hosts'][host]['gpu_uuid']) for host in EXPECTED})
    plan_sha = digest(json.dumps(plan, sort_keys=True).encode())
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get('schema') != 'v18_secondary_cpu_qualification_v1' or
                receipt.get('plan_sha256') != plan_sha or receipt.get('qualified') is not True or
                set(receipt.get('hosts', {})) != set(EXPECTED)):
            raise ValueError('existing CPU qualification receipt differs from frozen plan')
        return receipt
    sources = {}
    code = ('import hashlib,importlib,inspect,json,sys,torch; '
            'names=json.loads(sys.argv[1]); '
            'print(json.dumps(dict(sources={name:dict(path=inspect.getfile(importlib.import_module(name)), '
            'sha256=hashlib.sha256(open(inspect.getfile(importlib.import_module(name)),"rb").read()).hexdigest()) '
            'for name in names},torch=torch.__version__,torch_cuda=torch.version.cuda,torch_path=torch.__file__),sort_keys=True))')
    for host in EXPECTED:
        t.python(host, ['-m', 'pytest', '-q', *plan['tests']], timeout=900)
        actual = json.loads(t.python(host, ['-c', code,
                            json.dumps(sorted(config['qualification_sources']))], timeout=180))
        paths = {name: value.get('path') for name, value in actual.get('sources', {}).items()}
        hashes = {name: value.get('sha256') for name, value in actual.get('sources', {}).items()}
        expected_deploy = t.path(host, 'deploy', config['deploy'])
        expected_paths = {
            'scripts.v18_secondary': expected_deploy + '/scripts/v18_secondary.py',
            'scripts.v18_secondary_coordinate': expected_deploy + '/scripts/v18_secondary_coordinate.py',
            'experiments.value_direction_hopper.frontier_scope':
                expected_deploy + '/experiments/value_direction_hopper/frontier_scope.py'}
        if (hashes != config['qualification_sources'] or
                any(paths.get(name) != path for name, path in expected_paths.items()) or
                actual.get('torch') != config['torch_version'] or
                actual.get('torch_cuda') != config['torch_cuda']):
            raise ValueError(f'{host} loaded CPU source hash differs from frozen plan')
        if host == 'dllm':
            prefix = t.path(host, 'bridge', 'runtime')
            if (paths.get('transformers.integrations.sdpa_attention') !=
                    prefix + '/miniconda3/envs/ljy_dlm/lib/python3.10/site-packages/transformers/integrations/sdpa_attention.py' or
                    actual.get('torch_path') !=
                    prefix + '/.local/lib/python3.10/site-packages/torch/__init__.py'):
                raise ValueError('dllm imported outside the qualified private runtime')
        uuid = t.ssh(host, 'nvidia-smi --query-gpu=uuid --format=csv,noheader').splitlines()[0].strip()
        if uuid != config['hosts'][host]['gpu_uuid']:
            raise ValueError(f'{host} GPU UUID differs from frozen assignment')
        sources[host] = dict(gpu_uuid=uuid, source_hashes=hashes,
                             imported_paths=paths, torch_path=actual['torch_path'],
                             torch=actual['torch'], torch_cuda=actual['torch_cuda'],
                             cpu_tests_passed=True)
    receipt = dict(schema='v18_secondary_cpu_qualification_v1', qualified=True,
                   plan_sha256=plan_sha, hosts=sources)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    return receipt


def marker(t: Transport, stage: str, host: str, suffix: str) -> dict | None:
    raw = t.read(host, t.path(host, 'secondary', 'status', f'{host}_{stage}.{suffix}.json'))
    return json.loads(raw) if raw else None


def worker_state(t: Transport, stage: str, host: str) -> str:
    started, done = marker(t, stage, host, 'started'), marker(t, stage, host, 'done')
    if done:
        return 'complete' if started and done.get('start') == started.get('start') and done.get('rc') == 0 else 'failed'
    if started:
        pid = started.get('pid')
        if type(pid) is not int or pid <= 0 or t.ssh(host, f'kill -0 {pid} 2>/dev/null && echo yes || echo no') != 'yes':
            return 'orphaned'
        return 'running'
    return 'absent'


def launch_once(t: Transport, stage: str, host: str, args: list[str]) -> None:
    if worker_state(t, stage, host) != 'absent':
        return
    item = t.config['hosts'][host]
    status = t.path(host, 'secondary', 'status', f'{host}_{stage}')
    log = t.path(host, 'secondary', 'logs', f'{host}_{stage}.log')
    deploy = t.path(host, 'deploy', t.config['deploy'])
    env = ' '.join(f'{key}={shlex.quote(value)}' for key, value in sorted(item['env'].items()))
    command = ' '.join(shlex.quote(x) for x in [item['python'], *args])
    body = (f'set -eu; mkdir -p {shlex.quote(status.rsplit("/", 1)[0])} '
            f'{shlex.quote(log.rsplit("/", 1)[0])}; '
            f'test ! -e {shlex.quote(status + ".started.json")} || exit 91; '
            f'start=$(date +%s); '
            f'printf \'{{"pid":%s,"start":%s}}\\n\' "$$" "$start" > {shlex.quote(status + ".started.json")}; '
            f'cd {shlex.quote(deploy)}; set +e; {env} {command}; rc=$?; set -e; '
            f'printf \'{{"start":%s,"end":%s,"rc":%s}}\\n\' "$start" "$(date +%s)" "$rc" > '
            f'{shlex.quote(status + ".done.json")}; exit "$rc"')
    dispatch = f'setsid -f bash -lc {shlex.quote(body)} > {shlex.quote(log)} 2>&1 < /dev/null &'
    t.ssh(host, f'mkdir -p {shlex.quote(status.rsplit("/", 1)[0])} '
                f'{shlex.quote(log.rsplit("/", 1)[0])}')
    try:
        t.ssh(host, dispatch)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, TimeoutError):
        # An unknown dispatch is never repeated, even when no marker appeared.
        raise RuntimeError(f'{host}/{stage} dispatch uncertain; inspect host before recovery')
    until = time.monotonic() + 15
    while marker(t, stage, host, 'started') is None:
        if time.monotonic() > until:
            raise RuntimeError(f'{host}/{stage} dispatched without marker; no automatic retry')
        time.sleep(1)


def await_workers(t: Transport, stage: str, args_by_host: dict[str, list[str]],
                  deadline: float, poll: int) -> None:
    for host in EXPECTED:
        state = worker_state(t, stage, host)
        if state == 'absent':
            launch_once(t, stage, host, args_by_host[host])
        elif state in ('failed', 'orphaned'):
            raise RuntimeError(f'{host}/{stage} {state}; preserve ledger and inspect')
    while True:
        states = {host: worker_state(t, stage, host) for host in EXPECTED}
        print(json.dumps({'stage': stage, 'states': states}), flush=True)
        if all(x == 'complete' for x in states.values()):
            return
        if any(x not in ('running', 'complete') for x in states.values()):
            raise RuntimeError(f'{stage} incomplete worker; no retry')
        if time.time() >= deadline:
            raise RuntimeError('campaign GPU deadline reached; preserve live workers')
        time.sleep(poll)


def assert_same_protocol(t: Transport, stage: str) -> None:
    data = [json.loads(t.read(host, t.path(host, 'secondary', stage + '_protocol.json')) or b'null')
            for host in EXPECTED]
    if any(not isinstance(x, dict) or x.get('stage') != stage or x.get('status') != 'frozen'
           for x in data):
        raise ValueError('host-local secondary protocol missing')
    if (data[0].get('logical_protocol_sha256') != data[1].get('logical_protocol_sha256') or
            data[0].get('protocol_id') != data[1].get('protocol_id')):
        raise ValueError('host-local logical secondary protocol drift')


def make_input(t: Transport, group: str) -> dict:
    mpk = t.config['hosts']['mpk']
    stage = 'ruler_secondary70' if group == 'core70' else 'ruler_secondary_allocation'
    root = mpk['root']
    source = dict(group=group, manifest=mpk['calibration_manifest'],
                  old_scope_policies=mpk['old_scope_policies'],
                  primary_calibration=mpk['primary_calibration'],
                  primary_protocol=mpk['primary_protocol'], aime_protocol=mpk['aime_protocol'],
                  primary_ledgers=[f'{root}/evaluation/ledgers/{h}_ruler.jsonl' for h in EXPECTED],
                  aime_ledgers=[f'{root}/evaluation/ledgers/{h}_aime.jsonl' for h in EXPECTED],
                  ruler_summary=f'{root}/scoring/ruler_redacted_summary.json',
                  aime_summary=f'{root}/scoring/aime_redacted_summary.json',
                  calibration_host='mpk', calibration_gpu_uuid=mpk['gpu_uuid'],
                  model=mpk['model'], library=mpk['library'], torch_library=mpk['torch_library'],
                  policy_file=mpk['policy_file'], out=f'{root}/secondary/calibration_{group}',
                  lock=f'{root}/gpu.lock', authorization='v18-frozen-secondary')
    if group == 'controls':
        source.update(preceding_secondary=f'{root}/secondary/ruler_secondary70_protocol.json',
                      preceding_ledgers=[f'{root}/secondary/ledgers/{h}_ruler_secondary70.jsonl' for h in EXPECTED])
    return source


def run_pipeline(t: Transport, budget: dict, *, poll: int, folder: Path,
                 status_callback=None) -> None:
    if poll < 60:
        raise ValueError('poll interval must be at least 60 seconds')
    deadline = budget['deadline_epoch']
    t.read_deadline = deadline - budget['block_guard_s']
    while True:
        if time.time() >= t.read_deadline:
            raise PrimaryPendingDeadline('primary prerequisites did not close before GPU cutoff')
        try:
            ready = primary_ready(t)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
            ready = False
            pending = dict(primary='connection_pending', error_type=type(error).__name__)
            print(json.dumps(pending), flush=True)
            if status_callback:
                status_callback(status='waiting_primary', connection_pending=True,
                                last_read_error=type(error).__name__)
        else:
            if ready:
                break
            print(json.dumps({'primary': 'waiting_for_closed_stages_and_offline_summaries'}), flush=True)
            if status_callback:
                status_callback(status='waiting_primary', connection_pending=False)
        remaining = t.read_deadline - time.time()
        if remaining <= 0:
            raise PrimaryPendingDeadline('primary prerequisites did not close before GPU cutoff')
        time.sleep(min(poll, remaining))
    if status_callback:
        status_callback(status='running_secondary', connection_pending=False)
    t.read_deadline = None
    qualify_cpu(t)
    for host in EXPECTED:
        budget_path = t.path(host, 'deploy', t.config['deploy'],
                             'results/junyu_frontier_v18_20260926/campaign_budget.json')
        remote_budget = t.read(host, budget_path)
        if remote_budget is None or digest(remote_budget) != t.config['budget_sha256']:
            raise ValueError('remote immutable campaign budget differs')
    if all(worker_state(t, stage, host) == 'complete'
           for _, stage in STAGES for host in EXPECTED):
        score_secondary(t, folder, budget)
        return
    for dataset in ('ruler', 'aime'):
        name = dataset + '_redacted_summary.json'
        t.copy_verified('mpk', t.path('mpk', 'scoring', name), 'dllm',
                        t.path('dllm', 'scoring', name), folder)
    for group, stage in STAGES:
        if time.time() + budget['block_guard_s'] >= deadline:
            raise RuntimeError('no time for a complete secondary block')
        mpkroot = t.config['hosts']['mpk']['root']
        spec_input = f'{mpkroot}/secondary/calibration_{group}_input.json'
        spec_path = f'{mpkroot}/secondary/calibration_{group}_spec.json'
        t.write_immutable('mpk', spec_input,
                          (json.dumps(make_input(t, group), sort_keys=True, indent=2) + '\n').encode())
        t.python('mpk', ['-m', 'scripts.v18_secondary', 'freeze-calibration-spec',
                         '--input', spec_input, '--out', spec_path])
        cal = f'{mpkroot}/secondary/calibration_{group}/calibrated.json'
        prior = f'{mpkroot}/secondary/calibration_{group}/prerequisites.json'
        inventory = t.config['hosts']['mpk']['ledger_inventory']
        cal_args = ['-m', 'scripts.v18_secondary_coordinate', 'calibrate-group',
                    '--spec', spec_path, '--ledger-inventory', inventory,
                    '--budget', t.path('mpk', 'deploy', t.config['deploy'],
                                       'results/junyu_frontier_v18_20260926/campaign_budget.json'),
                    '--execute', '--wait-for-prerequisites']
        # Calibration has a durable started marker; an uncertain dispatch is terminal.
        cal_stage = 'calibrate_' + group
        if worker_state(t, cal_stage, 'mpk') == 'absent':
            launch_once(t, cal_stage, 'mpk', cal_args)
        while worker_state(t, cal_stage, 'mpk') != 'complete':
            state = worker_state(t, cal_stage, 'mpk')
            if state != 'running' or time.time() >= deadline:
                raise RuntimeError(f'{cal_stage} {state}; preserve history')
            time.sleep(poll)
        if t.read('mpk', cal) is None:
            raise RuntimeError('calibration exited without a frozen result')
        # The calibrated bytes are host-independent. Never recompute on dllm.
        dllmcal = t.path('dllm', 'secondary', 'calibration_' + group, 'calibrated.json')
        t.copy_verified('mpk', cal, 'dllm', dllmcal, folder)
        for point in range(5):
            name = f'secondary_calibration_p{point}_ledger.jsonl'
            source_ledger = f'{mpkroot}/secondary/calibration_{group}/{name}'
            if t.read('mpk', source_ledger) is not None:
                t.copy_verified('mpk', source_ledger, 'dllm',
                                t.path('dllm', 'secondary', 'calibration_' + group, name), folder)
        for host in EXPECTED:
            item = t.config['hosts'][host]
            root = item['root']
            args = ['-m', 'scripts.v18_secondary', 'freeze-eval', '--stage', stage,
                    '--draft', item['draft'], '--primary-protocol', item['primary_protocol'],
                    '--aime-protocol', item['aime_protocol'], '--calibration-manifest',
                    item['calibration_manifest'], '--secondary-calibration',
                    f'{root}/secondary/calibration_{group}/calibrated.json',
                    '--policy-file', item['policy_file'], '--model', item['model'],
                    '--library', item['library'], '--torch-library', item['torch_library'],
                    '--out', f'{root}/secondary']
            if group == 'controls':
                args.extend(['--preceding-secondary', f'{root}/secondary/ruler_secondary70_protocol.json'])
            t.python(host, args)
        assert_same_protocol(t, stage)
        # Per-host prerequisite and inventory files are immutable, with locally valid paths.
        for host in EXPECTED:
            item = t.config['hosts'][host]
            root = item['root']
            prior_map = {'ruler4k_primary': dict(protocol=item['primary_protocol'],
                       ledgers=[f'{root}/evaluation/ledgers/{h}_ruler.jsonl' for h in EXPECTED]),
                       'aime26_primary': dict(protocol=item['aime_protocol'],
                       ledgers=[f'{root}/evaluation/ledgers/{h}_aime.jsonl' for h in EXPECTED])}
            if group == 'controls':
                prior_map['ruler_secondary70'] = dict(
                    protocol=f'{root}/secondary/ruler_secondary70_protocol.json',
                    ledgers=[f'{root}/secondary/ledgers/{h}_ruler_secondary70.jsonl' for h in EXPECTED])
            t.write_immutable(host, f'{root}/secondary/{stage}_prerequisites.json',
                              (json.dumps(prior_map, sort_keys=True, indent=2) + '\n').encode())
        ledgers = {h: t.path(h, 'secondary', 'ledgers', f'{h}_{stage}.jsonl') for h in EXPECTED}
        for host in EXPECTED:
            for h in EXPECTED:
                path = t.path(host, 'secondary', 'ledgers', f'{h}_{stage}.jsonl')
                if t.read(host, path) is None:
                    t.write_immutable(host, path, b'')
        args_by_host = {}
        for host in EXPECTED:
            root = t.config['hosts'][host]['root']
            args_by_host[host] = ['-m', 'scripts.v18_secondary_coordinate', '--protocol',
                f'{root}/secondary/{stage}_protocol.json', '--prerequisites',
                f'{root}/secondary/{stage}_prerequisites.json', '--ledger-inventory',
                t.config['hosts'][host]['ledger_inventory'], '--budget',
                t.path(host, 'deploy', t.config['deploy'],
                       'results/junyu_frontier_v18_20260926/campaign_budget.json'),
                '--private', f'{root}/secondary/private/{stage}', '--ledger',
                f'{root}/secondary/ledgers/{host}_{stage}.jsonl', '--lock', f'{root}/gpu.lock',
                '--stage-ledger', f'{root}/secondary/ledgers/mpk_{stage}.jsonl',
                '--stage-ledger', f'{root}/secondary/ledgers/dllm_{stage}.jsonl',
                '--primary-summary', f'ruler4k_primary={root}/scoring/ruler_redacted_summary.json',
                '--primary-summary', f'aime26_primary={root}/scoring/aime_redacted_summary.json',
                '--execute', '--wait-for-prerequisites']
        await_workers(t, stage, args_by_host, deadline, poll)
        for host in EXPECTED:
            other = 'dllm' if host == 'mpk' else 'mpk'
            t.copy_verified(host, ledgers[host], other,
                            t.path(other, 'secondary', 'ledgers', f'{host}_{stage}.jsonl'), folder)
        # A zero exit can mean budget stop. Require every frozen execution to close.
        check = ('from pathlib import Path; import json,sys; '
                 'from scripts.v18_secondary import stage_completion; '
                 'p=json.loads(Path(sys.argv[1]).read_text()); '
                 'print(json.dumps(stage_completion(p,[Path(x) for x in sys.argv[2:]])))')
        completed = json.loads(t.python('mpk', ['-c', check,
            f'{mpkroot}/secondary/{stage}_protocol.json',
            f'{mpkroot}/secondary/ledgers/mpk_{stage}.jsonl',
            f'{mpkroot}/secondary/ledgers/dllm_{stage}.jsonl']))
        if completed.get('complete') is not True:
            raise RuntimeError(f'{stage} has unfinished blocks; preserve outputs and stop')
        print(json.dumps({'stage': stage, 'complete': True,
                          'recorded_executions': completed['completed']}), flush=True)
    score_secondary(t, folder, budget)


def closed_stages(t: Transport) -> list[str]:
    """Return stages with a frozen protocol and no live writers, including partials."""
    result = []
    for _, stage in STAGES:
        states = {host: worker_state(t, stage, host) for host in EXPECTED}
        if 'running' in states.values():
            raise RuntimeError('secondary GPU writer is still running; scoring deferred')
        for host in EXPECTED:
            started = marker(t, stage, host, 'started')
            if started:
                pid = started.get('pid')
                if type(pid) is not int or pid <= 0:
                    raise RuntimeError('invalid secondary supervisor PID')
                process_state = t.ssh(host, f'ps -o stat= -p {pid} 2>/dev/null || true').strip()
                if process_state and not process_state.startswith('Z'):
                    raise RuntimeError('secondary supervisor has not exited')
        if all(x == 'absent' for x in states.values()):
            continue
        if t.read('mpk', t.path('mpk', 'secondary', stage + '_protocol.json')) is None:
            raise RuntimeError('started secondary stage lacks a frozen protocol')
        result.append(stage)
    if result:
        for host in EXPECTED:
            active = t.ssh(host, 'nvidia-smi --query-compute-apps=pid --format=csv,noheader')
            if any(line.strip() and line.strip() != '[Not Supported]' for line in active.splitlines()):
                raise RuntimeError('GPU process remains; private receipts may still be changing')
    return result


def score_secondary(t: Transport, folder: Path, budget: dict) -> list[str]:
    """Move private receipts only after writers close; retain redacted partials."""
    stages = closed_stages(t)
    if not stages:
        return []
    dllm = t.config['hosts']['dllm']
    mpk = t.config['hosts']['mpk']
    for stage in stages:
        for host in EXPECTED:
            source = t.path(host, 'secondary', 'ledgers', f'{host}_{stage}.jsonl')
            if t.read(host, source) is not None:
                peer = 'dllm' if host == 'mpk' else 'mpk'
                t.copy_verified(host, source, peer,
                                t.path(peer, 'secondary', 'ledgers', f'{host}_{stage}.jsonl'), folder)
        for host in EXPECTED:
            local_ledger = t.path('mpk', 'secondary', 'ledgers', f'{host}_{stage}.jsonl')
            if t.read('mpk', local_ledger) is None:
                t.write_immutable('mpk', local_ledger, b'')
        t.ssh('dllm', 'mkdir -p ' + shlex.quote(
            t.path('dllm', 'secondary', 'private', stage)))
    gold = t.config['ruler_gold']
    if t.ssh('mpk', 'sha256sum ' + shlex.quote(gold)).split()[0] != GOLD_SHA:
        raise ValueError('original scorer-only RULER gold byte drift')
    for stage in stages:
        extracted = relay_private_stage(t, stage, folder)
        protocol = t.path('mpk', 'secondary', stage + '_protocol.json')
        scoring = t.path('mpk', 'secondary', 'scoring')
        lock = scoring + '/' + stage + '_scorer_lock.json'
        output = scoring + '/' + stage + '_redacted_summary.json'
        t.python('mpk', ['-m', 'scripts.v18_secondary', 'freeze-scorer-lock',
                         '--protocol', protocol, '--ruler-root', t.config['ruler_root'],
                         '--out', lock])
        args = ['-m', 'scripts.v18_secondary', 'score', '--protocol', protocol,
                '--primary-protocol', mpk['primary_protocol'], '--gold', gold,
                '--scorer-lock', lock, '--ruler-root', t.config['ruler_root'],
                '--out', output,
                '--ledger', t.path('mpk', 'secondary', 'ledgers', 'mpk_' + stage + '.jsonl'),
                '--ledger', t.path('mpk', 'secondary', 'ledgers', 'dllm_' + stage + '.jsonl'),
                '--primary-ledger', t.path('mpk', 'evaluation', 'ledgers', 'mpk_ruler.jsonl'),
                '--primary-ledger', t.path('mpk', 'evaluation', 'ledgers', 'dllm_ruler.jsonl'),
                '--private-root', 'dllm=' + extracted + '/' + stage,
                '--primary-private-root', 'dllm=' + t.path('mpk', 'scoring', 'dllm_private', 'ruler')]
        t.python('mpk', args, timeout=max(1, int(budget['deadline_epoch'] +
                         60 * budget['scoring_minutes_reserved'] - time.time())))
        result = json.loads(t.read('mpk', output) or b'null')
        if not isinstance(result, dict) or result.get('schema') != 'v18_secondary_redacted_summary_v1':
            raise RuntimeError(stage + ' redacted score is malformed; preserve outputs')
        print(json.dumps({'stage': stage, 'offline_summary': output,
                          'complete': result.get('complete'),
                          'unfinished_blocks': len(result.get('unfinished_blocks', [])),
                          'recorded_executions': result['recorded_executions']}), flush=True)
    return stages


def relay_private_stage(t: Transport, stage: str, folder: Path) -> str:
    """One immutable private archive per stage permits later partial recovery."""
    dllm, mpk = (t.config['hosts'][host] for host in ('dllm', 'mpk'))
    archive = folder / f'dllm_{stage}_private.tar.gz'
    target = t.path('mpk', 'secondary', 'scoring', archive.name)
    existing = t.ssh('mpk', f'if test -f {shlex.quote(target)}; '
                     f'then sha256sum {shlex.quote(target)} | cut -d " " -f1; fi')
    if existing:
        t.scp(f"{mpk['ssh']}:{target}", str(archive), timeout=3600)
    else:
        command = f'tar -C {shlex.quote(dllm["root"] + "/secondary/private")} -czf - {shlex.quote(stage)}'
        with archive.open('xb') as output:
            subprocess.run(['ssh', *SSH, dllm['ssh'], command], stdout=output,
                           check=True, timeout=3600)
    with tarfile.open(archive, 'r:gz') as stream:
        entries = stream.getmembers()
        if not entries or any(not (x.isfile() or x.isdir()) or
                              PurePosixPath(x.name).is_absolute() or
                              '..' in PurePosixPath(x.name).parts or
                              PurePosixPath(x.name).parts[0] != stage for x in entries):
            raise ValueError('unexpected stage private archive entry')
    sha = digest(archive.read_bytes())
    if existing and existing != sha:
        raise ValueError('previous stage archive has different bytes')
    if not existing:
        t.ssh('mpk', 'mkdir -p ' + shlex.quote(target.rsplit('/', 1)[0]))
        t.scp(str(archive), f"{mpk['ssh']}:{target}", timeout=3600)
    if t.ssh('mpk', 'sha256sum ' + shlex.quote(target)).split()[0] != sha:
        raise ValueError('stage archive relay changed bytes')
    extracted = t.path('mpk', 'secondary', 'scoring', 'dllm_private_' + stage)
    marker_path = extracted + '/.archive_sha256'
    prior = t.read('mpk', marker_path)
    if prior is None:
        if t.ssh('mpk', f'if test -e {shlex.quote(extracted)}; then echo exists; fi'):
            raise ValueError('unmarked stage extraction exists')
        t.ssh('mpk', f'mkdir {shlex.quote(extracted)} && '
                     f'tar -xzf {shlex.quote(target)} -C {shlex.quote(extracted)} && '
                     f'printf %s {shlex.quote(sha)} > {shlex.quote(marker_path)}', timeout=3600)
    if (t.read('mpk', marker_path) or b'').decode() != sha:
        raise ValueError('stage extraction identity drift')
    return extracted


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deploy', required=True)
    p.add_argument('--config', type=Path)
    p.add_argument('--freeze-config-input', type=Path,
                   help='CPU-only compact path input; writes --config immutably')
    p.add_argument('--poll-seconds', type=int, default=60)
    p.add_argument('--outcome', type=Path,
                   help='Durable redacted status; default CONFIG.outcome.json')
    a = p.parse_args()
    if a.config is None:
        p.error('--config required')
    if a.freeze_config_input:
        freeze_config(json.loads(a.freeze_config_input.read_text()), a.deploy, a.config)
        print(json.dumps({'frozen_config': str(a.config),
                          'sha256': digest(a.config.read_bytes())}), flush=True)
        return
    config_bytes = a.config.read_bytes()
    config = json.loads(config_bytes)
    budget = json.loads(Path(config['budget_local']).read_text())
    validate_config(config, a.deploy, budget, time.time())
    outcome_path = a.outcome or a.config.with_suffix('.outcome.json')
    config_sha = digest(config_bytes)
    previous = json.loads(outcome_path.read_text()) if outcome_path.exists() else {}
    if previous and (previous.get('config_sha256') != config_sha or
                     previous.get('deploy') != a.deploy):
        raise ValueError('previous outcome belongs to another frozen controller')
    def outcome(**fields):
        previous.update(fields)
        outcome_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = outcome_path.with_name(outcome_path.name + '.tmp')
        temporary.write_text(json.dumps(previous, sort_keys=True, indent=2) + '\n')
        os.replace(temporary, outcome_path)
    outcome(schema='v18_secondary_pipeline_outcome_v1', deploy=a.deploy,
            config_sha256=config_sha, start_epoch=previous.get('start_epoch', time.time()),
            status='running')
    print(json.dumps({'pipeline_config_sha256': digest(config_bytes),
                      'deploy': a.deploy, 'status': 'adopting'}), flush=True)
    with tempfile.TemporaryDirectory(prefix='v18_secondary_relay_') as path:
        transport = Transport(config)
        try:
            run_pipeline(transport, budget, poll=a.poll_seconds, folder=Path(path),
                         status_callback=outcome)
            outcome(status='complete', end_epoch=time.time(), partial_scoring='not_needed')
        except Exception as error:
            # A failed or budget-stopped GPU stage still has immutable first
            # receipts. Publish a redacted partial if every writer has closed.
            if isinstance(error, PrimaryPendingDeadline):
                partial, reason = 'not_started', 'primary_deadline'
            else:
                try:
                    scored = score_secondary(transport, Path(path), budget)
                    partial = 'published' if scored else 'deferred'
                    reason = None if scored else 'no_closed_secondary_stage'
                except Exception as scoring_error:
                    partial, reason = 'deferred', type(scoring_error).__name__
                    print(json.dumps({'partial_scoring': 'deferred',
                                      'reason': type(scoring_error).__name__}), flush=True)
            outcome(status='error', error_type=type(error).__name__,
                    error_message=str(error), partial_scoring=partial,
                    partial_scoring_reason=reason, end_epoch=time.time())
            raise


if __name__ == '__main__':
    main()
