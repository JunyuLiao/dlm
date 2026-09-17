"""50%-only scheduling/report overlay; all frozen numerical sources unchanged."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import xml.etree.ElementTree as ET

from experiments import diffusion_gemma_jl_fullbudget as fb
from experiments import diffusion_gemma_jl_accepted_round2 as accepted
from experiments import diffusion_gemma_jl_reference_first as rf
from experiments.diffusion_gemma_jl_focused import report as backend
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write

ROOT = fb.ROOT
MODULE = 'experiments.diffusion_gemma_jl_50_only_reference'
LABELS = ('full_centered_s50',)
CONDITIONS = ('dense', 'blasst_original_s50', 'blasst_aggressive_s50', 'mass_s50', *LABELS)
read, sha, frozen_write = fb.read, fb.sha, fb.frozen_write


def control(root):
    return root/'only50'


def scheduling_contract(root=ROOT, freeze=False):
    setup = deepcopy(fb.prepare(root)); contract = fb.execution(root)
    request = root/'user_scope_50_only.json'; authorization = read(request)
    if authorization['allowed_target_sparsities'] != [.5] or authorization['first_finish'] != 'full_centered_s50':
        raise ValueError('50%-only authorization does not match schedule')
    setup.update(conditions=list(CONDITIONS), targets=[.5],
        projected_methods_paused=True, follow_on_request=authorization)
    test_path = control(root)/'tests.xml'
    suites = ET.parse(test_path).getroot().findall('testsuite')
    counts = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=3, failures=0, errors=0, skipped=0):
        raise ValueError('50%-only scheduling tests incomplete')
    paths = [Path(__file__), Path('tests/test_jl_50_only_reference.py'), test_path, request,
        root/'execution_contract.json', root/'accepted_round2/extension_contract.json']
    paths += [root/'final_configs'/f'{label}.json' for label in CONDITIONS]
    extra = read(root/'accepted_round2/extension_contract.json')
    sources = {**contract['sources'], **extra['sources'], **{str(p): sha(p.read_bytes()) for p in paths}}
    fb.check_sources(sources)
    data = dict(schema='jl_50_only_reference_schedule_v1', labels=list(LABELS),
        conditions=list(CONDITIONS), expected_reference_outputs=80, expected_with_baselines=400,
        parent_execution_fingerprint=contract['fingerprint'], sources=sources,
        numerical_change=False, threshold_refitting=False, monitoring_interval_seconds=900,
        cancelled_75=True, follow_on_final_slots=240, follow_on_budget_clarification_pending=True)
    if freeze:
        frozen_write(control(root)/'scheduling_contract.json', data)
    elif (control(root)/'scheduling_contract.json').exists() and read(control(root)/'scheduling_contract.json') != data:
        raise ValueError('Frozen50-only scheduling contract changed')
    return setup, contract, data


def report_view(root, setup, contract):
    view = control(root)/'report'; view.mkdir(parents=True, exist_ok=True)
    for label in CONDITIONS:
        rf.link(view/'final_configs'/f'{label}.json', root/'final_configs'/f'{label}.json')
        stage = 'dense' if label == 'dense' else 'final'
        rf.link(view/stage/label, root/stage/label)
        if label != 'dense':
            for benchmark in ('aime26', 'longbench_v2'):
                rf.link(view/'policies'/benchmark/f'{label}.json', root/'policies'/benchmark/f'{label}.json')
    rf.link(view/'shared_state_index.json', root/'shared_state_index.json')
    index = []
    for item in read(root/'shared_diagnostics_index.json'):
        path = Path(item['path']); fb.check_sources({str(path): item['sha256']})
        ident = read(path)['identity']
        if ident['target'] == .5 and ident['name'] in {'full_centered', *fb.parent.BASELINES}:
            index.append(item)
    frozen_write(view/'shared_diagnostics_index.json', index)
    frozen_write(view/'scope.json', dict(conditions=list(CONDITIONS), expected=400,
        source_setup=str(root/'setup.json'), setup_sha256=sha((root/'setup.json').read_bytes()),
        excluded_targets=[.75], projected_shards_excluded=True,
        parent_execution_fingerprint=contract['fingerprint']))
    return view


def write_report(view, setup, rows, tasks, compared, diag, audit, policies):
    fb.write_report(view, setup, rows, tasks, compared, diag, audit, policies)
    audit.update(expected=400, complete=audit['completed'] == 400 and not audit['missing'] and not audit['violations'])
    path = view/'report.md'; text = path.read_text()
    text = text.replace('Status: INCOMPLETE, 400/720', 'Status: COMPLETE, 400/400') if audit['complete'] else text
    text = text.replace('/720 audited outputs:160 new reference outputs plus560 reused baselines',
        '/400 audited outputs:80 reference outputs plus320 reused baselines')
    text = text.replace('/400 audited outputs:160 new reference outputs plus560 reused baselines',
        '/400 audited outputs:80 reference outputs plus320 reused baselines')
    text = text.replace('Both50% and75% targets', 'Only the50% target')
    text = text.replace('The projected sweep is paused, not completed or scientifically excluded.',
        'The user cancelled all75% runs; three50%-only follow-on candidates are separately scheduled, not included in this reference report.')
    text = text.replace('Projected runs remain paused for an evidence-based decision.',
        'The user subsequently authorized centered Gaussian32, centered random-sign32 and one cancellation-guard follow-on, all at50%; these are not selected using the reference scores.')
    text = text.replace('experiments.diffusion_gemma_jl_fullbudget', MODULE)
    text += '\n##50%-only scope change\n\nThe side-chat request is pinned in user_scope_50_only.json. All75% final runs are cancelled, not scientific failures. Existing75% calibration, baselines, logs and superseded projection plans remain preserved outside this report. Full-centered50% thresholds, numerical kernels, prompts and budgets are unchanged. The running sample was allowed to checkpoint before replacing the two-target scheduler.\n'
    path.write_text(text)


def regenerate(root=ROOT):
    setup, contract, schedule = scheduling_contract(root)
    view = report_view(root, setup, contract)
    with patch.object(backend, 'prepare', lambda unused: setup), \
         patch.object(backend, 'execution', lambda unused: contract), \
         patch.object(backend, 'PROJECTED', {'full_centered': fb.CONFIG}), \
         patch.object(backend, 'TARGETS', (.5,)), \
         patch.object(backend, 'audit_policy', lambda unused, p, s, c: accepted.audit(root, p, s, c)), \
         patch.object(backend, 'smoke_audit', lambda unused, s, c: fb.inherited_smoke(root, s, c)), \
         patch.object(backend.common, 'shared_seed_summary', lambda unused: ([], {})), \
         patch.object(backend, 'write_report', write_report):
        audit = backend.regenerate(view)
    audit['sources'].update(schedule['sources'])
    for path in (control(root)/'scheduling_contract.json', view/'scope.json'):
        audit['sources'][str(path)] = sha(path.read_bytes())
    _write(view/'audit.json', audit)
    return audit


def verify(root=ROOT):
    view = control(root)/'report'; before = read(view/'audit.json')
    if not before['complete'] or before['completed'] != 400:
        raise ValueError('Require complete400-output reference50 report')
    fb.check_sources({str(view/p): v for p, v in before['artifacts'].items()})
    if regenerate(root) != before:
        raise ValueError('Raw-only50% report regeneration changed')
    proof = dict(passed=True, completed=400, inference_performed=False,
        audit_sha256=sha((view/'audit.json').read_bytes()))
    _write(view/'regeneration_verification.json', proof)
    return proof


def execute(root, command):
    if command == 'launch':
        scheduling_contract(root, freeze=True); fb.gpu_idle()
        with (control(root)/'supervisor.log').open('a', buffering=1) as log:
            child = subprocess.Popen([sys.executable, '-u', '-m', MODULE, 'supervise', '--output', str(root)],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(supervisor_pid=child.pid, reference_runs=80, cancelled_75=True)))
        return
    with patch.object(rf, 'control', control), patch.object(rf, 'MODULE', MODULE), \
         patch.object(rf, 'LABELS', LABELS), patch.object(rf, 'scheduling_contract', scheduling_contract), \
         patch.object(rf, 'regenerate', regenerate), patch.object(rf, 'verify', verify):
        result = {'work': rf.work, 'supervise': rf.supervise}[command](root)
    if command == 'work':
        path = control(root)/'terminal.json'; terminal = read(path)
        terminal.update(expected=400, cancelled_75=True)
        _write(path, terminal)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('launch', 'supervise', 'work', 'report', 'verify'))
    parser.add_argument('--output', type=Path, default=ROOT); args = parser.parse_args()
    if args.command in ('report', 'verify'):
        print(json.dumps((regenerate if args.command == 'report' else verify)(args.output)))
    else:
        execute(args.output, args.command)
