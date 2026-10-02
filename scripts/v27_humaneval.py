"""Private HumanEval preparation and isolated, first-output-only test scoring.

Chat-adapted complete-function prompt; native thinking/8192 decoding unchanged.
Dataset, prompts, tests and completions stay private. Public output is aggregate
or qualified redacted cells. No package or model installation is performed.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import secrets
import subprocess

DATA_REVISION = '6d43fb980f9fee3c892a914eda09951f772ad10d'
PROMPT_STYLE = 'chat_complete_function_v1'
EXTRACTOR = 'final_first_python_fence_or_plain_complete_function_v1'
SANDBOX = 'unprivileged_bwrap_readonly_net_pid_user_isolated_mpk_v1'
PREFIX = ('Complete the Python function below. Return the complete Python solution '
          'in a single fenced python code block, without a test harness.\n\n')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def contract(raw_sha, scorer_sha):
    return dict(dataset_revision=DATA_REVISION, dataset_sha256=raw_sha,
                prompt_style=PROMPT_STYLE, extractor=EXTRACTOR, sandbox=SANDBOX,
                scorer_sha256=scorer_sha, thinking=True, generation_budget=8192,
                test_wall_timeout_s=5, test_cpu_limit_s=3, test_memory_mb=512,
                quality_metric='test_pass@1; EOS not required; capped outputs reported')


def validate_contract(value):
    if not isinstance(value, dict):
        raise ValueError('HumanEval task contract absent')
    for key in ('dataset_sha256', 'scorer_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', value.get(key, '')):
            raise ValueError('HumanEval source hash absent')
    if value != contract(value['dataset_sha256'], value['scorer_sha256']):
        raise ValueError('HumanEval task/extractor/sandbox contract drift')


def prepare_rows(tasks, encode):
    manifests, gold = [], {}
    seen = set()
    for task in tasks:
        task_id = task['task_id']
        if not re.fullmatch(r'HumanEval/\d+', task_id) or task_id in seen:
            raise ValueError('invalid/duplicate HumanEval task identity')
        seen.add(task_id)
        qid = 'humaneval/' + task_id.split('/')[1]
        prompt = PREFIX + task['prompt']
        tokens = encode(prompt)
        manifests.append(dict(id=qid, source_id=task_id, benchmark='humaneval',
                              prompt=prompt, prompt_hash=sha(prompt.encode()),
                              prompt_tokens=tokens, prompt_token_count=len(tokens),
                              thinking=True, generation_budget=8192))
        gold[qid] = {k: task[k] for k in ('task_id', 'prompt', 'test', 'entry_point')}
    return manifests, gold


def final_text(raw):
    # Same final-channel boundary as the existing native thinking task scorer.
    if '<channel|>' not in raw:
        return ''
    text = raw.rsplit('<channel|>', 1)[1]
    for token in ('<turn|>', '<|endoftext|>', '<eos>'):
        text = text.split(token, 1)[0]
    return text.strip()


def extract_code(raw, entry_point):
    text = final_text(raw)
    if not text:
        return None, 'no_final_response'
    fences = re.findall(r'```([^\n`]*)\n(.*?)```', text, re.S)
    if fences:
        selected = next((body for lang, body in fences if lang.strip().lower() in ('', 'python', 'py')), None)
        if selected is None:
            return None, 'no_python_fence'
        text = selected.strip()
    try:
        module = ast.parse(text)
    except (SyntaxError, ValueError):
        return None, 'syntax_error'
    if not any(isinstance(n, ast.FunctionDef) and n.name == entry_point for n in module.body):
        return None, 'missing_entry_point'
    return text, 'parsed'


def program_for(problem, code):
    # Preserve original imports preceding the task's signature. Tests/check are
    # added only in the separate scoring process, never the generation manifest.
    stem = problem['prompt']
    header = stem[:re.search(r'^def\s+', stem, re.M).start()]
    return header + '\n' + code + '\n' + problem['test'] + '\ncheck(' + problem['entry_point'] + ')\n'


BOOTSTRAP = """
import io, json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
resource.setrlimit(resource.RLIMIT_AS, (512*1024*1024, 512*1024*1024))
resource.setrlimit(resource.RLIMIT_FSIZE, (32*1024*1024, 32*1024*1024))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
source = sys.stdin.read()
sys.stdin = io.StringIO('')
try:
    exec(compile(source, '<candidate>', 'exec'), {})
except BaseException:
    sys.exit(1)
else:
    print(sys.argv[1])
"""


def run_test(program):
    marker = 'HE_PASS_' + secrets.token_hex(24)
    # Unprivileged user-namespace bwrap (no sudo on the shared hosts). Only mpk allows unprivileged user
    # namespaces (AppArmor restricts dllm/dlm2), so test execution runs on mpk; sandbox_preflight() enforces it.
    cmd = ['/usr/bin/bwrap',
           '--ro-bind', '/usr', '/usr', '--ro-bind', '/lib', '/lib', '--ro-bind', '/lib64', '/lib64',
           '--unshare-all', '--die-with-parent', '--new-session', '--cap-drop', 'ALL',
           '--uid', '65534', '--gid', '65534', '--proc', '/proc', '--dev', '/dev',
           '--tmpfs', '/tmp', '--chdir', '/tmp', '--clearenv',
           '--setenv', 'PATH', '/usr/bin', '--setenv', 'HOME', '/tmp', '--setenv', 'TMPDIR', '/tmp',
           '/usr/bin/python3', '-I', '-c', BOOTSTRAP, marker]
    with __import__('tempfile').TemporaryFile() as out, __import__('tempfile').TemporaryFile() as err:
        try:
            r = subprocess.run(cmd, input=program.encode(), stdout=out, stderr=err, timeout=5)
        except subprocess.TimeoutExpired:
            return False, 'timeout'
        err.seek(0)
        if b'bwrap:' in err.read(4096) or r.returncode == 126:
            raise RuntimeError('HumanEval OS sandbox unavailable')
        out.seek(0, 2)
        length = out.tell()
        out.seek(max(0, length - 256))
        passed = r.returncode == 0 and marker.encode() in out.read().splitlines()
        return passed, 'passed' if passed else 'test_failed'


def sandbox_preflight():
    assert run_test('assert 2 + 2 == 4')[0]
    assert not run_test('assert False')[0]
    # Network is isolated; host paths and credentials are not mounted.
    assert not run_test("open('/home/exouser/.ssh/id_rsa').read()")[0]
    assert not run_test("import socket; socket.create_connection(('1.1.1.1', 80), timeout=.1)")[0]
    assert not run_test('import os; os._exit(0)')[0]


def score_panel(protocol_path, binding_path, ledger_paths, roots, gold_path, out_prefix):
    from scripts.v13_seed_runs import execution_key
    from scripts.v20_score import _first_receipt
    from scripts.v21_score import load_records, build_cells, cell_rows
    from scripts.v21_run import validate_protocol
    protocol = json.loads(protocol_path.read_bytes())
    validate_protocol(protocol)
    if set(protocol['ids']) != {'humaneval'}:
        raise ValueError('HumanEval scorer requires HumanEval-only protocol')
    binding = json.loads(binding_path.read_bytes())
    if (binding.get('schema') != 'v21_conditional_binding_v1' or binding.get('status') != 'frozen'
            or binding.get('panel_protocol_sha256') != sha(protocol_path.read_bytes())):
        raise ValueError('HumanEval scorer binding/protocol drift')
    c = protocol['task_contracts']['humaneval']
    if c['scorer_sha256'] != sha(Path(__file__).read_bytes()):
        raise ValueError('HumanEval scorer source drift')
    if sha(gold_path.read_bytes()) != protocol['extra_gold_sha256']['humaneval']:
        raise ValueError('HumanEval gold drift')
    gold = json.loads(gold_path.read_bytes())
    if not set(protocol['ids']['humaneval']) <= set(gold):
        raise ValueError('HumanEval gold coverage drift')
    records = load_records(protocol, binding_path, ledger_paths)
    sandbox_preflight()
    quality, evaluations = {}, []
    for spec in protocol['schedule']:
        r = records.get(execution_key(spec))
        if spec['role'] != 'attempt0' or not r or not r.get('ok'):
            continue
        receipt = _first_receipt(r, spec, roots)
        problem = gold[spec['id']]
        code, extraction = extract_code(receipt['raw_completion'], problem['entry_point'])
        passed, reason = run_test(program_for(problem, code)) if code else (False, extraction)
        capped = r['termination'] in ('length', 'max_new_tokens', 'max_tokens')
        quality[spec['cell_id']] = dict(score=float(passed), correct=passed, strict_correct=passed,
                                      task_correct=passed, parsed=code is not None,
                                      eos=r['termination'] == 'eos', capped=capped)
        evaluations.append(dict(dataset=spec['dataset'], id=spec['id'], seed=spec['seed'], arm=spec['arm'],
                                passed=passed, extraction=extraction, test_result=reason))
    cells = cell_rows(protocol, build_cells(protocol, records, quality))
    path = Path(str(out_prefix) + '.csv')
    with path.open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0]))
        w.writeheader()
        w.writerows(cells)
    result = dict(schema='v27_humaneval_scored_v1', protocol_id=protocol['protocol_id'],
                  planned=len(protocol['schedule']), recorded=len(records),
                  complete=len(records) == len(protocol['schedule']), metric=c['quality_metric'],
                  results=evaluations, task_contract=c, sandbox_preflight='passed')
    Path(str(out_prefix) + '.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'results'}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    b = sub.add_parser('prepare')
    b.add_argument('--data', type=Path, required=True)
    b.add_argument('--model', required=True)
    b.add_argument('--out', type=Path, required=True)
    sub.add_parser('sandbox-check')
    s = sub.add_parser('score')
    for name in ('protocol', 'binding', 'private-roots', 'gold', 'out-prefix'):
        s.add_argument('--' + name, type=Path, required=True)
    s.add_argument('--ledger', type=Path, action='append', required=True)
    a = p.parse_args()
    if a.action == 'prepare':
        from dllm.models import create_adapter
        adapter = create_adapter('diffusion_gemma', a.model, device='cpu', precision='bfloat16',
                                 revision='f7f5b7f5fa82ffc52addd066915886d497f5517b').load_tokenizer()
        raw = a.data.read_bytes()
        tasks = [json.loads(x) for x in gzip.decompress(raw).decode().splitlines() if x.strip()]
        if len(tasks) != 164:
            raise ValueError('expected complete official 164-task inventory')
        manifests, gold = prepare_rows(tasks, lambda text: adapter.encode_prompt(text, {'thinking': True}))
        a.out.mkdir(parents=True, exist_ok=False)
        for name, value in [('humaneval_pool_manifest.json', manifests), ('humaneval_gold_scorer_only.json', gold),
                            ('task_contract.json', contract(sha(raw), sha(Path(__file__).read_bytes())))]:
            (a.out / name).write_text(json.dumps(value, indent=2) + '\n')
        print(json.dumps(dict(tasks=164, min_prompt_tokens=min(r['prompt_token_count'] for r in manifests),
                              max_prompt_tokens=max(r['prompt_token_count'] for r in manifests))))
    elif a.action == 'sandbox-check':
        sandbox_preflight()
        print('OS sandbox preflight passed')
    else:
        roots = {k: Path(v) for k, v in json.loads(a.private_roots.read_bytes()).items()}
        score_panel(a.protocol, a.binding, a.ledger, roots, a.gold, a.out_prefix)


if __name__ == '__main__':
    main()
