"""The one rule shared by the v31 final-suite pool builders and scorers (official dataset protocols). CPU only, no torch.

Scoring rule (every v31 scorer):
  - the OFFICIAL per-sample metric of the dataset is the primary output, {arm label: {cell key: value}};
  - a boolean is secondary and named for exactly what it requires (e.g. `strict_all_correct_finished`);
  - a primary metric never requires a stop / eos finish; capped outputs (finish 'length') are counted separately;
  - a duplicate cell key raises (cell key = "dataset|index|panel_seed|repeat", arm label = the private file name
    <tag>_<label>.private.jsonl).
Answer text = `final_response(raw, thinking)` of experiments/diffusion_gemma_aime26_modes/protocol.py, compiled from that
file alone (importing the module would pull in torch): the text after the last '<channel|>' if one is present, cut at
the first '<turn|>' / '<|endoftext|>' / '<eos>', stripped; thinking ON without any '<channel|>' (an unfinished thought)
gives ''.
Prompt rendering:
  - every prompt is the model's chat template (adapter.encode_prompt, one user turn);
  - RULER (NVIDIA/RULER @ c3f5e3b, scripts/pred/call_api.py sends `input + answer_prefix`, the model template being part
    of `input`): the user turn holds the sample input WITHOUT the answer prefix (thinking off); the answer prefix opens the
    model's final response. DiffusionGemma's non-thinking turn opens with an empty thought block before its visible
    answer (RESPONSE_OPENER), so the prefill is that block followed by the official answer prefix, verbatim;
  - LongBench-v2 (THUDM/LongBench pred.py): the template filled with stripped fields; above 120,000 tokens of OUR
    tokenizer the first and last 60,000 ids are kept and decoded with skip_special_tokens=True (middle_truncate).
Never prints prompts, completions or gold.
"""
from __future__ import annotations

import ast
import collections
import hashlib
import importlib.util
import json
import os
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINISHED = ('stop', 'eos')
CAPPED = ('length', 'max_tokens', 'max_new_tokens')
FINAL_RESPONSE_MODULE = 'experiments.diffusion_gemma_aime26_modes.protocol'
AIME_MODULE = 'experiments.diffusion_gemma_aime30.protocol'

# ---------------------------------------------------------------- cells, labels, files


def sha(data) -> str:
    return hashlib.sha256(data if isinstance(data, bytes) else str(data).encode('utf-8')).hexdigest()


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def label_of(path) -> str:
    """Arm label of a private file <tag>_<label>.private.jsonl."""
    return Path(path).name.split('.private')[0].split('_', 1)[1]


def cell_key(record) -> str:
    return f"{record['dataset']}|{record['index']}|{record['panel_seed']}|{record['repeat']}"


def read_private(files):
    """[(label, key, record)] of every private record, in file order; a duplicate (label, cell key) raises."""
    seen, out = set(), []
    for f in files:
        label = label_of(f)
        with open(f, encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                r = json.loads(line)
                key = cell_key(r)
                if (label, key) in seen:
                    raise ValueError(f'duplicate cell key {key} for arm {label}')
                seen.add((label, key))
                out.append((label, key, r))
    return out


def put(table, label, key, value):
    """table[label][key] = value; a duplicate raises."""
    cells = table.setdefault(label, {})
    if key in cells:
        raise ValueError(f'duplicate cell key {key} for arm {label}')
    cells[key] = value


def finish_class(reason) -> str:
    return 'finished' if reason in FINISHED else 'capped' if reason in CAPPED else 'other'


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True) + '\n', encoding='utf-8')


def write_outputs(prefix, official, secondary, summary):
    """<prefix>.official.json (primary), <prefix>.<boolean name>.json (secondary), <prefix>.summary.json."""
    paths = [f'{prefix}.official.json']
    write_json(paths[0], official)
    for name, table in secondary.items():
        paths.append(f'{prefix}.{name}.json')
        write_json(paths[-1], table)
    paths.append(f'{prefix}.summary.json')
    write_json(paths[-1], summary)
    return paths


def load_rows(man_dir, dataset, fields):
    """Manifest rows of a dataset with only `fields` (index = manifest position); prefers the builder's small
    {dataset}_rowinfo.json, else reads {dataset}_generation_manifest.json."""
    man_dir = Path(man_dir)
    info = man_dir / f'{dataset}_rowinfo.json'
    rows = json.loads((info if info.exists() else man_dir / f'{dataset}_generation_manifest.json').read_text(encoding='utf-8'))
    return [{k: r[k] for k in fields if k in r} for r in rows]


def check_cell(rows, dataset, record):
    """The private record's id must be the manifest row at its index."""
    if rows[record['index']]['id'] != record['id']:
        raise ValueError(f'cell {cell_key(record)} id differs from manifest {dataset} at that index')
    return rows[record['index']]

# ---------------------------------------------------------------- answer text (compiled from the protocol sources)


def _source(module):
    try:
        spec = importlib.util.find_spec(module)
    except ModuleNotFoundError:
        spec = None
    path = Path(spec.origin) if spec is not None and spec.origin else ROOT / (module.replace('.', '/') + '.py')
    if not path.exists():
        raise FileNotFoundError(f'{module} source not found (run with cwd / PYTHONPATH holding experiments/)')
    return path


def compile_from_source(module, names, preamble=''):
    """Top-level functions / assignments `names` of a module's source file, compiled alone (no module import)."""
    path = _source(module)
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body
             if (isinstance(n, ast.FunctionDef) and n.name in names)
             or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
    namespace = {}
    exec(compile(preamble, f'<preamble {module}>', 'exec'), namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    missing = [n for n in names if n not in namespace]
    if missing:
        raise ImportError(f'{module} lacks {missing}')
    return [namespace[n] for n in names], dict(path=str(path), sha256=sha_file(path))


def load_final_response():
    (fn,), source = compile_from_source(FINAL_RESPONSE_MODULE, ['final_response'])
    return fn, source


def aime_numeric_score():
    """experiments/diffusion_gemma_aime30/protocol.py numeric_score (last \\boxed{} / answer marker / last number,
    |x - answer| <= 1e-6), compiled from that file alone."""
    fns, source = compile_from_source(AIME_MODULE, ['NUMBER', 'extract_number', 'numeric_score'],
                                      'import re\nfrom decimal import Decimal, InvalidOperation, DecimalException\n')
    return fns[2], source

# ---------------------------------------------------------------- RULER

RULER_COMMIT = 'c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a'
RULER_FILES = ('scripts/eval/synthetic/constants.py', 'scripts/synthetic.yaml', 'scripts/data/synthetic/constants.py',
               'scripts/pred/call_api.py', 'scripts/eval/evaluate.py')
# What a model's non-thinking turn emits before its visible answer, after the chat template's generation prompt.
# DiffusionGemma (Gemma 4 turn format): an empty thought block; 3972 / 3975 non-thinking RULER completions of the v31
# panels begin with exactly this text (the other 3 open it and stop without closing it), none with any other text.
RESPONSE_OPENER = {'diffusion_gemma': '<|channel>thought\n<channel|>'}


def ruler_prompt_ids(adapter, sample_input, answer_prefix, opener):
    """Official RULER placement: user turn = the sample input (chat template, thinking off); the response prefill
    (`opener` + the official answer prefix, verbatim) follows the generation prompt. Returns (ids, user_token_count)."""
    user = adapter.encode_prompt(sample_input, {'thinking': False})
    tail = adapter.tokenizer.encode(opener + answer_prefix, add_special_tokens=False)
    return list(user) + list(tail), len(user)


def ruler_verify_checkout(root) -> dict:
    """The pinned commit and unmodified metric / prompt sources (read-only git calls, no index refresh)."""
    root = Path(root)
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True, env=env).strip()
    if head != RULER_COMMIT:
        raise ValueError(f'RULER checkout at {head}, pinned {RULER_COMMIT}')
    out = {}
    for rel in RULER_FILES:
        committed = subprocess.check_output(['git', '-C', str(root), 'show', f'{RULER_COMMIT}:{rel}'], env=env)
        if (root / rel).read_bytes() != committed:
            raise ValueError(f'modified RULER source: {rel}')
        out[rel] = sha(committed)
    return dict(commit=head, sha256=out)


def ruler_metrics(root) -> dict:
    """{task base: official metric_fn} from RULER's scripts/eval/synthetic/constants.py (string_match_all / _part)."""
    path = Path(root) / 'scripts/eval/synthetic/constants.py'
    spec = importlib.util.spec_from_file_location('v31_ruler_eval_constants', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {name: cfg['metric_fn'] for name, cfg in module.TASKS.items()}


def ruler_task_bases(root) -> dict:
    """{task: task base} from RULER's scripts/synthetic.yaml (only needed for pools whose rows lack task_base)."""
    import yaml
    return {k: str(v['task']) for k, v in yaml.safe_load((Path(root) / 'scripts/synthetic.yaml').read_text()).items()}


def ruler_postprocess(prediction: str) -> str:
    """RULER scripts/eval/evaluate.py postprocess_pred (= dllm.evaluation.ruler.official.postprocess_prediction)."""
    return re.sub(r'[\x00-\x1f]', '\n', prediction.strip()).strip()


def ruler_official_aggregate(scores, task_of, length_of):
    """RULER's official aggregation on per-sample scores {cell: 0-100}: per length, the mean per task, then the
    unweighted mean over that length's tasks (every task weighs the same however many cells it has); overall = the
    mean over lengths. Returns {length: {'tasks': {task: mean}, 'avg': x, 'cwe': x, 'without_cwe': x}, '_overall': x}."""
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for k, v in scores.items():
        per[length_of(k)][task_of(k)].append(v)
    out = {}
    for ln in sorted(per):
        tasks = {t: statistics.mean(v) for t, v in sorted(per[ln].items())}
        rest = [v for t, v in tasks.items() if t != 'cwe']
        out[ln] = dict(tasks=tasks, avg=statistics.mean(tasks.values()), cwe=tasks.get('cwe'),
                       without_cwe=statistics.mean(rest) if rest else None, cells=sum(len(v) for v in per[ln].values()))
    out['_overall'] = statistics.mean(out[ln]['avg'] for ln in per) if per else None
    return out

# ---------------------------------------------------------------- LongBench-v2 (THUDM/LongBench pred.py / result.py)

LB_MAX_LEN = 120000                           # config/model2maxlen.json for every open model
LB_FIELDS = ('context', 'question', 'choice_A', 'choice_B', 'choice_C', 'choice_D')


def lb_fill(template: str, item: dict) -> str:
    """pred.py: the template with $DOC$ / $Q$ / $C_A$..$C_D$ replaced in this order by the stripped fields."""
    return (template.replace('$DOC$', item['context'].strip()).replace('$Q$', item['question'].strip())
            .replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip())
            .replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()))


def middle_truncate(ids, max_len=LB_MAX_LEN):
    """pred.py query_llm: above max_len ids keep the first max_len//2 and the last max_len//2."""
    return list(ids) if len(ids) <= max_len else list(ids[:max_len // 2]) + list(ids[-max_len // 2:])


def lb_extract_answer(response: str):
    """pred.py extract_answer, verbatim (None = unparsed, judged wrong)."""
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None


def lb_result(rows):
    """result.py on [{'judge', 'difficulty', 'length'}] (compensated = False): Overall / Easy / Hard / Short / Medium /
    Long, round(100 * acc / n, 1); an empty group gives None."""
    groups = collections.defaultdict(list)
    for r in rows:
        acc = int(r['judge'])
        groups['Overall'].append(acc)
        groups['Easy' if r['difficulty'] == 'easy' else 'Hard'].append(acc)
        groups['Short' if r['length'] == 'short' else 'Medium' if r['length'] == 'medium' else 'Long'].append(acc)
    return {g: (round(100 * sum(groups[g]) / len(groups[g]), 1) if groups[g] else None)
            for g in ('Overall', 'Easy', 'Hard', 'Short', 'Medium', 'Long')}
