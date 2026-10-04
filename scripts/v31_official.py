"""The one rule shared by the v31 final-suite pool builders, scorers and comparison tools (official dataset protocols).
CPU only, no torch.

Scoring rule (every v31 scorer):
  - the OFFICIAL per-sample metric of the dataset is the primary output, {arm label: {cell key: value}}, kept
    unrounded (aggregates are averaged before any rounding);
  - a boolean is secondary and named for exactly what it requires (e.g. `strict_all_correct_finished`);
  - a primary metric never requires a stop / eos finish; capped outputs (finish 'length') are counted separately;
  - a duplicate cell key raises (cell key = "dataset|index|panel_seed|repeat", arm label = the private file name
    <tag>_<label>.private.jsonl);
  - binding: every record must carry the manifest sha256, the sha256 of its prompt token ids and its generation budget
    (written by scripts/v31_vllm_paired_bench.py), and they must equal the pool the scorer is given; a record from
    another pool, manifest or budget is refused (older records without these fields only with --legacy-unbound);
  - planned cells: the scorer is given the planned cells file(s); a record outside the plan raises; a planned cell
    missing in any arm raises unless --allow-missing, which scores ONE explicit intersection of all arms and reports
    per arm planned / present / dropped.
Comparison tools additionally require, per cell, identical rng_seed, budget, prompt_tokens, max_model_len, chunk,
block_size, prompt and manifest hashes across arms (check_comparable).
Answer text:
  - `final_response(raw, thinking)` of experiments/diffusion_gemma_aime26_modes/protocol.py, compiled from that file
    alone (importing the module would pull in torch): the text after the last '<channel|>' if one is present, cut at
    the first '<turn|>' / '<|endoftext|>' / '<eos>', stripped; thinking ON without any '<channel|>' gives '';
  - MRCR and GraphWalks grade the raw response, as their READMEs do: answer_unstripped = the same boundary, no strip;
  - prefilled RULER pools (the prompt ends inside the answer): the completion cut at the first end token only.
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
import math
import os
import random
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINISHED = ('stop', 'eos')
CAPPED = ('length', 'max_tokens', 'max_new_tokens')
END_TOKENS = ('<turn|>', '<|endoftext|>', '<eos>')
FINAL_RESPONSE_MODULE = 'experiments.diffusion_gemma_aime26_modes.protocol'
AIME_MODULE = 'experiments.diffusion_gemma_aime30.protocol'
ROWINFO_SCHEMA = 'v31_rowinfo_v2'
BINDING_FIELDS = ('manifest_sha256', 'prompt_sha256', 'budget', 'rng_seed', 'prompt_tokens', 'max_model_len', 'chunk',
                  'block_size')
COMPARE_FIELDS = ('rng_seed', 'budget', 'prompt_tokens', 'max_model_len', 'chunk', 'block_size', 'prompt_sha256',
                  'manifest_sha256')

# ---------------------------------------------------------------- hashes, cells, labels, files


def sha(data) -> str:
    return hashlib.sha256(data if isinstance(data, bytes) else str(data).encode('utf-8')).hexdigest()


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def token_ids_sha256(ids) -> str:
    """sha256 of json.dumps(list of int ids) -- the bench's prompt_sha256 (same formula, computed there inline)."""
    return hashlib.sha256(json.dumps([int(t) for t in ids]).encode()).hexdigest()


def label_of(path) -> str:
    """Arm label of a private file <tag>_<label>.private.jsonl (or a public <tag>_<label>.jsonl)."""
    return Path(path).name.split('.private')[0].split('.jsonl')[0].split('_', 1)[1]


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


def natural_key(text):
    """Sort key that orders embedded numbers numerically ('ruler32k' < 'ruler64k' < 'ruler128k')."""
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', str(text))]


def finish_class(reason) -> str:
    return 'finished' if reason in FINISHED else 'capped' if reason in CAPPED else 'other'


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True) + '\n', encoding='utf-8')


def write_outputs(prefix, official, secondary, summary, binding=None):
    """<prefix>.official.json (primary), <prefix>.<boolean name>.json (secondary), <prefix>.binding.json (per-cell
    run settings for the comparison tools), <prefix>.summary.json."""
    paths = [f'{prefix}.official.json']
    write_json(paths[0], official)
    for name, table in secondary.items():
        paths.append(f'{prefix}.{name}.json')
        write_json(paths[-1], table)
    if binding is not None:
        paths.append(f'{prefix}.binding.json')
        write_json(paths[-1], binding)
    paths.append(f'{prefix}.summary.json')
    write_json(paths[-1], summary)
    return paths

# ---------------------------------------------------------------- pools: rows, binding, planned cells


def load_pool(man_dir, dataset, fields=()):
    """(rows, pool) of one dataset. rows (index = manifest position) hold id, generation_budget, prompt_sha256 and the
    requested `fields`; pool = {manifest_sha256, max_model_len (the pool's pinned value; None for older pools)}.
    Uses the builder's {dataset}_rowinfo.json (schema v31_rowinfo_v2), else derives both from the manifest itself."""
    man_dir = Path(man_dir)
    keep = ('id', 'generation_budget', 'prompt_sha256', *fields)
    info = man_dir / f'{dataset}_rowinfo.json'
    if info.exists():
        data = json.loads(info.read_text(encoding='utf-8'))
        if isinstance(data, dict) and data.get('schema') == ROWINFO_SCHEMA:
            if data['dataset'] != dataset:
                raise ValueError(f'{info.name} describes {data["dataset"]}')
            return ([{k: r[k] for k in keep if k in r} for r in data['rows']],
                    dict(manifest_sha256=data['manifest_sha256'], max_model_len=data.get('max_model_len')))
    raw = (man_dir / f'{dataset}_generation_manifest.json').read_bytes()
    rows = []
    for r in json.loads(raw):
        row = {k: r[k] for k in keep if k in r}
        row['prompt_sha256'] = token_ids_sha256(r['prompt_tokens'])
        rows.append(row)
    return rows, dict(manifest_sha256=sha(raw), max_model_len=None)


class Pools:
    """Lazy per-dataset (rows, pool) cache over one manifest directory."""

    def __init__(self, man_dir, fields=()):
        self.man_dir, self.fields, self.cache = Path(man_dir), tuple(fields), {}

    def __call__(self, dataset):
        if dataset not in self.cache:
            self.cache[dataset] = load_pool(self.man_dir, dataset, self.fields)
        return self.cache[dataset]


def bind(record, pools, legacy=False):
    """The manifest row of a record after checking that the record was generated from exactly this pool row: same id at
    the index, same manifest sha256, same prompt token ids (sha256), same generation budget. Returns (row, binding,
    verified); a record without the binding fields raises unless legacy (then verified = False)."""
    rows, pool = pools(record['dataset'])
    if not 0 <= int(record['index']) < len(rows) or rows[record['index']]['id'] != record['id']:
        raise ValueError(f'cell {cell_key(record)}: id differs from the manifest row at that index')
    row = rows[record['index']]
    binding = {f: record.get(f) for f in BINDING_FIELDS}
    missing = [f for f in ('manifest_sha256', 'prompt_sha256', 'budget') if record.get(f) is None]
    if missing:
        if legacy:
            return row, binding, False
        raise ValueError(f'cell {cell_key(record)}: record lacks {missing} (older bench); --legacy-unbound scores it '
                         'without verification')
    if record['manifest_sha256'] != pool['manifest_sha256']:
        raise ValueError(f'cell {cell_key(record)}: generated from another manifest than the pool given')
    if record['prompt_sha256'] != row['prompt_sha256']:
        raise ValueError(f'cell {cell_key(record)}: prompt token ids differ from the pool row')
    if int(record['budget']) != int(row['generation_budget']):
        raise ValueError(f'cell {cell_key(record)}: budget {record["budget"]} differs from the pool row '
                         f'({row["generation_budget"]})')
    return row, binding, True


def planned_cells(files, repeats=1):
    """{cell key: planned cell} of cells files (lists of {dataset, id, index, seed, ...}) x repeats 0..repeats-1."""
    plan = {}
    for f in files:
        for c in json.loads(Path(f).read_text(encoding='utf-8')):
            for rep in range(repeats):
                key = f"{c['dataset']}|{c['index']}|{c['seed']}|{rep}"
                if key in plan:
                    raise ValueError(f'duplicate planned cell {key}')
                plan[key] = c
    return plan


def common_cells(keys_by_label, plan, allow_missing=False):
    """One explicit intersection over ALL arms given: a cell outside the plan raises; a planned cell missing in an arm
    raises unless allow_missing. Returns (common cell keys, coverage {label: {planned, present, missing, dropped,
    scored}}) -- dropped = present in this arm but missing in another."""
    common = set(plan)
    for label, keys in keys_by_label.items():
        extra = set(keys) - set(plan)
        if extra:
            raise ValueError(f'arm {label}: {len(extra)} cells are not planned cells (e.g. {sorted(extra)[0]})')
        missing = len(plan) - len(set(keys))
        if missing and not allow_missing:
            raise ValueError(f'arm {label}: {missing} of {len(plan)} planned cells missing (--allow-missing scores the '
                             'cells common to all arms)')
        common &= set(keys)
    coverage = {label: dict(planned=len(plan), present=len(set(keys)), missing=len(plan) - len(set(keys)),
                            dropped=len(set(keys) - common), scored=len(common))
                for label, keys in sorted(keys_by_label.items())}
    return common, coverage


def select_cells(records, plan, allow_missing=False):
    """common_cells over private records (each must carry the planned id). Returns (records of the common cells,
    coverage)."""
    by_label = collections.defaultdict(set)
    for label, key, r in records:
        cell = plan.get(key)
        if cell is not None and cell['id'] != r['id']:
            raise ValueError(f'arm {label}: cell {key} has another id than the planned cell')
        by_label[label].add(key)
    common, coverage = common_cells(by_label, plan, allow_missing)
    return [(l, k, r) for l, k, r in records if k in common], coverage


def coverage_lines(coverage):
    return [f"{label}: planned {c['planned']} present {c['present']} dropped {c['dropped']} scored {c['scored']}"
            for label, c in sorted(coverage.items())]


def scorer_cli(doc, positionals):
    """Common scorer arguments: positionals, --cells (planned cells file, repeatable, required), --repeats,
    --allow-missing, --legacy-unbound, then the private files."""
    import argparse
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in positionals:
        p.add_argument(name)
    p.add_argument('--cells', action='append', required=True, help='planned cells file (repeatable)')
    p.add_argument('--repeats', type=int, default=1, help='bench REPEATS of the run (cell keys repeat 0..N-1)')
    p.add_argument('--allow-missing', action='store_true', help='score the cells common to all arms')
    p.add_argument('--legacy-unbound', action='store_true', help='accept records without binding fields (older bench)')
    p.add_argument('private', nargs='+')
    return p.parse_intermixed_args()


def planned_records(args):
    """Read the private files, apply the plan (one explicit intersection), print per-arm coverage."""
    kept, coverage = select_cells(read_private(args.private), planned_cells(args.cells, args.repeats), args.allow_missing)
    for line in coverage_lines(coverage):
        print(line)
    return kept, coverage


def check_comparable(binding, labels, keys, legacy=False):
    """Per cell, every arm must have run with identical COMPARE_FIELDS; a missing field raises unless legacy (then only
    the fields present in every arm are compared). Returns the number of cells compared."""
    for k in sorted(keys):
        seen = None
        for label in labels:
            b = binding[label][k]
            absent = [f for f in COMPARE_FIELDS if b.get(f) is None]
            if absent and not legacy:
                raise ValueError(f'arm {label}, cell {k}: no {absent} recorded (older bench); --legacy-unbound compares '
                                 'the fields present')
            vals = {f: b.get(f) for f in COMPARE_FIELDS}
            if seen is None:
                seen = vals
                continue
            diff = [f for f in COMPARE_FIELDS if vals[f] is not None and seen[f] is not None and vals[f] != seen[f]]
            if diff:
                raise ValueError(f'cell {k}: arms ran with different {diff}; refusing to compare')
    return len(keys)

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
    """Top-level functions / assignments `names` of a module's source file, compiled alone (no module import). The
    returned source identity holds the module name and file hash only (no host path)."""
    path = _source(module)
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body
             if (isinstance(n, ast.FunctionDef) and n.name in names)
             or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
    namespace = {}
    exec(compile(preamble, f'<preamble {module}>', 'exec'), namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), module.replace('.', '/') + '.py', 'exec'), namespace)
    missing = [n for n in names if n not in namespace]
    if missing:
        raise ImportError(f'{module} lacks {missing}')
    return [namespace[n] for n in names], dict(module=module, file=module.replace('.', '/') + '.py', sha256=sha_file(path))


def load_final_response():
    (fn,), source = compile_from_source(FINAL_RESPONSE_MODULE, ['final_response'])
    return fn, source


def cut_at_end(raw: str) -> str:
    """Cut at the first end token (the boundary final_response uses), nothing else."""
    for token in END_TOKENS:
        raw = raw.split(token, 1)[0]
    return raw


def answer_unstripped(raw: str, thinking: bool) -> str:
    """final_response without the final strip: the raw response the MRCR / GraphWalks READMEs grade."""
    if '<channel|>' in raw:
        raw = raw.rsplit('<channel|>', 1)[1]
    elif thinking:
        return ''
    return cut_at_end(raw)


def aime_numeric_score():
    """experiments/diffusion_gemma_aime30/protocol.py numeric_score (last \\boxed{} / answer marker / last number,
    |x - answer| <= 1e-6), compiled from that file alone."""
    fns, source = compile_from_source(AIME_MODULE, ['NUMBER', 'extract_number', 'numeric_score'],
                                      'import re\nfrom decimal import Decimal, InvalidOperation, DecimalException\n')
    return fns[2], source

# ---------------------------------------------------------------- RULER

RULER_COMMIT = 'c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a'
RULER_TASKS = ('niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1', 'niah_multikey_2',
               'niah_multikey_3', 'niah_multivalue', 'niah_multiquery', 'vt', 'cwe', 'fwe', 'qa_1', 'qa_2')
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
    """The pinned commit and unmodified metric / prompt sources (read-only git calls, GIT_OPTIONAL_LOCKS=0)."""
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


def ruler_sample_score(metric_fn, pred: str, refs) -> float:
    """RULER's per-sample score WITHOUT the metric's final round(.., 2): string_match_all = share of references found
    (case-insensitive substring), string_match_part = 1 if any is found, x 100, in RULER's own operation order.
    Asserted to round to exactly what the pinned metric function returns for the same sample."""
    hits = [1.0 if r.lower() in pred.lower() else 0.0 for r in refs]
    if metric_fn.__name__ == 'string_match_all':
        value = sum([sum(hits) / len(refs)]) / 1 * 100
    elif metric_fn.__name__ == 'string_match_part':
        value = sum([max(hits)]) / 1 * 100
    else:
        raise ValueError(f'unknown RULER metric {metric_fn.__name__}')
    if round(value, 2) != metric_fn([pred], [refs]):
        raise AssertionError('unrounded per-sample score disagrees with the pinned RULER metric')
    return value


def ruler_official_aggregate(scores, task_of, length_of):
    """RULER's official aggregation on per-sample scores {cell: 0-100}: per length, the mean per task, then the
    unweighted mean over that length's tasks (every task weighs the same however many cells it has); overall = the
    mean over lengths. Returns {length: {'tasks': {task: mean}, 'avg', 'cwe', 'without_cwe', 'cells'}, '_overall': x}."""
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for k, v in scores.items():
        per[length_of(k)][task_of(k)].append(v)
    out = {}
    for ln in sorted(per, key=natural_key):
        tasks = {t: statistics.mean(v) for t, v in sorted(per[ln].items())}
        rest = [v for t, v in tasks.items() if t != 'cwe']
        out[ln] = dict(tasks=tasks, avg=statistics.mean(tasks.values()), cwe=tasks.get('cwe'),
                       without_cwe=statistics.mean(rest) if rest else None, cells=sum(len(v) for v in per[ln].values()))
    out['_overall'] = statistics.mean(out[ln]['avg'] for ln in per) if per else None
    return out

# ---------------------------------------------------------------- LongBench-v2 (THUDM/LongBench pred.py / result.py)

LB_MAX_LEN = 120000                           # config/model2maxlen.json for every open model
LB_FIELDS = ('context', 'question', 'choice_A', 'choice_B', 'choice_C', 'choice_D')
LB_COLUMNS = ('Overall', 'Easy', 'Hard', 'Short', 'Medium', 'Long')


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


def lb_result(rows, digits=1):
    """result.py on [{'judge', 'difficulty', 'length'}] (compensated = False): Overall / Easy / Hard / Short / Medium /
    Long = 100 * acc / n, rounded to `digits` (None: unrounded); an empty group gives None."""
    groups = collections.defaultdict(list)
    for r in rows:
        acc = int(r['judge'])
        groups['Overall'].append(acc)
        groups['Easy' if r['difficulty'] == 'easy' else 'Hard'].append(acc)
        groups['Short' if r['length'] == 'short' else 'Medium' if r['length'] == 'medium' else 'Long'].append(acc)
    out = {}
    for g in LB_COLUMNS:
        value = 100 * sum(groups[g]) / len(groups[g]) if groups[g] else None
        out[g] = value if value is None or digits is None else round(value, digits)
    return out

# ---------------------------------------------------------------- paired statistics


def binomial_two_sided(a, b):
    """Exact two-sided sign / McNemar p for a vs b discordant counts (min(1, 2 * P[X <= min(a, b)]), X ~ Bin(a+b, 1/2))."""
    n = a + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def nested_mean(groups):
    """Mean over units of each unit's mean over its cells."""
    return statistics.mean(statistics.mean(v) for v in groups.values())


def paired_bootstrap(groups, reps=4000, seed=7, nested=True):
    """groups {unit: [per-cell paired differences]} -> (point, lo, hi): units resampled with replacement and, if nested,
    each drawn unit's cells resampled within it (seeds nested in problems); point = mean over units of unit means."""
    rng = random.Random(seed)
    units = sorted(groups)
    point = nested_mean(groups)
    boots = []
    for _ in range(reps):
        acc = 0.0
        for _ in units:
            v = groups[rng.choice(units)]
            acc += (sum(rng.choice(v) for _ in v) / len(v)) if nested and len(v) > 1 else sum(v) / len(v)
        boots.append(acc / len(units))
    boots.sort()
    return point, boots[int(0.025 * reps)], boots[int(0.975 * reps) - 1]
