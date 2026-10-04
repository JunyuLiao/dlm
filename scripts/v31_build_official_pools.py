"""Build the v31 final-suite pools under each dataset's OFFICIAL protocol (CPU tokenizer only, no weights, no GPU).

Rule: scripts/v31_official.py. Every pool writes <dataset>_generation_manifest.json (rows: id, benchmark, prompt,
prompt_hash = sha256(prompt), prompt_tokens, prompt_token_count, generation_budget, thinking, source_id + non-gold task
fields; json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=False) + '\\n'), a small <dataset>_rowinfo.json (the
same rows without prompt text / tokens, for the scorers), scorer-only <dataset>_gold.json, cells_<pool>.json
({dataset, id, index = manifest position, seed, prompt_tokens}) and readme.json (counts, token stats, sha256 only).
  ruler       Re-render of the v33 pool (same samples, ids, order, gold) with the official answer-prefix placement:
              user turn = the RULER sample `input` without its answer prefix (chat template, thinking off), then the
              response prefill = DiffusionGemma's empty thought block + the official answer prefix, verbatim, so
              generation starts inside the answer (RULER call_api.py sends input + answer_prefix with the model
              template inside input). Budget = RULER tokens_to_generate per task. Row field `prompt` is the user-turn
              text, `response_prefill` the appended text (prompt_tokens = encode_prompt(prompt) + tokens(prefill)).
  longbench   All 503 LongBench-v2 items (pinned dataset revision), THUDM/LongBench pred.py: template filled with the
              stripped fields (after the project's reserved-token escaping), tokenized with OUR tokenizer
              (tokenizer.encode, as pred.py); above 120,000 ids the first and last 60,000 are kept and decoded with
              skip_special_tokens=True; then adapter.encode_prompt. Variants (one dataset each):
                longbench_v2_0shot        prompts/0shot.txt,     thinking off, 128 new tokens   (w/o CoT)
                longbench_v2_0shot_think  prompts/0shot.txt,     thinking on, 16384             (w/ CoT, reasoning model)
                longbench_v2_cot_think    prompts/0shot_cot.txt, thinking on, 16384             (0shot_cot single stage)
  aime        The v27 final AIME26 manifest with generation_budget 32768, everything else unchanged; cells seeds 1-4.
  graphwalks  The pools_v31 GraphWalks pool with generation_budget 16384, everything else unchanged; cells seed 1.
usage: python v31_build_official_pools.py {ruler,longbench,aime,graphwalks} --out DIR [--src PATH] [--jobs N]
Run with cwd = a deployment holding src/ (PYTHONPATH=src:.), CUDA_VISIBLE_DEVICES= and HF_HUB_OFFLINE=1.
Never prints prompts, contexts, answers or gold.
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing
import os
import platform
import re
import shutil
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v31_official as op  # noqa: E402

MODEL = '/home/exouser/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b'
REVISION = 'f7f5b7f5fa82ffc52addd066915886d497f5517b'
RULER_LENGTHS = ((32768, 'ruler32k_v33'), (65536, 'ruler64k_v33'), (131072, 'ruler128k_v33'))
RULER_SRC = Path('/media/volume/dllm-1/dyh/ruler_long_v33')
LB_REPO = Path('/home/exouser/ljy/dlm/reference/LongBench')
LB_REPO_COMMIT = '2e00731f8d0bff23dc4325161044d0ed8af94c1e'
LB_REPO_FILES = ('pred.py', 'result.py', 'config/model2maxlen.json', 'prompts/0shot.txt', 'prompts/0shot_cot.txt',
                 'prompts/0shot_cot_ans.txt')
LB_VARIANTS = (('longbench_v2_0shot', '0shot', False, 128, 'w/o CoT: pred.py without --cot (temperature 0.1, 128 new tokens)'),
               ('longbench_v2_0shot_think', '0shot', True, 16384,
                'w/ CoT for a reasoning model: zero-shot prompt with native thinking (the paper evaluates the reasoning '
                'model o1-preview under zero-shot prompting; leaderboard Qwen3: thinking mode, 16K budget)'),
               ('longbench_v2_cot_think', '0shot_cot', True, 16384,
                'prompts/0shot_cot.txt with native thinking, single stage (pred.py --cot always adds the 0shot_cot_ans '
                'stage that asks for the answer format; this variant has no such stage)'))
AIME_SRC = Path('/media/volume/dllm-1/dyh/m3_output_numerics_v21_20260927/v23/v27_final_aime_f1_002/manifests/'
                'aime26_generation_manifest.json')
AIME_BUDGET, AIME_SEEDS = 32768, (1, 2, 3, 4)
GW_SRC = Path('/media/volume/dllm-1/dyh/pools_v31/graphwalks')
GW_DATASETS, GW_BUDGET = ('graphwalks_22k', 'graphwalks_45k', 'graphwalks_90k'), 16384
ENV_VARS = ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'HF_HUB_OFFLINE', 'OMP_NUM_THREADS', 'RAYON_NUM_THREADS',
            'TOKENIZERS_PARALLELISM', 'PYTHONDONTWRITEBYTECODE', 'PYTHONNOUSERSITE', 'TMPDIR', 'HF_HOME')


def log(stream, msg):
    stream.write(time.strftime('%H:%M:%S ') + msg + '\n')
    stream.flush()


def stats(values):
    v = sorted(values)
    if not v:
        return dict(n=0)
    return dict(n=len(v), min=v[0], median=statistics.median(v), mean=round(statistics.fmean(v), 1), max=v[-1])


def load_adapter():
    from dllm.models import create_adapter
    return create_adapter('diffusion_gemma', MODEL, device='cpu', precision='float32', revision=REVISION).load_tokenizer()


def dump_rows(path: Path, rows):
    path.write_bytes((json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8'))
    return op.sha_file(path)


def write_pool(out: Path, dataset, rows, gold, info_keys, outputs, gold_bytes=None):
    """Manifest + rowinfo + gold of one dataset; returns the cell list (seed filled in by the caller)."""
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids) or set(gold) != set(ids):
        raise AssertionError(f'{dataset}: ids must be unique and match the gold keys')
    for r in rows:
        if r['prompt_token_count'] != len(r['prompt_tokens']) or not all(type(t) is int for t in r['prompt_tokens']):
            raise AssertionError(f'{dataset}: prompt_tokens must be ints matching prompt_token_count')
        if r['prompt_hash'] != op.sha(r['prompt']):
            raise AssertionError(f'{dataset}: prompt_hash must be sha256(prompt)')
    outputs[f'{dataset}_generation_manifest.json'] = dump_rows(out / f'{dataset}_generation_manifest.json', rows)
    info = [dict({k: r[k] for k in info_keys if k in r}, index=i) for i, r in enumerate(rows)]
    path = out / f'{dataset}_rowinfo.json'
    path.write_text(json.dumps(info, indent=1, sort_keys=True) + '\n')
    outputs[path.name] = op.sha_file(path)
    path = out / f'{dataset}_gold.json'
    path.write_bytes(gold_bytes if gold_bytes is not None else (json.dumps(gold, sort_keys=True) + '\n').encode())
    outputs[path.name] = op.sha_file(path)
    return [dict(dataset=dataset, id=r['id'], index=i, prompt_tokens=r['prompt_token_count']) for i, r in enumerate(rows)]


def write_cells(out: Path, name, cells, outputs):
    path = out / f'cells_{name}.json'
    path.write_text(json.dumps(cells))
    outputs[path.name] = op.sha_file(path)


def identity(adapter, extra_code=()):
    import tokenizers
    import transformers
    model = Path(MODEL)
    code_root = Path.cwd()
    files = [Path(__file__).resolve(), Path(op.__file__).resolve(), *extra_code]
    return dict(
        tokenizer=dict(model_path=MODEL, model_revision=REVISION, processor_class=type(adapter.processor).__name__,
                       tokenizer_class=type(adapter.tokenizer).__name__, transformers=transformers.__version__,
                       tokenizers=tokenizers.__version__,
                       files_sha256={n: op.sha_file(model / n) for n in ('tokenizer.json', 'tokenizer_config.json',
                                                                          'chat_template.jinja', 'processor_config.json')
                                     if (model / n).exists()}),
        code=dict(cwd=str(code_root),
                  deploy_sha=(code_root / 'DEPLOY_SHA').read_text().strip() if (code_root / 'DEPLOY_SHA').exists() else None,
                  adapter_sha256=op.sha_file(code_root / 'src/dllm/models/adapters/diffusion_gemma.py'),
                  files_sha256={str(p): op.sha_file(p) for p in files}),
        environment=dict(python=sys.executable, python_version=platform.python_version(), host=platform.node(),
                         vars={k: os.environ.get(k) for k in ENV_VARS}, gpu_used=False, model_weights_loaded=False))


def write_readme(out: Path, readme):
    (out / 'readme.json').write_text(json.dumps(readme, indent=2, sort_keys=True) + '\n')

# ---------------------------------------------------------------- RULER


_W = {}


def _ruler_init():
    _W['adapter'] = load_adapter()


def _ruler_render(job):
    """(dataset, index, input, answer_prefix) -> ids + structural checks (no text returned)."""
    dataset, index, sample_input, answer_prefix = job
    adapter = _W['adapter']
    tok = adapter.tokenizer
    opener = op.RESPONSE_OPENER['diffusion_gemma']
    ids, n_user = op.ruler_prompt_ids(adapter, sample_input, answer_prefix, opener)
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    gen_prompt, opener_ids = enc('<|turn>model\n'), enc(opener)
    checks = dict(
        generation_prompt_ends_user_part=ids[n_user - len(gen_prompt):n_user] == gen_prompt,
        opener_follows=ids[n_user:n_user + len(opener_ids)] == opener_ids,
        prefix_tokens_equal_separate_encoding=ids[n_user + len(opener_ids):] == enc(answer_prefix),
        prefix_round_trip=dec(ids[n_user + len(opener_ids):]) == answer_prefix,
        tail_decodes_to_model_turn_opener_prefix=dec(ids[n_user - len(gen_prompt):]) == '<|turn>model\n' + opener + answer_prefix,
        whole_string_tokenization_identical=enc(dec(ids)) == ids,
        one_bos_first=ids[0] == tok.bos_token_id and ids.count(tok.bos_token_id) == 1,
        turn_tokens=ids.count(tok.convert_tokens_to_ids('<|turn>')) == 2 and ids.count(tok.convert_tokens_to_ids('<turn|>')) == 1,
        one_thought_block=ids.count(tok.convert_tokens_to_ids('<|channel>')) == 1 and ids.count(tok.convert_tokens_to_ids('<channel|>')) == 1,
        no_think_token=tok.convert_tokens_to_ids('<|think|>') not in ids)
    return dataset, index, ids, n_user, len(ids) - n_user, checks


def build_ruler(a, logf):
    src = Path(a.src or RULER_SRC)
    if 'wrote readme.json' not in (src / 'gen.log').read_text():
        raise RuntimeError('the v33 generator has not finished (no "wrote readme.json" in gen.log)')
    src_readme = json.loads((src / 'readme.json').read_text())
    ruler_root = Path(src_readme['ruler_root'])
    ruler_identity = op.ruler_verify_checkout(ruler_root)
    log(logf, f'RULER checkout {ruler_identity["commit"][:8]} verified; source {src}')
    out, outputs, plan, meta = a.out, {}, [], {}
    for L, bench in RULER_LENGTHS:
        old = json.loads((src / f'{bench}_generation_manifest.json').read_text())
        gold_bytes = (src / f'{bench}_gold.json').read_bytes()
        gold = json.loads(gold_bytes)
        raw_dir = src / 'raw' / str(L)
        raw_manifest = json.loads((raw_dir / 'manifest.json').read_text())
        samples_path = raw_dir / 'samples.jsonl'
        if op.sha_file(samples_path) != raw_manifest['samples']['sha256']:
            raise AssertionError(f'{bench}: samples.jsonl differs from its raw manifest hash')
        samples = [json.loads(x) for x in samples_path.read_text().splitlines() if x.strip()]
        shards = {t: [json.loads(x) for x in (raw_dir / 'official_raw' / str(L) / t / 'validation.jsonl').read_text().splitlines()
                      if x.strip()] for t in raw_manifest['tasks']}
        per_task, by_source = collections.Counter(), {}
        for rec in samples:
            p = per_task[rec['task']]
            per_task[rec['task']] += 1
            m = re.fullmatch(rf"ruler_{L}_{re.escape(rec['task'])}_(\d{{4}})", rec['sample_id'])
            shard = shards[rec['task']][int(m.group(1))]
            if shard['input'] + shard['answer_prefix'] != rec['prompt'] or [str(x) for x in shard['outputs']] != rec['outputs']:
                raise AssertionError(f'{bench}: sample does not match its official shard row')
            if shard['index'] != rec['official_index'] or not shard['answer_prefix']:
                raise AssertionError(f'{bench}: shard index / answer prefix mismatch')
            by_source[f"ruler_{L}_{rec['task']}_p{p:04d}"] = (rec, shard)
        if len(old) != len(samples) or {r['source_id'] for r in old} != set(by_source):
            raise AssertionError(f'{bench}: manifest rows and samples differ')
        rows = []
        for i, r in enumerate(old):
            rec, shard = by_source[r['source_id']]
            if (r['id'] != f"{bench}/{r['source_id']}" or r['prompt'] != rec['prompt'] or r['prompt_hash'] != op.sha(rec['prompt'])
                    or r['task'] != rec['task'] or r['generation_budget'] != rec['tokens_to_generate'] or r['thinking'] is not False
                    or gold[r['id']] != rec['outputs']):
                raise AssertionError(f'{bench}: v33 manifest row {i} does not match its sample / gold')
            rows.append(dict(answer_prefix=shard['answer_prefix'], benchmark=bench, generation_budget=int(rec['tokens_to_generate']),
                             id=r['id'], prompt=shard['input'], prompt_hash=op.sha(shard['input']),
                             response_prefill=op.RESPONSE_OPENER['diffusion_gemma'] + shard['answer_prefix'],
                             source_id=r['source_id'], task=rec['task'], task_base=rec['task_base'], thinking=False,
                             v33_prompt_hash=r['prompt_hash'], v33_prompt_token_count=r['prompt_token_count']))
            plan.append((bench, i, shard['input'], shard['answer_prefix']))
        meta[bench] = dict(L=L, rows=rows, gold=gold, gold_bytes=gold_bytes, raw_manifest=raw_manifest,
                           source_sha256={f'{bench}_generation_manifest.json': op.sha_file(src / f'{bench}_generation_manifest.json'),
                                          f'{bench}_gold.json': op.sha(gold_bytes), f'raw/{L}/samples.jsonl': op.sha_file(samples_path),
                                          f'raw/{L}/manifest.json': op.sha_file(raw_dir / 'manifest.json')})
        log(logf, f'{bench}: {len(rows)} rows mapped to samples, official shards and gold')
    t0 = time.time()
    checks = collections.Counter()
    with multiprocessing.Pool(a.jobs, initializer=_ruler_init) as pool:
        for n, (bench, i, ids, n_user, n_tail, ck) in enumerate(pool.imap_unordered(_ruler_render, plan, chunksize=1)):
            row = meta[bench]['rows'][i]
            row.update(prompt_tokens=ids, prompt_token_count=len(ids), user_token_count=n_user, prefill_token_count=n_tail)
            for k, v in ck.items():
                checks[k] += bool(v)
            if not all(ck.values()):
                raise AssertionError(f'{bench} row {i}: failed checks {[k for k, v in ck.items() if not v]}')
            if (n + 1) % 50 == 0:
                log(logf, f'rendered {n + 1}/{len(plan)} in {time.time() - t0:.0f}s')
    log(logf, f'rendered {len(plan)} rows in {time.time() - t0:.0f}s; all structural checks passed')
    cells, bins = [], {}
    adapter = load_adapter()
    for L, bench in RULER_LENGTHS:
        m = meta[bench]
        rows = m['rows']
        for r in rows:
            r.pop('user_token_count')
        cells += [dict(c, seed=1) for c in write_pool(out, bench, rows, m['gold'], ('id', 'task', 'task_base', 'generation_budget',
                                                                                      'thinking', 'prompt_token_count', 'prefill_token_count'),
                                                       outputs, gold_bytes=m['gold_bytes'])]
        by_task = collections.defaultdict(list)
        for r in rows:
            by_task[r['task']].append(r)
        bins[bench] = dict(
            context_length=m['L'], rows=len(rows), tasks=dict(collections.Counter(r['task'] for r in rows)),
            prompt_token_count=stats([r['prompt_token_count'] for r in rows]),
            delta_vs_v33_prompt_token_count=stats([r['prompt_token_count'] - r['v33_prompt_token_count'] for r in rows]),
            prefill_token_count_by_task={t: stats([r['prefill_token_count'] for r in v]) for t, v in sorted(by_task.items())},
            generation_budget_by_task={t: sorted({r['generation_budget'] for r in v}) for t, v in sorted(by_task.items())},
            total_tokens_max=max(r['prompt_token_count'] + r['generation_budget'] for r in rows),
            source_sha256=m['source_sha256'])
    write_cells(out, 'ruler_v33', cells, outputs)
    tok = adapter.tokenizer
    opener = op.RESPONSE_OPENER['diffusion_gemma']
    readme = dict(
        schema='pools_v31_official_readme_v1', pool='ruler_v33', built_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        rule=__doc__.split('  longbench')[0].strip(), source=str(src), source_readme_sha256=op.sha_file(src / 'readme.json'),
        ruler=dict(root=str(ruler_root), **ruler_identity, generator_seed=src_readme['seed'], length_mode=src_readme['length_mode']),
        placement=dict(user_turn='adapter.encode_prompt(sample input, {"thinking": False}) -> <bos><|turn>user\\n{input}<turn|>\\n<|turn>model\\n',
                       response_prefill='tokenizer.encode(opener + answer_prefix, add_special_tokens=False)',
                       opener=opener, opener_ids=tok.encode(opener, add_special_tokens=False),
                       generation_prompt_ids=tok.encode('<|turn>model\n', add_special_tokens=False),
                       answer_prefix='verbatim RULER answer_prefix (leading space kept, as RULER sends it)',
                       opener_evidence='3972 / 3975 non-thinking RULER completions of the v31 panels open with exactly the '
                                       'opener; the other 3 open it and stop without closing it'),
        row_format=dict(keys=sorted(meta[RULER_LENGTHS[0][1]]['rows'][0]),
                        prompt='RULER sample input (the user-turn text, answer prefix removed)',
                        prompt_tokens='encode_prompt(prompt, thinking off) + tokenizer.encode(response_prefill, add_special_tokens=False)',
                        v33_prompt_hash='prompt_hash of the same row in the v33 pool (input + answer prefix in the user turn)'),
        gold='byte-identical copies of the v33 gold files (same ids / samples)',
        checks=dict(checks, rows=len(plan)), counts={b: v['rows'] for b, v in bins.items()}, bins=bins,
        output_files_sha256=outputs, **identity(adapter))
    write_readme(out, readme)
    log(logf, f'done; outputs {outputs}')

# ---------------------------------------------------------------- LongBench-v2


def _lb_init(templates):
    _W['adapter'] = load_adapter()
    _W['templates'] = templates
    import v15_longbench_task as v15
    _W['escape'] = v15.escape


def lb_prompt(tok, template, item, escape):
    """pred.py prompt + query_llm truncation with OUR tokenizer: (text, original token count, truncated)."""
    escaped, escapes = escape(item, tuple(getattr(tok, 'all_special_tokens', ())))
    prompt = op.lb_fill(template, escaped)
    full = tok.encode(prompt)
    if len(full) <= op.LB_MAX_LEN:
        return prompt, len(full), False, escapes
    return tok.decode(op.middle_truncate(full), skip_special_tokens=True), len(full), True, escapes


def _lb_render(job):
    from array import array
    index, item = job
    adapter = _W['adapter']
    per_template, escapes = {}, {}
    for name, template in _W['templates'].items():
        text, n_orig, truncated, escapes = lb_prompt(adapter.tokenizer, template, item, _W['escape'])
        per_template[name] = (text, n_orig, truncated)
    rendered = {dataset: array('i', adapter.encode_prompt(per_template[tname][0], {'thinking': thinking}))
                for dataset, tname, thinking, _, _ in LB_VARIANTS}
    return index, per_template, rendered, escapes


def lb_repo_identity():
    import subprocess
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    head = subprocess.check_output(['git', '-C', str(LB_REPO), 'rev-parse', 'HEAD'], text=True, env=env).strip()
    if head != LB_REPO_COMMIT:
        raise ValueError(f'LongBench checkout at {head}, pinned {LB_REPO_COMMIT}')
    out = {}
    for rel in LB_REPO_FILES:
        committed = subprocess.check_output(['git', '-C', str(LB_REPO), 'show', f'{LB_REPO_COMMIT}:{rel}'], env=env)
        if (LB_REPO / rel).read_bytes() != committed:
            raise ValueError(f'modified LongBench source: {rel}')
        out[rel] = op.sha(committed)
    maxlen = json.loads((LB_REPO / 'config/model2maxlen.json').read_text())
    return dict(repo=str(LB_REPO), commit=head, files_sha256=out,
                max_len_open_models=sorted({v for k, v in maxlen.items() if 'claude' not in k}))


def build_longbench(a, logf):
    import v15_longbench_task as v15
    repo = lb_repo_identity()
    if repo['max_len_open_models'] != [op.LB_MAX_LEN]:
        raise AssertionError('config/model2maxlen.json does not give 120000 for the open models')
    templates = {t: (LB_REPO / 'prompts' / f'{t}.txt').read_text(encoding='utf-8') for t in ('0shot', '0shot_cot')}
    data_bytes = v15.LB_DATA.read_bytes()
    data = json.loads(data_bytes)
    if len(data) != 503 or len({r['_id'] for r in data}) != 503:
        raise AssertionError('expected 503 unique LongBench-v2 items')
    for r in data:
        if r['answer'] not in ('A', 'B', 'C', 'D') or r['difficulty'] not in ('easy', 'hard') or r['length'] not in ('short', 'medium', 'long'):
            raise AssertionError('unexpected LongBench-v2 answer / difficulty / length value')
    log(logf, f'LongBench repo {repo["commit"][:8]} verified; dataset {v15.LB_REVISION[:8]} sha256 {op.sha(data_bytes)[:12]}; {len(data)} items')
    adapter = load_adapter()
    tok = adapter.tokenizer
    probe = tok.encode('probe text')
    if probe != tok.encode('probe text', add_special_tokens=False):
        raise AssertionError('tokenizer.encode adds special tokens; the 120,000 count would include them (handle explicitly)')
    order = sorted(range(len(data)), key=lambda i: -len(data[i]['context']))     # longest first for load balance
    results, escapes_total = {}, collections.Counter()
    t0 = time.time()
    with multiprocessing.Pool(a.jobs, initializer=_lb_init, initargs=(templates,)) as pool:
        for n, (index, per_template, rendered, escapes) in enumerate(pool.imap_unordered(_lb_render, ((i, data[i]) for i in order), chunksize=1)):
            results[index] = (per_template, rendered)
            escapes_total.update(escapes)
            if (n + 1) % 25 == 0:
                log(logf, f'rendered {n + 1}/{len(data)} items in {time.time() - t0:.0f}s')
    log(logf, f'rendered all {len(data)} items in {time.time() - t0:.0f}s')
    outputs, variants = {}, {}
    for dataset, tname, thinking, budget, column in LB_VARIANTS:
        rows, gold = [], {}
        for i, item in enumerate(data):
            text, n_orig, truncated = results[i][0][tname]
            ids = list(results[i][1][dataset])
            rows.append(dict(benchmark=dataset, difficulty=item['difficulty'], domain=item['domain'], generation_budget=budget,
                             id=f"longbench_v2/{item['_id']}", length=item['length'], original_token_count=n_orig,
                             prompt=text, prompt_hash=op.sha(text), prompt_token_count=len(ids), prompt_tokens=ids,
                             source_id=item['_id'], sub_domain=item['sub_domain'], template=tname, thinking=thinking,
                             truncated=truncated))
            gold[rows[-1]['id']] = item['answer']
        cells = write_pool(a.out, dataset, rows, gold, ('id', 'difficulty', 'length', 'domain', 'generation_budget', 'thinking',
                                                         'template', 'truncated', 'original_token_count', 'prompt_token_count'),
                           outputs)
        write_cells(a.out, dataset, [dict(c, seed=1) for c in cells], outputs)
        tr = [r for r in rows if r['truncated']]
        variants[dataset] = dict(
            template=f'prompts/{tname}.txt', thinking=thinking, generation_budget=budget, column=column, rows=len(rows),
            truncated=len(tr), untruncated=len(rows) - len(tr),
            original_token_count=stats([r['original_token_count'] for r in rows]),
            original_token_count_truncated=stats([r['original_token_count'] for r in tr]),
            prompt_token_count=stats([r['prompt_token_count'] for r in rows]),
            prompt_token_count_truncated=stats([r['prompt_token_count'] for r in tr]),
            truncated_by_length_class=dict(collections.Counter(r['length'] for r in tr)),
            difficulty=dict(collections.Counter(r['difficulty'] for r in rows)),
            length_class=dict(collections.Counter(r['length'] for r in rows)),
            total_tokens_max=max(r['prompt_token_count'] + budget for r in rows))
        log(logf, f'{dataset}: rows={len(rows)} truncated={len(tr)} tokens={variants[dataset]["prompt_token_count"]}')
        del rows
    readme = dict(
        schema='pools_v31_official_readme_v1', pool='longbench_v2', built_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        rule=__doc__.split('  longbench')[1].split('  aime')[0].strip(), official=repo,
        dataset=dict(revision=v15.LB_REVISION, path=str(v15.LB_DATA), sha256=op.sha(data_bytes), items=len(data)),
        truncation=dict(max_len=op.LB_MAX_LEN, keep=f'first {op.LB_MAX_LEN // 2} + last {op.LB_MAX_LEN // 2} ids of tokenizer.encode(filled '
                        'template) (adds no special tokens for this tokenizer), decode(skip_special_tokens=True)',
                        tokenizer='ours (Gemma4Processor.tokenizer)'),
        escaping=dict(rule='scripts/v15_longbench_task.py escape: literal reserved special-token spellings in the item fields '
                           'become &lt;...&gt; before the template is filled (declared deviation)',
                      occurrences=dict(escapes_total)),
        row_format=dict(prompt='the (possibly truncated) filled template = the user-turn text',
                        original_token_count='len(tokenizer.encode(filled template)) before truncation',
                        prompt_token_count='len(adapter.encode_prompt(prompt, {"thinking": thinking}))'),
        variants=variants, output_files_sha256=outputs,
        **identity(adapter, extra_code=(Path(v15.__file__).resolve(),)))
    write_readme(a.out, readme)
    log(logf, f'done; outputs {outputs}')

# ---------------------------------------------------------------- AIME26 / GraphWalks budget variants


def build_aime(a, logf):
    src = Path(a.src or AIME_SRC)
    rows = json.loads(src.read_text())
    ids = [r['id'] for r in rows]
    if ids != [f'aime26/{i}' for i in range(1, 31)] or len(set(ids)) != 30:
        raise AssertionError('unexpected AIME26 ids')
    new = [dict(r, generation_budget=AIME_BUDGET) for r in rows]
    for r, n in zip(rows, new):
        if {k: v for k, v in r.items() if k != 'generation_budget'} != {k: v for k, v in n.items() if k != 'generation_budget'}:
            raise AssertionError('only generation_budget may change')
        if r['prompt_hash'] != op.sha(r['prompt']) or r['prompt_token_count'] != len(r['prompt_tokens']) or r['thinking'] is not True:
            raise AssertionError('source row inconsistent')
    outputs = {}
    outputs['aime26_generation_manifest.json'] = dump_rows(a.out / 'aime26_generation_manifest.json', new)
    info = [dict(id=r['id'], index=i, source_id=r['source_id'], generation_budget=r['generation_budget'], thinking=r['thinking'],
                 prompt_token_count=r['prompt_token_count']) for i, r in enumerate(new)]
    (a.out / 'aime26_rowinfo.json').write_text(json.dumps(info, indent=1, sort_keys=True) + '\n')
    outputs['aime26_rowinfo.json'] = op.sha_file(a.out / 'aime26_rowinfo.json')
    cells = [dict(dataset='aime26', id=r['id'], index=i, seed=s, prompt_tokens=r['prompt_token_count'])
             for s in AIME_SEEDS for i, r in enumerate(new)]
    write_cells(a.out, 'aime26', cells, outputs)
    readme = dict(schema='pools_v31_official_readme_v1', pool='aime26', built_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  rule='v27 final AIME26 manifest, generation_budget 8192 -> 32768; ids, prompts, prompt_tokens, thinking '
                       'unchanged (asserted per row); cells seeds 1-4 (avg@4), seed-major. Gold: the pinned math-ai/aime26 '
                       'file read by scripts/v31_score_aime.py (not copied here).',
                  source=str(src), source_sha256=op.sha_file(src), rows=len(new), cells=len(cells), seeds=list(AIME_SEEDS),
                  prompt_token_count=stats([r['prompt_token_count'] for r in new]), output_files_sha256=outputs,
                  code=dict(builder_sha256=op.sha_file(Path(__file__).resolve())))
    write_readme(a.out, readme)
    log(logf, f'aime26: {len(new)} rows, {len(cells)} cells; outputs {outputs}')


def build_graphwalks(a, logf):
    src = Path(a.src or GW_SRC)
    outputs, bins = {}, {}
    for ds in GW_DATASETS:
        rows = json.loads((src / f'{ds}_generation_manifest.json').read_text())
        gold_bytes = (src / f'{ds}_gold.json').read_bytes()
        new = [dict(r, generation_budget=GW_BUDGET) for r in rows]
        for r in rows:
            if r['prompt_hash'] != op.sha(r['prompt']) or r['prompt_token_count'] != len(r['prompt_tokens']) or r['thinking'] is not True:
                raise AssertionError(f'{ds}: source row inconsistent')
        write_pool(a.out, ds, new, json.loads(gold_bytes), ('id', 'problem_type', 'bin', 'generation_budget', 'thinking',
                                                              'prompt_token_count'), outputs, gold_bytes=gold_bytes)
        back = json.loads((a.out / f'{ds}_generation_manifest.json').read_text())
        same = all({k: v for k, v in x.items() if k != 'generation_budget'} == {k: v for k, v in y.items() if k != 'generation_budget'}
                   for x, y in zip(rows, back)) and len(back) == len(rows)
        if not same or {r['generation_budget'] for r in back} != {GW_BUDGET}:
            raise AssertionError(f'{ds}: written rows differ from the source beyond generation_budget')
        bins[ds] = dict(rows=len(new), problem_type=dict(collections.Counter(r['problem_type'] for r in new)),
                        prompt_token_count=stats([r['prompt_token_count'] for r in new]), identical_to_source_except_budget=same,
                        source_manifest_sha256=op.sha_file(src / f'{ds}_generation_manifest.json'),
                        source_gold_sha256=op.sha(gold_bytes))
    cells_src = src / 'cells_graphwalks.json'
    shutil.copyfile(cells_src, a.out / 'cells_graphwalks.json')
    cells = json.loads((a.out / 'cells_graphwalks.json').read_text())
    for c in cells:
        rows = json.loads((a.out / f"{c['dataset']}_rowinfo.json").read_text())
        if rows[c['index']]['id'] != c['id'] or c['seed'] != 1 or rows[c['index']]['prompt_token_count'] != c['prompt_tokens']:
            raise AssertionError('copied cells do not match the manifests')
    outputs['cells_graphwalks.json'] = op.sha_file(a.out / 'cells_graphwalks.json')
    readme = dict(schema='pools_v31_official_readme_v1', pool='graphwalks', built_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  rule=f'pools_v31 GraphWalks pool with generation_budget 8192 -> {GW_BUDGET} (no official budget: the '
                       'reasoning-model rule; 4 of 24 graphwalks_90k gold answer lines alone need > 2048 tokens); ids, prompts, '
                       'prompt_tokens, thinking, gold and cells byte-identical / unchanged (asserted)',
                  source=str(src), source_readme_sha256=op.sha_file(src / 'readme.json'), bins=bins,
                  cells=len(cells), output_files_sha256=outputs, code=dict(builder_sha256=op.sha_file(Path(__file__).resolve())))
    write_readme(a.out, readme)
    log(logf, f'graphwalks: {sum(b["rows"] for b in bins.values())} rows; outputs {outputs}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('pool', choices=('ruler', 'longbench', 'aime', 'graphwalks'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--src', default=None)
    p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if any(a.out.glob('*_generation_manifest.json')):
        raise FileExistsError(f'{a.out} already holds manifests; build into a fresh directory')
    with (a.out / 'build.log').open('a') as logf:
        log(logf, f'build {a.pool} -> {a.out} (jobs {a.jobs})')
        dict(ruler=build_ruler, longbench=build_longbench, aime=build_aime, graphwalks=build_graphwalks)[a.pool](a, logf)


if __name__ == '__main__':
    main()
