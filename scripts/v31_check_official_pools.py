"""Fresh-process checks of the pools_v31_official pools (CPU tokenizer only, no weights, no GPU).

For every manifest row: encode_prompt(prompt, {'thinking': thinking}) + tokens(response_prefill, if any) reproduces
prompt_tokens; prompt_hash = sha256(prompt); prompt_token_count = len(prompt_tokens); rowinfo, cells (index -> id,
prompt_tokens) and gold keys agree with the manifest. Per pool in addition:
  ruler_v33    the decoded tail is exactly '<|turn>model\\n' + response_prefill (= empty thought block + answer prefix),
               the user turn holds no answer prefix and closes with '<turn|>\\n<|turn>model\\n'; ids / gold equal the
               v33 source pool;
  longbench_v2 the truncation is re-derived from the source items (escape, fill, tokenizer.encode, first / last
               60,000 ids, decode): same text, flag and original token count; 503 rows per variant;
  aime26 / graphwalks  ids, prompts and prompt_tokens equal the source manifests; only generation_budget differs.
Writes counts only (OUT_JSON) and prints a summary; any mismatch is counted, never printed as text.
usage: python v31_check_official_pools.py OUT_JSON POOL_DIR [POOL_DIR ...] [--jobs N]
Run with cwd = a deployment holding src/ (PYTHONPATH=src:.), CUDA_VISIBLE_DEVICES=, HF_HUB_OFFLINE=1.
"""
from __future__ import annotations

import collections
import json
import multiprocessing
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v31_official as op  # noqa: E402
import v31_build_official_pools as bp  # noqa: E402

_W = {}


def _init():
    _W['adapter'] = bp.load_adapter()


def _render(job):
    """(key, prompt, thinking, prefill, expected ids) -> (key, reproduced, decoded-tail ok or None)."""
    key, prompt, thinking, prefill, expected = job
    adapter = _W['adapter']
    tok = adapter.tokenizer
    ids = list(adapter.encode_prompt(prompt, {'thinking': thinking}))
    tail_ok = None
    if prefill is not None:
        pre = tok.encode(prefill, add_special_tokens=False)
        n_user = len(ids)
        ids += pre
        dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        tail_ok = (dec(ids[n_user - 3:]) == '<|turn>model\n' + prefill and dec(ids[n_user - 5:n_user]).endswith('<turn|>\n<|turn>model\n')
                   and ids.count(tok.convert_tokens_to_ids('<|turn>')) == 2)
    return key, ids == list(expected), tail_ok


def _lb_truncation(job):
    index, item, templates = job
    tok = _W['adapter'].tokenizer
    import v15_longbench_task as v15
    return index, {t: bp.lb_prompt(tok, text, item, v15.escape)[:3] for t, text in templates.items()}


def check_pool(pool_dir: Path, jobs, report):
    c = collections.Counter()
    manifests = sorted(pool_dir.glob('*_generation_manifest.json'))
    cells = [x for f in sorted(pool_dir.glob('cells_*.json')) for x in json.loads(f.read_text())]
    cells_by_ds = collections.defaultdict(list)
    for x in cells:
        cells_by_ds[x['dataset']].append(x)
    plan, rows_by_ds = [], {}
    for man in manifests:
        ds = man.name[:-len('_generation_manifest.json')]
        rows = json.loads(man.read_text(encoding='utf-8'))
        rows_by_ds[ds] = rows
        info = json.loads((pool_dir / f'{ds}_rowinfo.json').read_text()) if (pool_dir / f'{ds}_rowinfo.json').exists() else None
        gold_path = pool_dir / f'{ds}_gold.json'
        if gold_path.exists():
            c['gold_keys_match'] += set(json.loads(gold_path.read_text())) == {r['id'] for r in rows}
            c['gold_files'] += 1
        c['ids_unique'] += len({r['id'] for r in rows}) == len(rows)
        for i, r in enumerate(rows):
            c['rows'] += 1
            c['hash_ok'] += r['prompt_hash'] == op.sha(r['prompt'])
            c['count_ok'] += r['prompt_token_count'] == len(r['prompt_tokens'])
            if info is not None:
                c['rowinfo_ok'] += (info[i]['id'] == r['id'] and info[i]['index'] == i and info[i]['prompt_token_count'] == r['prompt_token_count']
                                    and info[i]['generation_budget'] == r['generation_budget'])
            plan.append(((ds, i), r['prompt'], bool(r['thinking']), r.get('response_prefill'), r['prompt_tokens']))
        for x in cells_by_ds.get(ds, []):
            c['cells'] += 1
            c['cells_ok'] += rows[x['index']]['id'] == x['id'] and rows[x['index']]['prompt_token_count'] == x['prompt_tokens']
        c['manifests'] += 1
    c['cells_without_manifest'] = sum(len(v) for ds, v in cells_by_ds.items() if ds not in rows_by_ds)
    t0 = time.time()
    with multiprocessing.Pool(jobs, initializer=_init) as pool:
        for key, same, tail_ok in pool.imap_unordered(_render, plan, chunksize=1):
            c['rerender_identical'] += same
            if tail_ok is not None:
                c['ruler_tail_checked'] += 1
                c['ruler_tail_ok'] += tail_ok
        if pool_dir.name.startswith('longbench'):
            import v15_longbench_task as v15
            data = json.loads(v15.LB_DATA.read_bytes())
            templates = {t: (bp.LB_REPO / 'prompts' / f'{t}.txt').read_text(encoding='utf-8') for t in ('0shot', '0shot_cot')}
            order = sorted(range(len(data)), key=lambda i: -len(data[i]['context']))
            derived = dict(pool.imap_unordered(_lb_truncation, ((i, data[i], templates) for i in order), chunksize=1))
            for ds, rows in rows_by_ds.items():
                c[f'{ds}_rows'] = len(rows)
                c[f'{ds}_truncated'] = sum(r['truncated'] for r in rows)
                for i, r in enumerate(rows):
                    text, n_orig, truncated = derived[i][r['template']]
                    c['lb_truncation_rederived_identical'] += (text == r['prompt'] and n_orig == r['original_token_count']
                                                               and truncated == r['truncated'] and r['source_id'] == data[i]['_id'])
                    c['lb_rows'] += 1
    c['seconds'] = round(time.time() - t0)
    if pool_dir.name.startswith('ruler'):
        src = Path(json.loads((pool_dir / 'readme.json').read_text())['source'])
        for ds, rows in rows_by_ds.items():
            old = json.loads((src / f'{ds}_generation_manifest.json').read_text())
            c['ruler_ids_equal_v33'] += [r['id'] for r in rows] == [r['id'] for r in old]
            c['ruler_raw_prompt_equal_v33'] += sum(r['prompt'] + r['answer_prefix'] == o['prompt'] for r, o in zip(rows, old))
            c['ruler_user_turn_without_prefix'] += sum(not r['prompt'].rstrip().endswith(r['answer_prefix'].strip()) for r in rows)
            c['ruler_prefill_is_opener_plus_prefix'] += sum(r['response_prefill'] == op.RESPONSE_OPENER['diffusion_gemma'] + r['answer_prefix']
                                                            for r in rows)
            c['ruler_gold_bytes_equal_v33'] += (pool_dir / f'{ds}_gold.json').read_bytes() == (src / f'{ds}_gold.json').read_bytes()
    if pool_dir.name in ('aime26', 'graphwalks'):
        srcs = {'aime26': {'aime26': bp.AIME_SRC}, 'graphwalks': {d: bp.GW_SRC / f'{d}_generation_manifest.json' for d in bp.GW_DATASETS}}
        for ds, path in srcs[pool_dir.name].items():
            old = json.loads(path.read_text())
            new = rows_by_ds[ds]
            c['source_rows_equal_except_budget'] += sum({k: v for k, v in a.items() if k != 'generation_budget'} ==
                                                        {k: v for k, v in b.items() if k != 'generation_budget'} for a, b in zip(old, new))
            c['source_rows'] += len(old)
            c[f'{ds}_budgets'] = sorted({r['generation_budget'] for r in new})[0]
    report[pool_dir.name] = dict(c)
    return c


def main():
    args = sys.argv[1:]
    jobs = 4
    if '--jobs' in args:
        i = args.index('--jobs')
        jobs = int(args[i + 1])
        del args[i:i + 2]
    out, dirs = Path(args[0]), [Path(x) for x in args[1:]]
    report = {}
    for d in dirs:
        c = check_pool(d, jobs, report)
        print(d.name, dict(c), flush=True)
        out.write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
