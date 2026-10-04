"""Fresh-process checks of the pools_v31_official pools (CPU tokenizer only, no weights, no GPU, no git).

For every manifest row: encode_prompt(prompt, {'thinking': thinking}) + tokens(response_prefill, if any) reproduces
prompt_tokens; prompt_hash = sha256(prompt); prompt_token_count = len(prompt_tokens); benchmark = the dataset name.
Rowinfo (v31_rowinfo_v2): its manifest_sha256 is the manifest file's, every row's prompt_sha256 is the sha256 of its
token ids, budgets / counts agree, and the pinned max_model_len holds every row (prompt + budget). Cells (index -> id,
prompt_tokens) and gold keys agree with the manifest. Per pool in addition:
  ruler_v33ofc  decoded tail = '<|turn>model\\n' + empty thought block + answer prefix; ids / gold / raw prompts equal the
                v33 source pool;
  ruler_v33noop decoded tail = '<|turn>model\\n' + answer prefix, no thought block; ids / gold equal the ofc rows;
  longbench_v2  truncation re-derived from the source items with the dyh copies of the official files (escape, fill,
                tokenizer.encode, first / last 60,000 ids, decode): same text, flag and original token count;
  aime26_b32k / graphwalks_b16k  ids, prompts and prompt_tokens equal the sources; only the name and budget differ;
  mrcr_ofc      every row belongs to the dataset block of its README bin.
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
    """(key, prompt, thinking, prefill, expected ids, opener expected) -> (key, reproduced, tail ok or None)."""
    key, prompt, thinking, prefill, expected, opener = job
    adapter = _W['adapter']
    tok = adapter.tokenizer
    ids = list(adapter.encode_prompt(prompt, {'thinking': thinking}))
    tail_ok = None
    if prefill is not None:
        n_user = len(ids)
        ids += tok.encode(prefill, add_special_tokens=False)
        dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        n_channel = ids.count(tok.convert_tokens_to_ids('<|channel>'))
        tail_ok = (dec(ids[n_user - 3:]) == '<|turn>model\n' + prefill and dec(ids[n_user - 5:n_user]) == '<turn|>\n<|turn>model\n'
                   and ids.count(tok.convert_tokens_to_ids('<|turn>')) == 2 and prefill.startswith(opener)
                   and n_channel == (1 if opener else 0))
    return key, ids == list(expected), tail_ok


def _lb_truncation(job):
    index, item, templates = job
    tok = _W['adapter'].tokenizer
    import v15_longbench_task as v15
    return index, {t: bp.lb_prompt(tok, text, item, v15.escape)[:3] for t, text in templates.items()}


def check_pool(pool_dir: Path, jobs, report):
    c = collections.Counter()
    manifests = sorted(pool_dir.glob('*_generation_manifest.json'))
    cells = collections.defaultdict(list)
    for f in sorted(pool_dir.glob('cells_*.json')):
        for x in json.loads(f.read_text()):
            cells[x['dataset']].append(x)
    opener = op.RESPONSE_OPENER['diffusion_gemma'] if pool_dir.name == 'ruler_v33ofc' else ''
    plan, rows_by_ds = [], {}
    for man in manifests:
        ds = man.name[:-len('_generation_manifest.json')]
        rows = json.loads(man.read_text(encoding='utf-8'))
        rows_by_ds[ds] = rows
        info = json.loads((pool_dir / f'{ds}_rowinfo.json').read_text())
        c['rowinfo_v2'] += info.get('schema') == op.ROWINFO_SCHEMA and info['dataset'] == ds
        c['manifest_sha_ok'] += info['manifest_sha256'] == op.sha_file(man)
        pin = info['max_model_len']
        c['pin_holds_rows'] += all(r['prompt_token_count'] + r['generation_budget'] <= pin for r in rows)
        gold_path = pool_dir / f'{ds}_gold.json'
        if gold_path.exists():
            c['gold_keys_match'] += set(json.loads(gold_path.read_text())) == {r['id'] for r in rows}
            c['gold_files'] += 1
        c['ids_unique'] += len({r['id'] for r in rows}) == len(rows)
        for i, r in enumerate(rows):
            ri = info['rows'][i]
            c['rows'] += 1
            c['benchmark_is_dataset'] += r['benchmark'] == ds
            c['hash_ok'] += r['prompt_hash'] == op.sha(r['prompt'])
            c['count_ok'] += r['prompt_token_count'] == len(r['prompt_tokens'])
            c['rowinfo_ok'] += (ri['id'] == r['id'] and ri['index'] == i and ri['prompt_token_count'] == r['prompt_token_count']
                                and ri['generation_budget'] == r['generation_budget']
                                and ri['prompt_sha256'] == op.token_ids_sha256(r['prompt_tokens']))
            plan.append(((ds, i), r['prompt'], bool(r['thinking']), r.get('response_prefill'), r['prompt_tokens'], opener))
        for x in cells.get(ds, []):
            c['cells'] += 1
            c['cells_ok'] += rows[x['index']]['id'] == x['id'] and rows[x['index']]['prompt_token_count'] == x['prompt_tokens']
        c['manifests'] += 1
    c['cells_without_manifest'] = sum(len(v) for ds, v in cells.items() if ds not in rows_by_ds)
    t0 = time.time()
    with multiprocessing.Pool(jobs, initializer=_init) as pool:
        for key, same, tail_ok in pool.imap_unordered(_render, plan, chunksize=1):
            c['rerender_identical'] += same
            if tail_ok is not None:
                c['prefill_tail_checked'] += 1
                c['prefill_tail_ok'] += tail_ok
        if pool_dir.name.startswith('longbench'):
            import v15_longbench_task as v15
            bp.lb_identity()                                     # the dyh copies, verified by sha256
            data = json.loads(v15.LB_DATA.read_bytes())
            templates = {t: (bp.LB_SRC / 'prompts' / f'{t}.txt').read_text(encoding='utf-8') for t in ('0shot', '0shot_cot')}
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
    if pool_dir.name == 'ruler_v33ofc':
        for _, src_ds, ds in bp.RULER_LENGTHS:
            old = json.loads((bp.RULER_SRC / f'{src_ds}_generation_manifest.json').read_text())
            rows = rows_by_ds[ds]
            c['ruler_ids_equal_v33'] += [r['id'] for r in rows] == [r['id'] for r in old]
            c['ruler_raw_prompt_equal_v33'] += sum(r['prompt'] + r['answer_prefix'] == o['prompt'] for r, o in zip(rows, old))
            c['ruler_gold_bytes_equal_v33'] += (pool_dir / f'{ds}_gold.json').read_bytes() == (bp.RULER_SRC / f'{src_ds}_gold.json').read_bytes()
            c['ruler_user_turn_without_prefix'] += sum(not r['prompt'].rstrip().endswith(r['answer_prefix'].strip()) for r in rows)
    if pool_dir.name == 'ruler_v33noop':
        ofc = json.loads((bp.OFC / 'ruler_v33ofc' / 'ruler32k_v33ofc_generation_manifest.json').read_text())
        gold_ofc = json.loads((bp.OFC / 'ruler_v33ofc' / 'ruler32k_v33ofc_gold.json').read_text())
        gold = json.loads((pool_dir / f'{bp.NOOP_DATASET}_gold.json').read_text())
        for r in rows_by_ds[bp.NOOP_DATASET]:
            o = ofc[r['ofc_index']]
            c['noop_same_id_input_prefix_as_ofc'] += (o['id'] == r['id'] and o['prompt'] == r['prompt']
                                                      and o['answer_prefix'] == r['answer_prefix'] == r['response_prefill']
                                                      and gold[r['id']] == gold_ofc[r['id']])
            c['noop_tasks_ok'] += r['task'] in bp.NOOP_TASKS
    if pool_dir.name in ('aime26_b32k', 'graphwalks_b16k'):
        srcs = ({bp.AIME_DATASET: bp.AIME_SRC} if pool_dir.name == 'aime26_b32k'
                else {new: bp.GW_SRC / f'{old}_generation_manifest.json' for old, new in bp.GW_DATASETS})
        for ds, path in srcs.items():
            old = json.loads(path.read_text())
            new = rows_by_ds[ds]
            strip = lambda r: {k: v for k, v in r.items() if k not in ('benchmark', 'generation_budget')}
            c['source_rows_equal_except_name_budget'] += sum(strip(x) == strip(y) for x, y in zip(old, new))
            c['source_rows'] += len(old)
            c[f'{ds}_budget'] = sorted({r['generation_budget'] for r in new})[0]
    if pool_dir.name == 'mrcr_ofc':
        readme = json.loads((pool_dir / 'readme.json').read_text())
        for ds, rows in rows_by_ds.items():
            block = readme['bins'][ds]['dataset_block']
            for r in rows:
                file_index = 0 if r['source_id'].startswith('2needle_0') else 400
                c['mrcr_row_in_bin_block'] += (file_index + int(r['source_id'].rsplit('row', 1)[1])) // 100 == block
                c['mrcr_rows'] += 1
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
