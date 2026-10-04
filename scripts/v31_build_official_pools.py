"""Build the v31 final-suite pools under each dataset's OFFICIAL protocol (CPU tokenizer only, no weights, no GPU).

Rule: scripts/v31_official.py. Every pool has its OWN dataset names (never those of an earlier pool with other prompts
or budgets) and writes per dataset <dataset>_generation_manifest.json (rows: id, benchmark, prompt, prompt_hash =
sha256(prompt), prompt_tokens, prompt_token_count, generation_budget, thinking, source_id + non-gold task fields;
json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=False) + '\\n'), <dataset>_rowinfo.json (schema v31_rowinfo_v2:
the manifest's sha256, the pool's pinned max_model_len and per row id, generation_budget, prompt_sha256 = sha256 of the
prompt token ids, prompt_token_count and non-text task fields -- what the scorers bind completions to), scorer-only
<dataset>_gold.json, cells_<pool>.json ({dataset, id, index = manifest position, seed, prompt_tokens}) and readme.json.
Pinned max_model_len of a pool = min(262144, ((max prompt + budget + 4096) // 1024 + 1) * 1024) over its cells file (the
bench's own derivation), to be passed as MAX_MODEL_LEN to every arm.
  ruler       ruler32k_v33ofc / ruler64k_v33ofc / ruler128k_v33ofc: the v33 samples (same ids, order and gold) with
              the official answer-prefix placement: user turn = the RULER sample `input` without its answer prefix
              (chat template, thinking off), then the response prefill = DiffusionGemma's empty thought block + the
              official answer prefix, verbatim (RULER call_api.py sends input + answer_prefix with the model template
              inside input). Budget = RULER tokens_to_generate per task. Also writes cells_ruler32k_v33ofc_vt_niah.json
              (the vt + niah_* ids at 32K, the partner cells of the ruler_noop ablation).
  ruler_noop  ruler32k_v33noop (dense-only ablation): the vt + niah_* rows of ruler32k_v33ofc rendered WITHOUT the empty
              thought block: encode_prompt(input) ending in '<|turn>model\\n', then the answer prefix tokens, exactly
              as RULER sends input + answer_prefix. Pinned max_model_len = the ruler pool's.
  longbench   All 503 LongBench-v2 items (pinned dataset revision), THUDM/LongBench @ 2e00731f pred.py (the six files
              copied into the dyh tree and verified by sha256, never read through git): template filled with the
              stripped fields (after the project's reserved-token escaping), tokenized with OUR tokenizer
              (tokenizer.encode, as pred.py); above 120,000 ids the first and last 60,000 are kept and decoded with
              skip_special_tokens=True; then adapter.encode_prompt. One dataset per variant:
                longbench_v2_0shot        prompts/0shot.txt,     thinking off, 128 new tokens   headline w/o CoT
                longbench_v2_0shot_think  prompts/0shot.txt,     thinking on, 16384 (total cap)  headline w/ CoT
                longbench_v2_cot_think    prompts/0shot_cot.txt, thinking on, 16384             not a headline column
  aime        aime26_b32k: the v27 final AIME26 manifest with generation_budget 32768, prompts unchanged; seeds 1-4.
  graphwalks  graphwalks_22k_b16k / _45k_b16k / _90k_b16k: the pools_v31 GraphWalks rows with generation_budget 16384,
              prompts / gold unchanged; cells seed 1, plus cells_graphwalks_90k_b16k_pilot.json (90k bin, 24 cells).
  mrcr        mrcr2_32k_ofc / mrcr2_64k_ofc / mrcr2_128k_ofc: the README's bins (o200k_base tokens of prompt + answer):
              (16384, 32768], (32768, 65536], (65536, 131072]. The dataset holds 8 blocks of 100 rows in dataset order,
              one per README bin; a row's bin is its block, cross-checked with o200k counts from the vendored
              o200k_base vocabulary (the official BPE file is not on the host). 24 rows per bin by sha256(id); message
              list through the chat template (thinking off); budget 2048 (4096 if > 5% of the answers exceed 1500 of our
              tokens). Run with tiktoken importable (PYTHONPATH holding the dyh-local copy).
usage: python v31_build_official_pools.py {ruler,ruler_noop,longbench,aime,graphwalks,mrcr} --out DIR [--src PATH] [--jobs N]
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
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v31_official as op  # noqa: E402

MODEL = '/home/exouser/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b'
REVISION = 'f7f5b7f5fa82ffc52addd066915886d497f5517b'
OFC = Path('/media/volume/dllm-1/dyh/pools_v31_official')
RULER_SRC = Path('/media/volume/dllm-1/dyh/ruler_long_v33')
RULER_LENGTHS = ((32768, 'ruler32k_v33', 'ruler32k_v33ofc'), (65536, 'ruler64k_v33', 'ruler64k_v33ofc'),
                 (131072, 'ruler128k_v33', 'ruler128k_v33ofc'))
NOOP_DATASET = 'ruler32k_v33noop'
NOOP_TASKS = ('vt', 'niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1', 'niah_multikey_2',
              'niah_multikey_3', 'niah_multivalue', 'niah_multiquery')
LB_SRC = OFC / 'official_src/LongBench_2e00731f'
LB_COMMIT = '2e00731f8d0bff23dc4325161044d0ed8af94c1e'
LB_FILE_SHA256 = {'pred.py': 'ab63f77866a1c0dc770582bc3fe6b014c3b4be4667399b0ee267075780c6a138',
                  'result.py': 'b6675c1c0b72090c01e12c94d3b176c4991fe38ceed9a301d24d22b1997465bd',
                  'config/model2maxlen.json': 'c37ec6e5d77e68e4cf6280183c6e68e885852ced6904e049ed75d3e73fc61ebe',
                  'prompts/0shot.txt': '68a162252bc9ff71d5d7abca3d69bb31aac3c35f832d657a2866f2018b8a6950',
                  'prompts/0shot_cot.txt': '6cc33d92fd8f611662b756c1f10f302c541ca005b58c46d5f3188ac3a939bfe8',
                  'prompts/0shot_cot_ans.txt': '85ec59df4ac0e27aabbc619a7ca5cf3ae1787dd5a76cf51a18944eb52c91955f'}
LB_VARIANTS = (('longbench_v2_0shot', '0shot', False, 128,
                'headline w/o CoT: pred.py without --cot (zero-shot prompt, 128 new tokens), thinking off'),
               ('longbench_v2_0shot_think', '0shot', True, 16384,
                'headline w/ CoT: the leaderboard rule for hybrid reasoning models (Qwen3: w/o CoT = non-thinking mode, '
                'w/ CoT = thinking mode with a 16K thinking budget); zero-shot prompt, thinking on; our 16,384 is a total '
                'cap on thinking + answer, not a thinking budget'),
               ('longbench_v2_cot_think', '0shot_cot', True, 16384,
                'NOT a headline column: prompts/0shot_cot.txt single stage with thinking (pred.py --cot always adds the '
                '0shot_cot_ans stage that asks for the answer format; extract_answer undercounts this variant)'))
AIME_SRC = Path('/media/volume/dllm-1/dyh/m3_output_numerics_v21_20260927/v23/v27_final_aime_f1_002/manifests/'
                'aime26_generation_manifest.json')
AIME_DATASET, AIME_BUDGET, AIME_SEEDS = 'aime26_b32k', 32768, (1, 2, 3, 4)
GW_SRC = Path('/media/volume/dllm-1/dyh/pools_v31/graphwalks')
GW_DATASETS = (('graphwalks_22k', 'graphwalks_22k_b16k'), ('graphwalks_45k', 'graphwalks_45k_b16k'),
               ('graphwalks_90k', 'graphwalks_90k_b16k'))
GW_BUDGET, GW_PILOT = 16384, 'graphwalks_90k_b16k'
MRCR_README_BINS = ((4096, 8192), (8192, 16384), (16384, 32768), (32768, 65536), (65536, 131072), (131072, 262144),
                    (262144, 524288), (524288, 1048576))     # [4096, 8192] closed, the others (lo, hi]
MRCR_BINS = (('mrcr2_32k_ofc', '(16384, 32768]', 16384, 32768), ('mrcr2_64k_ofc', '(32768, 65536]', 32768, 65536),
             ('mrcr2_128k_ofc', '(65536, 131072]', 65536, 131072))
MRCR_PER_BIN, MRCR_BUDGETS, MRCR_LONG_ANSWER = 24, (2048, 4096), (1500, 0.05)
O200K_VENDORED = OFC / 'official_src/tiktoken_cache/o200k_base.vendored_copilot.bin'
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


def pin_max_model_len(rows):
    """The bench's max_model_len derivation over every row of a pool (its full cells file)."""
    need = max(r['prompt_token_count'] + r['generation_budget'] for r in rows)
    return min(262144, ((need + 4096) // 1024 + 1) * 1024), need


def write_pool(out: Path, dataset, rows, gold, info_keys, outputs, pin, gold_bytes=None):
    """Manifest + rowinfo (v2) + gold (unless None) of one dataset; returns the cell list (seed filled in by the caller)."""
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids) or (gold is not None and set(gold) != set(ids)):
        raise AssertionError(f'{dataset}: ids must be unique and match the gold keys')
    for r in rows:
        if r['prompt_token_count'] != len(r['prompt_tokens']) or not all(type(t) is int for t in r['prompt_tokens']):
            raise AssertionError(f'{dataset}: prompt_tokens must be ints matching prompt_token_count')
        if r['prompt_hash'] != op.sha(r['prompt']) or r['benchmark'] != dataset:
            raise AssertionError(f'{dataset}: prompt_hash must be sha256(prompt) and benchmark the dataset name')
        if r['prompt_token_count'] + r['generation_budget'] > pin:
            raise AssertionError(f'{dataset}: a row exceeds the pinned max_model_len')
    man = out / f'{dataset}_generation_manifest.json'
    man.write_bytes((json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8'))
    outputs[man.name] = op.sha_file(man)
    info = dict(schema=op.ROWINFO_SCHEMA, dataset=dataset, manifest_sha256=outputs[man.name], max_model_len=pin,
                rows=[dict({k: r[k] for k in info_keys if k in r}, id=r['id'], index=i, generation_budget=r['generation_budget'],
                           prompt_sha256=op.token_ids_sha256(r['prompt_tokens']), prompt_token_count=r['prompt_token_count'],
                           thinking=r['thinking']) for i, r in enumerate(rows)])
    path = out / f'{dataset}_rowinfo.json'
    path.write_text(json.dumps(info, indent=1, sort_keys=True) + '\n')
    outputs[path.name] = op.sha_file(path)
    if gold is not None:
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
        tokenizer=dict(model_revision=REVISION, processor_class=type(adapter.processor).__name__,
                       tokenizer_class=type(adapter.tokenizer).__name__, transformers=transformers.__version__,
                       tokenizers=tokenizers.__version__,
                       files_sha256={n: op.sha_file(model / n) for n in ('tokenizer.json', 'tokenizer_config.json',
                                                                          'chat_template.jinja', 'processor_config.json')
                                     if (model / n).exists()}),
        code=dict(deploy_sha=(code_root / 'DEPLOY_SHA').read_text().strip() if (code_root / 'DEPLOY_SHA').exists() else None,
                  adapter_sha256=op.sha_file(code_root / 'src/dllm/models/adapters/diffusion_gemma.py'),
                  files_sha256={p.name: op.sha_file(p) for p in files}),
        environment=dict(python_version=platform.python_version(), host=platform.node(),
                         vars={k: os.environ.get(k) for k in ENV_VARS}, gpu_used=False, model_weights_loaded=False))


def write_readme(out: Path, readme):
    (out / 'readme.json').write_text(json.dumps(readme, indent=2, sort_keys=True) + '\n')


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

# ---------------------------------------------------------------- RULER (official placement and the no-opener ablation)


_W = {}


def _ruler_init():
    _W['adapter'] = load_adapter()


def _ruler_render(job):
    """(dataset, index, input, answer_prefix, opener) -> ids + structural checks (no text returned)."""
    dataset, index, sample_input, answer_prefix, opener = job
    adapter = _W['adapter']
    tok = adapter.tokenizer
    ids, n_user = op.ruler_prompt_ids(adapter, sample_input, answer_prefix, opener)
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    gen_prompt, opener_ids = enc('<|turn>model\n'), enc(opener) if opener else []
    n_channel = 1 if opener else 0
    checks = dict(
        generation_prompt_ends_user_part=ids[n_user - len(gen_prompt):n_user] == gen_prompt,
        opener_follows=ids[n_user:n_user + len(opener_ids)] == opener_ids,
        prefix_tokens_equal_separate_encoding=ids[n_user + len(opener_ids):] == enc(answer_prefix),
        prefix_round_trip=dec(ids[n_user + len(opener_ids):]) == answer_prefix,
        tail_decodes_to_model_turn_opener_prefix=dec(ids[n_user - len(gen_prompt):]) == '<|turn>model\n' + opener + answer_prefix,
        whole_string_tokenization_identical=enc(dec(ids)) == ids,
        one_bos_first=ids[0] == tok.bos_token_id and ids.count(tok.bos_token_id) == 1,
        turn_tokens=ids.count(tok.convert_tokens_to_ids('<|turn>')) == 2 and ids.count(tok.convert_tokens_to_ids('<turn|>')) == 1,
        thought_blocks=(ids.count(tok.convert_tokens_to_ids('<|channel>')) == n_channel
                        and ids.count(tok.convert_tokens_to_ids('<channel|>')) == n_channel),
        no_think_token=tok.convert_tokens_to_ids('<|think|>') not in ids)
    return dataset, index, ids, n_user, len(ids) - n_user, checks


def ruler_source(src: Path, L, src_bench, logf):
    """[(old manifest row, sample record, official shard row)] in the v33 manifest order + the gold bytes."""
    old = json.loads((src / f'{src_bench}_generation_manifest.json').read_text())
    gold_bytes = (src / f'{src_bench}_gold.json').read_bytes()
    gold = json.loads(gold_bytes)
    raw_dir = src / 'raw' / str(L)
    raw_manifest = json.loads((raw_dir / 'manifest.json').read_text())
    samples_path = raw_dir / 'samples.jsonl'
    if op.sha_file(samples_path) != raw_manifest['samples']['sha256']:
        raise AssertionError(f'{src_bench}: samples.jsonl differs from its raw manifest hash')
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
            raise AssertionError(f'{src_bench}: sample does not match its official shard row')
        if shard['index'] != rec['official_index'] or not shard['answer_prefix']:
            raise AssertionError(f'{src_bench}: shard index / answer prefix mismatch')
        by_source[f"ruler_{L}_{rec['task']}_p{p:04d}"] = (rec, shard)
    if len(old) != len(samples) or {r['source_id'] for r in old} != set(by_source):
        raise AssertionError(f'{src_bench}: manifest rows and samples differ')
    out = []
    for i, r in enumerate(old):
        rec, shard = by_source[r['source_id']]
        if (r['id'] != f"{src_bench}/{r['source_id']}" or r['prompt'] != rec['prompt'] or r['prompt_hash'] != op.sha(rec['prompt'])
                or r['task'] != rec['task'] or r['generation_budget'] != rec['tokens_to_generate'] or r['thinking'] is not False
                or gold[r['id']] != rec['outputs']):
            raise AssertionError(f'{src_bench}: v33 manifest row {i} does not match its sample / gold')
        out.append((r, rec, shard))
    source_sha = {f'{src_bench}_generation_manifest.json': op.sha_file(src / f'{src_bench}_generation_manifest.json'),
                  f'{src_bench}_gold.json': op.sha(gold_bytes), f'raw/{L}/samples.jsonl': op.sha_file(samples_path),
                  f'raw/{L}/manifest.json': op.sha_file(raw_dir / 'manifest.json')}
    log(logf, f'{src_bench}: {len(out)} rows mapped to samples, official shards and gold')
    return out, gold, gold_bytes, source_sha


def render_ruler(plan, jobs, rows_by, logf):
    t0, checks = time.time(), collections.Counter()
    with multiprocessing.Pool(jobs, initializer=_ruler_init) as pool:
        for n, (ds, i, ids, n_user, n_tail, ck) in enumerate(pool.imap_unordered(_ruler_render, plan, chunksize=1)):
            rows_by[ds][i].update(prompt_tokens=ids, prompt_token_count=len(ids), prefill_token_count=n_tail)
            for k, v in ck.items():
                checks[k] += bool(v)
            if not all(ck.values()):
                raise AssertionError(f'{ds} row {i}: failed checks {[k for k, v in ck.items() if not v]}')
            if (n + 1) % 50 == 0:
                log(logf, f'rendered {n + 1}/{len(plan)} in {time.time() - t0:.0f}s')
    log(logf, f'rendered {len(plan)} rows in {time.time() - t0:.0f}s; all structural checks passed')
    return dict(checks, rows=len(plan))


RULER_INFO = ('task', 'task_base', 'prefilled', 'prefill_token_count', 'v33_dataset', 'ofc_index')


def ruler_bins(rows_by, meta):
    out = {}
    for ds, rows in rows_by.items():
        by_task = collections.defaultdict(list)
        for r in rows:
            by_task[r['task']].append(r)
        out[ds] = dict(rows=len(rows), tasks=dict(collections.Counter(r['task'] for r in rows)),
                       prompt_token_count=stats([r['prompt_token_count'] for r in rows]),
                       delta_vs_v33_prompt_token_count=stats([r['prompt_token_count'] - r['v33_prompt_token_count'] for r in rows]),
                       prefill_token_count_by_task={t: stats([r['prefill_token_count'] for r in v]) for t, v in sorted(by_task.items())},
                       generation_budget_by_task={t: sorted({r['generation_budget'] for r in v}) for t, v in sorted(by_task.items())},
                       total_tokens_max=max(r['prompt_token_count'] + r['generation_budget'] for r in rows), **meta.get(ds, {}))
    return out


def build_ruler(a, logf, noop=False):
    src = Path(a.src or RULER_SRC)
    if 'wrote readme.json' not in (src / 'gen.log').read_text():
        raise RuntimeError('the v33 generator has not finished (no "wrote readme.json" in gen.log)')
    src_readme = json.loads((src / 'readme.json').read_text())
    ruler_root = Path(src_readme['ruler_root'])
    ruler_identity = op.ruler_verify_checkout(ruler_root)
    log(logf, f'RULER checkout {ruler_identity["commit"][:8]} verified; source v33 pool')
    opener = '' if noop else op.RESPONSE_OPENER['diffusion_gemma']
    lengths = RULER_LENGTHS[:1] if noop else RULER_LENGTHS
    rows_by, golds, meta, plan = {}, {}, {}, []
    for L, src_bench, ds in lengths:
        mapped, gold, gold_bytes, source_sha = ruler_source(src, L, src_bench, logf)
        if noop:
            ds = NOOP_DATASET
            ofc_index = {r['id']: i for i, (r, _, _) in enumerate(mapped)}
            mapped = [m for m in mapped if m[0]['task'] in NOOP_TASKS]
            gold = {m[0]['id']: gold[m[0]['id']] for m in mapped}
            gold_bytes = None
        rows = []
        for i, (r, rec, shard) in enumerate(mapped):
            rows.append(dict(answer_prefix=shard['answer_prefix'], benchmark=ds, generation_budget=int(rec['tokens_to_generate']),
                             id=r['id'], prompt=shard['input'], prompt_hash=op.sha(shard['input']), prefilled=True,
                             response_prefill=opener + shard['answer_prefix'], source_id=r['source_id'], task=rec['task'],
                             task_base=rec['task_base'], thinking=False, v33_dataset=src_bench, v33_prompt_hash=r['prompt_hash'],
                             v33_prompt_token_count=r['prompt_token_count'],
                             **(dict(ofc_index=ofc_index[r['id']]) if noop else {})))
            plan.append((ds, i, shard['input'], shard['answer_prefix'], opener))
        rows_by[ds], golds[ds] = rows, (gold, gold_bytes)
        meta[ds] = dict(context_length=L, source_sha256=source_sha)
    checks = render_ruler(plan, a.jobs, rows_by, logf)
    all_rows = [r for rows in rows_by.values() for r in rows]
    if noop:
        ofc = json.loads((OFC / 'ruler_v33ofc' / 'ruler32k_v33ofc_rowinfo.json').read_text())
        pin, need = ofc['max_model_len'], max(r['prompt_token_count'] + r['generation_budget'] for r in all_rows)
        pin_rule = 'the ruler_v33ofc pool pin, so the ablation runs with the main pool setting'
    else:
        pin, need = pin_max_model_len(all_rows)
        pin_rule = 'pin_max_model_len over all three lengths'
    outputs, cells = {}, []
    for ds, rows in rows_by.items():
        gold, gold_bytes = golds[ds]
        cells += [dict(c, seed=1) for c in write_pool(a.out, ds, rows, gold, RULER_INFO, outputs, pin, gold_bytes=gold_bytes)]
    write_cells(a.out, 'ruler_v33noop' if noop else 'ruler_v33ofc', cells, outputs)
    if not noop:
        sub = [c for c in cells if c['dataset'] == 'ruler32k_v33ofc' and rows_by['ruler32k_v33ofc'][c['index']]['task'] in NOOP_TASKS]
        write_cells(a.out, 'ruler32k_v33ofc_vt_niah', sub, outputs)
    tok = load_adapter().tokenizer
    readme = dict(
        schema='pools_v31_official_readme_v2', pool='ruler_v33noop' if noop else 'ruler_v33ofc', built_utc=now(),
        rule=__doc__.split('  ruler       ')[1].split('  longbench')[0].strip(),
        ruler=dict(**ruler_identity, generator_seed=src_readme['seed'], length_mode=src_readme['length_mode']),
        placement=dict(user_turn='adapter.encode_prompt(sample input, {"thinking": False}) -> <bos><|turn>user\\n{input}<turn|>\\n<|turn>model\\n',
                       response_prefill='tokenizer.encode(opener + answer_prefix, add_special_tokens=False)',
                       opener=opener, opener_ids=tok.encode(opener, add_special_tokens=False) if opener else [],
                       generation_prompt_ids=tok.encode('<|turn>model\n', add_special_tokens=False),
                       answer_prefix='verbatim RULER answer_prefix (leading space kept, as RULER sends it)',
                       opener_evidence='3972 / 3975 non-thinking RULER completions of the v31 panels open with the empty thought '
                                       'block; the other 3 open it and stop without closing it'),
        max_model_len=dict(pinned=pin, need=need, rule=pin_rule),
        gold='the v33 gold (byte-identical copies for the ofc pool; the subset for the ablation pool)',
        checks=checks, counts={ds: len(r) for ds, r in rows_by.items()}, bins=ruler_bins(rows_by, meta),
        output_files_sha256=outputs, **identity(load_adapter()))
    write_readme(a.out, readme)
    log(logf, f'done; pin {pin}; outputs {outputs}')

# ---------------------------------------------------------------- LongBench-v2


def lb_identity():
    """The six pinned official files, copied into the dyh tree (plain copy) and verified by sha256 (no git)."""
    out = {}
    for rel, digest in LB_FILE_SHA256.items():
        got = op.sha_file(LB_SRC / rel)
        if got != digest:
            raise ValueError(f'LongBench file {rel} differs from the pinned sha256')
        out[rel] = got
    maxlen = json.loads((LB_SRC / 'config/model2maxlen.json').read_text())
    return dict(commit=LB_COMMIT, files_sha256=out, max_len_open_models=sorted({v for k, v in maxlen.items() if 'claude' not in k}),
                source='copies of the THUDM/LongBench @ 2e00731f files, verified by sha256 against the pinned values')


def _lb_init(templates):
    _W['adapter'] = load_adapter()
    _W['templates'] = templates
    import v15_longbench_task as v15
    _W['escape'] = v15.escape


def lb_prompt(tok, template, item, escape):
    """pred.py prompt + query_llm truncation with OUR tokenizer: (text, original token count, truncated, escapes)."""
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


LB_INFO = ('difficulty', 'length', 'domain', 'template', 'truncated', 'original_token_count')


def build_longbench(a, logf):
    import v15_longbench_task as v15
    official = lb_identity()
    if official['max_len_open_models'] != [op.LB_MAX_LEN]:
        raise AssertionError('config/model2maxlen.json does not give 120000 for the open models')
    templates = {t: (LB_SRC / 'prompts' / f'{t}.txt').read_text(encoding='utf-8') for t in ('0shot', '0shot_cot')}
    data_bytes = v15.LB_DATA.read_bytes()
    data = json.loads(data_bytes)
    if len(data) != 503 or len({r['_id'] for r in data}) != 503:
        raise AssertionError('expected 503 unique LongBench-v2 items')
    for r in data:
        if r['answer'] not in ('A', 'B', 'C', 'D') or r['difficulty'] not in ('easy', 'hard') or r['length'] not in ('short', 'medium', 'long'):
            raise AssertionError('unexpected LongBench-v2 answer / difficulty / length value')
    log(logf, f'LongBench files verified (sha256); dataset {v15.LB_REVISION[:8]} sha256 {op.sha(data_bytes)[:12]}; {len(data)} items')
    tok = load_adapter().tokenizer
    if tok.encode('probe text') != tok.encode('probe text', add_special_tokens=False):
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
        pin, need = pin_max_model_len(rows)
        cells = write_pool(a.out, dataset, rows, gold, LB_INFO, outputs, pin)
        write_cells(a.out, dataset, [dict(c, seed=1) for c in cells], outputs)
        tr = [r for r in rows if r['truncated']]
        variants[dataset] = dict(
            template=f'prompts/{tname}.txt', thinking=thinking, generation_budget=budget, column=column, rows=len(rows),
            truncated=len(tr), untruncated=len(rows) - len(tr), max_model_len=dict(pinned=pin, need=need),
            original_token_count=stats([r['original_token_count'] for r in rows]),
            original_token_count_truncated=stats([r['original_token_count'] for r in tr]),
            prompt_token_count=stats([r['prompt_token_count'] for r in rows]),
            prompt_token_count_truncated=stats([r['prompt_token_count'] for r in tr]),
            truncated_by_length_class=dict(collections.Counter(r['length'] for r in tr)),
            difficulty=dict(collections.Counter(r['difficulty'] for r in rows)),
            length_class=dict(collections.Counter(r['length'] for r in rows)),
            total_tokens_max=max(r['prompt_token_count'] + budget for r in rows))
        log(logf, f'{dataset}: rows={len(rows)} truncated={len(tr)} pin={pin} tokens={variants[dataset]["prompt_token_count"]}')
        del rows
    readme = dict(
        schema='pools_v31_official_readme_v2', pool='longbench_v2', built_utc=now(),
        rule=__doc__.split('  longbench   ')[1].split('  aime')[0].strip(), official=official,
        dataset=dict(revision=v15.LB_REVISION, sha256=op.sha(data_bytes), items=len(data)),
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
        **identity(load_adapter(), extra_code=(Path(v15.__file__).resolve(),)))
    write_readme(a.out, readme)
    log(logf, f'done; outputs {outputs}')

# ---------------------------------------------------------------- AIME26 / GraphWalks budget variants (own names)


def rename_rows(rows, dataset, budget):
    """Rows with the new dataset name and budget; everything else must stay byte-for-byte the source's."""
    new = [dict(r, benchmark=dataset, generation_budget=budget) for r in rows]
    for r, n in zip(rows, new):
        if {k: v for k, v in r.items() if k not in ('benchmark', 'generation_budget')} != \
                {k: v for k, v in n.items() if k not in ('benchmark', 'generation_budget')}:
            raise AssertionError('only benchmark and generation_budget may change')
        if r['prompt_hash'] != op.sha(r['prompt']) or r['prompt_token_count'] != len(r['prompt_tokens']) or r['thinking'] is not True:
            raise AssertionError('source row inconsistent')
    return new


def build_aime(a, logf):
    src = Path(a.src or AIME_SRC)
    rows = json.loads(src.read_text())
    ids = [r['id'] for r in rows]
    if ids != [f'aime26/{i}' for i in range(1, 31)] or len(set(ids)) != 30:
        raise AssertionError('unexpected AIME26 ids')
    new = rename_rows(rows, AIME_DATASET, AIME_BUDGET)
    pin, need = pin_max_model_len(new)
    outputs = {}
    # no gold file: the scorer reads the answers from the pinned math-ai/aime26 file via each row's source_id
    cells = write_pool(a.out, AIME_DATASET, new, None, ('source_id',), outputs, pin)
    cells = [dict(c, seed=s) for s in AIME_SEEDS for c in cells]
    write_cells(a.out, AIME_DATASET, cells, outputs)
    readme = dict(schema='pools_v31_official_readme_v2', pool=AIME_DATASET, built_utc=now(),
                  rule='v27 final AIME26 manifest, generation_budget 8192 -> 32768 under its own dataset name; ids, prompts, '
                       'prompt_tokens, thinking unchanged (asserted per row); cells seeds 1-4 (avg@4), seed-major. No gold '
                       'file: answers come from the pinned math-ai/aime26 file via each row\'s source_id.',
                  source_sha256=op.sha_file(src), rows=len(new), cells=len(cells), seeds=list(AIME_SEEDS),
                  max_model_len=dict(pinned=pin, need=need),
                  prompt_token_count=stats([r['prompt_token_count'] for r in new]), output_files_sha256=outputs,
                  code=dict(builder_sha256=op.sha_file(Path(__file__).resolve())))
    write_readme(a.out, readme)
    log(logf, f'{AIME_DATASET}: {len(new)} rows, {len(cells)} cells, pin {pin}; outputs {outputs}')


def build_graphwalks(a, logf):
    src = Path(a.src or GW_SRC)
    outputs, bins, all_rows, staged = {}, {}, [], []
    for old_ds, ds in GW_DATASETS:
        rows = json.loads((src / f'{old_ds}_generation_manifest.json').read_text())
        gold_bytes = (src / f'{old_ds}_gold.json').read_bytes()
        new = rename_rows(rows, ds, GW_BUDGET)
        all_rows += new
        staged.append((old_ds, ds, rows, new, gold_bytes))
    pin, need = pin_max_model_len(all_rows)
    cells = []
    for old_ds, ds, rows, new, gold_bytes in staged:
        cells += [dict(c, seed=1) for c in write_pool(a.out, ds, new, json.loads(gold_bytes), ('problem_type', 'bin'), outputs,
                                                     pin, gold_bytes=gold_bytes)]
        bins[ds] = dict(source_dataset=old_ds, rows=len(new), problem_type=dict(collections.Counter(r['problem_type'] for r in new)),
                        prompt_token_count=stats([r['prompt_token_count'] for r in new]),
                        source_manifest_sha256=op.sha_file(src / f'{old_ds}_generation_manifest.json'),
                        source_gold_sha256=op.sha(gold_bytes))
    write_cells(a.out, 'graphwalks_b16k', cells, outputs)
    write_cells(a.out, f'{GW_PILOT}_pilot', [c for c in cells if c['dataset'] == GW_PILOT], outputs)
    readme = dict(schema='pools_v31_official_readme_v2', pool='graphwalks_b16k', built_utc=now(),
                  rule=f'pools_v31 GraphWalks rows under their own dataset names with generation_budget 8192 -> {GW_BUDGET} (no '
                       'official budget: the reasoning-model rule; 4 of 24 graphwalks_90k gold answer lines alone need > 2048 '
                       'tokens); ids, prompts, prompt_tokens, thinking and gold unchanged (asserted); cells seed 1; '
                       f'cells_{GW_PILOT}_pilot.json = the 90k bin for the dense cap-rate pilot',
                  bins=bins, cells=len(cells), max_model_len=dict(pinned=pin, need=need), output_files_sha256=outputs,
                  code=dict(builder_sha256=op.sha_file(Path(__file__).resolve())))
    write_readme(a.out, readme)
    log(logf, f'graphwalks: {len(all_rows)} rows, pin {pin}; outputs {outputs}')

# ---------------------------------------------------------------- MRCR 2-needle on the README's bins


def o200k_vendored(path):
    """tiktoken's o200k_base pattern and special tokens with the vendored rank list (length-prefixed token bytes in rank
    order); returns (encoding, identity)."""
    import tiktoken
    import tiktoken_ext.openai_public as pub
    data = Path(path).read_bytes()
    ranks, i = {}, 0
    while i < len(data):
        n = data[i]
        ranks[data[i + 1:i + 1 + n]] = len(ranks)
        i += 1 + n
    original = pub.load_tiktoken_bpe
    pub.load_tiktoken_bpe = lambda *args, **kwargs: ranks          # no download: the vendored ranks
    try:
        spec = pub.o200k_base()
    finally:
        pub.load_tiktoken_bpe = original
    return tiktoken.Encoding(**spec), dict(tiktoken=tiktoken.__version__, vendored_file_sha256=op.sha_file(path),
                                           vendored_ranks=len(ranks), official_ranks=199998,
                                           note='the official o200k_base BPE file is not on the host; counts are approximate')


def readme_bin(n):
    """Index into MRCR_README_BINS ([4096, 8192] closed, the others (lo, hi]) or None."""
    for i, (lo, hi) in enumerate(MRCR_README_BINS):
        if (lo <= n <= hi) if i == 0 else (lo < n <= hi):
            return i
    return None


def build_mrcr(a, logf):
    import v31_build_pools as vb
    adapter = load_adapter()
    specials = vb.Specials(adapter.tokenizer)
    source = vb.source_identity('mrcr', vb.MRCR_FILES)
    rows = vb.load_mrcr(specials, logf)
    if specials.found:
        raise AssertionError(f'literal reserved special-token spellings in MRCR text: {dict(specials.found)}')
    if [r['dataset_index'] for r in rows] != list(range(800)):
        raise AssertionError('expected 800 MRCR rows in dataset order')
    enc, o200k = o200k_vendored(O200K_VENDORED)
    t0 = time.time()
    for r in rows:
        messages = json.loads(r['prompt'])
        r['o200k_prompt'] = sum(len(enc.encode(m['content'])) for m in messages)       # the README's n_tokens(messages)
        r['o200k_total'] = r['o200k_prompt'] + len(enc.encode(r['answer']))
        r['block'] = r['dataset_index'] // 100
    log(logf, f'o200k counts (vendored ranks) for {len(rows)} rows in {time.time() - t0:.0f}s')
    blocks = {}
    for b in range(8):
        members = [r for r in rows if r['block'] == b]
        bins = collections.Counter(readme_bin(r['o200k_total']) for r in members)
        main_bin = statistics.median_low(sorted(r['o200k_total'] for r in members))
        blocks[b] = dict(readme_bin=readme_bin(main_bin), rows=len(members),
                         count_bins={('outside' if k is None else str(k)): v for k, v in bins.items()},
                         o200k_total=stats([r['o200k_total'] for r in members]),
                         rows_outside_block_bin=sum(readme_bin(r['o200k_total']) != readme_bin(main_bin) for r in members))
    if sorted(b['readme_bin'] for b in blocks.values()) != list(range(8)):
        raise AssertionError('the 8 dataset blocks do not map one-to-one onto the README bins')
    bin_block = {b['readme_bin']: k for k, b in blocks.items()}
    log(logf, f'blocks -> README bins {[(k, v["readme_bin"], v["rows_outside_block_bin"]) for k, v in blocks.items()]}')
    tok = adapter.tokenizer
    chosen_by, outputs, readme_bins = {}, {}, {}
    for ds, label, lo, hi in MRCR_BINS:
        b = bin_block[MRCR_README_BINS.index((lo, hi))]
        cands = [dict(id=f'{ds}/{r["source_id"]}', row=r) for r in rows if r['block'] == b]
        for c in cands:
            c['id_sha256'] = op.sha(c['id'].encode('utf-8'))
        cands.sort(key=lambda c: c['id_sha256'])
        chosen_by[ds] = cands[:MRCR_PER_BIN]
        readme_bins[ds] = dict(label=label, dataset_block=b, candidates=len(cands), chosen=len(chosen_by[ds]))
    answer_tokens = {c['id']: len(tok.encode(c['row']['answer'], add_special_tokens=False)) for v in chosen_by.values() for c in v}
    limit, share = MRCR_LONG_ANSWER
    frac = sum(n > limit for n in answer_tokens.values()) / len(answer_tokens)
    budget = MRCR_BUDGETS[1] if frac > share else MRCR_BUDGETS[0]
    staged, all_rows = [], []
    for ds, label, lo, hi in MRCR_BINS:
        out_rows, gold = [], {}
        for c in chosen_by[ds]:
            r = c['row']
            messages = json.loads(r['prompt'])
            ids = vb.render_messages(adapter, messages, False)
            specials.check_ids(ids, vb.turns_of(messages))
            out_rows.append(dict(benchmark=ds, bin=label, date_added=r['date_added'], generation_budget=budget, id=c['id'],
                                 n_chars=r['n_chars'], n_needles=r['n_needles'], o200k_total=r['o200k_total'],
                                 prompt=r['prompt'], prompt_hash=op.sha(r['prompt']), prompt_token_count=len(ids),
                                 prompt_tokens=ids, source_id=r['source_id'], thinking=False, total_messages=r['total_messages']))
            gold[c['id']] = dict(answer=r['answer'], random_string_to_prepend=r['random_string_to_prepend'])
        staged.append((ds, out_rows, gold))
        all_rows += out_rows
        readme_bins[ds].update(prompt_token_count=stats([x['prompt_token_count'] for x in out_rows]),
                               o200k_total=stats([x['o200k_total'] for x in out_rows]),
                               chosen_outside_bin=sum(not (lo < x['o200k_total'] <= hi) for x in out_rows),
                               answer_tokens=stats([answer_tokens[x['id']] for x in out_rows]))
        log(logf, f'{ds}: {len(out_rows)} rows rendered, tokens {readme_bins[ds]["prompt_token_count"]}')
    pin, need = pin_max_model_len(all_rows)
    cells = []
    for ds, out_rows, gold in staged:
        cells += [dict(c, seed=1) for c in write_pool(a.out, ds, out_rows, gold, ('bin', 'o200k_total', 'n_chars'), outputs, pin)]
    write_cells(a.out, 'mrcr_ofc', cells, outputs)
    readme = dict(schema='pools_v31_official_readme_v2', pool='mrcr_ofc', built_utc=now(),
                  rule=__doc__.split('  mrcr        ')[1].split('usage:')[0].strip(), source=dict(
                      repo=source['repo'], revision=source['revision'], files={k: v['sha256'] for k, v in source['files'].items()}),
                  o200k=o200k, blocks=blocks, bins=readme_bins, budget=dict(value=budget, fraction_answers_over_1500=round(frac, 4)),
                  max_model_len=dict(pinned=pin, need=need), cells=len(cells), output_files_sha256=outputs,
                  **identity(adapter, extra_code=(Path(vb.__file__).resolve(),)))
    write_readme(a.out, readme)
    log(logf, f'mrcr: {len(all_rows)} rows, budget {budget}, pin {pin}; outputs {outputs}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('pool', choices=('ruler', 'ruler_noop', 'longbench', 'aime', 'graphwalks', 'mrcr'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--src', default=None)
    p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if any(a.out.glob('*_generation_manifest.json')):
        raise FileExistsError(f'{a.out} already holds manifests; build into a fresh directory')
    with (a.out / 'build.log').open('a') as logf:
        log(logf, f'build {a.pool} -> {a.out} (jobs {a.jobs})')
        if a.pool in ('ruler', 'ruler_noop'):
            build_ruler(a, logf, noop=a.pool == 'ruler_noop')
        else:
            dict(longbench=build_longbench, aime=build_aime, graphwalks=build_graphwalks, mrcr=build_mrcr)[a.pool](a, logf)


if __name__ == '__main__':
    main()
