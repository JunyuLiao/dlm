"""Build the private v31 MRCR 2-needle and GraphWalks pools + scorer-only gold (CPU tokenizer only, no weights, no GPU).

Sources (revisions and file sha256 pinned in /media/volume/dllm-1/dyh/datasets_v31/download_manifest.json, re-verified):
  openai/mrcr       2needle/2needle_{0,1}.parquet (800 rows). Source key '<file stem>_row<NNN>'; the README's pd.concat
                    index (= the file's stored RangeIndex start + row) is recorded as dataset_index.
  openai/graphwalks graphwalks_128k_and_shorter.parquet (750 rows, problem_type bfs / parents). Source key 'row<NNN>'.
Rendering: no truncation and no edit of the source text. The source text holds no literal model-reserved special-token
spelling (asserted), so the project's LongBench escape rule would be a no-op and is not applied.
  mrcr       manifest `prompt` = the row's own `prompt` field verbatim (the JSON-serialized message list). The messages
             are rendered by the processor chat template through exactly the call DiffusionGemmaAdapter._prompt_tensor
             makes for encode_prompt (apply_chat_template(messages, add_generation_prompt=True,
             enable_thinking=<thinking>, tokenize=True, return_tensors='pt')), thinking OFF. This messages path is
             verified to reproduce adapter.encode_prompt ids exactly on single-user-turn inputs (both thinking values).
  graphwalks the row's `prompt` as one user turn: adapter.encode_prompt(prompt, {'thinking': True}).
Bins on the rendered request token count: 32k = [28000, 40000), 64k = [56000, 76000), 128k = [116000, 136000)
(--bins overrides). Selection: id = '<benchmark>/<source key>'; a bin's candidates are ordered by sha256(id) (UTF-8, hex
ascending) and the first 24 are taken (all if fewer). GraphWalks takes 12 bfs + 12 parents per bin, each type in its own
sha256 order; if one type has fewer than 12, the other type's continuing sha256 order fills the remaining slots.
Length prefilter (mrcr only): only rows whose `n_chars` (asserted = total message characters) lies in [84000, 884000)
are rendered, a chars/token guard of [3.0, 6.5] around the bins; the readme records the observed chars/token range of
every rendered row as the check that no row outside the window can reach a bin. GraphWalks renders every row (cheap;
its `prompt_chars` field predates the 2026-02-27 prompt fix and is shorter than the prompt by a constant).
Generation budget: graphwalks 8192 (thinking ON); mrcr 2048, or 4096 if more than 5% of the chosen rows' answers exceed
1500 tokens. A bin with no candidate writes no manifest / gold / cells rows.
Never prints prompt text, conversations, answers or gold.
"""
from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from collections.abc import Mapping
from pathlib import Path

MODEL = '/home/exouser/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b'
REVISION = 'f7f5b7f5fa82ffc52addd066915886d497f5517b'
DATA = Path('/media/volume/dllm-1/dyh/datasets_v31')
BINS = (('32k', '28-40K', 28000, 40000), ('64k', '56-76K', 56000, 76000), ('128k', '116-136K', 116000, 136000))
PER_BIN = 24
MRCR_FILES = ('2needle/2needle_0.parquet', '2needle/2needle_1.parquet')
MRCR_CHAR_WINDOW = (84000, 884000)
MRCR_GUARD = (3.0, 6.5)                      # chars/token bounds the window assumes: 84000 = 28000 * 3.0, 884000 = 136000 * 6.5
MRCR_BUDGETS = (2048, 4096)
MRCR_LONG_ANSWER = (1500, 0.05)              # > 5% of chosen answers above 1500 tokens -> the larger budget
GW_FILE = 'graphwalks_128k_and_shorter.parquet'
GW_TYPES = ('bfs', 'parents')
GW_BUDGET = 8192
POOLS = dict(mrcr=dict(repo='openai/mrcr', dir='openai__mrcr', prefix='mrcr2', thinking=False),
             graphwalks=dict(repo='openai/graphwalks', dir='openai__graphwalks', prefix='graphwalks', thinking=True))
ROW_TYPES = dict(
    mrcr=dict(benchmark=str, bin=str, date_added=str, generation_budget=int, id=str, n_chars=int, n_needles=int,
              prompt=str, prompt_hash=str, prompt_token_count=int, prompt_tokens=list, source_id=str, thinking=bool,
              total_messages=int),
    graphwalks=dict(benchmark=str, bin=str, date_added=str, generation_budget=int, id=str, problem_type=str, prompt=str,
                    prompt_hash=str, prompt_token_count=int, prompt_tokens=list, source_id=str, thinking=bool))
ENV_VARS = ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'HF_HUB_OFFLINE', 'OMP_NUM_THREADS', 'RAYON_NUM_THREADS',
            'TOKENIZERS_PARALLELISM', 'PYTHONDONTWRITEBYTECODE', 'PYTHONNOUSERSITE')


def sha(x) -> str:
    return hashlib.sha256(x if isinstance(x, bytes) else str(x).encode()).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def log(stream, msg):
    stream.write(time.strftime('%H:%M:%S ') + msg + '\n')
    stream.flush()


def stats(values, extra=False):
    v = sorted(values)
    if not v:
        return dict(n=0)
    out = dict(n=len(v), min=v[0], median=statistics.median(v), max=v[-1])
    if extra:
        out.update(mean=round(statistics.fmean(v), 1), p90=v[min(len(v) - 1, int(0.9 * len(v)))],
                   p95=v[min(len(v) - 1, int(0.95 * len(v)))])
    return out


def parse_bins(text):
    if not text:
        return BINS
    out = []
    for part in text.split(','):
        key, label, lo, hi = part.split(':')
        out.append((key, label, int(lo), int(hi)))
    return tuple(out)


def render_messages(adapter, messages, thinking: bool) -> list[int]:
    """DiffusionGemmaAdapter._prompt_tensor / encode_prompt with the message list in place of the one user turn."""
    import torch
    thinking = adapter.prompt_configuration({'thinking': thinking})['thinking']
    encoded = adapter.processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=thinking,
                                                    tokenize=True, return_tensors='pt')
    if isinstance(encoded, Mapping):
        encoded = encoded['input_ids']
    if not isinstance(encoded, torch.Tensor):
        encoded = torch.tensor(encoded, dtype=torch.long)
    if encoded.ndim == 1:
        encoded = encoded.unsqueeze(0)
    if encoded.ndim != 2 or encoded.shape[0] != 1:
        raise ValueError('DiffusionGemma text generation expects one tokenized prompt')
    return encoded[0].tolist()


def verify_messages_path(adapter, texts) -> dict:
    """The messages path must reproduce encode_prompt ids byte for byte on single-user-turn inputs."""
    checks = 0
    for text in texts:
        for thinking in (False, True):
            if render_messages(adapter, [{'role': 'user', 'content': text}], thinking) != \
                    adapter.encode_prompt(text, {'thinking': thinking}):
                raise AssertionError('messages path differs from encode_prompt on a single-user-turn input')
            checks += 1
    return dict(inputs=len(texts), comparisons=checks, all_identical=True,
                description='render_messages([{"role": "user", "content": x}], t) == adapter.encode_prompt(x, '
                            '{"thinking": t}) for t in (False, True), on synthetic strings and real source texts')


class Specials:
    def __init__(self, tok):
        added = {t.content for t in tok.added_tokens_decoder.values()}
        self.reserved = tuple(sorted({s for s in (*tok.all_special_tokens, *added) if s.startswith('<') and s.endswith('>')}))
        ids = tok.convert_tokens_to_ids
        self.bos, self.turn_open, self.turn_close = tok.bos_token_id, ids('<|turn>'), ids('<turn|>')
        self.found = collections.Counter()

    def scan(self, text):
        if '<' in text:
            for s in self.reserved:
                if s in text:
                    self.found[s] += text.count(s)

    def check_ids(self, ids, turns):
        """One BOS at position 0; `turns` closed turns plus the generation-prompt turn."""
        if ids[0] != self.bos or ids.count(self.bos) != 1:
            raise AssertionError('unexpected BOS layout')
        if ids.count(self.turn_open) != turns + 1 or ids.count(self.turn_close) != turns:
            raise AssertionError('unexpected turn-token count')


def source_identity(pool, files):
    manifest_path = DATA / 'download_manifest.json'
    entry = json.loads(manifest_path.read_text())[POOLS[pool]['repo']]
    root = DATA / 'raw' / POOLS[pool]['dir']
    out = {}
    for name in (*files, 'README.md'):
        digest = sha_file(root / name)
        if digest != entry['files'][name]['sha256']:
            raise AssertionError(f'source file hash differs from the download manifest: {name}')
        out[name] = dict(sha256=digest, bytes=(root / name).stat().st_size)
    return dict(repo=POOLS[pool]['repo'], revision=entry['revision'], root=str(root), files=out,
                download_manifest=str(manifest_path), download_manifest_sha256=sha_file(manifest_path))


def choose(cands, typed):
    for c in cands:
        c['id_sha256'] = sha(c['id'].encode('utf-8'))
    order = sorted(cands, key=lambda c: c['id_sha256'])
    if not typed:
        return order[:PER_BIN], order, {}
    per_type = {t: [c for c in order if c['problem_type'] == t] for t in GW_TYPES}
    if set(c['problem_type'] for c in order) - set(GW_TYPES):
        raise ValueError('unexpected GraphWalks problem_type')
    share = PER_BIN // len(GW_TYPES)
    picked = {t: v[:share] for t, v in per_type.items()}
    fill = collections.Counter()
    for t in GW_TYPES:                                   # fill a short type from the other's continuing sha256 order
        missing = share - len(picked[t])
        for other in GW_TYPES:
            if other != t and missing > 0:
                extra = per_type[other][len(picked[other]):len(picked[other]) + missing]
                picked[other] += extra
                fill[other] += len(extra)
                missing -= len(extra)
    chosen = sorted((c for v in picked.values() for c in v), key=lambda c: c['id_sha256'])
    return chosen, order, dict(per_type_candidates={t: len(v) for t, v in per_type.items()},
                               per_type_chosen=dict(collections.Counter(c['problem_type'] for c in chosen)),
                               filled_from_other_type=dict(fill))


def load_mrcr(specials, logf):
    import pyarrow.parquet as pq
    rows = []
    for name in MRCR_FILES:
        path = DATA / 'raw' / POOLS['mrcr']['dir'] / name
        pf = pq.ParquetFile(path)
        index = json.loads(pf.schema_arrow.metadata[b'pandas'])['index_columns'][0]
        table = pf.read().to_pydict()
        n = len(table['prompt'])
        if index['kind'] != 'range' or index['stop'] - index['start'] != n or index['step'] != 1:
            raise ValueError(f'unexpected stored pandas index in {name}')
        stem = Path(name).stem
        for i in range(n):
            messages = json.loads(table['prompt'][i])
            if not isinstance(messages, list) or not messages or messages[-1]['role'] != 'user':
                raise ValueError('MRCR prompt is not a message list ending in a user turn')
            for m in messages:
                if set(m) != {'role', 'content'} or m['role'] not in ('user', 'assistant') or not isinstance(m['content'], str):
                    raise ValueError('unexpected MRCR message format')
                specials.scan(m['content'])
            specials.scan(table['answer'][i])
            if sum(len(m['content']) for m in messages) != table['n_chars'][i] or len(messages) != table['total_messages'][i]:
                raise ValueError('n_chars / total_messages do not describe the message list')
            if not table['answer'][i].startswith(table['random_string_to_prepend'][i]):
                raise ValueError('MRCR answer does not start with its random string')
            rows.append(dict(source_id=f'{stem}_row{i:03d}', file=name, row=i, dataset_index=index['start'] + i,
                             prompt=table['prompt'][i], answer=table['answer'][i],
                             random_string_to_prepend=table['random_string_to_prepend'][i],
                             n_needles=int(table['n_needles'][i]), total_messages=int(table['total_messages'][i]),
                             n_chars=int(table['n_chars'][i]), date_added=str(table['date_added'][i])))
        log(logf, f'loaded {name}: {n} rows, pandas index [{index["start"]}, {index["stop"]})')
    if len({r['dataset_index'] for r in rows}) != len(rows):
        raise ValueError('duplicate MRCR dataset_index')
    return rows


def turns_of(messages):
    """Turns the template opens: consecutive assistant messages continue one model turn; user turns never merge."""
    return sum(1 for i, m in enumerate(messages) if not (m['role'] == 'assistant' and i and messages[i - 1]['role'] == 'assistant'))


def build_mrcr(adapter, specials, bins, out, logf):
    rows = load_mrcr(specials, logf)
    if specials.found:
        raise AssertionError(f'literal reserved special-token spellings in MRCR text: {dict(specials.found)}')
    tok = adapter.tokenizer
    lo_c, hi_c = MRCR_CHAR_WINDOW
    window = sorted((r for r in rows if lo_c <= r['n_chars'] < hi_c), key=lambda r: r['n_chars'])
    # verify the messages path against encode_prompt on single-user-turn inputs (synthetic + real last user turns)
    real = [json.loads(r['prompt'])[-1]['content'] for r in sorted(window, key=lambda r: sha(r['source_id']))[:4]]
    verification = verify_messages_path(adapter, ['Synthetic check sentence.', 'Line one\n\nLine two  '] + real)
    log(logf, f'messages path verified: {verification["comparisons"]} single-turn comparisons identical')
    counts, ratios, rendered = {}, {}, {}
    t0 = time.time()
    for k, r in enumerate(window):
        s = time.time()
        messages = json.loads(r['prompt'])
        ids = render_messages(adapter, messages, POOLS['mrcr']['thinking'])
        specials.check_ids(ids, turns_of(messages))
        n = len(ids)
        counts[r['source_id']], ratios[r['source_id']] = n, r['n_chars'] / n
        if any(lo <= n < hi for _, _, lo, hi in bins):
            rendered[r['source_id']] = ids
        log(logf, f'{k + 1}/{len(window)} src={r["source_id"]} n_chars={r["n_chars"]} msgs={len(messages)} tokens={n} '
                  f'dt={time.time() - s:.2f}s')
    log(logf, f'rendered {len(window)} rows in {time.time() - t0:.0f}s')
    all_ratios = list(ratios.values())
    lo_r, hi_r = min(all_ratios), max(all_ratios)
    if not (MRCR_GUARD[0] < lo_r and hi_r < MRCR_GUARD[1]):
        raise AssertionError('observed chars/token range reaches the prefilter guard; widen MRCR_CHAR_WINDOW')
    by_src = {r['source_id']: r for r in rows}
    chosen_all, readme_bins, outputs, cells = [], {}, {}, []
    for key, label, lo, hi in bins:
        bench = f'{POOLS["mrcr"]["prefix"]}_{key}'
        cands = [dict(id=f'{bench}/{s}', source_id=s) for s, n in counts.items() if lo <= n < hi]
        chosen, order, _ = choose(cands, typed=False)
        chosen_all += [(key, label, bench, c) for c in chosen]
        readme_bins[key] = dict(label=label, benchmark=bench, token_range=f'[{lo}, {hi})', target=PER_BIN,
                                candidates_in_bin=len(order), chosen=len(chosen), shortfall=max(0, PER_BIN - len(chosen)),
                                candidates_in_sha256_order=[dict(id=c['id'], prompt_token_count=counts[c['source_id']],
                                                                 chosen=i < PER_BIN) for i, c in enumerate(order)])
    # generation budget from the chosen rows' answer token lengths (stats only)
    ans_tokens = {c['id']: len(tok.encode(by_src[c['source_id']]['answer'], add_special_tokens=False))
                  for _, _, _, c in chosen_all}
    limit, share = MRCR_LONG_ANSWER
    over = sum(n > limit for n in ans_tokens.values())
    frac = over / len(ans_tokens) if ans_tokens else 0.0
    budget = MRCR_BUDGETS[1] if frac > share else MRCR_BUDGETS[0]
    log(logf, f'answer tokens over {limit}: {over}/{len(ans_tokens)} -> generation_budget {budget}')
    for key, label, lo, hi in bins:
        bench = readme_bins[key]['benchmark']
        chosen = [c for k2, _, _, c in chosen_all if k2 == key]
        out_rows, gold, meta = [], {}, []
        for c in chosen:
            r, ids = by_src[c['source_id']], rendered[c['source_id']]
            out_rows.append(dict(benchmark=bench, bin=label, date_added=r['date_added'], generation_budget=budget,
                                 id=c['id'], n_chars=r['n_chars'], n_needles=r['n_needles'], prompt=r['prompt'],
                                 prompt_hash=sha(r['prompt']), prompt_token_count=len(ids), prompt_tokens=ids,
                                 source_id=c['source_id'], thinking=POOLS['mrcr']['thinking'],
                                 total_messages=r['total_messages']))
            gold[c['id']] = dict(answer=r['answer'], random_string_to_prepend=r['random_string_to_prepend'])
            rem = r['answer'][len(r['random_string_to_prepend']):]
            meta.append(dict(id=c['id'], source_id=c['source_id'], id_sha256=c['id_sha256'], file=r['file'], row=r['row'],
                             dataset_index=r['dataset_index'], prompt_token_count=len(ids), mod16=len(ids) % 16,
                             mod2=len(ids) % 2, n_chars=r['n_chars'], chars_per_token=round(ratios[c['source_id']], 4),
                             total_messages=r['total_messages'], date_added=r['date_added'], prompt_sha256=sha(r['prompt']),
                             answer_tokens=ans_tokens[c['id']],
                             perfect_reply_ceiling=round(difflib.SequenceMatcher(None, rem.strip(), rem).ratio(), 6)))
        files = write_pool(out, 'mrcr', bench, out_rows, gold, cells)
        outputs.update(files)
        toks = [m['prompt_token_count'] for m in meta]
        ceil = [m['perfect_reply_ceiling'] for m in meta]
        readme_bins[key].update(
            manifest=f'{bench}_generation_manifest.json' if files else None, gold=f'{bench}_gold.json' if files else None,
            prompt_token_count=stats(toks), answer_tokens=stats([m['answer_tokens'] for m in meta], extra=True),
            answers_over_1500_tokens=sum(m['answer_tokens'] > 1500 for m in meta),
            perfect_reply_ceiling=dict(description='SequenceMatcher ratio of the stripped answer remainder (what the '
                                                   'template shows and final_response returns) vs the unstripped gold '
                                                   'remainder', min=min(ceil) if ceil else None,
                                       below_0_99=sum(x < 0.99 for x in ceil), below_1=sum(x < 1 for x in ceil)),
            date_added=dict(collections.Counter(m['date_added'] for m in meta)), chosen_items=meta)
        log(logf, f'bin {key}: in_bin={readme_bins[key]["candidates_in_bin"]} chosen={len(meta)} tokens={stats(toks)}')
    write_cells(out, 'mrcr', cells, outputs)
    edges = sorted(window, key=lambda r: r['n_chars'])
    band = collections.Counter()
    for n in counts.values():
        band[next((lab for _, lab, lo, hi in bins if lo <= n < hi), None) or
             ('<' + bins[0][1].split('-')[0] + 'K' if n < bins[0][2] else 'between/above bins')] += 1
    extra = dict(
        prefilter=dict(field='n_chars (asserted == sum of message content lengths for all 800 rows)',
                       window=dict(lo_inclusive=lo_c, hi_exclusive=hi_c), guard_chars_per_token=list(MRCR_GUARD),
                       rows_total=len(rows), rows_below_window=sum(r['n_chars'] < lo_c for r in rows),
                       rows_in_window_rendered=len(window), rows_above_window=sum(r['n_chars'] >= hi_c for r in rows),
                       observed_chars_per_token=dict(min=round(lo_r, 4), max=round(hi_r, 4), n=len(all_ratios)),
                       smallest_window_rows=[dict(source_id=r['source_id'], n_chars=r['n_chars'],
                                                  tokens=counts[r['source_id']]) for r in edges[:3]],
                       largest_window_rows=[dict(source_id=r['source_id'], n_chars=r['n_chars'],
                                                 tokens=counts[r['source_id']]) for r in edges[-3:]],
                       soundness=f'a row below the window would need < {lo_c / bins[0][2]:.2f} chars/token to reach '
                                 f'{bins[0][2]} tokens and a row above it > {hi_c / bins[-1][3]:.2f} chars/token to stay '
                                 f'under {bins[-1][3]}; every rendered row lies in [{lo_r:.3f}, {hi_r:.3f}]'),
        rendered_by_token_band=dict(band),
        rendered_token_count_distribution=stats(list(counts.values())),
        generation_budget=dict(value=budget, rule=f'{MRCR_BUDGETS[0]} unless more than {share:.0%} of the chosen rows\' '
                                                  f'answers exceed {limit} tokens, then {MRCR_BUDGETS[1]}',
                               answer_token_count='len(tokenizer.encode(answer, add_special_tokens=False)), answer '
                                                  'including the random prefix',
                               chosen_answers=len(ans_tokens), over_limit=over, fraction_over_limit=round(frac, 4),
                               answer_tokens_all_chosen=stats(list(ans_tokens.values()), extra=True)),
        messages_path_verification=verification,
        structure_checks='per rendered row: one BOS at position 0, <|turn> count = template turns + 1, <turn|> count = '
                         'template turns (every conversation opens with two user messages, then alternates)',
        source_notes=dict(source_messages_with_outer_whitespace='the chat template trims each message (user: trim, '
                                                                'model: strip_thinking + trim), as for any client',
                          desired_msg_index='not used (does not index the message list directly)'))
    return readme_bins, outputs, extra, verification


def build_graphwalks(adapter, specials, bins, out, logf):
    import pyarrow.parquet as pq
    path = DATA / 'raw' / POOLS['graphwalks']['dir'] / GW_FILE
    table = pq.read_table(path).to_pydict()
    n_rows = len(table['prompt'])
    for p in table['prompt']:
        specials.scan(p)
    for nodes in table['answer_nodes']:
        for x in nodes:
            specials.scan(x)
    if specials.found:
        raise AssertionError(f'literal reserved special-token spellings in GraphWalks text: {dict(specials.found)}')
    rows = [dict(source_id=f'row{i:03d}', row=i, prompt=table['prompt'][i], answer_nodes=list(table['answer_nodes'][i]),
                 prompt_chars_field=int(table['prompt_chars'][i]), problem_type=str(table['problem_type'][i]),
                 date_added=str(table['date_added'][i])) for i in range(n_rows)]
    if any(not all(isinstance(x, str) for x in r['answer_nodes']) for r in rows):
        raise ValueError('GraphWalks answer nodes must be strings')
    log(logf, f'loaded {GW_FILE}: {n_rows} rows {dict(collections.Counter(r["problem_type"] for r in rows))}')
    shortest = sorted(rows, key=lambda r: len(r['prompt']))[:3]
    verification = verify_messages_path(adapter, ['Synthetic check sentence.'] + [r['prompt'] for r in shortest])
    log(logf, f'messages path verified: {verification["comparisons"]} single-turn comparisons identical')
    counts, rendered = {}, {}
    t0 = time.time()
    for k, r in enumerate(sorted(rows, key=lambda r: len(r['prompt']))):
        s = time.time()
        ids = adapter.encode_prompt(r['prompt'], {'thinking': POOLS['graphwalks']['thinking']})
        specials.check_ids(ids, 2)                       # system (thinking) turn + user turn
        counts[r['source_id']] = len(ids)
        if any(lo <= len(ids) < hi for _, _, lo, hi in bins):
            rendered[r['source_id']] = ids
        log(logf, f'{k + 1}/{n_rows} src={r["source_id"]} type={r["problem_type"]} chars={len(r["prompt"])} '
                  f'tokens={len(ids)} dt={time.time() - s:.2f}s')
    log(logf, f'rendered {n_rows} rows in {time.time() - t0:.0f}s')
    by_src = {r['source_id']: r for r in rows}
    readme_bins, outputs, cells = {}, {}, []
    for key, label, lo, hi in bins:
        bench = f'{POOLS["graphwalks"]["prefix"]}_{key}'
        cands = [dict(id=f'{bench}/{s}', source_id=s, problem_type=by_src[s]['problem_type'])
                 for s, n in counts.items() if lo <= n < hi]
        chosen, order, typing = choose(cands, typed=True)
        out_rows, gold, meta = [], {}, []
        for c in chosen:
            r, ids = by_src[c['source_id']], rendered[c['source_id']]
            out_rows.append(dict(benchmark=bench, bin=label, date_added=r['date_added'], generation_budget=GW_BUDGET,
                                 id=c['id'], problem_type=r['problem_type'], prompt=r['prompt'],
                                 prompt_hash=sha(r['prompt']), prompt_token_count=len(ids), prompt_tokens=ids,
                                 source_id=c['source_id'], thinking=POOLS['graphwalks']['thinking']))
            gold[c['id']] = r['answer_nodes']
            meta.append(dict(id=c['id'], source_id=c['source_id'], id_sha256=c['id_sha256'], row=r['row'],
                             problem_type=r['problem_type'], prompt_token_count=len(ids), mod16=len(ids) % 16,
                             mod2=len(ids) % 2, prompt_chars=len(r['prompt']), date_added=r['date_added'],
                             prompt_sha256=sha(r['prompt']), answer_node_count=len(r['answer_nodes'])))
        files = write_pool(out, 'graphwalks', bench, out_rows, gold, cells)
        outputs.update(files)
        toks = [m['prompt_token_count'] for m in meta]
        readme_bins[key] = dict(label=label, benchmark=bench, token_range=f'[{lo}, {hi})', target=PER_BIN,
                                candidates_in_bin=len(order), chosen=len(chosen), shortfall=max(0, PER_BIN - len(chosen)),
                                balance=typing, manifest=f'{bench}_generation_manifest.json' if files else None,
                                gold=f'{bench}_gold.json' if files else None, prompt_token_count=stats(toks),
                                answer_node_count=stats([m['answer_node_count'] for m in meta]), chosen_items=meta,
                                candidates_in_sha256_order=[dict(id=c['id'], problem_type=c['problem_type'],
                                                                 prompt_token_count=counts[c['source_id']],
                                                                 chosen=c in chosen) for c in order])
        log(logf, f'bin {key}: in_bin={len(order)} chosen={len(chosen)} balance={typing} tokens={stats(toks)}')
    write_cells(out, 'graphwalks', cells, outputs)
    clusters = collections.defaultdict(list)                 # the file holds 8 discrete prompt sizes (2.7K-220K chars)
    for r in rows:
        clusters[(round(math.log2(len(r['prompt']))), r['problem_type'])].append((len(r['prompt']), counts[r['source_id']]))
    extra = dict(
        prefilter=dict(rule='none: every row rendered', rows_total=n_rows,
                       prompt_chars_field_minus_len_prompt=dict(collections.Counter(
                           r['prompt_chars_field'] - len(r['prompt']) for r in rows))),
        rendered_token_count_distribution=stats(list(counts.values())),
        length_clusters=[dict(log2_prompt_chars=c, problem_type=t, prompt_chars=stats([x for x, _ in v]),
                              tokens=stats([y for _, y in v])) for (c, t), v in sorted(clusters.items())],
        per_row_token_counts={r['source_id']: dict(problem_type=r['problem_type'], prompt_chars=len(r['prompt']),
                                                   tokens=counts[r['source_id']], date_added=r['date_added'])
                              for r in rows},
        messages_path_verification=verification,
        structure_checks='per rendered row: one BOS at position 0, 3 <|turn> (system/thinking, user, generation prompt) '
                         'and 2 <turn|>')
    return readme_bins, outputs, extra, verification


def write_pool(out, pool, bench, rows, gold, cells):
    if not rows:
        return {}
    types = ROW_TYPES[pool]
    for r in rows:
        if list(r) != sorted(types) or any(type(r[k]) is not types[k] for k in types):
            raise AssertionError('row format differs from the declared pool row keys / types')
        if not all(type(t) is int for t in r['prompt_tokens']) or len(r['prompt_tokens']) != r['prompt_token_count']:
            raise AssertionError('prompt_tokens must be ints matching prompt_token_count')
    if len({r['id'] for r in rows}) != len(rows) or set(gold) != {r['id'] for r in rows}:
        raise AssertionError('ids must be unique and match the gold keys')
    man_path, gold_path = out / f'{bench}_generation_manifest.json', out / f'{bench}_gold.json'
    man_path.write_bytes((json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8'))
    gold_path.write_text(json.dumps(gold, sort_keys=True) + '\n')
    cells += [dict(dataset=bench, id=r['id'], index=i, seed=1, prompt_tokens=r['prompt_token_count'])
              for i, r in enumerate(rows)]
    return {man_path.name: sha_file(man_path), gold_path.name: sha_file(gold_path)}


def write_cells(out, pool, cells, outputs):
    if not cells:
        return
    path = out / f'cells_{pool}.json'
    path.write_text(json.dumps(cells))
    outputs[path.name] = sha_file(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool', choices=sorted(POOLS), required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--bins', default='', help='override: key:label:lo:hi[,...] (token range [lo, hi))')
    a = p.parse_args()
    bins = parse_bins(a.bins)
    a.out.mkdir(parents=True, exist_ok=True)
    logf = (a.out / 'build.log').open('a')
    t0 = time.time()
    code_root = Path.cwd()
    from dllm.models import create_adapter
    import transformers
    import tokenizers
    files = MRCR_FILES if a.pool == 'mrcr' else (GW_FILE,)
    source = source_identity(a.pool, files)
    log(logf, f'pool {a.pool}: source {source["repo"]}@{source["revision"][:8]} hashes match the download manifest')
    adapter = create_adapter('diffusion_gemma', MODEL, device='cpu', precision='float32', revision=REVISION).load_tokenizer()
    log(logf, f'tokenizer loaded: {type(adapter.processor).__name__}/{type(adapter.tokenizer).__name__}')
    specials = Specials(adapter.tokenizer)
    build = build_mrcr if a.pool == 'mrcr' else build_graphwalks
    readme_bins, outputs, extra, _ = build(adapter, specials, bins, a.out, logf)
    model = Path(MODEL)
    tok_files = {n: sha_file(model / n) for n in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                                                  'processor_config.json') if (model / n).exists()}
    adapter_file = code_root / 'src/dllm/models/adapters/diffusion_gemma.py'
    cfg = POOLS[a.pool]
    readme = dict(
        schema='pools_v31_readme_v1', pool=a.pool, built_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        host=platform.node(),
        purpose=f'Private {cfg["repo"]} pool manifests (natural prompt length, untruncated) for the DiffusionGemma '
                f'sparse-attention study, plus scorer-only gold.',
        rule=__doc__.split('Never prints')[0].strip(),
        selection=dict(hash=f'sha256 of UTF-8 id string "{cfg["prefix"]}_<bin>/<source key>", hex digest ascending',
                       per_bin=PER_BIN, balanced_problem_types=list(GW_TYPES) if a.pool == 'graphwalks' else None,
                       bins={k: dict(label=lab, lo_inclusive=lo, hi_exclusive=hi) for k, lab, lo, hi in bins},
                       truncation=False, thinking=cfg['thinking']),
        row_format=dict(keys=sorted(ROW_TYPES[a.pool]),
                        serialization='json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=False) + "\\n" (UTF-8), '
                                      'as /media/volume/dllm-1/dyh/lb_long_v31_128k/longbench_v2_128k_pool_manifest.json',
                        prompt=('the source JSON-serialized message list string, verbatim' if a.pool == 'mrcr'
                                else 'the source prompt string, verbatim (one user turn)'),
                        prompt_hash='sha256 of the UTF-8 prompt string',
                        files='<benchmark>_generation_manifest.json + <benchmark>_gold.json per non-empty bin; '
                              f'cells_{a.pool}.json lists every row (dataset, id, index = manifest position, seed 1, '
                              'prompt_tokens)'),
        gold_format=('flat dict {id: {"answer": str, "random_string_to_prepend": str}} (official MRCR grading inputs)'
                     if a.pool == 'mrcr' else 'flat dict {id: [answer node, ...]} (official GraphWalks answer_nodes)')
                    + ', json.dumps(gold, sort_keys=True) + "\\n" (scorer only)',
        counts=dict(bins={k: dict(candidates_in_bin=v['candidates_in_bin'], chosen=v['chosen'], shortfall=v['shortfall'])
                          for k, v in readme_bins.items()}),
        bins=readme_bins, source=source, special_token_scan=dict(reserved_spellings=list(specials.reserved),
                                                                 occurrences=dict(specials.found)),
        tokenizer=dict(model_path=MODEL, model_revision=REVISION, loaded='create_adapter("diffusion_gemma", MODEL, device="cpu", '
                       'precision="float32", revision=REVISION).load_tokenizer() -> AutoProcessor only, no weights',
                       processor_class=type(adapter.processor).__name__, tokenizer_class=type(adapter.tokenizer).__name__,
                       files_sha256=tok_files, transformers=transformers.__version__, tokenizers=tokenizers.__version__),
        code=dict(root=str(code_root), deploy_sha=(code_root / 'DEPLOY_SHA').read_text().strip() if (code_root / 'DEPLOY_SHA').exists() else None,
                  adapter_sha256=sha_file(adapter_file), builder=str(Path(__file__).resolve()),
                  builder_sha256=sha_file(Path(__file__).resolve())),
        environment=dict(python=sys.executable, python_version=platform.python_version(),
                         vars={k: os.environ.get(k) for k in ENV_VARS}, gpu_used=False, model_weights_loaded=False),
        runtime_seconds=round(time.time() - t0, 1), output_files_sha256=outputs, **extra)
    (a.out / 'readme.json').write_text(json.dumps(readme, indent=2, sort_keys=True) + '\n')
    log(logf, f'done in {time.time() - t0:.0f}s; outputs {outputs}')


if __name__ == '__main__':
    main()
