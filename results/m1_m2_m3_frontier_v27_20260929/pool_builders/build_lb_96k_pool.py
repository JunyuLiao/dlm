"""Build the private LongBench-v2 96K pool manifest + scorer-only gold (CPU tokenizer only, no weights).

Rule: every LongBench-v2 item (pinned revision) is rendered exactly as scripts/v15_longbench_task.py:render
(NeMo eval/longbench/default @ bcf059af, reserved-token escaping, chat template thinking ON via
adapter.encode_prompt(prompt, {'thinking': True})), with NO truncation and NO length prefilter.
Bin on the rendered request token count: 84-104K = [84000, 104000) (v27 96K extension; same rule as the
32K / 64K bins built by lb_long_v27/build_lb_long_pool.py). Only items whose token count in that earlier full render
(lb_long_v27/build.log, sha256 pinned in readme) falls in the bin are re-rendered here; each re-rendered count must equal
the logged count.
The 12 ids of the existing v27 LongBench-v2 pool are excluded. Candidates of each bin are ordered by
sha256(id) (id = 'longbench_v2/<_id>', UTF-8, hex ascending) and the first 24 are taken (all if fewer).
Never prints prompt text, contexts, questions, choices or answers.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

MODEL = '/home/exouser/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b'
REVISION = 'f7f5b7f5fa82ffc52addd066915886d497f5517b'
BINS = (('96k', '84-104K', 84000, 104000),)
FULL_LOG = Path('/media/volume/dllm-1/dyh/lb_long_v27/build.log')
PER_BIN = 24
POOL_KEYS = ['benchmark', 'bin', 'domain', 'generation_budget', 'id', 'prompt', 'prompt_hash', 'prompt_token_count',
             'prompt_tokens', 'source_id', 'sub_domain', 'thinking']
POOL_TYPES = dict(benchmark=str, bin=str, domain=str, generation_budget=int, id=str, prompt=str, prompt_hash=str,
                  prompt_token_count=int, prompt_tokens=list, source_id=str, sub_domain=str, thinking=bool)


def sha(x) -> str:
    return hashlib.sha256(x if isinstance(x, bytes) else str(x).encode()).hexdigest()


def log(stream, msg):
    stream.write(time.strftime('%H:%M:%S ') + msg + '\n')
    stream.flush()


def stats(values):
    v = sorted(values)
    return dict(n=len(v), min=v[0], median=statistics.median(v), max=v[-1]) if v else dict(n=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--existing-manifest', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    logf = (a.out / 'build.log').open('a')
    t0 = time.time()
    code_root = Path.cwd()
    from scripts import v15_longbench_task as task
    from dllm.models import create_adapter
    import transformers
    import tokenizers
    contract = task.contract_identity()
    log(logf, f'contract ok nemo={contract["nemo_revision"][:8]} dataset={contract["dataset_revision"][:8]}')
    data = json.loads(task.LB_DATA.read_text())
    if len({r['_id'] for r in data}) != len(data):
        raise ValueError('duplicate LongBench IDs')
    adapter = create_adapter('diffusion_gemma', MODEL, device='cpu', precision='float32', revision=REVISION).load_tokenizer()
    log(logf, f'tokenizer loaded: {type(adapter.processor).__name__}/{type(adapter.tokenizer).__name__}')

    # ---- reproduction check: re-render the existing pool items, require exact prompt_hash + token equality
    existing = json.loads(a.existing_manifest.read_text())
    existing_ids = sorted(r['id'] for r in existing)
    by_src = {r['_id']: r for r in data}
    repro = []
    for r in existing:
        x = task.render(adapter, by_src[r['source_id']])
        ok = x['prompt_hash'] == r['prompt_hash'] and x['prompt_tokens'] == r['prompt_tokens'] and x['prompt'] == r['prompt']
        repro.append(dict(id=r['id'], token_count=x['prompt_tokens_n'], identical=ok))
        if not ok:
            raise AssertionError(f'existing pool item not reproduced: {r["id"]}')
    log(logf, f'reproduction check: {len(repro)}/{len(existing)} existing pool items byte/token identical')

    # ---- render every item, no truncation, no prefilter
    import re
    logged = {}
    for line in FULL_LOG.read_text().splitlines():
        m = re.search(r' idx=(\d+) ctx_chars=(\d+) tokens=(\d+) ', line)
        if m:
            logged[int(m.group(1))] = int(m.group(3))
    if len(logged) != len(data):
        raise AssertionError('full render log does not cover every item')
    order = sorted((i for i in range(len(data)) if any(lo <= logged[i] < hi for _, _, lo, hi in BINS)),
                   key=lambda i: len(data[i]['context']))
    counts, rendered = {data[i]['_id']: logged[i] for i in range(len(data))}, {}
    for k, index in enumerate(order):
        item = data[index]
        s = time.time()
        x = task.render(adapter, item)
        n = x['prompt_tokens_n']
        if n != logged[index]:
            raise AssertionError(f'token count differs from the full render log at idx={index}')
        counts[item['_id']] = n
        if any(lo <= n < hi for _, _, lo, hi in BINS):
            rendered[item['_id']] = x
        log(logf, f'{k + 1}/{len(order)} idx={index} ctx_chars={len(item["context"])} tokens={n} dt={time.time() - s:.2f}s')
        del x
    log(logf, f'all rendered in {time.time() - t0:.0f}s')

    overall = collections.Counter()
    for n in counts.values():
        name = next((lab for _, lab, lo, hi in BINS if lo <= n < hi), None)
        overall[name or ('<84K' if n < 84000 else '>=104K')] += 1

    outputs, readme_bins = {}, {}
    index_of = {r['_id']: j for j, r in enumerate(data)}
    for key, label, lo, hi in BINS:
        cands = [dict(source_id=i, id=f'longbench_v2/{i}', index=index_of[i]) for i, n in counts.items() if lo <= n < hi]
        in_bin_total = len(cands)
        excluded = sorted(c['id'] for c in cands if c['id'] in existing_ids)
        cands = [c for c in cands if c['id'] not in existing_ids]
        for c in cands:
            c['id_sha256'] = sha(c['id'].encode('utf-8'))
        cands.sort(key=lambda c: c['id_sha256'])
        chosen = cands[:PER_BIN]
        rows, gold, meta = [], {}, []
        for c in chosen:
            item, x = data[c['index']], rendered[c['source_id']]
            assert len(x['prompt_tokens']) == x['prompt_tokens_n'] and sha(x['prompt']) == x['prompt_hash']
            rows.append(dict(benchmark='longbench_v2', bin=label, domain=item['domain'], generation_budget=8192, id=c['id'],
                             prompt=x['prompt'], prompt_hash=x['prompt_hash'], prompt_token_count=x['prompt_tokens_n'],
                             prompt_tokens=x['prompt_tokens'], source_id=c['source_id'], sub_domain=item['sub_domain'],
                             thinking=True))
            if item['answer'] not in ('A', 'B', 'C', 'D'):
                raise ValueError(f'gold not in A-D for {c["id"]}')
            gold[c['id']] = item['answer']
            n = x['prompt_tokens_n']
            meta.append(dict(id=c['id'], source_id=c['source_id'], id_sha256=c['id_sha256'], dataset_index=c['index'],
                             prompt_token_count=n, mod16=n % 16, mod2=n % 2, domain=item['domain'],
                             sub_domain=item['sub_domain'], difficulty=item['difficulty'], length=item['length'],
                             prompt_sha256=x['prompt_hash'], raw_prompt_sha256=x['raw_prompt_hash'],
                             literal_special_token_escapes=x['literal_special_token_escapes']))
        for r in rows:
            if list(r) != POOL_KEYS or any(type(r[k]) is not POOL_TYPES[k] for k in POOL_KEYS):
                raise AssertionError('row format differs from existing pool')
            if not all(type(t) is int for t in r['prompt_tokens']):
                raise AssertionError('prompt_tokens must be ints')
        man_path = a.out / f'longbench_v2_{key}_pool_manifest.json'
        gold_path = a.out / f'longbench_v2_{key}_gold_scorer_only.json'
        man_path.write_bytes((json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8'))
        gold_path.write_text(json.dumps(gold, sort_keys=True) + '\n')
        outputs[man_path.name] = sha(man_path.read_bytes())
        outputs[gold_path.name] = sha(gold_path.read_bytes())
        toks = [m['prompt_token_count'] for m in meta]
        readme_bins[key] = dict(
            label=label, token_range=f'[{lo}, {hi})', target=PER_BIN, candidates_in_bin=in_bin_total,
            excluded_existing_pool_ids=excluded, candidates_after_exclusion=len(cands), chosen=len(chosen),
            shortfall=PER_BIN - len(chosen), manifest=man_path.name, gold=gold_path.name,
            prompt_token_count=stats(toks),
            domain=dict(collections.Counter(m['domain'] for m in meta).most_common()),
            sub_domain=dict(collections.Counter(m['sub_domain'] for m in meta).most_common()),
            difficulty=dict(collections.Counter(m['difficulty'] for m in meta).most_common()),
            length=dict(collections.Counter(m['length'] for m in meta).most_common()),
            candidate_domain=dict(collections.Counter(data[c['index']]['domain'] for c in cands).most_common()),
            candidate_difficulty=dict(collections.Counter(data[c['index']]['difficulty'] for c in cands).most_common()),
            chosen_items=meta,
            candidates_in_sha256_order=[dict(id=c['id'], prompt_token_count=counts[c['source_id']],
                                             chosen=i < PER_BIN) for i, c in enumerate(cands)])
        log(logf, f'bin {key}: in_bin={in_bin_total} after_excl={len(cands)} chosen={len(chosen)} tokens={readme_bins[key]["prompt_token_count"]}')

    model = Path(MODEL)
    tok_files = {n: sha((model / n).read_bytes()) for n in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                                                              'processor_config.json') if (model / n).exists()}
    adapter_file = code_root / 'src/dllm/models/adapters/diffusion_gemma.py'
    readme = dict(
        schema='lb_long_v27_pool_readme_v1', built_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        host=platform.node(), purpose='Private LongBench-v2 96K pool manifest (natural prompt length, untruncated) '
                                      'for the DiffusionGemma sparse-attention study, plus scorer-only gold.',
        rule=__doc__.split('Never prints')[0].strip(),
        selection=dict(hash='sha256 of UTF-8 id string "longbench_v2/<_id>", hex digest ascending', per_bin=PER_BIN,
                       bins={k: dict(label=lab, lo_inclusive=lo, hi_exclusive=hi) for k, lab, lo, hi in BINS},
                       excluded_pool=str(a.existing_manifest), excluded_ids=existing_ids,
                       truncation=False, length_prefilter=False,
                       render_prefilter=dict(rule='re-render only items whose token count in the earlier full render falls in the bin; '
                                                  'each re-rendered count asserted equal to the logged one',
                                             full_render_log=str(FULL_LOG), full_render_log_sha256=sha(FULL_LOG.read_bytes()),
                                             rerendered=len(order)), thinking=True, generation_budget=8192),
        row_format=dict(keys=POOL_KEYS, serialization='json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=False) + "\\n" (UTF-8), '
                                                     'identical to E:/dlm/v27_private/pool/longbench_v2_pool_manifest.json',
                        bin_labels='range labels in the existing pool style ("10-15K" -> "28-40K", "56-76K")',
                        extra_fields_note='difficulty and length (LongBench length band) are not pool-row keys; recorded per id here'),
        gold_format='flat dict {id: answer_letter}, json.dumps(gold, sort_keys=True) + "\\n", same schema as '
                    '/media/volume/dllm-1/dyh/numerical_qk_longcontext_scored_20260926/private/gold_scorer_only.json (scorer only)',
        counts=dict(dataset_items=len(data), rendered=len(counts), by_token_band=dict(overall), bins={
            k: dict(candidates_in_bin=v['candidates_in_bin'], candidates_after_exclusion=v['candidates_after_exclusion'],
                    chosen=v['chosen'], shortfall=v['shortfall']) for k, v in readme_bins.items()}),
        dataset_token_count_distribution=stats(list(counts.values())),
        bins=readme_bins,
        reproduction_check=dict(description='the existing pool items were re-rendered with this exact code/tokenizer and '
                                            'compared field by field (prompt, prompt_hash, prompt_tokens)',
                                reference=str(a.existing_manifest), reference_sha256=sha(a.existing_manifest.read_bytes()),
                                all_identical=all(r['identical'] for r in repro), items=repro),
        contract=contract,
        tokenizer=dict(model_path=MODEL, model_revision=REVISION, loaded='create_adapter("diffusion_gemma", MODEL, device="cpu", '
                       'precision="float32", revision=REVISION).load_tokenizer() -> AutoProcessor only, no weights',
                       processor_class=type(adapter.processor).__name__, tokenizer_class=type(adapter.tokenizer).__name__,
                       files_sha256=tok_files, transformers=transformers.__version__, tokenizers=tokenizers.__version__),
        code=dict(root=str(code_root), deploy_sha=(code_root / 'DEPLOY_SHA').read_text().strip() if (code_root / 'DEPLOY_SHA').exists() else None,
                  v15_longbench_task_sha256=sha(Path(task.__file__).read_bytes()), adapter_sha256=sha(adapter_file.read_bytes()),
                  builder=str(Path(__file__).resolve()), builder_sha256=sha(Path(__file__).read_bytes())),
        environment=dict(python=sys.executable, python_version=platform.python_version(),
                         vars={k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'HF_HUB_OFFLINE', 'OMP_NUM_THREADS',
                                                              'RAYON_NUM_THREADS', 'TOKENIZERS_PARALLELISM', 'PYTHONDONTWRITEBYTECODE')},
                         gpu_used=False, model_weights_loaded=False),
        runtime_seconds=round(time.time() - t0, 1),
        output_files_sha256=outputs)
    (a.out / 'readme.json').write_text(json.dumps(readme, indent=2, sort_keys=True) + '\n')
    log(logf, f'done in {time.time() - t0:.0f}s; outputs {outputs}')


if __name__ == '__main__':
    main()
