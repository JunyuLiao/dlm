"""Pinned AIME30 plus real LongBench v2 multiple-choice dataset contract."""
import argparse
import ast
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import re
import subprocess

from experiments.diffusion_gemma_value_aware.protocol import (
    MODEL, REVISION, ROOT as PREVIOUS, sha, frozen_write,
)

ROOT = Path('results/diffusion_gemma_value_aware_followup')
LB_REVISION = '2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9'
LB_CHECKOUT = Path('reference/LongBench')
LB_CODE_REVISION = '2e00731f8d0bff23dc4325161044d0ed8af94c1e'
LB_DATA = Path('/home/exouser/.cache/huggingface/hub/datasets--THUDM--LongBench-v2/snapshots')/LB_REVISION/'data.json'
DOMAINS = ('Single-Document QA', 'Multi-Document QA', 'Code Repository Understanding')
INPUT_BUDGET = 32768
TARGETS = (.25, .5, .75, .9)


def official_prompt(item, template):
    # Same sequential substitution and stripping as the pinned pred.py.
    for marker, field in (('DOC', 'context'), ('Q', 'question'),
            ('C_A', 'choice_A'), ('C_B', 'choice_B'), ('C_C', 'choice_C'), ('C_D', 'choice_D')):
        template = template.replace(f'${marker}$', item[field].strip())
    return template


def check_official_checkout():
    head = subprocess.check_output(['git', '-C', str(LB_CHECKOUT), 'rev-parse', 'HEAD'], text=True).strip()
    if head != LB_CODE_REVISION:
        raise ValueError('LongBench checkout revision changed')
    for name in ('pred.py', 'prompts/0shot.txt', 'result.py'):
        expected = subprocess.check_output(['git', '-C', str(LB_CHECKOUT), 'show', f'{head}:{name}'])
        if (LB_CHECKOUT/name).read_bytes() != expected:
            raise ValueError(f'LongBench official source was modified: {name}')


@lru_cache
def official_extractor():
    """Load the exact small official extractor without pred.py's server setup."""
    check_official_checkout()
    tree = ast.parse((LB_CHECKOUT/'pred.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'extract_answer']
    if len(nodes) != 1:
        raise ValueError('official answer extractor missing or ambiguous')
    namespace = {'re': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(LB_CHECKOUT/'pred.py'), 'exec'), namespace)
    return namespace['extract_answer']


def score(row, prediction):
    if row['benchmark'] == 'aime26':
        from experiments.diffusion_gemma_value_aware.protocol import score as aime_score
        return aime_score(row, prediction)
    if row['benchmark'] != 'longbench_v2':
        raise ValueError('unexpected benchmark; do not substitute LongBench v1')
    return float(official_extractor()(prediction.strip()) == row['expected'])


def choose_samples(items, seed=42):
    """Score-blind, short-first, difficulty-balanced selection with disjoint IDs.

    Per domain/difficulty: one calibration, five final, one development. Prefer
    official short contexts; next length band only fills an insufficient pool.
    With v2's six easy/six hard short code items, its two development examples
    therefore come from the next available band, not the final set.
    """
    result = {'calibration': [], 'final': [], 'development': []}
    bands = {'short': 0, 'medium': 1, 'long': 2}
    for domain in DOMAINS:
        for difficulty in ('easy', 'hard'):
            pool = [r for r in items if r['domain'] == domain and r['difficulty'] == difficulty]
            if len({r['_id'] for r in pool}) != len(pool):
                raise ValueError('duplicate source IDs')
            pool.sort(key=lambda r: (bands[r['length']], sha(f'{seed}|{domain}|{difficulty}|{r["_id"]}')))
            if len(pool) < 7:
                raise ValueError(f'insufficient distinct examples: {domain}/{difficulty}')
            result['calibration'].append(pool[0])
            result['final'].extend(pool[1:6])
            result['development'].append(pool[6])
    return result


def limit_prompt(adapter, text, limit=INPUT_BUDGET):
    ids = adapter.encode_prompt(text, {'thinking': False})
    original_count = len(ids)
    if original_count <= limit:
        return text, ids, original_count, False
    raw = adapter.tokenizer.encode(text, add_special_tokens=False)
    # Reserve chat-template overhead, and verify the actual model input length.
    keep = limit - 128
    while keep > 256:
        clipped = adapter.tokenizer.decode(raw[:keep//2] + raw[-keep//2:], skip_special_tokens=True)
        ids = adapter.encode_prompt(clipped, {'thinking': False})
        if len(ids) <= limit:
            return clipped, ids, original_count, True
        keep -= 128
    raise ValueError('could not respect the model-input budget')


def audit(setup):
    sets = [setup[k] for k in ('final', 'calibration', 'development')]
    final, calibration, development = sets
    if Counter(r['benchmark'] for r in final) != {'aime26': 30, 'longbench_v2': 30}:
        raise ValueError('final benchmark sample counts')
    if Counter(r['benchmark'] for r in calibration) != {'aime26': 6, 'longbench_v2': 6}:
        raise ValueError('calibration counts')
    if len(development) != 6 or any(r['benchmark'] != 'longbench_v2' for r in development):
        raise ValueError('development counts')
    for rows in sets:
        for field in ('id', 'prompt_hash'):
            if len({r[field] for r in rows}) != len(rows):
                raise ValueError(f'duplicate {field}')
        if any(sha(r['prompt']) != r['prompt_hash'] for r in rows):
            raise ValueError('prompt hash mismatch')
    heldout = [r for r in final if not r['calibration']]
    for field in ('id', 'prompt_hash'):
        if {r[field] for r in heldout} & {r[field] for r in calibration+development}:
            raise ValueError('final versus tuning contamination')
        if {r[field] for r in calibration} & {r[field] for r in development}:
            raise ValueError('calibration versus development contamination')
    for domain in DOMAINS:
        lb = [r for r in final if r['task'] == domain]
        if len(lb) != 10 or Counter(r['difficulty'] for r in lb) != {'easy': 5, 'hard': 5}:
            raise ValueError('LongBench v2 domain/difficulty balance')
    for row in final+calibration+development:
        if row['benchmark'] == 'longbench_v2':
            if not isinstance(row['prompt_tokens'], list) or len(row['prompt_tokens']) > setup['input_budget']:
                raise ValueError('LongBench v2 token budget/IDs')
            if row['generation_budget'] != 128 or row['expected'] not in tuple('ABCD'):
                raise ValueError('LongBench v2 official budget/answer')
    if {int(r['source_id']) for r in calibration if r['benchmark']=='aime26'} != {2,8,14,20,23,30}:
        raise ValueError('original AIME calibration IDs changed')
    return dict(passed=True, final=60, calibration=12, development=6,
        aime_full30_includes_calibration6=True, aime_heldout24_previously_exposed=True,
        longbench_v2_tasks={d:10 for d in DOMAINS},
        input_budget=setup['input_budget'],
        truncated_final_by_task={d:sum(r['truncated'] for r in final if r['task']==d) for d in DOMAINS})


def prepare(root=ROOT):
    if (root/'setup.json').exists():
        setup=json.loads((root/'setup.json').read_text());audit(setup);return setup
    from dllm.models import create_adapter
    check_official_checkout()
    previous=json.loads((PREVIOUS/'setup.json').read_text())
    groups=choose_samples(json.loads(LB_DATA.read_text()))
    tokenizer=create_adapter('diffusion_gemma',MODEL,device='cpu',precision='float32',revision=REVISION).load_tokenizer()
    template=(LB_CHECKOUT/'prompts/0shot.txt').read_text()
    prepared={k:[dict(r) for r in previous[k] if r['benchmark']=='aime26'] for k in groups}
    selection=[]
    for split,items in groups.items():
        for item in items:
            original=official_prompt(item,template)
            prompt,ids,original_tokens,truncated=limit_prompt(tokenizer,original)
            row=dict(id=f'longbench_v2/{item["_id"]}',source_id=item['_id'],benchmark='longbench_v2',
                task=item['domain'],sub_domain=item['sub_domain'],difficulty=item['difficulty'],length_band=item['length'],
                prompt=prompt,prompt_hash=sha(prompt),prompt_tokens=ids,
                original_prompt_hash=sha(original),original_prompt_tokens=original_tokens,truncated=truncated,
                expected=item['answer'],seed=42,generation_budget=128,split=split,calibration=split=='calibration')
            prepared[split].append(row)
            selection.append({k:v for k,v in row.items() if k not in ('prompt','prompt_tokens','expected')})
    setup=dict(schema='value_aware_followup_v1',model=MODEL,revision=REVISION,precision='bfloat16',
        tile_size=[128,64],regions='prefix_plus_canvas',input_budget=INPUT_BUDGET,
        decoding=previous['decoding'],**prepared,
        dataset_revision=LB_REVISION,official_code_revision=LB_CODE_REVISION,
        selection='Per v2 domain/difficulty: short-first, SHA256 seed42 order, first1 calibration, next5 final, next1 development; no answer/score-based selection. Five easy/five hard final per domain.',
        scoring='Exact pinned LongBench v2 zero-shot extractor, no CoT second call; unparsed answers score zero. Prior AIME scorer unchanged.',
        context_limitation='32,768 model-input tokens with official-style head/tail truncation; not full-context LongBench v2.',
        exposure='AIME heldout24 excludes fitting examples but was evaluated in prior iterations; new mechanisms are exploratory on AIME. LongBench v2 finals are not used for selection.',
        source_hashes={str(p):sha(p.read_bytes()) for p in [PREVIOUS/'setup.json',LB_DATA,
            LB_CHECKOUT/'pred.py',LB_CHECKOUT/'prompts/0shot.txt',LB_CHECKOUT/'result.py']})
    checked=audit(setup)
    for name,rows in prepared.items():frozen_write(root/f'{name}.json',rows)
    frozen_write(root/'dataset_selection.json',selection)
    frozen_write(root/'setup.json',setup);frozen_write(root/'dataset_audit.json',checked)
    return setup


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT)
    args=p.parse_args();s=prepare(args.output);print(json.dumps(audit(s),indent=2))


if __name__=='__main__':main()
