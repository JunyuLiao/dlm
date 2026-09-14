"""New isolated 30-AIME/50-v2 scope; frozen parent files stay unchanged."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from experiments.diffusion_gemma_value_aware_followup import protocol as prior
from experiments.diffusion_gemma_value_aware.protocol import frozen_write, sha
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _fingerprint

ROOT = Path('results/diffusion_gemma_value_aware_gpu_50_75')
PARENT = prior.ROOT
METHODS = ('blasst_original', 'blasst_aggressive', 'value', 'mass', 'risk', 'aligned')
TARGETS = (.5, .75)
QUOTAS = dict(zip(prior.DOMAINS, ((9, 8), (8, 9), (8, 8))))
CONFIGS = dict(blasst_original=dict(method='blasst'), blasst_aggressive=dict(method='blasst'),
               value=dict(method='value', pooling='vector_mean'), mass=dict(method='mass'),
               risk=dict(method='risk', pooling='mean'), aligned=dict(method='aligned'))


def select_additions(items, previous):
    final = [r for r in previous['final'] if r['benchmark'] == 'longbench_v2']
    excluded = {r['source_id'] for split in ('final', 'calibration', 'development')
                for r in previous[split] if r['benchmark'] == 'longbench_v2'}
    if len({r['_id'] for r in items}) != len(items):
        raise ValueError('Duplicate source dataset IDs')
    additions = []
    for domain, quotas in QUOTAS.items():
        for difficulty, total in zip(('easy', 'hard'), quotas):
            existing = sum(r['task'] == domain and r['difficulty'] == difficulty for r in final)
            pool = [r for r in items if r['_id'] not in excluded and r['domain'] == domain and r['difficulty'] == difficulty]
            pool.sort(key=lambda r: ({'short': 0, 'medium': 1, 'long': 2}[r['length']], sha(f'gpu50/42/{r["_id"]}')))
            if total < existing or len(pool) < total-existing:
                raise ValueError('Insufficient disjoint domain/difficulty samples')
            additions.extend(pool[:total-existing])
    return additions


def audit(setup):
    rows = setup['final']
    if Counter(r['benchmark'] for r in rows) != {'aime26': 30, 'longbench_v2': 50}:
        raise ValueError('Final scope must be all30 AIME and50 v2')
    for split in ('final', 'calibration', 'development'):
        group = setup[split]
        for key in ('id', 'prompt_hash'):
            if len({r[key] for r in group}) != len(group):
                raise ValueError(f'Duplicate {split}/{key}')
        if any(sha(r['prompt']) != r['prompt_hash'] for r in group):
            raise ValueError('Prompt hash mismatch')
    heldout = [r for r in rows if not r['calibration']]
    for key in ('id', 'prompt_hash'):
        if {r[key] for r in heldout} & {r[key] for r in setup['calibration']+setup['development']}:
            raise ValueError('Final versus tuning overlap')
    if {int(r['source_id']) for r in setup['calibration'] if r['benchmark'] == 'aime26'} != {2, 8, 14, 20, 23, 30}:
        raise ValueError('AIME calibration IDs changed')
    for domain, quotas in QUOTAS.items():
        counts = Counter(r['difficulty'] for r in rows if r['task'] == domain)
        if counts != dict(zip(('easy', 'hard'), quotas)):
            raise ValueError('V2 quotas changed')
    return dict(passed=True, expected_per_condition=80, expected_final=1040,
        aime_full30=30, aime_noncalibration24=24, v2_tasks={d:sum(q) for d, q in QUOTAS.items()},
        v2_difficulty=dict(Counter(r['difficulty'] for r in rows if r['benchmark'] == 'longbench_v2')),
        v2_truncated=sum(r['truncated'] for r in rows if r['benchmark'] == 'longbench_v2'))


def prepare(root=ROOT):
    if (root/'setup.json').exists():
        data = json.loads((root/'setup.json').read_text())
        audit(data)
        return data
    prior.check_official_checkout()
    previous = json.loads((PARENT/'setup.json').read_text())
    prior.audit(previous)
    data = deepcopy(previous)
    items = json.loads(prior.LB_DATA.read_text())
    additions = select_additions(items, previous)
    from dllm.models import create_adapter
    adapter = create_adapter('diffusion_gemma', prior.MODEL, device='cpu', precision='float32', revision=prior.REVISION).load_tokenizer()
    template = (prior.LB_CHECKOUT/'prompts/0shot.txt').read_text()
    for item in additions:
        full = prior.official_prompt(item, template)
        prompt, ids, original_tokens, truncated = prior.limit_prompt(adapter, full)
        data['final'].append(dict(id=f'longbench_v2/{item["_id"]}', source_id=item['_id'], benchmark='longbench_v2',
            task=item['domain'], sub_domain=item['sub_domain'], difficulty=item['difficulty'], length_band=item['length'],
            prompt=prompt, prompt_hash=sha(prompt), prompt_tokens=ids, original_prompt_hash=sha(full),
            original_prompt_tokens=original_tokens, truncated=truncated, expected=item['answer'],
            seed=42, generation_budget=128, split='final', calibration=False))
    old_ids = {r['id'] for r in previous['final']}
    for row in data['final']:
        row['previous_manifest_member'] = row['id'] in old_ids
    data.update(schema='value_aware_gpu_50_75_v1', targets=list(TARGETS), configs=CONFIGS,
        source_hashes={**previous['source_hashes'], str(PARENT/'setup.json'): sha((PARENT/'setup.json').read_bytes())},
        selection='Preserve prior AIME30/v2-30; add20 score-blind short-first SHA256 gpu50/42 IDs, excluding all prior final/calibration/development IDs; v2 quotas17/17/16 with25easy/25hard.',
        exposure='AIME previously exposed; report full30 and noncalibration24 separately. V2 retains prior30 and adds20; exploratory follow-up, not fresh confirmatory evaluation.',
        optimization_scope='Custom GPU engineering only; same routing/native BF16 operator/threshold rules. No FlashAttention comparison or system-performance claim.',
        conditions=['dense']+[f'{n}_s{int(t*100)}' for t in TARGETS for n in METHODS])
    frozen_write(root/'setup.json', data)
    frozen_write(root/'dataset_audit.json', audit(data))
    for split in ('final', 'calibration', 'development'):
        frozen_write(root/f'{split}_manifest.json', data[split])
    return data


def execution(root=ROOT, freeze=False):
    import torch
    import triton
    import transformers
    prepare(root)
    paths = [Path(__file__), Path(__file__).with_name('kernels.py'), Path(__file__).with_name('routing.py'), Path(__file__).with_name('runner.py')]
    paths += [Path('experiments/diffusion_gemma_value_aware')/n for n in ('operators.py', 'routing.py')]
    paths += [Path('src/dllm/attention/blasst/core.py'), Path('src/dllm/models/adapters/diffusion_gemma.py'),
              Path('experiments/diffusion_gemma_solattn_blasst_multibench/runner.py')]
    old = json.loads((PARENT/'execution_contract.json').read_text())
    data = dict(schema='value_aware_gpu_execution_v1', previous_fingerprint=old['previous_fingerprint'],
        parent_fingerprint=old['fingerprint'], setup_sha256=sha((root/'setup.json').read_bytes()),
        sources={str(p):sha(p.read_bytes()) for p in paths},
        runtime=dict(torch=torch.__version__, triton=triton.__version__, transformers=transformers.__version__),
        native_QK_softmax_BF16_PV_unchanged=True, diagnostics='Original dense and sparse diagnostic operators/reduction order reused without regrouping',
        engineering_only=True, speedup_claims=False)
    data['fingerprint'] = _fingerprint(data)
    if freeze:
        frozen_write(root/'execution_contract.json', data)
    elif (root/'execution_contract.json').exists():
        if data != json.loads((root/'execution_contract.json').read_text()):
            raise ValueError('Frozen GPU execution source changed')
    return data


if __name__ == '__main__':
    print(json.dumps(audit(prepare()), indent=2))
