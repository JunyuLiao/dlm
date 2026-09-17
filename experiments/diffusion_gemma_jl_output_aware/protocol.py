"""Reuse AIME30/v2-50 IDs; explicitly upgrade only LongBench prompt/output policy."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import copy, deepcopy
from functools import lru_cache
import json
from pathlib import Path
from threading import local

from experiments.diffusion_gemma_value_aware.protocol import frozen_write, sha
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _fingerprint, _write
from experiments.diffusion_gemma_value_aware_followup.protocol import MODEL, REVISION, LB_DATA
from experiments.diffusion_gemma_longbench_v2_100.protocol import render
from experiments.diffusion_gemma_longbench_v2_100 import nemo
from .config import METHODS, PROJECTED, BASELINES, TARGETS, PRIMARY_SEED, SENSITIVITY_SEEDS

ROOT = Path('results/diffusion_gemma_jl_output_aware_v1')
PARENT = Path('results/diffusion_gemma_value_aware_gpu_50_75')
LATEST = Path('results/diffusion_gemma_longbench_v2_100_nemo')


def prerequisites():
    for root in (PARENT, LATEST):
        audit = json.loads((root/'audit.json').read_text())
        proof = json.loads((root/'regeneration_verification.json').read_text())
        if not audit['complete'] or not proof['passed'] or sha((root/'audit.json').read_bytes()) != proof['audit_sha256']:
            raise ValueError(f'Prior study has not completed its final audited regeneration: {root}')
        for name, digest in audit['artifacts'].items():
            if sha((root/name).read_bytes()) != digest:
                raise ValueError(f'Prior completed artifact changed: {root/name}')
    if json.loads((LATEST/'supervisor_terminal.json').read_text())['exit_code'] != 0:
        raise ValueError('Prior supervisor has not completed successfully')


def audit(data):
    prior = json.loads((PARENT/'setup.json').read_text())
    if {r['id'] for r in data['final']} != {r['id'] for r in prior['final']}:
        raise ValueError('Must preserve all 30 AIME and the same 50 LongBench IDs')
    if Counter(r['benchmark'] for r in data['final']) != {'aime26': 30, 'longbench_v2': 50}:
        raise ValueError('Incorrect final sample counts')
    for split in ('final', 'calibration', 'development'):
        rows = data[split]
        for key in ('id', 'prompt_hash'):
            if len({r[key] for r in rows}) != len(rows):
                raise ValueError(f'Duplicate {split}/{key}')
        for row in rows:
            if sha(row['prompt']) != row['prompt_hash']:
                raise ValueError('Prompt hash changed')
            if row['benchmark'] == 'longbench_v2' and row['generation_budget'] != 4096:
                raise ValueError('LongBench output budget must be 4096')
    for benchmark in ('aime26', 'longbench_v2'):
        cal = [r for r in data['calibration'] if r['benchmark'] == benchmark]
        if len(cal) != 6 or any(r['split'] != 'calibration' for r in cal):
            raise ValueError('Preserve six calibration rows per benchmark')
    held = [r for r in data['final'] if not r['calibration']]
    if {r['id'] for r in held} & {r['id'] for r in data['calibration']+data['development']}:
        raise ValueError('Noncalibration final examples overlap fitting/development')
    old_aime = {r['id']: r for r in prior['final'] if r['benchmark'] == 'aime26'}
    for row in data['final']:
        if row['benchmark'] == 'aime26' and any(row[k] != old_aime[row['id']][k]
                for k in ('prompt', 'prompt_tokens', 'seed', 'generation_budget')):
            raise ValueError('AIME protocol must remain compatible with cached results')
    return dict(passed=True, final=80, calibration=12, development=6,
        aime_calibration_overlap=[r['id'] for r in data['final'] if r['calibration']],
        final_previously_examined=80, fresh_heldout_claim=False,
        longbench_domains=dict(Counter(r['task'] for r in data['final'] if r['benchmark'] == 'longbench_v2')),
        longbench_context_truncated=sum(r.get('truncated', False) for r in data['final'] if r['benchmark'] == 'longbench_v2'))


def prepare(root=ROOT):
    path = root/'setup.json'
    if path.exists():
        data = json.loads(path.read_text()); audit(data); return data
    prerequisites()
    prior = json.loads((PARENT/'setup.json').read_text())
    from dllm.models import create_adapter
    adapter = create_adapter('diffusion_gemma', MODEL, device='cpu', precision='float32', revision=REVISION).load_tokenizer()
    items = {r['_id']: r for r in json.loads(LB_DATA.read_text())}
    nemo.initialize()
    worker = local()
    source_sha = sha(Path(__file__).read_bytes())
    def one(pair):
        split, old = pair
        if old['benchmark'] == 'aime26':
            return deepcopy(old)
        identity = dict(id=old['id'], split=split, renderer=sha(Path(render.__code__.co_filename).read_bytes()),
                        protocol=source_sha, input_budget=32768, output_budget=4096, nemo_revision=nemo.REVISION)
        dest = root/'prepared_rows'/f'{_fingerprint(identity)}.json'
        if dest.exists():
            cached = json.loads(dest.read_text())
            if cached['identity'] != identity:
                raise ValueError('Prepared row identity mismatch')
            return cached['row']
        if not hasattr(worker, 'adapter'):
            worker.adapter = copy(adapter)
            worker.adapter.processor = deepcopy(adapter.processor)
            worker.adapter.tokenizer = worker.adapter.processor.tokenizer
        row = render(worker.adapter, items[old['source_id']], 32768, 4096, split)
        row.update(previous_manifest_member=True, historical_prompt_hash=old['prompt_hash'],
                   historical_generation_budget=old['generation_budget'])
        _write(dest, dict(identity=identity, row=row))
        print('prepared', split, row['id'], len(row['prompt_tokens']), flush=True)
        return row
    groups = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for split in ('calibration', 'development', 'final'):
            groups[split] = list(pool.map(one, [(split, r) for r in prior[split]]))
    data = dict(schema='jl_output_aware_v1', model=MODEL, revision=REVISION, precision='bfloat16',
        tile_size=[128, 64], regions='prefix_plus_canvas', input_budget=32768,
        decoding=prior['decoding'], configs={**BASELINES, **PROJECTED}, targets=list(TARGETS),
        conditions=['dense']+[f'{n}_s{int(t*100)}' for t in TARGETS for n in METHODS],
        primary_projection_seed=PRIMARY_SEED, sensitivity_projection_seeds=list(SENSITIVITY_SEEDS),
        sensitivity_scope='All six primary family/dimension combinations on shared calibration states and predeclared development generation checks; primary seed is never selected by score',
        selection='Exactly the historical AIME30/LongBench-v2-50 IDs; historical LongBench domain counts17/17/16, not full-benchmark proportions',
        exposure='All80 final examples previously examined. AIME full30 includes the same six calibration examples; separately report noncalibration24. Not fresh held-out confirmation.',
        longbench_update='User approved actual NeMo prompt/scorer plus4096 output tokens; same50 IDs, context-only32K truncation. New LongBench dense baseline is required.',
        operator='Exact online alpha times projected attention-weighted block mean minus retained running projected output; strict all-row physical gate; first support retained; no compensation',
        source_hashes={str(p):sha(p.read_bytes()) for p in [PARENT/'setup.json', PARENT/'audit.json',
            LATEST/'audit.json', LATEST/'regeneration_verification.json', LB_DATA,
            Path(render.__code__.co_filename), *nemo.sources()]}, **groups)
    checked = audit(data)
    for split in groups:
        frozen_write(root/f'{split}_manifest.json', data[split])
    frozen_write(root/'setup.json', data)
    frozen_write(root/'dataset_audit.json', checked)
    return data


def execution(root=ROOT, freeze=False):
    import torch, triton, transformers
    setup = prepare(root)
    parent = json.loads((PARENT/'execution_contract.json').read_text())
    for p, expected in parent['sources'].items():
        if sha(Path(p).read_bytes()) != expected:
            raise ValueError(f'Parent inference source changed: {p}')
    names = ('config.py', 'projections.py', 'reference.py', 'kernels.py', 'routing.py',
             'protocol.py', 'runner.py', 'diagnostics.py')
    paths = [Path(__file__).with_name(n) for n in names]
    data = dict(schema='jl_execution_v1', parent_fingerprint=parent['fingerprint'],
        setup_sha256=sha((root/'setup.json').read_bytes()),
        sources={**parent['sources'], **{str(p):sha(p.read_bytes()) for p in paths}},
        runtime=dict(torch=torch.__version__, triton=triton.__version__, transformers=transformers.__version__),
        projection_dtype='float32', matmul_allow_tf32=False, numerical_guard=8e-5,
        diagnostics='Direct FP32 dense and masked outputs on corresponding shared QKV; separate from generation divergence',
        hardware_speedup_claim=False)
    data['fingerprint'] = _fingerprint(data)
    path = root/'execution_contract.json'
    if freeze:
        frozen_write(path, data)
    elif path.exists() and json.loads(path.read_text()) != data:
        raise ValueError('Frozen JL execution changed')
    return data


def score(row, prediction):
    if row['benchmark'] == 'longbench_v2':
        return nemo.score(row, prediction)
    from experiments.diffusion_gemma_value_aware.protocol import score as aime_score
    return aime_score(row, prediction)


if __name__ == '__main__':
    print(json.dumps(audit(prepare()), indent=2))
