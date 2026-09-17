"""Score-blind subset selection and immutable focused execution contract."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from experiments.diffusion_gemma_value_aware.protocol import frozen_write, sha
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _fingerprint
from experiments.diffusion_gemma_value_aware_followup.evidence import check_sources
from experiments.diffusion_gemma_longbench_v2_100.protocol import apportion
from experiments.diffusion_gemma_jl_output_aware.protocol import prerequisites, MODEL, REVISION

ROOT = Path('results/diffusion_gemma_jl_focused_lb100_subset50_v2')
OLD = Path('results/diffusion_gemma_jl_output_aware_v1')
LB = Path('results/diffusion_gemma_longbench_v2_100_nemo')
AIME = Path('results/diffusion_gemma_value_aware_gpu_50_75')
SELECTION_SEED = 20260915
TARGETS = (.5, .75)
PROJECTED = dict(jl_gaussian_r32=dict(family='gaussian', rank=32),
    contribution_gaussian_r32=dict(method='contribution', family='gaussian', rank=32),
    full_centered=dict(family='identity'))
BASELINES = dict(blasst_original=dict(method='blasst'), blasst_aggressive=dict(method='blasst'), mass=dict(method='mass'))
CONFIGS = {**BASELINES, **PROJECTED}
CONDITIONS = ['dense']+[f'{n}_s{int(t*100)}' for t in TARGETS for n in CONFIGS]


def read(path):
    return json.loads(Path(path).read_text())


def select(rows, total=50, seed=SELECTION_SEED):
    """Hamilton domain then subtask quotas; SHA order uses only seed and ID."""
    if len(rows) != 100 or len({r['id'] for r in rows}) != 100:
        raise ValueError('Expected the completed newer100 unique source rows')
    quotas = apportion(Counter(r['task'] for r in rows), total)
    selected, subtasks = [], {}
    for domain, n in sorted(quotas.items()):
        pool = [r for r in rows if r['task'] == domain]
        sub = apportion(Counter(r['sub_domain'] for r in pool), n)
        subtasks[domain] = sub
        for task, count in sorted(sub.items()):
            group = [r for r in pool if r['sub_domain'] == task]
            group.sort(key=lambda r:(sha(f'jl-focused/{seed}/{r["id"]}'), r['id']))
            selected.extend(deepcopy(group[:count]))
    return sorted(selected, key=lambda r:r['id']), dict(seed=seed, domains=quotas, subtasks=subtasks,
        rule='Largest-remainder domain and within-domain subtask quotas from completed100; SHA256(seed,ID) order; no answers, scores, lengths or predictions used')


def audit(data):
    latest, previous, old = read(LB/'setup.json'), read(AIME/'setup.json'), read(OLD/'setup.json')
    chosen, selection = select(latest['final'])
    expected = [r for r in previous['final'] if r['benchmark']=='aime26']+chosen
    if data['final'] != expected or data['selection'] != selection:
        raise ValueError('Selected IDs or cached prompts/settings changed')
    if data['calibration'] != old['calibration'] or data['development'] != old['development']:
        raise ValueError('Calibration/development set changed')
    if data['conditions'] != CONDITIONS or data['configs'] != CONFIGS:
        raise ValueError('Unrequested method/configuration')
    for split in ('final','calibration','development'):
        rows=data[split]
        for key in ('id','prompt_hash'):
            if len({r[key] for r in rows}) != len(rows):
                raise ValueError(f'Duplicate {split}/{key}')
        if any(sha(r['prompt']) != r['prompt_hash'] for r in rows):
            raise ValueError('Prompt hash mismatch')
    held=[r for r in data['final'] if not r['calibration']]
    tuning=data['calibration']+data['development']+latest['calibration']+latest['development']
    for key in ('id','prompt_hash'):
        if {r[key] for r in held} & {r[key] for r in tuning}:
            raise ValueError('Noncalibration evaluation overlaps fitting/development')
    for row in data['final']:
        if row['generation_budget'] != (4096 if row['benchmark']=='longbench_v2' else 2048):
            raise ValueError('Unrequested generation cap')
    return dict(passed=True, expected=1040, cached_baselines=560, new_generations=480,
        longbench_domains=dict(Counter(r['task'] for r in chosen)),
        longbench_subtasks=dict(Counter(r['sub_domain'] for r in chosen)),
        longbench_truncated=sum(r['truncated'] for r in chosen),
        longbench_difficulty=dict(Counter(r['difficulty'] for r in chosen)),
        aime_calibration_overlap=[r['id'] for r in expected if r['calibration']],
        longbench_calibration_overlap=[], all_final_previously_examined=True, fresh_heldout_claim=False)


def prepare(root=ROOT):
    path=root/'setup.json'
    if path.exists():
        data=read(path); audit(data); return data
    prerequisites()
    latest, previous, old = read(LB/'setup.json'), read(AIME/'setup.json'), read(OLD/'setup.json')
    chosen, selection = select(latest['final'])
    for key in ('model','revision','precision','tile_size','regions'):
        if not (latest[key] == previous[key] == old[key]):
            raise ValueError(f'Incompatible inference setting: {key}')
    if old['decoding']!=previous['decoding'] or latest['decoding']!=dict(canvas=256,max_denoising_steps=48,
            seed=42,temperature='native0.4-0.8 schedule; not greedy',thinking=False):
        raise ValueError('Incompatible decoding declaration')
    # The newer100 setup uses short aliases; cached raw generation_metadata is
    # additionally compared field-for-field when pairing every final result.
    paths=[p/'setup.json' for p in (LB,AIME,OLD)]+[p/'execution_contract.json' for p in (LB,AIME,OLD)]
    data=dict(schema='jl_focused_lb100_subset50_v2', **{k:old[k] for k in
        ('model','revision','precision','tile_size','regions','decoding','input_budget')},
        final=[deepcopy(r) for r in previous['final'] if r['benchmark']=='aime26']+chosen,
        calibration=old['calibration'], development=old['development'],
        selection=selection, configs=CONFIGS, conditions=CONDITIONS, targets=list(TARGETS),
        projection_seed=1729, projection_policy='Fixed Gaussian rank32 for both projected controls; no final-score direction or seed selection',
        exposure='Previously examined samples; AIME full30 includes6 calibration problems, noncalibration24 reported separately',
        calibration_policy='Reuse verified JL32/full-centered policies and540 shared calibration QKV states; calibrate only contribution32 with unchanged six-per-benchmark512-token empirical-CDF search, at most3 verified joint points. Baselines retain their source policies (LB12 across6 domains; JL LB6 across3 domains). No final retuning.',
        baseline_sources={'aime26':str(AIME),'longbench_v2':str(LB)},
        source_hashes={str(p):sha(p.read_bytes()) for p in paths})
    checked=audit(data)
    for split in ('final','calibration','development'):
        frozen_write(root/f'{split}_manifest.json',data[split])
    frozen_write(root/'selection.json',dict(selection,ids=[r['id'] for r in chosen],source=str(LB/'setup.json')))
    frozen_write(path,data); frozen_write(root/'dataset_audit.json',checked)
    return data


def execution(root=ROOT, freeze=False):
    import torch, triton, transformers
    setup=prepare(root)
    old=read(OLD/'execution_contract.json')
    check_sources(old['sources']); check_sources(setup['source_hashes'])
    runtime=dict(torch=torch.__version__,triton=triton.__version__,transformers=transformers.__version__)
    if runtime != old['runtime']:
        raise ValueError('Cached execution runtime mismatch')
    paths=sorted(Path(__file__).parent.glob('*.py'))
    paths += [Path('experiments/diffusion_gemma_jl_output_aware')/n for n in
        ('calibration.py','screen.py','validation.py','shared_analysis.py','trace_kernels.py','report.py')]
    data=dict(schema='jl_focused_execution_v2',parent_fingerprint=old['fingerprint'],runtime=runtime,
        setup_sha256=sha((root/'setup.json').read_bytes()),
        sources={**old['sources'],**{str(p):sha(p.read_bytes()) for p in paths}},
        projection_dtype='float32',matmul_allow_tf32=False,numerical_guard=8e-5,
        hardware_speedup_claim=False, frozen_algorithms_unchanged=True)
    data['fingerprint']=_fingerprint(data)
    path=root/'execution_contract.json'
    if path.exists() and read(path)!=data:
        raise ValueError('Focused frozen execution changed; version the study instead')
    if freeze: frozen_write(path,data)
    return data
