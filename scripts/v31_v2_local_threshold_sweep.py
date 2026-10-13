"""Frozen, resumable accuracy-first local V2 sweep on three native workloads.

Execution is copied before launch; no selector or sampler source is modified.
All inputs/completions stay private. Retained FA4 rectangles and selection work
are accounted independently. Full-panel tuning is explicitly exploratory.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / 'results/v31_v2_local_threshold_sweep_20261014_attempt002'
PYTHON = '/home/exouser/dyh/v31_env_20261005/bin/python'
MODEL = Path('/media/volume/new-dllm/dlm-migrated-20260921/cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b')
GRID = [.018, .0195, .021, .0225, .024]
REFERENCES = ['current_v31_control', 'dense_full_fix51994', 'dense_piecewise', 'allkept_fa4']
SUITES = {
    'longbench_v2': dict(dataset='longbench_v2_0shot_think', count=503, shards=4,
        max_len=141312, block=32, chunk=16384, memory=.85, mage_k=8192),
    'aime26': dict(dataset='aime26', count=90, shards=3,
        max_len=9216, block=64, chunk=4096, memory=.80, mage_k=1728),
    'math500': dict(dataset='math500', count=100, shards=4,
        max_len=16384, block=64, chunk=4096, memory=.80, mage_k=1728),
}


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda: f.read(1 << 22), b''): h.update(x)
    return h.hexdigest()


def write(p, obj, immutable=False):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if immutable and p.exists():
        if p.read_text() != data: raise RuntimeError(f'frozen content differs: {p}')
        return
    tmp = p.with_suffix(p.suffix + '.tmp'); tmp.write_text(data); tmp.replace(p)


def key(r):
    return f"{r['dataset']}|{r['index']}|{r['panel_seed']}|{r['repeat']}"


def label(t):
    return 'v2_tau_' + format(t, '.6f').rstrip('0').replace('.', 'p')


def jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def check_inputs(pool, cells):
    data = read(cells); datasets = {r['dataset'] for r in data}
    manifests = {ds: read(pool / f'{ds}_generation_manifest.json') for ds in datasets}
    for c in data:
        row = manifests[c['dataset']][c['index']]
        tokens = c['prompt_tokens']
        token_match = (len(row['prompt_tokens']) == tokens if isinstance(tokens, int)
                       else row['prompt_tokens'] == tokens)
        if row['id'] != c['id'] or not token_match:
            raise RuntimeError('schedule/manifest input mismatch')
    return data


def runtime_identity():
    import vllm, torch
    sources = {}
    vp = Path(vllm.__file__).parent
    for rel in ['model_executor/models/diffusion_gemma.py',
                'model_executor/layers/attention/diffusion_gemma_attention.py',
                'vllm_flash_attn/cute/flash_fwd_sm90.py', 'vllm_flash_attn/flash_attn_interface.py']:
        p = vp / rel
        if p.exists(): sources['vllm/' + rel] = sha(p)
    return dict(packages={x: importlib.metadata.version(x) for x in
                        ['vllm', 'torch', 'math-verify', 'sympy', 'latex2sympy2-extended']},
                torch_cuda=torch.version.cuda, runtime_sources=sources,
                gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version',
                                             '--format=csv,noheader'], text=True).strip(),
                host_sha256=hashlib.sha256(socket.gethostname().encode()).hexdigest())


def prepare(study):
    protocol_file = study / 'protocol.json'
    if protocol_file.exists():
        protocol = read(protocol_file); verify(study, protocol); return protocol
    if study.exists(): raise RuntimeError('choose a new study root; preparation is already partial')
    study.mkdir(parents=True)
    execution = study / 'private/execution'
    # Freeze production code separately from the shared mutable worktree.
    for directory in ['src', 'scripts', 'experiments']:
        shutil.copytree(ROOT / directory, execution / directory,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pt', '*.npz', '*.png', '*.pdf', '*.jsonl', '*.safetensors', '*.so'))
    inventory = ROOT / 'results/v31_value_v2_equal_work_20261011/model_source_fingerprints.json'
    shutil.copyfile(inventory, study / 'model_source_fingerprints.json')
    inv = read(inventory)
    for f in inv['files']:
        p = MODEL / f['file']
        if not p.exists() or p.stat().st_size != f['bytes']:
            raise RuntimeError('model inventory size mismatch')
        if not f['file'].endswith('.safetensors') and sha(p) != f['sha256']:
            raise RuntimeError('model metadata fingerprint mismatch')
    fingerprints = {str(p.relative_to(execution)): sha(p) for p in execution.rglob('*') if p.is_file()}
    data_sources = {}
    schedules = {}
    for suite, settings in SUITES.items():
        pool = study / f'private/pools/{suite}'; pool.mkdir(parents=True)
        if suite == 'longbench_v2':
            source = Path('/home/exouser/dyh/ljy_value_20261007/pools/longbench_v2_ofc')
            for suffix in ['generation_manifest.json', 'rowinfo.json', 'gold.json']:
                name = settings['dataset'] + '_' + suffix
                shutil.copyfile(source / name, pool / name)
            cells = read(ROOT / 'results/v31_value_v2_equal_work_20261011/private/cells_target_full503.json')
        elif suite == 'aime26':
            source = ROOT / 'results/v31_value_performance_20261010/private/aime26'
            shutil.copyfile(source / 'aime26_generation_manifest.json', pool / 'aime26_generation_manifest.json')
            shutil.copyfile(source / 'aime2026_answers.jsonl', pool / 'aime2026_answers.jsonl')
            cells = read(source / 'cells_aime26_seed42_43_44.json')
        else:
            source = ROOT / 'results/v31_value_matrix_matched_20261011/private/quick100'
            shutil.copyfile(source / 'math500_generation_manifest.json', pool / 'math500_generation_manifest.json')
            manifest = read(pool / 'math500_generation_manifest.json')
            cells = [dict(dataset='math500', id=m['id'], index=i,
                          prompt_tokens=m['prompt_tokens'], seed=1) for i, m in enumerate(manifest)]
            old = {r['id']: r['gold'] for r in read(source / 'math500_rowinfo.json')['rows']}
            # Rebuild ONLY malformed scoring metadata (not prompts or generation tokens).
            # The previous quick rowinfo used compact-token hashes and omitted budget/thinking.
            rows = [dict(id=m['id'], index=i, thinking=m['thinking'], source_id=m['source_id'],
                         generation_budget=m['generation_budget'],
                         prompt_sha256=hashlib.sha256(json.dumps(m['prompt_tokens']).encode()).hexdigest(),
                         gold=old[m['id']]) for i, m in enumerate(manifest)]
            write(pool / 'math500_rowinfo.json', dict(schema='v31_rowinfo_v2', dataset='math500',
                manifest_sha256=sha(pool / 'math500_generation_manifest.json'), max_model_len=16384, rows=rows), True)
        # Old schedules use the native runner's seed key; keep it unchanged.
        if len(cells) != settings['count']: raise RuntimeError('unexpected panel size')
        if suite == 'aime26' and {c['seed'] for c in cells} != {42, 43, 44}:
            raise RuntimeError('AIME seed scope changed')
        cp = study / f'private/schedules/{suite}/cells.json'; write(cp, cells, True)
        check_inputs(pool, cp)
        for i in range(settings['shards']):
            # Keep paired items in exactly the same ordered shard for every arm.
            a, b = len(cells) * i // settings['shards'], len(cells) * (i + 1) // settings['shards']
            write(cp.parent / f'shard{i+1:03d}.json', cells[a:b], True)
        write(cp.parent / 'smoke.json', cells[:2], True)
        schedules[suite] = dict(cells_sha256=sha(cp), seeds=sorted({c['seed'] for c in cells}),
                               count=len(cells), problem_count=len({c['index'] for c in cells}))
        for p in pool.iterdir(): data_sources[str(p.relative_to(study))] = sha(p)
        for p in cp.parent.iterdir(): data_sources[str(p.relative_to(study))] = sha(p)
    protocol = dict(schema='v31_v2_local_threshold_v1', date='2026-10-14',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        execution_fingerprints=fingerprints, input_fingerprints=data_sources,
        model_inventory_sha256=sha(study/'model_source_fingerprints.json'),
        model_revision=inv['model_revision'], runtime=runtime_identity(), suites=SUITES,
        schedules=schedules, stage1_thresholds=GRID, reference_arms=REFERENCES,
        selector='value_v2_online_preserve_mass', scope='GLOBAL only; LOCAL native dense',
        output_path='unchanged FA4; projected V only for routing', exact_greedy='excluded',
        tuning='exploratory full-panel tuning; no held-out accuracy claim', timing='paused',
        selection=dict(primary='mean of three workload accuracies (AIME average across seeds)',
            secondary='mean workload N/problem divided by matched V31 N/problem',
            tertiary='mean workload kept GLOBAL tiles divided by matched V31 tiles',
            refinement='two adjacent midpoint thresholds around the stage1 macro-accuracy winner; no unbounded search',
            per_workload_winners='reported separately; never pool 503 LongBench cells against 30 AIME problems'),
        frozen_native=dict(seed_base=31, canvas=256, denoise_steps=48, thinking=True,
            confidence=.1, stability=1, entropy=.005, temperature=[.8,.4],
            select_step=1, carry=True, progress_trigger=.15, sticky=1.386,
            projection_rank=32, projection_seed=1729, mass_anchor=0))
    write(protocol_file, protocol, True)
    write(study / 'receipts/source_fingerprints.json', dict(source_commit=protocol['source_commit'],
        execution_fingerprints=fingerprints, protocol_sha256=sha(protocol_file)), True)
    write(study / 'thresholds.json', dict(stage1=GRID, reference=.021,
        stage2_rule=protocol['selection']['refinement']), True)
    text = '# V2 local threshold sweep\n\nFive initial uniform GLOBAL thresholds: 0.018, 0.0195, 0.021, 0.0225, 0.024. Two adjacent midpoints are selected after the complete stage-1 grid. Workloads: official 503-item LongBench-v2 (seed 1), 30 AIME26 problems (seeds 42/43/44), deterministic 100-item MATH500 subset (sampling seed 20261011, generation seed 1).\n\nThis is exploratory accuracy tuning on the evaluation panels. Fresh V31, fixed FULL dense, PIECEWISE dense, and all-kept FA4 controls are source/runtime matched. Native decoding and V2 formula are unchanged. Clean timing remains paused. All prompts, gold, tokens, and completions are private.\n'
    (study / 'README.md').write_text(text)
    return protocol


def verify(study, protocol):
    execution = study / 'private/execution'
    for rel, expected in protocol['execution_fingerprints'].items():
        if sha(execution / rel) != expected: raise RuntimeError('frozen execution source drift: ' + rel)
    for rel, expected in protocol['input_fingerprints'].items():
        if sha(study / rel) != expected: raise RuntimeError('frozen input drift: ' + rel)


def env_for(study, suite, arm, threshold=None):
    s = SUITES[suite]; execution = study / 'private/execution'
    env = dict(os.environ)
    for k in list(env):
        if k.startswith(('MAGE_', 'VALUE_', 'LOCAL_', 'CACHED_REFRESH_', 'STABLE_', 'V27_')):
            env.pop(k)
    for k in ['CG_STOP', 'STALL_RESCUE', 'FORCE_REF', 'FORCE_RECORD', 'LIMIT', 'SHARD',
              'TRACE', 'DENSE_WHEN', 'DENSE_BELOW', 'RISK_GROUP', 'CANVAS_RESEED',
              'RESIDUAL', 'DROP_GUARD', 'DRIFT_DIAG', 'REGROUP_DIAG', 'DATASETS', 'REPEATS']:
        env.pop(k, None)
    cache = study / 'private/cache'
    for k in ['vllm', 'triton', 'inductor', 'cuda', 'fa4_local_fix', 'tmp', 'flashinfer']:
        (cache/k).mkdir(parents=True, exist_ok=True)
    env.update(PYTHONPATH=f'{execution}/src:{execution}', PYTHONNOUSERSITE='1',
        HF_HUB_OFFLINE='1', VLLM_ENABLE_V1_MULTIPROCESSING='0', CUDA_VISIBLE_DEVICES='0',
        FIX_51994='1', FA4_LOCAL_FIX='1', FA4_LOCAL_FIX_DIR=str(cache/'fa4_local_fix'),
        VLLM_CACHE_ROOT=str(cache/'vllm'), TRITON_CACHE_DIR=str(cache/'triton'),
        TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'), CUDA_CACHE_PATH=str(cache/'cuda'),
        TMPDIR=str(cache/'tmp'), FLASHINFER_WORKSPACE_BASE=str(cache/'flashinfer'),
        BLOCK=str(s['block']), CHUNK=str(s['chunk']), MEM=str(s['memory']),
        MAX_MODEL_LEN=str(s['max_len']), SEED_BASE='31', MAGE_K=str(s['mage_k']),
        MAGE_SELECT='fa4', MAGE_GRAN='qblock_max', MAGE_STEP='1', MAGE_CARRY='1',
        MAGE_RESELECT_TRIGGER='0.15', MAGE_TRIGGER_SIGNAL='settle', MAGE_STICKY='1.386',
        LOGIT_STATS='fused', DP_BUILD='chunked', OBSERVE='fa4', KV_COPY='triton', MERGE='triton',
        VALUE_MASS_ANCHOR='0', VALUE_AUDIT='1', VALUE_CLEAN_TIMING='0')
    if threshold is not None:
        env.update(VALUE_SELECTOR='value_v2_online_preserve_mass', VALUE_THRESHOLD=str(threshold))
    return env


def wait_idle():
    # External experiments are read-only: wait for resource, never terminate them.
    consecutive = 0
    while consecutive < 3:
        out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        consecutive = consecutive + 1 if not out.strip() else 0
        if consecutive < 3: time.sleep(10)


def validate_records(rows, suite, arm, threshold, planned):
    expected = SUITES[suite]; errors = []
    if len(rows) != len(planned): errors.append('incomplete coverage')
    if len({key(r) for r in rows}) != len(rows): errors.append('duplicate cell')
    planned_keys = {f"{r['dataset']}|{r['index']}|{r['seed']}|0" for r in planned}
    if {key(r) for r in rows} != planned_keys: errors.append('schedule/seed mismatch')
    for r in rows:
        a = (r.get('receipts') or {}).get('adapter') or {}
        selector = 'value_v2_online_preserve_mass' if threshold is not None else None
        checks = [
          (r.get('value_selector') == selector, 'wrong selector'),
          (r.get('value_threshold') == threshold if selector else True, 'wrong threshold'),
          (r.get('cuda_graph_captures') == 0, 'CUDA capture during requests'),
          (r.get('local_kv_budget') is None and a.get('native_local_skipped_tiles',0)==0, 'LOCAL sparsity'),
          (a.get('global_calls') == 5*r['denoise_forwards'], 'GLOBAL scope'),
          (a.get('native_local_calls_audited') == 25*r['denoise_forwards'], 'LOCAL scope'),
          (not a.get('value_nonfinite_logits_calls') and not a.get('value_nonfinite_attention_calls'), 'nonfinite output'),
          (not a.get('order_errors') and not a.get('value_invalid_rows'), 'invalid selector/lifecycle'),
          (a.get('cached_refresh_policy') is None and not a.get('stable_refresh_enabled',False), 'unexpected cache variant'),
          (r.get('block_size')==expected['block'] and r.get('chunk')==expected['chunk'] and r.get('max_model_len')==expected['max_len'], 'runtime geometry'),
          (sum(a.get('decode_cudagraph_modes',{}).values())==r['denoise_forwards'], 'missing graph dispatch audit'),
          (a.get('global_eligible_tiles',-1)>=a.get('global_kept_tiles',0)>=0, 'invalid work accounting'),
        ]
        if selector:
            checks += [(a.get('value_output_path')=='unchanged_fa4' and a.get('value_state_role')=='selector_only', 'attention path changed'),
                       (a.get('value_scope')=='global_only' and not a.get('value_local_router_installed'), 'selector scope')]
        if arm in ('dense_full_fix51994','dense_piecewise','allkept_fa4'):
            checks += [(a.get('global_kept_tiles')==a.get('global_eligible_tiles'), 'allkept/dense skipped tiles')]
        phases = a.get('value_phase_tiles',{})
        checks += [(sum(x['eligible_tiles'] for x in phases.values())==a.get('global_eligible_tiles'), 'phase denominator'),
                   (sum(x['kept_tiles'] for x in phases.values())==a.get('global_kept_tiles'), 'phase kept tiles')]
        errors.extend(name for ok,name in checks if not ok)
    return sorted(set(errors))


def run_shard(study, protocol, suite, arm, threshold, shard, smoke=False):
    name = label(threshold) if threshold is not None else arm
    parent = study / ('qualification' if smoke else 'runs') / suite / name / shard.stem
    parent.mkdir(parents=True, exist_ok=True)
    for p in sorted(parent.glob('attempt[0-9][0-9][0-9]')):
        if (p/'attempt_complete.json').exists():
            cfg = read(p/'config.json'); receipt = read(p/'receipt.json')
            if cfg['protocol_sha256'] != sha(study/'protocol.json'): raise RuntimeError('attempt protocol drift')
            if receipt['validation_errors']: raise RuntimeError('invalid completed receipt')
            return p
    ordinal = max([int(p.name[-3:]) for p in parent.glob('attempt[0-9][0-9][0-9]')], default=0)+1
    attempt = parent / f'attempt{ordinal:03d}'
    verify(study, protocol); wait_idle()
    if runtime_identity() != protocol['runtime']: raise RuntimeError('runtime/host/GPU identity drift')
    attempt.mkdir(); (attempt/'private').mkdir()
    env = env_for(study,suite,arm,threshold)
    execution = study/'private/execution'
    public = attempt/'records.jsonl'; private = attempt/'private'/f'run_{name}.private.jsonl'
    bench_arm = 'mage' if arm == 'current_v31_control' or threshold is not None else ('allkept' if arm=='allkept_fa4' else 'dense')
    cg = 'default' if arm=='dense_full_fix51994' else 'PIECEWISE'
    settings = {k:v for k,v in env.items() if k.startswith(('MAGE_','VALUE_')) or k in ['BLOCK','CHUNK','MEM','MAX_MODEL_LEN','SEED_BASE','FIX_51994','FA4_LOCAL_FIX','KV_COPY','MERGE']}
    write(attempt/'config.json', dict(arm=name, suite=suite, threshold=threshold,
        protocol_sha256=sha(study/'protocol.json'), source_commit=protocol['source_commit'],
        cells_sha256=sha(shard), expected_cells=len(read(shard)), runtime=settings,
        source_fingerprints_sha256=sha(study/'receipts/source_fingerprints.json')), True)
    write(study/'state.json',dict(status='running',suite=suite,arm=name,threshold=threshold,
        attempt=str(attempt.relative_to(study)), updated_unix=time.time()))
    print(json.dumps(dict(event='launch',suite=suite,arm=name,threshold=threshold,
        attempt=str(attempt.relative_to(study)),planned=len(read(shard)))),flush=True)
    start=time.monotonic()
    with (attempt/'private/worker.log').open('w') as log:
        proc=subprocess.run([PYTHON,str(execution/'scripts/v31_vllm_paired_bench.py'),str(MODEL),
            str(study/f'private/pools/{suite}'),str(shard),str(public),str(private),bench_arm,cg],
            cwd=execution,env=env,stdout=log,stderr=subprocess.STDOUT)
    rows=jsonl(public) if public.exists() else []
    errors=validate_records(rows,suite,arm,threshold,read(shard)); verify(study,protocol)
    receipt=dict(status='complete' if proc.returncode==0 and not errors else 'failed',
        returncode=proc.returncode,records=len(rows),planned=len(read(shard)),
        worker_reservation_seconds=time.monotonic()-start,validation_errors=errors,
        unexpected_cuda_captures=sum(r.get('cuda_graph_captures',0) for r in rows),
        protocol_sha256=sha(study/'protocol.json'),source_commit=protocol['source_commit'])
    write(attempt/'receipt.json',receipt,True)
    if receipt['status']!='complete': raise RuntimeError('attempt failed: '+str(attempt))
    write(attempt/'attempt_complete.json',receipt,True)
    return attempt


def quantiles(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return {'n':0}
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),
        min=float(a.min()),p10=float(np.quantile(a,.1)),p25=float(np.quantile(a,.25)),
        p75=float(np.quantile(a,.75)),p90=float(np.quantile(a,.9)),max=float(a.max()))


def ordinal_canvas(a):
    # Adapter canvas IDs are lifecycle IDs, not a guaranteed zero-based ordinal.
    # Rank only the actual observed IDs; never fill invented early zero canvases.
    canvas=a.get('value_canvas_phase_tiles')
    if not canvas:return None
    result={}
    for ordinal,cid in enumerate(sorted(canvas,key=int),1):
        result[ordinal]={k:sum(x.get(k,0) for x in canvas[cid].values())
                         for k in ['eligible_tiles','kept_tiles']}
    if sum(x['kept_tiles'] for x in result.values())!=a['global_kept_tiles']:
        raise RuntimeError('canvas accounting fails to sum to GLOBAL work')
    return result


def aggregate(rows):
    adapters=[r['receipts']['adapter'] for r in rows]
    e=sum(a['global_eligible_tiles'] for a in adapters);k=sum(a['global_kept_tiles'] for a in adapters)
    oe=sum(a['overall_eligible_rectangles'] for a in adapters);oskip=sum(a['overall_skipped_rectangles'] for a in adapters)
    n=sum(r['denoise_forwards'] for r in rows);c=sum(r['canvases'] for r in rows);t=sum(r['output_tokens'] for r in rows)
    phases={}
    for phase in ['dense','initial','refresh','held','carried']:
        p=[a.get('value_phase_tiles',{}).get(phase,{}) for a in adapters]
        pe=sum(x.get('eligible_tiles',0) for x in p);pk=sum(x.get('kept_tiles',0) for x in p)
        phases[phase]=dict(calls=sum(x.get('calls',0) for x in p),eligible_tiles=pe,kept_tiles=pk,
                          skipped_tiles=pe-pk,kept_fraction=pk/pe if pe else None,sparsity=1-pk/pe if pe else None)
    cum=[]
    cmap=[ordinal_canvas(a) for a in adapters]
    if all(x is not None for x in cmap):
        for prefix in range(1,min(max(x) for x in cmap)+1):
            eligible=[sum(v['eligible_tiles'] for i,v in x.items() if i<=prefix) for x in cmap]
            kept=[sum(v['kept_tiles'] for i,v in x.items() if i<=prefix) for x in cmap]
            cum.append(dict(canvas_prefix=prefix,items=len(rows),eligible_tiles=sum(eligible),
                kept_tiles=sum(kept),sparsity=1-sum(kept)/sum(eligible),
                item_kept_distribution=quantiles(kept)))
    out=dict(items=len(rows),N=n,C=c,T=t,mean_N=n/len(rows),median_N=statistics.median(r['denoise_forwards'] for r in rows),
        mean_C=c/len(rows),N_per_C=n/c if c else None,mean_T=t/len(rows),median_T=statistics.median(r['output_tokens'] for r in rows),
        audit_wall_s=sum(r['wall_s'] for r in rows),audit_decode_s=sum(r['decode_s'] for r in rows),
        global_eligible_tiles=e,global_kept_tiles=k,global_skipped_tiles=e-k,global_sparsity=1-k/e,
        overall_eligible_rectangles=oe,overall_skipped_rectangles=oskip,overall_sparsity=oskip/oe,
        fa4_consumer_kept_global_tiles=k,local_skipped_tiles=0,phases=phases,
        value_selected_prefix_tiles=sum(a.get('value_selected_prefix_tiles',0) or 0 for a in adapters),
        selector_discovery_qk_rectangles=sum(a.get('value_statistics_qk_rectangles',0) or 0 for a in adapters),
        selector_candidate_evaluations=sum(a.get('value_candidate_evaluations',0) or 0 for a in adapters),
        selector_discovery_ms=sum((a.get('value_audit_timings') or {}).get('discovery_ms',0) for a in adapters),
        selector_selection_ms=sum((a.get('value_audit_timings') or {}).get('selection_ms',0) for a in adapters),
        finish_reasons=dict(collections.Counter(r['finish_reason'] for r in rows)),cumulative_common_prefix=cum)
    return out


def score_arm(study,suite,name,attempts):
    execution=study/'private/execution'; pool=study/f'private/pools/{suite}'
    dest=study/f'scoring/{suite}/{name}';prefix=dest/'official';summary=Path(str(prefix)+'.summary.json')
    if summary.exists():return read(summary),read(str(prefix)+'.official.json')[name]
    dest.mkdir(parents=True,exist_ok=True)
    files=[]
    for p in attempts: files.extend(p.glob('private/*.private.jsonl'))
    cells=study/f'private/schedules/{suite}/cells.json'
    if suite=='longbench_v2':
        cmd=[PYTHON,str(execution/'scripts/v31_score_longbench_official.py'),str(prefix),str(pool),str(pool)]
    elif suite=='aime26':
        cmd=[PYTHON,str(execution/'scripts/v31_score_aime.py'),str(prefix),str(pool),str(pool/'aime2026_answers.jsonl')]
    else:
        cmd=[PYTHON,str(execution/'scripts/v31_score_math500_bound.py'),str(prefix),str(pool)]
    cmd += ['--cells',str(cells),*[str(p) for p in files]]
    env=dict(os.environ,PYTHONPATH=f'{execution}/src:{execution}',PYTHONNOUSERSITE='1')
    with (dest/'scorer.log').open('w') as log:
        subprocess.run(cmd,cwd=execution,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    return read(summary),read(str(prefix)+'.official.json')[name]


def paired(control,candidate,work_c,work_v):
    if control.keys()!=candidate.keys():raise RuntimeError('paired scorer keys differ')
    keys=sorted(control); clusters=collections.defaultdict(list)
    for k in keys:clusters[tuple(k.split('|')[:2])].append(float(candidate[k])-float(control[k]))
    delta=np.asarray([statistics.mean(x) for x in clusters.values()]);rng=np.random.default_rng(1729)
    draws=rng.integers(len(delta),size=(20000,len(delta)));boot=delta[draws].mean(1)*100
    a=sum(control[k] and not candidate[k] for k in keys);b=sum(candidate[k] and not control[k] for k in keys);nd=a+b
    p=min(1.,2*sum(math.comb(nd,i) for i in range(min(a,b)+1))/2**nd) if nd else 1.
    c={key(x):x for x in work_c};v={key(x):x for x in work_v}
    diff=[v[k]['receipts']['adapter']['global_kept_tiles']-c[k]['receipts']['adapter']['global_kept_tiles'] for k in keys]
    return dict(delta_pp=float(delta.mean()*100),problem_cluster_bootstrap_ci95_pp=list(map(float,np.quantile(boot,[.025,.975]))),
        problem_clusters=len(delta),control_only_correct=a,candidate_only_correct=b,
        mcnemar_exact_p=p,mcnemar_scope='cell-level exploratory; repeated AIME seeds are clustered in the primary bootstrap',
        output_hash_mismatches=sum(c[k]['output_hash']!=v[k]['output_hash'] for k in keys),
        output_token_count_mismatches=sum(c[k]['output_tokens']!=v[k]['output_tokens'] for k in keys),
        denoising_call_mismatches=sum(c[k]['denoise_forwards']!=v[k]['denoise_forwards'] for k in keys),
        canvas_count_mismatches=sum(c[k]['canvases']!=v[k]['canvases'] for k in keys),
        per_item_global_kept_delta=quantiles(diff))


def summarize_arm(study,suite,arm,threshold):
    name=label(threshold) if threshold is not None else arm
    paths=sorted((study/f'private/schedules/{suite}').glob('shard*.json'))
    attempts=[run_shard(study,read(study/'protocol.json'),suite,arm,threshold,p) for p in paths]
    rows=[r for p in attempts for r in jsonl(p/'records.jsonl')]
    score,correct=score_arm(study,suite,name,attempts)
    s=score['coverage'][name]
    if s['scored']!=SUITES[suite]['count'] or s['missing'] or s['dropped']:raise RuntimeError('scorer coverage')
    dest=study/f'reports/{suite}/{name}'
    a=aggregate(rows)
    result=dict(status='complete',suite=suite,arm=name,threshold=threshold,correct=sum(correct.values()),
                items=len(correct),accuracy=sum(correct.values())/len(correct),work=a,official_scorer=score)
    if name!='current_v31_control':
        control_summary=read(study/f'reports/{suite}/current_v31_control/summary.json')
        cpaths=control_summary['attempts'];crows=[r for p in cpaths for r in jsonl(study/p/'records.jsonl')]
        cscores=read(study/f'scoring/{suite}/current_v31_control/official.official.json')['current_v31_control']
        result['paired_vs_v31']=paired(cscores,correct,crows,rows)
        cb=read(study/f'scoring/{suite}/current_v31_control/official.binding.json')
        vb=read(study/f'scoring/{suite}/{name}/official.binding.json')
        # Exact per-cell prompt/runtime/seed parity, enforced independently of receipts.
        sys.path.insert(0,str(study/'private/execution/scripts'))
        import v31_official as op
        op.check_comparable({**cb,**vb},['current_v31_control',name],set(correct))
        result['N_ratio_vs_v31']=a['N']/control_summary['work']['N']
        result['global_kept_ratio_vs_v31']=a['global_kept_tiles']/control_summary['work']['global_kept_tiles']
    result['attempts']=[str(p.relative_to(study)) for p in attempts]
    write(dest/'summary.json',result,True)
    write(dest/'receipt.json',dict(status='complete',protocol_sha256=sha(study/'protocol.json'),
        records=len(rows),scored=len(correct),selector='value_v2_online_preserve_mass' if threshold is not None else None,
        threshold=threshold,local_skipped_tiles=0,validation_errors=[]),True)
    write(dest/'final_complete.json',dict(status='complete',records=len(rows),scored=len(correct)),True)
    print(json.dumps(dict(event='arm_complete',suite=suite,arm=name,correct=result['correct'],items=result['items'],
                         N=a['N'],global_kept_tiles=a['global_kept_tiles'])),flush=True)
    publish(study)
    return result


def rank_candidates(results):
    return sorted(results,key=lambda x:(-x['macro_accuracy'],x['mean_N_ratio'],x['mean_kept_ratio'],x['threshold']))


def stage2_thresholds(best):
    # Midpoints around a boundary winner may extend by half a grid step.
    return [round(best-.00075,6),round(best+.00075,6)]


def choose(study,thresholds,phase):
    rows=[]
    for t in thresholds:
        ss=[read(study/f'reports/{s}/{label(t)}/summary.json') for s in SUITES]
        rows.append(dict(threshold=t,macro_accuracy=statistics.mean(x['accuracy'] for x in ss),
            mean_N_ratio=statistics.mean(x['N_ratio_vs_v31'] for x in ss),
            mean_kept_ratio=statistics.mean(x['global_kept_ratio_vs_v31'] for x in ss),
            workload_accuracy={x['suite']:x['accuracy'] for x in ss}))
    ranked=rank_candidates(rows);out=dict(status='complete',phase=phase,ranked=ranked,
        best_shared_threshold=ranked[0]['threshold'],selection='macro accuracy first; normalized calls then kept work break exact ties')
    write(study/f'selection/{phase}.json',out,True)
    return ranked[0]['threshold']


def publish(study):
    paths=sorted((study/'reports').glob('*/*/summary.json')); summaries=[read(p) for p in paths]
    fields=['suite','arm','threshold','correct','items','accuracy','N','C','T','mean_N','median_N','mean_C','N_per_C','mean_T',
            'global_eligible_tiles','global_kept_tiles','global_skipped_tiles','global_sparsity','overall_sparsity',
            'N_ratio_vs_v31','global_kept_ratio_vs_v31','selector_discovery_qk_rectangles','selector_candidate_evaluations',
            'selector_discovery_ms','selector_selection_ms','audit_wall_s']
    p=study/'curve.csv'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for x in summaries:
            row={**x,**x['work']};w.writerow({k:row.get(k) for k in fields})
    write(study/'curve.json',dict(status='partial',arms=summaries))
    lines=['# V2 local threshold sweep results','',
        'Full-panel threshold tuning is exploratory. Clean timing remains paused. Physical FA4 tile work excludes additional selector discovery/evaluation.',
        '', '| workload | arm | accuracy | N/problem | N/canvas | output tokens/problem | GLOBAL kept tiles | GLOBAL sparsity | kept ratio vs V31 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in summaries:
        a=x['work'];lines.append(f"| {x['suite']} | {x['arm']} | {x['correct']}/{x['items']} ({x['accuracy']*100:.2f}%) | {a['mean_N']:.2f} | {a['N_per_C']:.2f} | {a['mean_T']:.1f} | {a['global_kept_tiles']:,} | {a['global_sparsity']:.2%} | {x.get('global_kept_ratio_vs_v31',1):.4f} |")
    if (study/'selection/final.json').exists():
        select=read(study/'selection/final.json');lines+=['',f"Best shared tested threshold: {select['best_shared_threshold']}. Accuracy-based ranking is exploratory and does not establish a global optimum or noninferiority."]
    (study/'summary.md').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,default=DEFAULT_STUDY)
    ap.add_argument('--stage',choices=['prepare','qualify','run','report'],default='run');args=ap.parse_args()
    study=args.study.resolve()
    if not study.is_relative_to(ROOT/'results'):raise ValueError('canonical result root required')
    protocol=prepare(study)
    if args.stage=='prepare':return
    if args.stage=='report':publish(study);return
    # Gate every reference plus exact V2 in all workloads before the large grid.
    for suite in ['aime26','math500','longbench_v2']:
        shard=study/f'private/schedules/{suite}/smoke.json'
        for arm in REFERENCES+['value_v2_online_preserve_mass']:
            run_shard(study,protocol,suite,arm,.021 if arm.startswith('value_v2') else None,shard,True)
    write(study/'qualification/final_complete.json',dict(status='complete',workloads=3,arms=5,cells_per_arm_workload=2),True)
    if args.stage=='qualify':return
    # Shorter workloads first, retaining complete source-matched controls.
    for suite in ['aime26','math500','longbench_v2']:
        for arm in REFERENCES:summarize_arm(study,suite,arm,None)
        for t in GRID:summarize_arm(study,suite,'value_v2_online_preserve_mass',t)
    best=choose(study,GRID,'stage1');fine=stage2_thresholds(best)
    write(study/'selection/refinement_thresholds.json',dict(stage1_best=best,thresholds=fine,selection_data='stage1 complete three-workload grid'),True)
    for suite in ['aime26','math500','longbench_v2']:
        for t in fine:summarize_arm(study,suite,'value_v2_online_preserve_mass',t)
    best=choose(study,GRID+fine,'final')
    winners={}
    for suite in SUITES:
        xs=[read(study/f'reports/{suite}/{label(t)}/summary.json') for t in GRID+fine]
        wins=sorted(xs,key=lambda x:(-x['accuracy'],x['work']['mean_N'],x['work']['global_kept_tiles'],x['threshold']))
        winners[suite]=dict(threshold=wins[0]['threshold'],correct=wins[0]['correct'],items=wins[0]['items'],accuracy=wins[0]['accuracy'])
    write(study/'selection/workload_winners.json',winners,True)
    publish(study);verify(study,protocol)
    write(study/'receipts/fairness_audit.json',dict(status='complete',same_source_model_runtime_seed_per_suite=True,
        native_sampler_unchanged=True,selector_formula_unchanged=True,global_only=True,
        local_skipped_tiles=0,unexpected_request_graph_captures=0,thresholds=GRID+fine,
        scoring='official LongBench/AIME; bound MATH500 symbolic math-verify',timing='paused'),True)
    write(study/'final_complete.json',dict(status='complete',best_shared_threshold=best,
        workload_winners=winners,thresholds=GRID+fine,source_commit=protocol['source_commit'],
        protocol_sha256=sha(study/'protocol.json'),cells_per_threshold=693,timing='paused'),True)
    write(study/'state.json',dict(status='complete',updated_unix=time.time()))
    print(json.dumps(dict(event='complete',study=str(study),best_shared_threshold=best)),flush=True)

if __name__=='__main__':main()
