"""Immutable single-worker launch wrapper for the independent selector study.

Uses supplied frozen manifests and cells verbatim. Does not build prompts,
change budgets, score answers, or reuse a killed attempt directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


ARM_CONFIG = {
    'dense_full_fix51994': ('dense', 'default', None, 'qblock_max'),
    'dense_piecewise': ('dense', 'PIECEWISE', None, 'qblock_max'),
    'allkept_fa4': ('allkept', 'PIECEWISE', None, 'qblock_max'),
    'current_v31_control': ('mage', 'PIECEWISE', None, 'qblock_max'),
    'attention_mass_kvhead': ('mage', 'PIECEWISE', None, 'kvhead'),
    **{name: ('mage', 'PIECEWISE', name, 'qblock_max') for name in (
        'value_v1_online_discard_mass', 'value_v2_online_preserve_mass',
        'value_v3a_singleton_delete', 'value_v3b_greedy_exact', 'value_v3b_approx_batch8')},
    'value_v2_online_mass_anchor32': ('mage', 'PIECEWISE', 'value_v2_online_preserve_mass', 'qblock_max'),
    'value_v2_online_mass_anchor64': ('mage', 'PIECEWISE', 'value_v2_online_preserve_mass', 'qblock_max'),
    'value_v2_online_jcgate': ('mage', 'PIECEWISE', 'value_v2_online_preserve_mass', 'qblock_max'),
    'value_v2_online_jcgate_anchor16': ('mage', 'PIECEWISE', 'value_v2_online_preserve_mass', 'qblock_max'),
}

# Accuracy-first fixed-budget follow-ups.  Each arm keeps an equal prefix
# budget per (query head, 128-row) unit; pool_factor and swap_count only alter
# which tiles are exchanged inside that fixed budget.
FIXED_VALUE_CONFIG = {
    'value_mass_gated_online_swap8': ('value_mass_gated_online', 2, 8, None),
    'value_mass_gated_online_swap16': ('value_mass_gated_online', 2, 16, None),
    'value_mass_gated_full_swap16': ('value_mass_gated_full', 2, 16, None),
    'value_mass_gated_online_phase_swap16': ('value_mass_gated_online_phase', 2, 16, None),
    'value_mass_gated_full_phase_swap16': ('value_mass_gated_full_phase', 2, 16, None),
    'value_mass_gated_online_cgate_swap16': ('value_mass_gated_online_cgate', 2, 16, 'cgate'),
    'value_mass_gated_full_cgate_swap16': ('value_mass_gated_full_cgate', 2, 16, 'cgate'),
    'value_intersection_online': ('value_intersection_online', 2, 0, None),
    'value_intersection_full': ('value_intersection_full', 2, 0, None),
}
ARM_CONFIG.update({name: ('mage', 'PIECEWISE', cfg[0], 'qblock_max')
                   for name, cfg in FIXED_VALUE_CONFIG.items()})

# Cached-statistics refresh variants.  The frontier size is the total global
# candidate pool (K+4B, 2K, or 4K), never an additional outside pool.  The
# decay arm is the primary ablation: half-life two cheap events, floor 0.25.
CACHED_REFRESH_CONFIG = {
    'cached_refresh_global_topk_m160': ('global_topk', 160, 0, 0.0, 0.25),
    'cached_refresh_global_topk_m256': ('global_topk', 256, 0, 0.0, 0.25),
    'cached_refresh_global_topk_m512': ('global_topk', 512, 0, 0.0, 0.25),
    'cached_refresh_bounded_m160': ('bounded_swaps', 160, 8, 0.0, 0.25),
    'cached_refresh_bounded_m256': ('bounded_swaps', 256, 8, 0.0, 0.25),
    'cached_refresh_bounded_m512': ('bounded_swaps', 512, 8, 0.0, 0.25),
    'cached_refresh_bounded_m256_decay': ('bounded_swaps', 256, 8, 0.3465735903, 0.25),
    'cached_refresh_bounded_m256_phase_protected': ('bounded_swaps', 256, 8, 0.0, 0.25),
    'cached_refresh_bounded_m256_cgate': ('bounded_swaps', 256, 8, 0.0, 0.25),
    'cached_refresh_bounded_m256_online_value_tiebreak': ('bounded_swaps', 256, 8, 0.0, 0.25),
    'cached_refresh_bounded_m256_full_value_tiebreak': ('bounded_swaps', 256, 8, 0.0, 0.25),
    'cached_refresh_bounded_m256_every_1_step': ('bounded_swaps', 256, 8, 0.0, 0.25),
    'cached_refresh_bounded_m256_every_4_steps': ('bounded_swaps', 256, 8, 0.0, 0.25),
    'cached_refresh_bounded_m256_progress_triggered': ('bounded_swaps', 256, 8, 0.0, 0.25),
    'cached_refresh_intersection_online': ('bounded_swaps', 256, 8, 0.0, 0.25),
    'cached_refresh_intersection_full': ('bounded_swaps', 256, 8, 0.0, 0.25),
}
ARM_CONFIG.update({name: ('mage', 'PIECEWISE', None, 'qblock_max') for name in CACHED_REFRESH_CONFIG})


STABLE_ARMS = {
    'v31_progress_dense_control': ('raw', 0, True, False),
    'v31_no_progress_dense_hold': ('raw', 0, False, False),
    'raw_global_progress_dense_s2': ('raw', 2, True, True),
    'stable_global_progress_dense_s2': ('stable', 2, True, True),
    'observation_only_progress_dense_s2': ('raw', 2, True, False),
    **{f'{mode}_global_no_progress_dense_s{period}': (mode, period, False, True)
       for mode in ('raw', 'stable') for period in (1, 4, 8)},
}
ARM_CONFIG = {name: ('mage', 'PIECEWISE', None, 'qblock_max') for name in STABLE_ARMS} | {
    name: ARM_CONFIG[name] for name in ('dense_full_fix51994', 'dense_piecewise', 'allkept_fa4')}


def main():
    p = argparse.ArgumentParser()
    for name in ('python', 'model', 'manifests', 'cells', 'attempt'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--arm', required=True, choices=ARM_CONFIG)
    p.add_argument('--threshold', type=float)
    p.add_argument('--mage-k', type=int, default=8192,
                   help='per-qblock kept-token budget; set per suite/regime')
    p.add_argument('--max-model-len', type=int, default=141312)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    p.add_argument('--purpose', choices=('smoke', 'development', 'audit'), required=True)
    p.add_argument('--result-root', type=Path,
                   default=Path(__file__).resolve().parents[1]/'results/v31_value_selectors_20261008')
    p.add_argument('--stable-parameters', type=Path)
    p.add_argument('--trajectory-dir', type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    result_root = args.result_root.resolve()
    attempt = Path(args.attempt).resolve()
    if not attempt.is_relative_to(result_root) or attempt.exists():
        raise ValueError('a new attempt below the canonical study result root is required')
    if (args.arm.startswith(('value_v1_', 'value_v2_')) or args.arm in FIXED_VALUE_CONFIG) and args.threshold is None:
        raise ValueError('supply the development trial or frozen uniform threshold')
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    if gpu.strip():
        raise RuntimeError('H100 is occupied; launch refused')
    cells_bytes = Path(args.cells).read_bytes()
    cells = json.loads(cells_bytes)
    if not cells:
        raise ValueError('empty cell schedule')
    attempt.mkdir(parents=True)
    (attempt/'private').mkdir()
    cache = result_root/'private/cache'
    for name in ('vllm', 'triton', 'inductor', 'cuda', 'fa4_local_fix', 'tmp', 'flashinfer'):
        (cache/name).mkdir(parents=True, exist_ok=True)
    arm, graph, selector, granularity = ARM_CONFIG[args.arm]
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(('MAGE_','VALUE_','LOCAL_','STABLE_')):
            env.pop(key)
    for key in ('LOCAL_KV_BUDGET', 'LOCAL_KERNEL', 'VALUE_SELECTOR', 'VALUE_THRESHOLD',
                'VALUE_POOL_FACTOR', 'VALUE_SWAP_COUNT',
                'MAGE_FRAC', 'MAGE_POOL', 'MAGE_ROWW', 'MAGE_RESELECT', 'MAGE_RESELECT_K',
                'MAGE_RESELECT_KMIN', 'MAGE_KCOVER', 'RESIDUAL', 'DROP_GUARD', 'TRACE',
                'DRIFT_DIAG', 'CG_STOP', 'STALL_RESCUE', 'FORCE_REF', 'FORCE_RECORD', 'LIMIT', 'SHARD',
                'DENSE_WHEN','DENSE_BELOW','REGROUP_DIAG','RISK_GROUP','CANVAS_RESEED','V27_ADAPTER_DIR','DATASETS','REPEATS',
                'CACHED_REFRESH_POLICY','CACHED_REFRESH_PERIOD','CACHED_REFRESH_FRONTIER_M',
                'CACHED_REFRESH_SWAP_BUDGET','CACHED_REFRESH_DECAY_LAMBDA','CACHED_REFRESH_DECAY_FLOOR',
                'CACHED_REFRESH_VALUE_TIEBREAK','CACHED_REFRESH_MASS_TOLERANCE','CACHED_REFRESH_DRIFT_THRESHOLD',
                'CACHED_REFRESH_PHASE_PROTECTED', 'CACHED_REFRESH_FREQUENCY'):
        env.pop(key, None)
    env.update(PYTHONPATH=f'{root}/src:{root}', PYTHONNOUSERSITE='1',
        HF_HUB_OFFLINE='1', VLLM_ENABLE_V1_MULTIPROCESSING='0', CUDA_VISIBLE_DEVICES='0',
        FIX_51994='1', FA4_LOCAL_FIX='1', FA4_LOCAL_FIX_DIR=str(cache/'fa4_local_fix'),
        VLLM_CACHE_ROOT=str(cache/'vllm'), TRITON_CACHE_DIR=str(cache/'triton'),
        TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'), CUDA_CACHE_PATH=str(cache/'cuda'),
        TMPDIR=str(cache/'tmp'), FLASHINFER_WORKSPACE_BASE=str(cache/'flashinfer'),
        BLOCK='32', CHUNK='16384', MEM=str(args.gpu_memory_utilization), MAX_MODEL_LEN=str(args.max_model_len),
        SEED_BASE='31', MAGE_K=str(args.mage_k), MAGE_SELECT='fa4', MAGE_GRAN=granularity,
        MAGE_STEP='1', MAGE_CARRY='1', MAGE_RESELECT_TRIGGER='0.15',
        MAGE_TRIGGER_SIGNAL='settle', LOGIT_STATS='fused', DP_BUILD='chunked',
        OBSERVE='fa4', KV_COPY='triton', MERGE='triton')
    # The mass-mean control cannot accept qblock-specific sticky in this adapter.
    # It is therefore an explicitly labelled ablation, never the matched control.
    if granularity == 'qblock_max':
        env['MAGE_STICKY'] = '1.386'
    else:
        env.pop('MAGE_STICKY', None)
    if selector:
        env['VALUE_SELECTOR'] = selector
    if args.arm in FIXED_VALUE_CONFIG:
        _, pool_factor, swap_count, row_weight = FIXED_VALUE_CONFIG[args.arm]
        env['VALUE_POOL_FACTOR'] = str(pool_factor)
        env['VALUE_SWAP_COUNT'] = str(swap_count)
        if row_weight is not None:
            env['MAGE_ROWW'] = row_weight
    if args.arm in CACHED_REFRESH_CONFIG:
        policy, frontier_m, swaps, decay_lambda, decay_floor = CACHED_REFRESH_CONFIG[args.arm]
        period = '2'
        if args.arm.endswith('_every_1_step'):
            period = '1'
        elif args.arm.endswith('_every_4_steps'):
            period = '4'
        env.update(CACHED_REFRESH_POLICY=policy, CACHED_REFRESH_PERIOD=period,
                   CACHED_REFRESH_FRONTIER_M=str(frontier_m), CACHED_REFRESH_SWAP_BUDGET=str(swaps),
                   CACHED_REFRESH_DECAY_LAMBDA=str(decay_lambda), CACHED_REFRESH_DECAY_FLOOR=str(decay_floor),
                   CACHED_REFRESH_VALUE_TIEBREAK='1' if ('value_tiebreak' in args.arm or 'cgate' in args.arm) else '0',
                   CACHED_REFRESH_MASS_TOLERANCE='0.05' if ('value_tiebreak' in args.arm or 'cgate' in args.arm) else '0',
                   CACHED_REFRESH_DRIFT_THRESHOLD='10000000',
                   CACHED_REFRESH_PHASE_PROTECTED='1' if 'phase_protected' in args.arm else '0',
                   CACHED_REFRESH_FREQUENCY='progress' if args.arm.endswith('_progress_triggered') else 'periodic')
        if 'cgate' in args.arm:
            env['MAGE_ROWW'] = 'cgate'
    stable_config = None
    if args.arm in STABLE_ARMS:
        mode, period, progress_dense, updates = STABLE_ARMS[args.arm]
        parameters = (json.loads(args.stable_parameters.read_text())['selector']
                      if args.stable_parameters else {})
        if mode == 'stable' and args.stable_parameters is None:
            raise ValueError('stable production arm requires operator-frozen parameters')
        parameters.update(mode=mode, updates_enabled=updates)
        stable_config = dict(selector=parameters, period=period, progress_dense=progress_dense)
        env['STABLE_REFRESH_CONFIG'] = json.dumps(stable_config, sort_keys=True)
    if args.trajectory_dir:
        env['STABLE_TRAJECTORY_DIR'] = str(args.trajectory_dir.resolve())
    env['VALUE_MASS_ANCHOR'] = str({'value_v2_online_mass_anchor32': 32,
                                    'value_v2_online_mass_anchor64': 64,
                                    'value_v2_online_jcgate_anchor16': 16}.get(args.arm, 0))
    if args.arm in ('value_v2_online_jcgate', 'value_v2_online_jcgate_anchor16'):
        env['MAGE_ROWW'] = 'jcgate'
    env['VALUE_AUDIT'] = '1' if args.purpose in ('audit', 'smoke', 'development') else '0'
    env['VALUE_CLEAN_TIMING'] = '1' if args.purpose == 'clean' else '0'
    env.pop('VALUE_DIAGNOSTIC_DIR', None)
    # Stable trajectory capture is explicit and restricted to qualification.
    if args.threshold is not None:
        env['VALUE_THRESHOLD'] = str(args.threshold)
    # Hash each frozen dataset manifest once.  A target schedule can contain
    # hundreds of cells from one dataset; rereading a 653 MB manifest per cell
    # needlessly turns preparation into a multi-hundred-GB scan.
    datasets = {c['dataset'] for c in cells}
    manifest_hashes = {dataset: hashlib.sha256(
        (Path(args.manifests)/(dataset+'_generation_manifest.json')).read_bytes()).hexdigest()
        for dataset in datasets}
    source_files = [root/'experiments/numerical_qk_reuse'/name for name in
        ('vllm_adapter.py', 'v31_value_selectors.py', 'v31_value_kernels.py', 'v31_value_summary.py', 'v31_value_workspace.py', 'v31_value_projection.py', 'v31_value_snapshots.py', 'v31_value_native_audit.py', 'v31_cached_refresh.py', 'v27_fa4.py', 'v31_fa4_observe.py', 'v29_paged_copy.py', 'v29_lse_merge.py', 'v31_logit_stats.py')]
    source_files += [root/'experiments/diffusion_gemma_jl_output_aware/projections.py',
                     root/'experiments/value_direction_hopper/projection.py']
    source_files += [root/'scripts/v31_vllm_paired_bench.py', root/'scripts/v31_value_campaign.py']
    source_files += [root/'experiments/numerical_qk_reuse'/n for n in ('v31_stable_refresh.py','v31_stable_observe.py','v31_stable_trajectory.py')]
    source_files += [root/'scripts/v31_stable_refresh_campaign.py']
    gpu_identity=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,driver_version','--format=csv,noheader'],text=True).strip()
    freeze = dict(schema='independent_v31_attempt_v1', purpose=args.purpose, arm=args.arm,
        cell_count=len(cells), cells_sha256=hashlib.sha256(cells_bytes).hexdigest(),
        manifests_sha256=manifest_hashes, source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        source_sha256={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files},
        host_fingerprint=hashlib.sha256(socket.gethostname().encode()).hexdigest(),
        gpu_identity=gpu_identity,
        model_source_inventory_sha256=hashlib.sha256((result_root/'model_source_fingerprints.json').read_bytes()).hexdigest(),
        runtime_settings={k:env[k] for k in ('BLOCK','CHUNK','MEM','MAX_MODEL_LEN','MAGE_K','MAGE_STEP','MAGE_CARRY','MAGE_RESELECT_TRIGGER','MAGE_TRIGGER_SIGNAL','MAGE_GRAN','MAGE_STICKY','MAGE_ROWW','VALUE_MASS_ANCHOR','VALUE_POOL_FACTOR','VALUE_SWAP_COUNT','CACHED_REFRESH_POLICY','CACHED_REFRESH_PERIOD','CACHED_REFRESH_FRONTIER_M','CACHED_REFRESH_SWAP_BUDGET','CACHED_REFRESH_DECAY_LAMBDA','CACHED_REFRESH_DECAY_FLOOR','CACHED_REFRESH_VALUE_TIEBREAK','CACHED_REFRESH_MASS_TOLERANCE','CACHED_REFRESH_DRIFT_THRESHOLD','CACHED_REFRESH_PHASE_PROTECTED','CACHED_REFRESH_FREQUENCY','KV_COPY','MERGE','VALUE_AUDIT','VALUE_CLEAN_TIMING') if k in env},
        stable_refresh_config=stable_config, threshold=args.threshold, local_scope='native_dense', model_revision='f7f5b7f5fa82ffc52addd066915886d497f5517b')
    (attempt/'config.json').write_text(json.dumps(freeze,indent=2)+'\n')
    start = time.perf_counter()
    with (attempt/'private/worker.log').open('w') as log:
        result = subprocess.run([args.python, str(root/'scripts/v31_vllm_paired_bench.py'),
            args.model, args.manifests, args.cells, str(attempt/'records.jsonl'),
            str(attempt/'private'/('run_'+args.arm+'.private.jsonl')), arm, graph], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    rows = [json.loads(line) for line in (attempt/'records.jsonl').read_text().splitlines()] if (attempt/'records.jsonl').exists() else []
    errors = []
    if any(hashlib.sha256((root/f).read_bytes()).hexdigest()!=expected for f,expected in freeze['source_sha256'].items()):
        errors.append('source changed during generation')
    for row in rows:
        if row.get('cuda_graph_captures') != 0:
            errors.append('timed CUDA graph capture')
        if row.get('local_kv_budget') is not None:
            errors.append('unexpected LOCAL routing')
        if row.get('value_selector') != selector:
            errors.append('selector identity mismatch')
        if row.get('value_clean_timing')!=(args.purpose=='clean') or row.get('value_audit')!=(args.purpose!='clean'):
            errors.append('clean/audit mode mismatch')
        if args.purpose!='clean':
            audit=row['receipts']['adapter']
            expected_mode='FULL' if args.arm=='dense_full_fix51994' else 'PIECEWISE'
            modes=audit.get('decode_cudagraph_modes',{})
            if (sum(modes.values())!=row['denoise_forwards']
                    or any(mode.split('.')[-1]!=expected_mode for mode in modes)):
                errors.append('unexpected actual decode CUDA graph path')
            if audit.get('native_local_calls_audited')!=25*row['denoise_forwards']:
                errors.append('native LOCAL layer scope mismatch')
            if audit.get('global_calls')!=5*row['denoise_forwards']:
                errors.append('GLOBAL layer scope mismatch')
            if audit.get('value_nonfinite_logits_calls'):
                errors.append('nonfinite native logits')
        if arm in ('mage','allkept') and args.purpose != 'clean':
            audit = row['receipts']['adapter']
            if audit.get('order_errors') or audit.get('value_nonfinite_attention_calls'):
                errors.append('attention output or lifecycle validation failure')
            if audit.get('native_local_calls_audited') != 25*row['denoise_forwards']:
                errors.append('native LOCAL layer scope mismatch')
            phases = audit.get('value_phase_tiles',{})
            if sum(x['eligible_tiles'] for x in phases.values()) != audit.get('global_eligible_tiles'):
                errors.append('GLOBAL phase denominator mismatch')
            if sum(x['kept_tiles'] for x in phases.values()) != audit.get('global_kept_tiles'):
                errors.append('GLOBAL phase kept accounting mismatch')
            if selector and (audit.get('value_scope') != 'global_only' or audit.get('value_local_router_installed')):
                errors.append('value selector layer scope mismatch')
        if args.arm in CACHED_REFRESH_CONFIG and args.purpose != 'clean':
            policy, frontier_m, swaps, decay_lambda, decay_floor = CACHED_REFRESH_CONFIG[args.arm]
            if audit.get('cached_refresh_policy') != policy:
                errors.append('cached refresh policy identity mismatch')
            if audit.get('cached_refresh_frontier_m') != frontier_m:
                errors.append('cached refresh frontier mismatch')
            if audit.get('cached_refresh_swap_budget') != swaps:
                errors.append('cached refresh swap budget mismatch')
            if audit.get('cached_refresh_decay_lambda') != decay_lambda:
                errors.append('cached refresh decay mismatch')
            if audit.get('cached_refresh_decay_floor') != decay_floor:
                errors.append('cached refresh floor mismatch')
            if audit.get('native_local_calls_audited') != 25*row['denoise_forwards']:
                errors.append('cached refresh LOCAL routing changed')
            if audit.get('cached_refresh_fallbacks', 0) < 0:
                errors.append('cached refresh fallback counter invalid')
    if stable_config is not None:
        for row in rows:
            audit = row['receipts']['adapter']
            if audit.get('stable_refresh_config') != stable_config:
                errors.append('stable configuration mismatch')
            reasons = audit.get('stable_refresh_reasons', {})
            if not stable_config['progress_dense'] and reasons.get('progress_selection_dense', 0):
                errors.append('progress dense was not disabled')
            if reasons.get('emergency_dense', 0):
                errors.append('unexpected unlabelled dense rescue')
            if stable_config['selector']['mode'] == 'stable' and audit.get('stable_max_cumulative_swaps', 0) > stable_config['selector'].get('cumulative_cap', 8):
                errors.append('cumulative stability budget violation')
            if not stable_config['selector']['updates_enabled'] and reasons.get('actual_updates', 0):
                errors.append('updates-disabled changed a map')
    receipt = dict(status='complete' if result.returncode==0 and len(rows)==len(cells) else 'failed',
        returncode=result.returncode, records=len(rows), planned=len(cells), gpu_seconds=time.perf_counter()-start,
        timed_cuda_captures=sum(r.get('cuda_graph_captures', 0) for r in rows), source_commit=freeze['source_commit'])
    receipt['validation_errors'] = sorted(set(errors))
    if errors:
        receipt['status'] = 'failed_validation'
    (attempt/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if receipt['status']=='complete':
        (attempt/'attempt_complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    raise SystemExit(result.returncode or (receipt['status']!='complete'))


if __name__=='__main__':
    main()
