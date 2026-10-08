"""Immutable single-worker launch wrapper for the independent selector study.

Uses supplied frozen manifests and cells verbatim. Does not build prompts,
change budgets, score answers, or reuse a killed attempt directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
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
}


def main():
    p = argparse.ArgumentParser()
    for name in ('python', 'model', 'manifests', 'cells', 'attempt'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--arm', required=True, choices=ARM_CONFIG)
    p.add_argument('--threshold', type=float)
    p.add_argument('--max-model-len', type=int, default=141312)
    p.add_argument('--purpose', choices=('smoke', 'development', 'audit', 'clean'), required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    result_root = root/'results/v31_value_selectors_20261008'
    attempt = Path(args.attempt).resolve()
    if not attempt.is_relative_to(result_root) or attempt.exists():
        raise ValueError('a new attempt below the canonical study result root is required')
    if args.arm.startswith(('value_v1_', 'value_v2_')) and args.threshold is None:
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
    for key in ('LOCAL_KV_BUDGET', 'LOCAL_KERNEL', 'VALUE_SELECTOR', 'VALUE_THRESHOLD',
                'MAGE_FRAC', 'MAGE_POOL', 'MAGE_ROWW', 'MAGE_RESELECT', 'MAGE_RESELECT_K',
                'MAGE_RESELECT_KMIN', 'MAGE_KCOVER', 'RESIDUAL', 'DROP_GUARD', 'TRACE',
                'DRIFT_DIAG', 'CG_STOP', 'STALL_RESCUE', 'FORCE_REF', 'FORCE_RECORD', 'LIMIT', 'SHARD'):
        env.pop(key, None)
    env.update(PYTHONPATH=f'{root}/src:{root}', PYTHONNOUSERSITE='1',
        HF_HUB_OFFLINE='1', VLLM_ENABLE_V1_MULTIPROCESSING='0', CUDA_VISIBLE_DEVICES='0',
        FIX_51994='1', FA4_LOCAL_FIX='1', FA4_LOCAL_FIX_DIR=str(cache/'fa4_local_fix'),
        VLLM_CACHE_ROOT=str(cache/'vllm'), TRITON_CACHE_DIR=str(cache/'triton'),
        TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'), CUDA_CACHE_PATH=str(cache/'cuda'),
        TMPDIR=str(cache/'tmp'), FLASHINFER_WORKSPACE_BASE=str(cache/'flashinfer'),
        BLOCK='32', CHUNK='16384', MEM='0.85', MAX_MODEL_LEN=str(args.max_model_len),
        SEED_BASE='31', MAGE_K='8192', MAGE_SELECT='fa4', MAGE_GRAN=granularity,
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
    env['VALUE_AUDIT'] = '1' if args.purpose in ('audit', 'smoke', 'development') else '0'
    if args.threshold is not None:
        env['VALUE_THRESHOLD'] = str(args.threshold)
    manifest_hashes = {c['dataset']: hashlib.sha256((Path(args.manifests)/(c['dataset']+'_generation_manifest.json')).read_bytes()).hexdigest() for c in cells}
    source_files = [root/'experiments/numerical_qk_reuse'/name for name in
        ('vllm_adapter.py', 'v31_value_selectors.py', 'v31_value_kernels.py', 'v27_fa4.py', 'v31_fa4_observe.py')]
    source_files += [root/'scripts/v31_vllm_paired_bench.py', root/'scripts/v31_value_campaign.py']
    freeze = dict(schema='independent_v31_attempt_v1', purpose=args.purpose, arm=args.arm,
        cell_count=len(cells), cells_sha256=hashlib.sha256(cells_bytes).hexdigest(),
        manifests_sha256=manifest_hashes, source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        source_sha256={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files},
        runtime_settings={k:env[k] for k in ('BLOCK','CHUNK','MEM','MAX_MODEL_LEN','MAGE_K','MAGE_STEP','MAGE_CARRY','MAGE_RESELECT_TRIGGER','MAGE_TRIGGER_SIGNAL','MAGE_GRAN')},
        threshold=args.threshold, local_scope='native_dense', model_revision='f7f5b7f5fa82ffc52addd066915886d497f5517b')
    (attempt/'config.json').write_text(json.dumps(freeze,indent=2)+'\n')
    start = time.perf_counter()
    with (attempt/'private/worker.log').open('w') as log:
        result = subprocess.run([args.python, str(root/'scripts/v31_vllm_paired_bench.py'),
            args.model, args.manifests, args.cells, str(attempt/'records.jsonl'),
            str(attempt/'private'/('run_'+args.arm+'.private.jsonl')), arm, graph], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    rows = [json.loads(line) for line in (attempt/'records.jsonl').read_text().splitlines()] if (attempt/'records.jsonl').exists() else []
    receipt = dict(status='complete' if result.returncode==0 and len(rows)==len(cells) else 'failed',
        returncode=result.returncode, records=len(rows), planned=len(cells), gpu_seconds=time.perf_counter()-start,
        timed_cuda_captures=sum(r.get('cuda_graph_captures', 0) for r in rows), source_commit=freeze['source_commit'])
    (attempt/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if receipt['status']=='complete':
        (attempt/'attempt_complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    raise SystemExit(result.returncode or (receipt['status']!='complete'))


if __name__=='__main__':
    main()
