"""Read-only diagnostic of the persistent causal flag in frozen native runs."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('default', 'PIECEWISE'), default='default')
    args = p.parse_args()
    study, out = args.study.resolve(), args.output.resolve()
    if out.exists():
        raise RuntimeError('new diagnostic directory required')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU occupied')
    out.mkdir(parents=True)
    (out/'private').mkdir()
    protocol = json.loads((study/'specs/protocol.json').read_text())
    execution = study/'private/execution'
    for rel, expected in protocol['execution_sha256'].items():
        if hashlib.sha256((execution/rel).read_bytes()).hexdigest() != expected:
            raise RuntimeError('frozen source drift')
    for key in list(os.environ):
        if key.startswith(('MAGE_', 'VALUE_', 'STABLE_', 'CACHED_REFRESH_', 'LOCAL_')):
            os.environ.pop(key)
    cache = study/'private/cache'
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0', PYTHONNOUSERSITE='1',
        HF_HUB_OFFLINE='1', CUDA_VISIBLE_DEVICES='0', FIX_51994='1', FA4_LOCAL_FIX='1',
        FA4_LOCAL_FIX_DIR=str(cache/'fa4_local_fix'), VLLM_CACHE_ROOT=str(cache/'vllm'),
        TRITON_CACHE_DIR=str(cache/'triton'), TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),
        CUDA_CACHE_PATH=str(cache/'cuda'), TMPDIR=str(cache/'tmp'),
        FLASHINFER_WORKSPACE_BASE=str(cache/'flashinfer'),
        BLOCK='32', CHUNK='16384', MEM='.80', MAX_MODEL_LEN='141312', SEED_BASE='31',
        VALUE_AUDIT='1', VALUE_CLEAN_TIMING='0')
    for key in ('FORCE_REF', 'FORCE_RECORD', 'CANVAS_RESEED', 'LIMIT', 'SHARD', 'REPEATS', 'DATASETS'):
        os.environ.pop(key, None)
    sys.path[:0] = [str(execution/'src'), str(execution)]
    import torch
    import vllm.model_executor.models.diffusion_gemma as dg
    original = dg.DiffusionGemmaModelState.prepare_attn
    counts = dict(capture_calls=0, runtime_calls=0, metadata_checks=0, alias_errors=0, dtype_errors=0)

    def prepare(state, *a, **kw):
        metadata = original(state, *a, **kw)
        counts['capture_calls' if kw.get('for_capture', False) else 'runtime_calls'] += 1
        if state._causal_buf.dtype != torch.int32:
            counts['dtype_errors'] += 1
            raise RuntimeError('causal buffer is not int32')
        for value in metadata.values():
            causal = value.causal
            counts['metadata_checks'] += 1
            if not isinstance(causal, torch.Tensor) or causal.dtype != torch.int32:
                counts['dtype_errors'] += 1
                raise RuntimeError('causal metadata is not int32')
            if causal.data_ptr() != state._causal_buf.data_ptr():
                counts['alias_errors'] += 1
                raise RuntimeError('metadata detached from persistent causal buffer')
        return metadata

    dg.DiffusionGemmaModelState.prepare_attn = prepare
    bench = execution/'scripts/v31_vllm_paired_bench.py'
    spec = importlib.util.spec_from_file_location('frozen_dense_bench', bench)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = [str(bench), protocol['model'], str(study/'private/manifests'),
                str(study/'private/cells_qualification02.json'), str(out/'records.jsonl'),
                str(out/'private/completions.jsonl'), 'dense', args.mode]
    status = 'failed'
    try:
        mod.main()
        if not counts['capture_calls'] or not counts['runtime_calls'] or not counts['metadata_checks']:
            raise RuntimeError('capture/runtime causal checks did not run')
        status = 'complete'
    finally:
        receipt = dict(status=status, mode=args.mode, source_commit=protocol['source_commit'],
            bench_sha256=hashlib.sha256(bench.read_bytes()).hexdigest(),
            diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            persistent_causal_alias_required=True, **counts)
        (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
