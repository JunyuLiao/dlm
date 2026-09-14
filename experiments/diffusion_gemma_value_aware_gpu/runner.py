"""Reuse native model generation; isolate accelerated shards and provenance."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time

import torch
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _install_dense, _set_context, _request, _write, _append
from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_value_aware_followup.engine import check_result
from experiments.diffusion_gemma_value_aware_followup.evidence import read_result, DECODING_FIELDS
from experiments.diffusion_gemma_value_aware_followup.protocol import score, MODEL, REVISION
from .protocol import ROOT, PARENT, CONFIGS, METHODS, prepare, execution, sha
from .routing import Attention


def generate(adapter, row, config=None, thresholds=None, validate=False, reference=False):
    binding = _install_dense(adapter)
    if reference:
        from experiments.diffusion_gemma_value_aware.routing import Attention as Reference
        router = Reference(config, thresholds)
    else:
        router = Attention(config, thresholds, validate=validate)
    binding.runtime.attention_override = router
    try:
        _set_context(binding, row)
        output = adapter.generate(_request(row))
    finally:
        binding.close()
    expected = row['prompt_tokens']
    if (output.prompt_tokens != expected if isinstance(expected, list) else len(output.prompt_tokens) != expected):
        raise ValueError('Cached prompt tokens changed')
    records = router.records
    if {r['layer'] for r in records} != set(range(30)) or {r['head'] for r in records} != set(range(16)):
        raise ValueError('Incomplete model layer/head coverage')
    result = dict(id=row['id'], prompt_hash=row['prompt_hash'], seed=row['seed'], generation_budget=row['generation_budget'],
        config=config or {}, thresholds=thresholds, screen=False, prediction=output.text,
        completion_tokens=output.completion_tokens, generation_metadata=output.metadata,
        termination_reason=output.termination_reason, records=records, distributions=[],
        finite_calls=router.calls, metadata_cache=dict(reused_blocks=router.cache.reused_blocks,
        refreshed_blocks=router.cache.refreshed_blocks), score=score(row, output.text),
        backend='reference' if reference else 'custom_gpu_routing_native_attention',
        guard_fallback_calls=getattr(router, 'fallback_calls', 0),
        kernel_validation=getattr(router, 'validation', []))
    return result


def cached(adapter, root, row, stage, name, config, thresholds, contract):
    path = shard_path(root, stage, name, row['id'])
    if path.exists():
        out = json.loads(path.read_text())
        check_result(out, row, contract['fingerprint'], config, thresholds)
        return out
    if adapter is None:
        raise FileNotFoundError(path)
    _write(root/'progress.json', dict(pid=os.getpid(), stage=stage, condition=name, id=row['id'],
                                    path=str(path), started=time.time()))
    out = generate(adapter, row, config, thresholds)
    out['fingerprint'] = contract['fingerprint']
    check_result(out, row, contract['fingerprint'], config, thresholds)
    _write(path, out)
    _append(root/'completed.jsonl', dict(stage=stage, condition=name, id=row['id'], path=str(path),
        finished=time.time(), backend=out['backend'], guard_fallback_calls=out['guard_fallback_calls']))
    print(time.strftime('%FT%TZ', time.gmtime()), stage, name, row['id'], 'complete', flush=True)
    return out


def dense(adapter, root, row, contract):
    path = shard_path(root, 'dense', 'dense', row['id'])
    if path.exists():
        return cached(None, root, row, 'dense', 'dense', {}, None, contract)
    parent = json.loads((PARENT/'execution_contract.json').read_text())
    try:
        out, source = read_result(PARENT, row, 'dense', 'dense', {}, None, parent)
    except FileNotFoundError:
        return cached(adapter, root, row, 'dense', 'dense', {}, None, contract)
    out = dict(out, fingerprint=contract['fingerprint'], imported_dense=source,
               backend='audited_cached_native_dense', guard_fallback_calls=0)
    check_result(out, row, contract['fingerprint'], {}, None)
    _write(path, out)
    _append(root/'completed.jsonl', dict(stage='dense', condition='dense', id=row['id'],
        path=str(path), finished=time.time(), backend=out['backend']))
    return out


def smoke(adapter, root, setup, contract):
    path = root/'validation'/'smoke.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['fingerprint'] == contract['fingerprint'] and previous['passed']:
            for source, digest in previous['sources'].items():
                if sha(Path(source).read_bytes()) != digest:
                    raise ValueError('CUDA validation source changed')
            return previous
    examples = [next(r for r in setup['calibration'] if r['benchmark'] == 'aime26'),
                max((r for r in setup['calibration'] if r['benchmark'] == 'longbench_v2'), key=lambda r:len(r['prompt_tokens']))]
    proposals = json.loads((PARENT/'threshold_proposals.json').read_text())['policies']
    cases, sources = [], {}
    for example in examples:
        row = dict(example, generation_budget=16)
        binding = _install_dense(adapter)
        try:
            _set_context(binding, row)
            native = adapter.generate(_request(row))
        finally:
            binding.close()
        conditions = [('dense', {}, None)]
        for name in METHODS:
            policy = json.loads((PARENT/'verified_policies'/row['benchmark']/f'{name}_s50.json').read_text())['policy']
            conditions.append((f'{name}_s50', CONFIGS[name], policy))
            probe = 'blasst' if name.startswith('blasst_') else name+('_'+CONFIGS[name]['pooling'] if name in ('value','risk') else '')
            candidate = {}
            for kind in ('local','global'):
                entry = proposals[row['benchmark']][probe][kind]
                if probe == 'blasst':
                    candidate[kind] = dict(log_scale=entry['targets']['0.75']['log_scale'], cap_one=name == 'blasst_original')
                else:
                    candidate[kind] = dict(log_threshold=entry['0.75']['log_threshold'])
            conditions.append((f'{name}_s75_proposal', CONFIGS[name], candidate))
        for name, config, policy in conditions:
            out = generate(adapter, row, config, policy, validate=True)
            if not config and out['completion_tokens'] != native.completion_tokens:
                raise AssertionError('Native dense token parity failed')
            for key in DECODING_FIELDS:
                if out['generation_metadata'].get(key) != native.metadata.get(key):
                    raise AssertionError(f'Unrelated decoding setting changed: {key}')
            parity = not config
            # Additional independent end-to-end reference replays for both
            # benchmark prompts and both50% leading mechanisms.
            if name in ('value_s50', 'mass_s50'):
                ref = generate(adapter, row, config, policy, reference=True)
                if out['completion_tokens'] != ref['completion_tokens']:
                    raise AssertionError('Sparse generated-token parity failed')
                refpath = shard_path(root, 'validation', name+'_reference', row['id'])
                _write(refpath, ref)
                sources[str(refpath)] = sha(refpath.read_bytes())
                parity = True
            dest = shard_path(root, 'validation', name, row['id'])
            out['fingerprint'] = contract['fingerprint']
            _write(dest, out)
            sources[str(dest)] = sha(dest.read_bytes())
            cases.append(dict(id=row['id'], condition=name, passed=True,
                exact_generation_parity=parity, reference_masks_and_diagnostics_checked=len(out['kernel_validation']),
                guard_fallback_calls=out['guard_fallback_calls'], finite_calls=out['finite_calls'],
                prefix_eligible=sum(r['prefix_eligible'] for r in out['records']),
                canvas_eligible=sum(r['canvas_eligible'] for r in out['records'])))
            print('CUDA model validation', row['id'], name, 'passed', flush=True)
            _write(root/'validation'/'progress.json', dict(cases=cases, expected=26))
            torch.cuda.empty_cache()
    if len(cases) != 26 or any(not c['prefix_eligible'] or not c['canvas_eligible'] for c in cases):
        raise AssertionError('Incomplete real-model coverage')
    result = dict(passed=True, fingerprint=contract['fingerprint'], expected=26, cases=cases, sources=sources,
                  native_output_math_unchanged=True, sparse_reference_replays=4,
                  tested75_thresholds='Calibration-only proposals; final75 policy verification follows')
    _write(path, result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('smoke',))
    p.add_argument('--output', type=Path, default=ROOT)
    args = p.parse_args()
    setup = prepare(args.output)
    contract = execution(args.output)
    from dllm.models import create_adapter
    adapter = create_adapter('diffusion_gemma', MODEL, device='cuda', precision='bfloat16', revision=REVISION).load()
    smoke(adapter, args.output, setup, contract)


if __name__ == '__main__':
    main()
