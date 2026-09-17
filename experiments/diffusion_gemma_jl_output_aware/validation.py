"""Actual-model CUDA validation before threshold fitting or final deployment."""
import argparse
import json
from pathlib import Path
import shutil

import torch

from dllm.models import create_adapter
from experiments.diffusion_gemma_value_aware.protocol import frozen_write, sha
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _install_dense, _set_context, _request, _write, _fingerprint
from .protocol import ROOT, MODEL, REVISION, prepare, execution
from .config import PROJECTED
from .runner import cached


def smoke(adapter, root, setup, contract):
    folder = root/'validation'/contract['fingerprint']
    summary = folder/'smoke.json'
    if summary.exists():
        out = json.loads(summary.read_text())
        if out['passed'] and out['fingerprint'] == contract['fingerprint']:
            return out
    # A diagnostic-only extension cannot change a smoke run with observer=None.
    # Reuse only when reconstructing the full old execution fingerprint proves
    # identical setup/runtime and every non-diagnostic source is byte-identical.
    for previous_path in sorted((root/'validation').glob('*/smoke.json')):
        old = json.loads(previous_path.read_text())
        if not old.get('passed') or len(old.get('cases', [])) != 4*len(PROJECTED):
            continue
        before = dict(contract, sources=old['sources'])
        before.pop('fingerprint')
        changed = {p for p in set(old['sources']) | set(contract['sources'])
                   if old['sources'].get(p) != contract['sources'].get(p)}
        allowed = {str(Path(__file__).with_name('diagnostics.py'))}
        if changed <= allowed and _fingerprint(before) == old['fingerprint']:
            imported = dict(old, fingerprint=contract['fingerprint'], sources=contract['sources'],
                imported_validation=dict(path=str(previous_path), sha256=sha(previous_path.read_bytes()),
                    original_fingerprint=old['fingerprint'], changed_sources=sorted(changed),
                    reason='Only diagnostic observer/analysis changed; this smoke used observer=None; all routing, kernels, projection, native generation, setup and runtime sources are identical'))
            frozen_write(summary, imported)
            return imported
    # Preserve source versions used by an unsuccessful development implementation.
    for path, digest in contract['sources'].items():
        source = Path(path)
        dest = folder/'sources'/source.resolve().relative_to(Path.cwd())
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
        if sha(dest.read_bytes()) != digest:
            raise ValueError('Archived validation source mismatch')
    examples = [next(r for r in setup['calibration'] if r['benchmark'] == 'aime26'),
                max(setup['development'], key=lambda r:len(r['prompt_tokens']))]
    cases = []
    for original in examples:
        row = dict(original, generation_budget=16)
        native_path = folder/(sha(row['id'])+'.native.json')
        if native_path.exists():
            native = json.loads(native_path.read_text())
        else:
            binding = _install_dense(adapter)
            try:
                _set_context(binding, row)
                generated = adapter.generate(_request(row))
            finally:
                binding.close()
            native = dict(completion_tokens=generated.completion_tokens, generation_metadata=generated.metadata)
            frozen_write(native_path, native)
        for name, config in PROJECTED.items():
            for unpruned in (True, False):
                # Explicit tau0 maps to log -inf in memory, retaining even
                # exact zero-risk ties. JSON policy scalars remain finite.
                policy = {kind:dict(log_threshold=-1000., unpruned=True, tau=0.) if unpruned
                          else dict(log_threshold=-2.) for kind in ('local', 'global')}
                label = f'{name}/{"unpruned" if unpruned else "pruned"}'
                result = cached(adapter, folder, row, 'smoke', label, name, config, policy, contract, validate=True)
                if unpruned and result['completion_tokens'] != native['completion_tokens']:
                    raise AssertionError(f'Native unpruned generation parity: {name}/{row["id"]}')
                if unpruned and any(r['skipped'] for r in result['records']):
                    raise AssertionError('Unpruned validation unexpectedly skipped tiles')
                for key in ('native_canvas_length', 'thinking', 'sampling', 'denoising_configuration'):
                    if result['generation_metadata'].get(key) != native['generation_metadata'].get(key):
                        raise AssertionError(f'Unrelated decoding setting changed: {key}')
                exact_generation = None
                if not unpruned:
                    reference_result = cached(adapter, folder, row, 'reference', label, name, config,
                                              policy, contract, trusted=True)
                    exact_generation = result['completion_tokens'] == reference_result['completion_tokens']
                    if not exact_generation:
                        raise AssertionError(f'Projected GPU/trusted generation mismatch: {name}/{row["id"]}')
                    for kind in ('local', 'global'):
                        for key in ('eligible', 'skipped'):
                            a = sum(r[key] for r in result['records'] if r['attention_type'] == kind)
                            b = sum(r[key] for r in reference_result['records'] if r['attention_type'] == kind)
                            if a != b:
                                raise AssertionError('Projected GPU/trusted physical counts differ')
                cases.append(dict(id=row['id'], name=name, unpruned=unpruned, passed=True,
                    native_dense_parity=True if unpruned else None, trusted_generation_parity=exact_generation,
                    checked_calls=len(result['kernel_validation']), finite_calls=result['finite_calls'],
                    prefix_skipped=sum(r['prefix_skipped'] for r in result['records']),
                    canvas_skipped=sum(r['canvas_skipped'] for r in result['records'])))
                _write(folder/'progress.json', dict(cases=cases, expected=2*2*len(PROJECTED)))
                print('validated', row['id'], label, flush=True)
    out = dict(passed=True, fingerprint=contract['fingerprint'], cases=cases,
               expected=2*2*len(PROJECTED), sources=contract['sources'])
    frozen_write(summary, out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    setup = prepare(args.output)
    contract = execution(args.output)
    adapter = create_adapter('diffusion_gemma', MODEL, device='cuda', precision='bfloat16', revision=REVISION).load()
    print(json.dumps(smoke(adapter, args.output, setup, contract)))


if __name__ == '__main__':
    main()
