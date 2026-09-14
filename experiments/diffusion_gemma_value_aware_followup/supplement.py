"""Collect missing base-pooling and matched no-value calibration diagnostics.

The prior JointScreen covers refinements and guarded rankings, not base pools.
This separately versioned observer replays tuning examples only, requiring
exact parity with their cached dense tokens. Final dense runs are untouched.
"""
import argparse
import json
from pathlib import Path
import time
import traceback
from unittest.mock import patch

from experiments.diffusion_gemma_value_aware import run as original
from experiments.diffusion_gemma_value_aware import routing
from experiments.diffusion_gemma_value_aware.operators import Config, screen_risks
from experiments.diffusion_gemma_value_aware.report import inspect_output
from experiments.diffusion_gemma_value_aware.protocol import sha
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write, _append
from .controls import CONFIG, unity_metadata, source_sha as control_sha
from .analysis import screen_source
from .engine import contract
from .protocol import ROOT, MODEL, REVISION, prepare, score


def source_sha():
    return sha(Path(__file__).read_bytes())


class SupplementalScreen(routing.Attention):
    def _screen(self, state, meta, q, k, valid, scale, layer, step, kind, prefix, length):
        super()._screen(state, meta, q, k, valid, scale, layer, step, kind, prefix, length)
        c = Config(**CONFIG)
        risks, _ = screen_risks(state, unity_metadata(meta), c)
        self.risk_arrays.setdefault(f'no_value_control__{kind}', []).append(
            risks[state['eligible']].float().cpu().numpy())
        for target in (.25, .5, .75, .9):
            keep, empty = routing.rescue(routing.select(risks, state['eligible'], 'topk', target), state)
            self.record(state, state['eligible'] & ~keep, c, meta, layer, step, kind,
                f'screen/no_value_control/s{int(target*100)}', prefix, empty)


def audit_raw(data, row, execution, dense):
    if row['split'] not in ('calibration', 'development'):
        raise ValueError('supplemental observer cannot use final examples')
    if (data.get('supplement_sha256') != source_sha()
            or data.get('control_sha256') != control_sha()):
        raise ValueError('supplemental observer source changed')
    if not data['screen']:
        raise ValueError('supplemental diagnostics missing screen flag')
    _, _, errors = inspect_output(data, row, execution['fingerprint'],
        dict(config={}, thresholds={row['benchmark']: None}))
    if errors:
        raise ValueError(f'supplemental raw audit: {errors}')
    if data['completion_tokens'] != dense['completion_tokens']:
        raise ValueError('supplemental observer changed cached dense tokens')
    probes = {r['probe'] for r in data['records']}
    if not all(f'screen/{name}/s50' in probes for name in
            ('value_max', 'mass_value_rms', 'risk_vector_mean', 'no_value_control')):
        raise ValueError('supplemental pooling/control probes missing')


def source(root, row, execution, dense):
    if row['split'] not in ('calibration', 'development'):
        raise ValueError('supplemental observer cannot use final examples')
    path = original.shard_path(root, 'supplemental_screen', 'dense', row['id'])
    raw = path.read_bytes(); data = json.loads(raw)
    audit_raw(data, row, execution, dense)
    arrays = path.with_suffix('.npz')
    return data, arrays, dict(path=str(path), sha256=sha(raw), arrays_path=str(arrays),
        arrays_sha256=sha(arrays.read_bytes()), split=row['split'])


def execute(root):
    import torch
    from dllm.models import create_adapter
    setup = prepare(root); execution = contract(root)
    imports = json.loads((root/'imported_sources.json').read_text())
    rows = sorted(setup['calibration']+setup['development'], key=lambda r:(r['generation_budget'], r['id']))
    adapter = None; failures = []; sources = {}
    for row in rows:
        try:
            dense, _, _ = screen_source(root, row, execution, imports)
            path = original.shard_path(root, 'supplemental_screen', 'dense', row['id'])
            if not path.exists():
                if adapter is None:
                    adapter = create_adapter('diffusion_gemma', MODEL, device='cuda',
                        precision='bfloat16', revision=REVISION).load()
                original_generate = original.generate
                def generate(*args, **kwargs):
                    with patch.object(original, 'Attention', SupplementalScreen):
                        data, arrays = original_generate(*args, **kwargs)
                    data.update(supplement_sha256=source_sha(), control_sha256=control_sha())
                    audit_raw(dict(data, fingerprint=execution['fingerprint']), row, execution, dense)
                    return data, arrays
                with patch.object(original, 'score', score), patch.object(original, 'generate', generate):
                    original.cache(adapter, row, path, execution['fingerprint'], screen=True, root=root)
            _, _, sources[row['id']] = source(root, row, execution, dense)
        except Exception:
            error = dict(stage='supplemental_screen', id=row['id'], traceback=traceback.format_exc())
            failures.append(error); _append(root/'failures.jsonl', error); print(error, flush=True)
        finally:
            torch.cuda.empty_cache()
    _write(root/'supplemental_audit.json', dict(complete=not failures and len(sources)==18,
        expected=18, sources=sources, failures=failures, fingerprint=execution['fingerprint'],
        supplement_sha256=source_sha(), control_sha256=control_sha(), finished=time.time(),
        cached_dense_token_parity=True, heldout_used=False))
    if failures:
        raise RuntimeError('supplemental diagnostics incomplete; independent examples continued')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, default=ROOT)
    args = p.parse_args(); execute(args.output)


if __name__ == '__main__':
    main()
