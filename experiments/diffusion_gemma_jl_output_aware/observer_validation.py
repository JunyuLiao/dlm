"""Check that shared-state capture leaves native dense generation unchanged."""
import json
from pathlib import Path

from experiments.diffusion_gemma_value_aware.protocol import sha, frozen_write
from experiments.diffusion_gemma_value_aware_followup.evidence import check_sources
from .runner import cached
from .diagnostics import SnapshotObserver


def validate_observer(adapter, root, setup, contract, smoke):
    folder = root/'validation'/contract['fingerprint']
    path = folder/'observer.json'
    if path.exists():
        result = json.loads(path.read_text())
        check_sources(result['sources'])
        if result['fingerprint'] != contract['fingerprint'] or not result['passed']:
            raise ValueError('Dense observer validation identity changed')
        return result
    native_folder = Path(smoke['imported_validation']['path']).parent if smoke.get('imported_validation') else folder
    sources = {str(Path(__file__)): sha(Path(__file__).read_bytes())}
    cases = []
    examples = [next(r for r in setup['calibration'] if r['benchmark'] == 'aime26'),
                max(setup['development'], key=lambda r:len(r['prompt_tokens']))]
    for original in examples:
        row = dict(original, generation_budget=16)
        native_path = native_folder/(sha(row['id'])+'.native.json')
        native = json.loads(native_path.read_text())
        sources[str(native_path)] = sha(native_path.read_bytes())
        observer = SnapshotObserver(folder/'observer_capture', row, 0)
        out = cached(adapter, folder, row, 'observer', 'dense', 'dense', {}, None, contract, observer=observer)
        if out['completion_tokens'] != native['completion_tokens']:
            raise AssertionError('Dense observation changed generated tokens')
        for key in ('native_canvas_length', 'thinking', 'sampling', 'denoising_configuration'):
            if out['generation_metadata'].get(key) != native['generation_metadata'].get(key):
                raise AssertionError('Dense observation changed decoding settings')
        states = out['shared_state_sources']
        if {s['layer'] for s in states} != set(range(30)) or {s['attention_type'] for s in states} != {'local', 'global'}:
            raise AssertionError('Shared observer lacks complete layer/type coverage')
        sources.update({s['path']: s['sha256'] for s in states})
        cases.append(dict(id=row['id'], passed=True, native_dense_parity=True, snapshots=len(states)))
    result = dict(passed=True, fingerprint=contract['fingerprint'], sources=sources, cases=cases)
    frozen_write(path, result)
    return result
