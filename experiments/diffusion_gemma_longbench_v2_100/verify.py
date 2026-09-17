"""Independent CPU-only report regeneration, following the parent study audit."""
import argparse
import json
from pathlib import Path

from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
from .protocol import ROOT, sha
from .report import regenerate


def verify(root=ROOT):
    before = json.loads((root / 'audit.json').read_text())
    if not before['complete'] or before['completed'] != 900:
        raise ValueError('A complete 900-result report is required before verification')
    for name, expected in before['artifacts'].items():
        if sha((root / name).read_bytes()) != expected:
            raise ValueError(f'Reported artifact changed before verification: {name}')
    after = regenerate(root)
    if not after['complete'] or before != after:
        raise ValueError('Raw-only regeneration changed the audit or artifact hashes')
    proof = dict(
        passed=True, completed=900, inference_performed=False,
        identical_audit_and_artifact_hashes=True,
        artifact_count=len(after['artifacts']),
        audit_sha256=sha((root / 'audit.json').read_bytes()),
        reporting_sources={str(p): sha(p.read_bytes()) for p in (
            Path(__file__), Path(__file__).with_name('report.py'),
            Path(__file__).with_name('nemo.py'))})
    _write(root / 'regeneration_verification.json', proof)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT)
    print(json.dumps(verify(parser.parse_args().output)))
