"""Independent CPU-only audit/report reproduction; never initiates inference."""
import argparse
import json
from pathlib import Path

from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
from .protocol import ROOT, sha
from .report import regenerate


def verify(root=ROOT):
    before = json.loads((root/'audit.json').read_text())
    if not before['complete'] or before['completed'] != 2960:
        raise ValueError('A complete 2960-result report plus supporting checks is required')
    for name, expected in before['artifacts'].items():
        if sha((root/name).read_bytes()) != expected:
            raise ValueError(f'Reported artifact changed before verification: {name}')
    after = regenerate(root)
    if not after['complete'] or before != after:
        raise ValueError('Raw-only regeneration changed audit or artifacts')
    proof = dict(passed=True, completed=2960, inference_performed=False,
        identical_audit_and_artifact_hashes=True, artifact_count=len(after['artifacts']),
        audit_sha256=sha((root/'audit.json').read_bytes()),
        reporting_sources={str(p):sha(p.read_bytes()) for p in (Path(__file__), Path(__file__).with_name('report.py'))})
    _write(root/'regeneration_verification.json', proof)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT)
    print(json.dumps(verify(parser.parse_args().output)))
