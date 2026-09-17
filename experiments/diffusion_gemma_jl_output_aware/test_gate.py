"""Freeze and require successful CUDA-test evidence before a model worker."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from experiments.diffusion_gemma_value_aware.protocol import frozen_write, sha
from experiments.diffusion_gemma_value_aware_followup.evidence import check_sources
from .protocol import ROOT, execution


def record(root=ROOT):
    path = root/'gpu_tests.xml'
    suites = ET.parse(path).getroot().findall('testsuite')
    totals = {k:sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    if totals != dict(tests=51, failures=0, errors=0, skipped=3):
        raise ValueError(f'Incomplete CUDA test matrix: {totals}')
    contract = execution(root)
    paths = [Path('tests/test_jl_output_aware.py'), Path('tests/test_jl_experiment.py'), path]
    paths += [Path(__file__).with_name(n) for n in ('screen.py','shared_analysis.py','trace_kernels.py','calibration.py')]
    sources = {**contract['sources'], **{str(p):sha(p.read_bytes()) for p in paths}}
    result = dict(passed=True, fingerprint=contract['fingerprint'], passed_tests=48, **totals,
        sources=sources, skips='Three inapplicable high-dimensional cancellation-guard combinations',
        scope='Real CUDA kernels plus CPU orchestration and full37x80 synthetic report roundtrip; synthetic scores are not benchmark evidence')
    frozen_write(root/'tensor_validation.json', result)
    return result


def require(root, contract):
    result = json.loads((root/'tensor_validation.json').read_text())
    if not result['passed'] or result['fingerprint'] != contract['fingerprint']:
        raise ValueError('CUDA test proof execution mismatch')
    check_sources(result['sources'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT)
    print(json.dumps(record(parser.parse_args().output)))
