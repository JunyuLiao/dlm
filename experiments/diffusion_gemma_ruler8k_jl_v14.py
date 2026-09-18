"""Version14: normalize fresh/cache-hit shard metadata before smoke auditing.

Preserves version13 and its failed smoke evidence. Numerical inference,
selection, calibration and reporting algorithms are byte-identical imports.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

from experiments import diffusion_gemma_ruler8k_jl as base

ROOT = Path('results/diffusion_gemma_ruler8k_jl130_v14')
MODULE = 'experiments.diffusion_gemma_ruler8k_jl_v14'


def execution(root=ROOT):
    import torch, transformers, triton
    setup = base.prepare(root)
    prior = base.read(base.ROOT/'execution_contract.json')
    base.evidence.check_sources(prior['sources'])
    runtime = dict(torch=torch.__version__, transformers=transformers.__version__, triton=triton.__version__)
    if runtime != prior['runtime']: raise ValueError('Frozen runtime changed')
    suites = ET.parse(root/'tests.xml').getroot().findall('testsuite')
    counts = {k: sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts != dict(tests=19, failures=0, errors=0, skipped=0): raise ValueError('Require complete19-test CUDA/CPU gate')
    files = [Path(__file__), Path('tests/test_ruler8k_jl_v14.py'), root/'tests.xml', root/'setup.json', base.ROOT/'execution_contract.json']
    sources = base.evidence.merge_sources(prior['sources'], {str(p):base.sha(p.read_bytes()) for p in files})
    data = dict(prior, schema='ruler8k_execution_v14', sources=sources,
        superseded_execution_fingerprint=prior['fingerprint'],
        change='Fresh cached() results are reloaded from their completed lossless shard before metadata audits; no numerical changes')
    data.pop('fingerprint'); data['fingerprint'] = base._fingerprint(data)
    base.frozen_write(root/'execution_contract.json', data)
    return data


@contextmanager
def complete_metadata():
    original = base.runner.cached
    def cached(adapter, root, row, stage, label, name, config, thresholds, contract, **kwargs):
        out = original(adapter, root, row, stage, label, name, config, thresholds, contract, **kwargs)
        if 'records_source' not in out:
            # write_output returns no value and intentionally leaves the fresh
            # in-memory result unchanged. Reload exactly what cache hits read.
            out = base.runner.load_output(base.shard_path(root, stage, label, row['id']))
            base.runner.check_result(out, row, contract['fingerprint'], config, thresholds)
        return out
    with patch.object(base.runner, 'cached', cached): yield


@contextmanager
def active():
    with patch.object(base, 'execution', execution), patch.object(base, 'MODULE', MODULE), complete_metadata():
        yield


def prepare(root=ROOT):
    return base.prepare(root)


def work(root=ROOT):
    with active(): return base.work(root)


def launch(root=ROOT):
    with active(): return base.launch(root)


def supervise(root=ROOT):
    with active(): return base.supervise(root)


def report(root=ROOT):
    with active(): return base.report(root)


def verify(root=ROOT):
    with active(): return base.verify(root)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=('prepare','launch','work','supervise','report','verify'))
    p.add_argument('--output', type=Path, default=ROOT); args = p.parse_args()
    result = globals()[args.command](args.output)
    if args.command in ('prepare','report','verify'): print(json.dumps(dict(command=args.command, complete=result.get('complete'), passed=result.get('passed'))))
