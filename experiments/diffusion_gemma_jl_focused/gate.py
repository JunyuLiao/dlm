"""Bind successful focused CPU and real CUDA tests to the exact source version."""
import xml.etree.ElementTree as ET

from experiments.diffusion_gemma_value_aware_followup.evidence import check_sources
from .protocol import ROOT,read,sha,frozen_write,execution


def record(root=ROOT):
    expected={'focused_tests.xml':dict(tests=5,failures=0,errors=0,skipped=0),
              'cuda_kernel_tests.xml':dict(tests=29,failures=0,errors=0,skipped=3)}
    sources={}
    for filename,counts in expected.items():
        path=root/filename
        suites=ET.parse(path).getroot().findall('testsuite')
        actual={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in counts}
        if actual!=counts:raise ValueError(f'Incomplete test gate {filename}: {actual}')
        sources[str(path)]=sha(path.read_bytes())
    from pathlib import Path
    for path in (Path('tests/test_jl_focused.py'),Path('tests/test_jl_output_aware.py')):
        sources[str(path)]=sha(path.read_bytes())
    contract=execution(root)
    result=dict(passed=True,fingerprint=contract['fingerprint'],tests=expected,
        sources={**contract['sources'],**sources},
        note='5 focused CPU tests plus26 passing real CUDA kernel tests;3 inapplicable high-dimensional guard cases skipped. Unchanged prior tensor proof additionally covers cache/masks/GQA/reference semantics.')
    frozen_write(root/'test_gate.json',result)
    return result


def require(root,contract):
    proof=read(root/'test_gate.json')
    if not proof['passed'] or proof['fingerprint']!=contract['fingerprint']:
        raise ValueError('Focused test proof fingerprint mismatch')
    check_sources(proof['sources'])
    return proof


if __name__=='__main__':
    print(record())
