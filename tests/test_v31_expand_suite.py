"""v31_expand_suite: S2 is disjoint from S1, deterministic, takes the stated items, and leaves the rest (CPU)."""
import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = next(p for p in (_HERE.parent / 'scripts' / 'v31_expand_suite.py', _HERE / 'v31_expand_suite.py') if p.exists())
spec = importlib.util.spec_from_file_location('v31_expand_suite', _SRC)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def _pools():
    ruler = [dict(dataset=f'ruler{L}k', index=i, id=f'ruler{L}k_t{t}_p{p:04d}')
             for L in (32, 64, 128) for i, (t, p) in enumerate((t, p) for t in range(13) for p in range(15))]
    lbt = [dict(dataset='longbench_v2_0shot_think', index=i, id=f'lb{i}') for i in range(160)]
    mrcr = [dict(dataset=f'mrcr{L}', index=i, id=f'm{L}_{i}') for L in (32, 64, 128) for i in range(24)]
    gw = [dict(dataset=f'gw{L}', index=i, id=f'g{L}_{i}') for L in (22, 45, 90) for i in range(24)]
    return dict(ruler=ruler, lbt=lbt, mrcr=mrcr, gw=gw)


def _s1(load):
    return dict(ruler=[c for c in load['ruler'] if c['id'].endswith(('_p0000', '_p0001'))],
                lbt=load['lbt'][:32], mrcr=[c for c in load['mrcr'] if c['index'] < 8],
                gw=[c for c in load['gw'] if c['index'] < 4])


def test_sizes_disjoint_and_rest():
    load = _pools()
    s1 = _s1(load)
    s2, rest = S.build(load, s1, [1, 2], 20261005)
    assert {k: len(v) for k, v in s2.items()} == dict(ruler=156, lbt=128, mrcr=24, gw=24)
    for k in s2:
        a, b, r = ({S.key(c) for c in x} for x in (s1[k], s2[k], rest[k]))
        assert not (a & b) and not (a & r) and not (b & r) and len(a | b | r) == len(load[k])
    assert {k: len(v) for k, v in rest.items()} == dict(ruler=585 - 78 - 156, lbt=0, mrcr=72 - 24 - 24, gw=72 - 12 - 24)
    assert all(sum(c['dataset'] == ds for c in s2['mrcr']) == 8 for ds in ('mrcr32', 'mrcr64', 'mrcr128'))


def test_deterministic():
    load = _pools()
    s1 = _s1(load)
    assert S.build(load, s1, [1, 2], 7) == S.build(load, s1, [1, 2], 7)
    assert S.build(load, s1, [1, 2], 7)[0]['mrcr'] != S.build(load, s1, [1, 2], 8)[0]['mrcr']
