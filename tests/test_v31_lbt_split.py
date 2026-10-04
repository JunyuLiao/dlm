"""v31_lbt_split: proportional stratified exploration sample, disjoint from the hold-out, deterministic (CPU)."""
import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent                     # repository layout (tests/) or a flat overlay
_SRC = next(p for p in (_HERE.parent / 'scripts' / 'v31_lbt_split.py', _HERE / 'v31_lbt_split.py') if p.exists())
spec = importlib.util.spec_from_file_location('v31_lbt_split', _SRC)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def _rows():
    rows, i = [], 0
    for cls, n_long, n_short in (('short', 7, 11), ('medium', 21, 0), ('long', 11, 0)):
        for k in range(n_long + n_short):
            rows.append(dict(index=i, length=cls, prompt_token_count=40000 if k < n_long else 20000))
            i += 1
    return rows


def test_allocation_is_proportional_and_exact():
    take = S.allocate({'short': 70, 'medium': 215, 'long': 108}, 160)
    assert sum(take.values()) == 160 and take == {'short': 28, 'medium': 88, 'long': 44}


def test_split_is_eligible_stratified_and_deterministic():
    rows = _rows()
    a, eligible, take = S.split(rows, 13, 5)
    b, _, _ = S.split(rows, 13, 5)
    assert a == b and len(a) == 13 and eligible == {'short': 7, 'medium': 21, 'long': 11}
    by = {r['index']: r for r in rows}
    assert all(by[i]['prompt_token_count'] >= S.MIN_PROMPT for i in a)
    assert {c: sum(by[i]['length'] == c for i in a) for c in S.CLASSES} == take
    assert S.split(rows, 13, 6)[0] != a


if __name__ == '__main__':
    import sys
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name)
    sys.exit(0)
