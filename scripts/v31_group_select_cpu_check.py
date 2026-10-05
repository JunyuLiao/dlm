"""CPU logic check of the group-shared units and the MAGE first-call carry (FA4 entry points stubbed; no GPU).
usage (deployment cwd, CUDA_VISIBLE_DEVICES empty):
  PYTHONPATH=src:. python -c "import experiments.numerical_qk_reuse as p; p.__path__.insert(0, '<overlay>');
      exec(open('<this file>').read())"
"""
import math
import sys

import torch

sys.path.insert(0, '.')
from experiments.numerical_qk_reuse import v27_fa4  # noqa: E402
from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter  # noqa: E402

TYPES = ['sliding_attention'] * 5 + ['full_attention']
H, HK, G = 16, 2, 8


def adapter(**kw):
    a = VllmMethodAdapter(TYPES, arm='mage', mage_select='fa4', **kw)
    a.calls.update(mage_kept_prefix_tiles=0, mage_prefix_tiles=0, mage_selections=0, mage_reused_calls=0)
    return a


# ---- units on synthetic per-row prefix log-mass
torch.manual_seed(0)
n, pt, kt = 256, 60, 64
qb = 2
head_lse = torch.randn(H, n, pt)
head_lse[2, 7, 40] = 30.0          # head 2 (group 0), row 7 (block 0): needle in tile 40
head_lse[13, 200, 17] = 30.0       # head 13 (group 1), row 200 (block 1): needle in tile 17
mass = torch.softmax(torch.cat([head_lse, torch.randn(H, n, kt - pt)], -1), -1)
for gran in ('kvblock_max', 'kvhead_max', 'qblock_max'):
    a = adapter(mage_granularity=gran)
    kept = a._mage_units(mass, head_lse, H, n, qb, kt, pt, 1, G)
    assert kept.shape == (1, H, qb, kt), kept.shape
    assert bool(kept[0, :, :, pt:].all())
    assert bool((kept[0, :, :, :pt].sum(-1) == 1).all()), gran
    assert a.calls['mage_kept_prefix_tiles'] == H * qb and a.calls['mage_prefix_tiles'] == H * qb * pt
    if gran == 'qblock_max':
        assert bool(kept[0, 2, 0, 40]) and bool(kept[0, 13, 1, 17])
        assert not bool(kept[0, 3, 0, 40])
        continue
    for grp in range(HK):
        assert bool((kept[0, grp * G:(grp + 1) * G] == kept[0, grp * G:grp * G + 1]).all()), gran
    if gran == 'kvblock_max':
        assert bool(kept[0, :G, 0, 40].all()) and bool(kept[0, G:, 1, 17].all())
        assert not bool(kept[0, :G, 1, 40].any())          # block 1 of group 0 has no needle row for tile 40
    else:
        assert bool(kept[0, :G, :, 40].all()) and bool(kept[0, G:, :, 17].all())
        assert bool((kept[0, :, 0] == kept[0, :, 1]).all())  # one set for the whole canvas
# padded rows (n = 200: block 1 has 72 real rows) never decide
n2 = 200
hl = torch.randn(H, n2, pt)
hl[0, 199, 3] = 30.0
a = adapter(mage_granularity='kvblock_max')
kept = a._mage_units(torch.softmax(torch.cat([hl, torch.randn(H, n2, kt - pt)], -1), -1), hl, H, n2, 2, kt, pt, 1, G)
assert bool(kept[0, :G, 1, 3].all()) and not bool(kept[0, :G, 0, 3].any())
# keep-fraction budget
k_tiles = pt - int(math.floor((1.0 - 0.12) * pt + 1e-9))
kept = adapter(mage_granularity='kvblock_max', mage_keep_frac=0.12)._mage_units(mass, head_lse, H, n, qb, kt, pt, k_tiles, G)
assert bool((kept[0, :, :, :pt].sum(-1) == k_tiles).all())
print('units ok')

# ---- carry: stub the FA4 entry points
log = []
v27_fa4.block_sparse_tensors = lambda kept, q_block=128: ('lists', kept.clone())
v27_fa4.dense = lambda q, k, v, s: (log.append(('dense', None)), 'dense')[1]
v27_fa4.sparse_lists = lambda q, k, v, lists, s: (log.append(('sparse', lists[1])), 'sparse')[1]
sel = {}


def fake_select(self, q, k, v, scale, prefix, n):
    kt_ = -(-(prefix + n) // 64)
    kept = torch.zeros((1, H, -(-n // 128), kt_), dtype=torch.bool)
    kept[..., prefix // 64:] = True
    kept[..., (self.canvas_id * 3) % (prefix // 64)] = True   # a canvas-dependent decision
    sel[self.canvas_id] = kept
    return 'select', kept


VllmMethodAdapter._mage_select_fa4 = fake_select
q = torch.zeros(1, H, 256, 4)
b = dict(k=None, v=None)
prefix, n = 64 * 40, 256
a = adapter(mage_select_step=1, mage_carry_first=True)
a.canvas_id = 1
assert [a._mage(5, q, b, 1.0, prefix, n) for _ in range(3)] == ['dense', 'select', 'sparse']
a.canvas_id = 2
assert a._mage(5, q, b, 1.0, prefix + n, n) == 'sparse'
carried = log[-1][1]
want = torch.ones((1, H, 2, (prefix + 2 * n) // 64), dtype=torch.bool)
want[..., :prefix // 64] = sel[1][..., :prefix // 64]
assert torch.equal(carried, want) and a.calls['mage_carried_calls'] == 1
assert a._mage(5, q, b, 1.0, prefix + n, n) == 'select' and a._mage(5, q, b, 1.0, prefix + n, n) == 'sparse'
a.canvas_id = 3
assert a._mage(5, q, b, 1.0, prefix + n, n) == 'dense'                 # prefix not grown by the previous canvas
assert a._mage(5, q[:, :, :128], b, 1.0, prefix + 2 * n, 128) == 'dense'   # different block layout
a.canvas_id = 5
assert a._mage(5, q, b, 1.0, prefix + 2 * n, n) == 'dense'             # a skipped canvas
assert a.calls['mage_carried_calls'] == 1 and a.calls['mage_warm_dense_calls'] == 4
# step 2 selection: only call 0 is carried, call 1 stays exact
a2 = adapter(mage_select_step=2, mage_carry_first=True)
a2.canvas_id = 1
assert [a2._mage(5, q, b, 1.0, prefix, n) for _ in range(4)] == ['dense', 'dense', 'select', 'sparse']
a2.canvas_id = 2
assert [a2._mage(5, q, b, 1.0, prefix + n, n) for _ in range(4)] == ['sparse', 'dense', 'select', 'sparse']
# without the carry nothing changes
a3 = adapter(mage_select_step=1)
a3.canvas_id = 1
[a3._mage(5, q, b, 1.0, prefix, n) for _ in range(3)]
a3.canvas_id = 2
assert a3._mage(5, q, b, 1.0, prefix + n, n) == 'dense' and a3.mage_state[5]['kept'] is None
for bad in (dict(mage_carry_first=True), dict(mage_carry_first=True, mage_select_step=0)):
    try:
        adapter(**bad)
        raise AssertionError('accepted ' + str(bad))
    except ValueError:
        pass
print('carry ok')
