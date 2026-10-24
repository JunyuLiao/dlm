import importlib.util
from pathlib import Path
from types import SimpleNamespace
from collections import deque
import numpy as np
import torch

p = Path(__file__).resolve().parents[1]/'code/state_restore.py'
spec = importlib.util.spec_from_file_location('state_restore_test', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_storage_alias_addresses_rng_and_python_cycles_restore():
    base = torch.arange(12, dtype=torch.float32)
    view = base.as_strided((2, 3), (0, 2), 1)
    a = SimpleNamespace(base=base, view=view, array=np.arange(4), queue=deque([1, 2]))
    a.same = a
    a.mapping = {'a': a, 'value': [3, 4]}
    pointer = base.data_ptr()
    state = m.Snapshot({'root': a})
    expected_draw = torch.rand(4)
    base.fill_(37)
    a.view = torch.zeros(1)
    a.queue.popleft()
    a.mapping.clear()
    a.array[:] = 0
    a.new = True
    state.restore()
    assert a.same is a and a.mapping['a'] is a
    assert base.data_ptr() == pointer and a.view is view
    assert torch.equal(base, torch.arange(12, dtype=torch.float32))
    assert list(a.queue) == [1, 2] and not hasattr(a, 'new')
    np.testing.assert_array_equal(a.array, np.arange(4))
    assert torch.equal(torch.rand(4), expected_draw)
    assert state.verify()['passed']

def test_restore_slots_and_shared_objects_are_explicit():
    class Slot:
        __slots__ = ('x',)
    s = Slot()
    s.x = [1]
    ignored = SimpleNamespace(x=3)
    snap = m.Snapshot({'slot': s, 'ignored': ignored}, shared=[ignored])
    s.x.append(2)
    ignored.x = 4
    snap.restore()
    assert s.x == [1] and ignored.x == 4
