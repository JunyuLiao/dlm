"""v14: bounded compute-sanitizer target — one v5 call with export+protect and one plain call
(prefix 1645, Q256, H16, HK2, D512). Run under compute-sanitizer --tool {memcheck,racecheck,synccheck}."""
import os
import sys

import torch

sys.path[:0] = ['.', 'src']
from tests.test_v14_cvm import fresh_inputs, v5_call  # noqa: E402
from experiments.value_direction_hopper.cvm import load_v5  # noqa: E402

load_v5(os.environ['V14_V5_BUILD'])
x = fresh_inputs(1645, 7)
out, exported = v5_call(*x, export=True, protect=1645 // 64, threshold=0.0)
plain, _ = v5_call(*x, threshold=-3.1366905212402343)
torch.cuda.synchronize()
print('SANITIZE_TARGET_OK', bool(torch.isfinite(out.output.float()).all()), tuple(exported.shape))
