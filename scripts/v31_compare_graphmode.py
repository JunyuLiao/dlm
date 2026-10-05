"""Compare dumped temperature-scaled sampler logits between vLLM cudagraph modes for the same cell and seed.
usage: python v31_compare_graphmode.py DUMP_DIR TAG   (TAG like longbench_v2_32k_2_1111)"""
import itertools
import json
import sys
from pathlib import Path

import torch

d, tag = Path(sys.argv[1]), sys.argv[2]
res = {}
for call in (1, 2, 3):
    x = {m: torch.load(d / f'{m}_{tag}_call{call}.pt').float() for m in ('eager', 'PIECEWISE', 'default')
         if (d / f'{m}_{tag}_call{call}.pt').exists()}
    for a, b in itertools.combinations(sorted(x), 2):
        diff = (x[a] - x[b]).abs()
        pa, pb = torch.log_softmax(x[a], -1), torch.log_softmax(x[b], -1)
        kl = (pa.exp() * (pa - pb)).sum(-1)
        res[f'call{call} {a} vs {b}'] = dict(max_abs=round(float(diff.max()), 4), mean_abs=round(float(diff.mean()), 5),
                                            argmax_mismatch=int((x[a].argmax(-1) != x[b].argmax(-1)).sum()),
                                            kl_mean=round(float(kl.mean()), 6), kl_max=round(float(kl.max()), 5))
print(json.dumps(res, indent=1))
