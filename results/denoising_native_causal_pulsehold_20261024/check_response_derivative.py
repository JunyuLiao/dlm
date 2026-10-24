"""Independent CPU central-difference check of the Q/K/V bias differential."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / 'code'))
    source = root / 'code/support_bank.py'
    spec = importlib.util.spec_from_file_location('frozen_support_bank', source)
    bank = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bank)
    gen = torch.Generator(device='cpu').manual_seed(8921)
    errors = []
    for _ in range(10):
        q = torch.randn(4, 5, generator=gen, dtype=torch.float64)
        k = torch.randn(192, 5, generator=gen, dtype=torch.float64)
        v = torch.randn(192, 7, generator=gen, dtype=torch.float64)
        dq, dk, dv = [torch.randn(x.shape, generator=gen, dtype=x.dtype) * .1 for x in (q, k, v)]
        keep = torch.tensor([False, True])
        scale = 5 ** -.5
        p = (q @ k.T * scale).softmax(-1)
        measured = bank._response_residual(p, q, k, v, keep, 2, dq, dk, dv, scale)
        exact = measured['_private_signed_vectors']['joint']['residual']

        def bias(t):
            scores = ((q + t * dq) @ (k + t * dk).T) * scale
            dense = scores.softmax(-1) @ (v + t * dv)
            masked = scores.clone()
            masked[:, :64] = -torch.inf
            sparse = masked.softmax(-1) @ (v + t * dv)
            return sparse - dense

        eps = 1e-5
        finite = (bias(eps) - bias(-eps)) / (2 * eps)
        errors.append(float((finite - exact).abs().max()))
    result = dict(passed=max(errors) < 1e-7, trials=10, seed=8921,
        max_absolute_error=max(errors), errors=errors, tolerance=1e-7,
        scope='Independent FP64 CPU bias directional derivative; no native predictive evidence',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        check_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (root / 'closeout001/response_finite_difference002.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))
    assert result['passed']


if __name__ == '__main__':
    main()
