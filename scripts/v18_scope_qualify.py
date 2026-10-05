"""GPU qualification of native-legal masks on saved real decoder Q/K/V captures.

Input is a trusted torch file from v18_scope_capture.py, containing real
decoder Q/K/V and one labeled padded-mask derivative. Qualification requires
mask=None, padded-mask, and partial 64-key edge cases. This is diagnostic,
never accepted request timing.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch

NATIVE_SDPA_SHA256 = '87f933d1a2d8508df572da5c0748c6b24c22ff2b625796949957dcd86cc57564'


def unpack(mask, nk):
    words = mask.tensor.detach().cpu()
    b, h, q, tiles = words.shape
    result = torch.zeros(b, h, q, nk, dtype=torch.bool)
    for j in range(nk):
        values = words[..., j // 64].reshape(-1).tolist()
        result[..., j] = torch.tensor([bool(int(value) & (1 << (j % 64))) for value in values]).reshape(b, h, q)
    return result


def qualify(captures: Path, library: Path, torch_library: Path) -> dict:
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from experiments.value_direction_hopper.cuda import Kernel
    from experiments.value_direction_hopper.masks import geometry, pack

    sdpa_source = Path(inspect.getfile(sdpa_attention_forward))
    sdpa_hash = hashlib.sha256(sdpa_source.read_bytes()).hexdigest()
    if sdpa_hash != NATIVE_SDPA_SHA256:
        raise RuntimeError(f'native SDPA source changed: {sdpa_hash}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    states = torch.load(captures, map_location='cpu', weights_only=True)
    if not isinstance(states, list) or not states:
        raise ValueError('captures must be a nonempty list')
    kernel = Kernel(str(library), torch_library=str(torch_library))
    report = []
    categories = set()
    for i, state in enumerate(states):
        if state.get('is_causal') is not False:
            raise ValueError('native-legal capture requires explicit is_causal=False')
        q, k, v = (state[key].to('cuda').contiguous() for key in ('q', 'k', 'v'))
        mask = state.get('mask')
        mask = None if mask is None else mask.to('cuda').contiguous()
        b, h, nq, d = q.shape
        nk, hk = k.shape[-2], k.shape[1]
        scaling = state.get('scaling')
        if mask is None:
            packed, validkv = geometry(b, nq, nk, device=q.device, causal=False, window=0)
            expected = torch.ones(b, 1, nq, nk, dtype=torch.bool)
            categories.add('mask_none')
        elif mask.dtype == torch.bool:
            expected = torch.broadcast_to(mask[..., :nk], (b, h, nq, nk)).cpu()
            packed = pack(expected.to('cuda').contiguous())
            validkv = expected.any((1, 2)).to('cuda')[:, None, :].expand(b, hk, nk)
            if not bool(expected.all()):
                categories.add('padded')
        else:
            raise ValueError('qualification expects native None or boolean mask')
        if nk % 64:
            categories.add('partial_edge')
        actual = unpack(packed, nk)
        if not torch.equal(actual.expand(b, h, nq, nk), expected.expand(b, h, nq, nk)):
            raise AssertionError(f'legal K bitmap mismatch on capture {i}')
        # With -inf threshold every legal tile is retained. Projection values
        # are irrelevant to the dense output; use explicit finite sentinels.
        z = torch.zeros(b, hk, nk, 32, device='cuda')
        ref = torch.ones(b, hk, device='cuda')
        result = kernel(q, k, v, z, ref, mask=packed, scale=scaling, log_threshold=-float('inf'),
                        mode='value', precision='tf32x3_register', tma=True)
        module = type('Module', (), {'num_key_value_groups': h // hk, 'is_causal': False})()
        native, _ = sdpa_attention_forward(module, q, k, v, mask, scaling=scaling,
                                           is_causal=False, sliding_window=state.get('sliding_window'))
        torch.cuda.synchronize()
        if not torch.isfinite(result.output).all() or not torch.isfinite(native).all():
            raise AssertionError(f'nonfinite dense output on capture {i}')
        if result.skipped.any():
            raise AssertionError(f'dense kernel skipped a tile on capture {i}')
        max_abs = float((result.output.transpose(1, 2) - native).abs().max())
        report.append(dict(source_id=state.get('source_id'), layer_kind=state.get('layer_kind'),
                           nk=nk, nq=nq, mask_none=mask is None, partial_edge=bool(nk % 64),
                           scaling=scaling, derived_mask='derived_padded_mask' in str(state.get('source_id')),
                           legal_pairs=int(expected.sum()), max_abs_dense_vs_native=max_abs))
    missing = {'mask_none', 'padded', 'partial_edge'} - categories
    if missing:
        raise ValueError(f'missing real capture categories: {sorted(missing)}')
    return dict(schema='v18_native_scope_qualification_v1', native_sdpa_source=str(sdpa_source),
                native_sdpa_sha256=sdpa_hash, captures_sha256=hashlib.sha256(captures.read_bytes()).hexdigest(),
                categories=sorted(categories), states=report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--captures', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--torch-library', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = qualify(args.captures, args.library, args.torch_library)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(categories=result['categories'], states=len(result['states']))))


if __name__ == '__main__':
    main()
