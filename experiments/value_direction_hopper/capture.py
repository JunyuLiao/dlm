"""Capture post-RoPE native QKV from a fixed development prompt, without routing.

These are diagnostic shared states, not benchmark generations or final scores.
The source manifest and prompt hash are recorded and no source files are changed.
"""
import argparse
import json
from pathlib import Path

import torch

from dllm.models import create_adapter
from dllm.attention.blasst.core import _attention_type, _attention_validity
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _install_dense, _set_context, _request
from experiments.diffusion_gemma_value_aware_followup.protocol import MODEL, REVISION
from experiments.diffusion_gemma_jl_output_aware.projections import Projections

SOURCE = Path('results/diffusion_gemma_ruler4k_value_direction_s70_v19')


class Complete(Exception):
    pass


def capture(output, layers=(0, 5, 29)):
    if (output/'index.json').exists():
        raise FileExistsError(output/'index.json')
    output.mkdir(parents=True, exist_ok=True)
    row = json.loads((SOURCE/'development_manifest.json').read_text())[0]
    adapter = create_adapter('diffusion_gemma', MODEL, device='cuda', precision='bfloat16', revision=REVISION).load()
    matrices = Projections()
    saved = []

    @torch.no_grad()
    def observe(module, q, k, v, mask, **kwargs):
        layer = int(module.layer_idx)
        if layer in layers and not any(r['layer']==layer for r in saved):
            scale = kwargs.get('scaling', q.shape[-1]**-.5)
            causal = kwargs.get('is_causal', False)
            window = kwargs.get('sliding_window')
            valid = _attention_validity(mask, q, k, is_causal=bool(causal), sliding_window=window)
            valid = valid.expand(q.shape[0], q.shape[1], q.shape[-2], k.shape[-2])
            keyvalid = valid.reshape(q.shape[0], k.shape[1], q.shape[1]//k.shape[1], q.shape[-2], k.shape[-2]).any((2,3))
            matrix = matrices.get(layer, v.shape[1], v.shape[-1], 'gaussian', 32, 1729, v.device)
            z = v.float() @ matrix
            reference = (v.float().norm(dim=-1).square().masked_fill(~keyvalid, 0.).sum(-1)/keyvalid.sum(-1).clamp_min(1)).sqrt().clamp_min(1e-12)
            payload = dict(q=q, k=k, v=v, z=z, reference=reference, valid=valid,
                           mask=mask, scale=scale, layer=layer, kind=_attention_type(module,window))
            payload = {key:val.detach().cpu().contiguous() if isinstance(val,torch.Tensor) else val for key,val in payload.items()}
            dest=output/f'layer{layer}.pt'
            if dest.exists(): raise FileExistsError(dest)
            torch.save(payload,dest)
            saved.append(dict(layer=layer,kind=payload['kind'],shape=list(q.shape),kv_shape=list(k.shape),scale=scale,path=str(dest)))
            print(json.dumps(saved[-1]),flush=True)
            if len(saved)==len(layers): raise Complete()

    binding=_install_dense(adapter)
    binding.runtime.attention_observer=observe
    try:
        _set_context(binding,row)
        adapter.generate(_request(dict(row,generation_budget=1)))
    except Complete:
        pass
    finally:
        binding.close()
    if len(saved)!=len(layers): raise RuntimeError('Missing native capture layers')
    (output/'index.json').write_text(json.dumps(dict(source=str(SOURCE/'development_manifest.json'),
        id=row['id'],prompt_hash=row['prompt_hash'],model=MODEL,revision=REVISION,
        projection_matrices=matrices.manifest,states=saved,scope='first denoising step; development diagnostic only'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    capture(p.parse_args().output)
