"""Counterfactual masks on ONE dense trajectory, using the existing routers.

Only sufficient statistics are cached; dense QK and FP32 probabilities are
shared by all threshold probes and discarded after each attention call.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from dllm.attention.blasst.core import (
    Blasst2DConfig, Blasst2DStats, _attention_type, _prepare_attention_scores,
    _finish_eager_attention, _region_tile_counts, apply_blasst_2d,
)
from experiments.diffusion_attention_threshold_modeling.routing import FreshRoutingConfig, _keep_vector
from experiments.diffusion_attention_threshold_modeling.proxy import sol_attention_proxy_rows
from .config import condition_map
from .runner import _policy_for_target


def sol_config(target):
    return FreshRoutingConfig(mode='gaussian', target_density=1-target,
        q_block_size=64, kv_block_size=64, region='all', combined_region_population=True,
        execution='logical', adapter='diffusion_gemma', corpus='multibench')


def tile_mass(scores, valid, width=64):
    """Sum dense probability mass within each *physical* Q/K tile."""
    has = valid.any(-1)
    probs = torch.softmax(torch.where(has[..., None], scores, 0), -1, dtype=torch.float32)
    probs = probs * valid * has[..., None]
    q, k = scores.shape[-2:]
    padded = F.pad(probs, (0, (-k) % width, 0, (-q) % width))
    return padded.reshape(*scores.shape[:2], (q+width-1)//width, width,
        (k+width-1)//width, width).sum((-1, -3)), int(has.sum().item())


class DenseDiagnostics:
    def __init__(self):
        self.calls = {name: [] for name in condition_map()}
        self.calls['blasst_lambda1'] = []
        self.configs = {name: sol_config(c.target_sparsity) for name,c in condition_map().items() if c.method=='sol'}
        self.bconfigs = {name: Blasst2DConfig(q_tile_size=64,kv_tile_size=64,
            length_aware_policy=_policy_for_target(c.target_sparsity))
            for name,c in condition_map().items() if c.method=='blasst'}
        self.bconfigs['blasst_lambda1'] = Blasst2DConfig(q_tile_size=64,kv_tile_size=64,blasst_lambda=1.)

    @torch.no_grad()
    def __call__(self,module,query,key,value,attention_mask,*,dropout=0.,scaling=None,is_causal=None,sliding_window=None,**kwargs):
        expanded_key, expanded_value, scores, valid = _prepare_attention_scores(module,query,key,value,attention_mask,
            scaling=scaling,is_causal=is_causal,sliding_window=sliding_window)
        mass, nrows = tile_mass(scores,valid)
        # One small device-to-host transfer, not one synchronization per tile.
        masses = mass.cpu().numpy()
        runtime = module._blasst_2d_runtime
        kind = _attention_type(module,sliding_window)
        prefix = max(0,key.shape[-2]-query.shape[-2])
        length = int(valid.any(dim=(1,2)).sum(-1).item())
        meta = dict(layer=int(module.layer_idx),head_ids=list(range(query.shape[1])),
            denoising_step=runtime.current_denoising_iteration,attention_type=kind,
            query_length=query.shape[-2],sequence_length=key.shape[-2],valid_kv_length=length,
            retained_attention_mass_rows=nrows,valid_rows=nrows)
        rows = sol_attention_proxy_rows(module,query,key,value,attention_mask,q_block_size=64,kv_block_size=64,
            prefix_length=prefix,scaling=scaling,is_causal=is_causal,sliding_window=sliding_window,
            prepared_key=expanded_key,valid_pair_mask=valid,global_kv_tiling=True)
        dense_regions = {r:dict(eligible_tiles=0,skipped_tiles=0) for r in ('prefix','canvas')}
        for name,config in self.configs.items():
            total=skipped=degenerate=fallback=0; kept_mass=0.
            regions={r:dict(eligible_tiles=0,skipped_tiles=0) for r in ('prefix','canvas')}
            for row in rows:
                entries=[(region,span,float(v)) for region in ('prefix','canvas')
                    for span,v in zip(row[region+'_spans'],row[region])]
                if not entries: continue
                values=np.asarray([e[2] for e in entries],dtype=np.float64)
                degen=len(values)<2 or values.std(ddof=0)<1.e-8
                if degen: keep=np.ones(len(values),dtype=bool); fall=False
                else: keep,_,fall=_keep_vector(values,config,kind)
                degenerate+=int(degen); fallback+=int(fall); total+=len(values); skipped+=int((~keep).sum())
                b,h,qi=int(row['batch']),int(row['head']),int(row['query_start'])//64
                for retained,(region,(start,end),_) in zip(keep,entries):
                    regions[region]['eligible_tiles']+=1
                    regions[region]['skipped_tiles']+=int(not retained)
                    if retained: kept_mass+=float(masses[b,h,qi,int(start)//64])
            self.calls[name].append(dict(meta,eligible_tiles=total,skipped_tiles=skipped,retained_tiles=total-skipped,
                retained_dense_attention_mass=kept_mass/nrows if nrows else 0.,
                degenerate_rows=degenerate,fallback_rows=fallback,routing_rows=len(rows),region_counts=regions))
            dense_regions={r:dict(eligible_tiles=v['eligible_tiles'],skipped_tiles=0) for r,v in regions.items()}
        self.calls['dense'].append(dict(meta,eligible_tiles=total,skipped_tiles=0,retained_tiles=total,
            retained_dense_attention_mass=1.,region_counts=dense_regions))
        for name,config in self.bconfigs.items():
            lam=config.lambda_for(kind,runtime.current_denoising_iteration,length)
            _,decisions=apply_blasst_2d(scores,valid,runtime.active_query_mask,config,blasst_lambda=lam,sparse_kv_start=0)
            counts=Blasst2DStats._tensor_counts(decisions)
            kept_mass=float((mass*(~decisions.skip_mask)).sum().item())
            self.calls[name].append(dict(meta,**counts,effective_blasst_lambda=lam,
                retained_dense_attention_mass=kept_mass/nrows if nrows else 0.,
                region_counts=_region_tile_counts(decisions,kv_tile_size=64,sequence_length=key.shape[-2],prefix_length=prefix)))
        output=_finish_eager_attention(query,expanded_value,scores,valid,dropout,module.training)
        if not torch.isfinite(output[0]).all(): raise FloatingPointError('nonfinite dense attention')
        return output


class CheckedAttention:
    """Finite-output assertion without changing the underlying algorithm."""
    def __init__(self,fn): self.fn=fn; self.calls=0
    def __call__(self,*args,**kwargs):
        result=self.fn(*args,**kwargs); self.calls+=1
        if not torch.isfinite(result[0]).all(): raise FloatingPointError('nonfinite attention')
        return result
