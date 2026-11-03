"""Diagnostic-only logit probes with identical per-head tile counts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os

import torch

from scripts.llada22_eos_trajectory_diagnostic import (
    V2, TAU, pinned_modules, eos_stats, immutable, sha,
)
from scripts.llada22_snapshot_attention_diagnostic import mass_count_matched


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('base',type=Path);p.add_argument('source',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();protocol=json.loads((a.out/'protocol.json').read_text())
    if sha(__file__)!=protocol['runner_sha256'] or (a.out/'final_complete.json').exists():
        raise RuntimeError('fresh frozen protocol required')
    if not (a.source/'final_complete.json').exists():
        raise RuntimeError('source diagnostic incomplete')
    os.environ['HF_MODULES_CACHE']=str(a.base.resolve()/'private/hf_modules')
    os.environ['TRITON_CACHE_DIR']=str(a.base.resolve()/'private/triton_cache')
    attention,runtime=pinned_modules(a.base.resolve())
    kernels=__import__('experiments.numerical_qk_reuse.v31_value_kernels',fromlist=['statistics_cuda'])
    h=runtime.ModelHarness(a.base.resolve(),'llada22').load()
    results=[]
    paths=sorted((a.source/'private').glob('*_snapshots.pt'))
    if len(paths)!=protocol['source_item_count']:raise RuntimeError('wrong source cohort')
    for path in paths:
        snapshots=torch.load(path,map_location='cpu',weights_only=False)
        s=next(x for x in snapshots if x['label']=='first_eos_write')
        s['input_ids']=s['input_ids'].cuda()
        s['cache']={layer:(k.cuda(),v.cuda()) for layer,(k,v) in s['cache'].items()}
        saved_maps={layer:x['keep'].cuda() for layer,x in s['layers'].items()}
        positions=s['eos_positions'];reference=s['logits'][positions].float()
        item=path.stem.replace('_snapshots','').replace('humaneval_','humaneval/')
        for mode in ['saved_v2_map','mass_same_count_each_head','swap_forced_first_tile']:
            rt=attention.AttentionRuntime(V2,TAU)
            rt.block(s['block_start'],s['block_end']);rt.call_in_block=1
            native_attention=rt.attention
            swaps=[]
            def forced(q,k,v,layer,scale,**kwargs):
                keep=saved_maps[layer].clone().reshape(q.shape[1],-1)
                if mode!='saved_v2_map':
                    z=torch.zeros(1,k.shape[1],k.shape[2],32,device=q.device)
                    nu=torch.ones(k.shape[1],device=q.device)
                    stats=kernels.statistics_cuda(q,k,z,nu,scale,keep.shape[-1])
                    lm=stats.log_mass;den=torch.logsumexp(lm,1,keepdim=True)
                    score=(lm-torch.where(torch.isfinite(den),den,0.)).amax(-1)
                    mandatory=torch.zeros_like(keep)
                    mandatory[:,s['block_start']//64:(s['block_end']+63)//64]=True
                    if mode=='mass_same_count_each_head':
                        keep=mass_count_matched(score,mandatory,keep.sum(-1))
                    else:
                        # Test the first-support requirement only: at most one
                        # equal-count swap/head, selected by current native mass.
                        for head in range(q.shape[1]):
                            outside=score[head].masked_fill(keep[head],-torch.inf)
                            j=int(outside.argmax())
                            if not mandatory[head,0] and keep[head,0] and outside[j]>score[head,0]:
                                keep[head,0]=False;keep[head,j]=True
                                swaps.append(dict(layer=layer,head=head,replacement_tile=j))
                target=saved_maps[layer].reshape(q.shape[1],-1).sum(-1)
                if not torch.equal(keep.sum(-1),target):raise RuntimeError('per-head count changed')
                keep=keep.reshape_as(saved_maps[layer])
                key=(layer,q.shape[2],k.shape[2])
                rt.maps[key]=(keep,attention.v27_fa4.block_sparse_tensors(keep))
                return native_attention(q,k,v,layer,scale,**kwargs)
            rt.attention=forced
            h.runtime,h.cache=rt,s['cache']
            h.in_joint,h.query_offset,h.store_cache=True,s['block_start'],False
            logits=h._llada_forward(s['input_ids'][:,:s['block_end']]).logits.tensor[0,positions].float().cpu()
            record=dict(id=item,route=mode,**eos_stats(logits,h.tokenizer.eos_token_id),
                        max_abs_logit_delta=float((logits-reference).abs().max()),
                        per_head_tile_counts_identical=True,counts=dict(rt.counts),swaps=swaps)
            if mode=='saved_v2_map' and record['max_abs_logit_delta']!=0:
                immutable(a.out/'parity_failure.json',record)
                raise RuntimeError('saved-map logit reproduction failed')
            results.append(record)
        print(json.dumps(dict(event='matched_count_complete',id=item)),flush=True)
    immutable(a.out/'receipt.json',dict(status='complete',results=results,model_forwards=len(results),
                                       native_continuations=0,protocol_sha256=sha(a.out/'protocol.json')))
    immutable(a.out/'final_complete.json',dict(status='complete',items=len(paths),receipt_sha256=sha(a.out/'receipt.json')))


if __name__=='__main__':main()
