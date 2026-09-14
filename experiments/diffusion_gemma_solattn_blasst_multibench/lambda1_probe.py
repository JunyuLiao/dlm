"""Small exact-replay audit and row-block snapshots for the lambda=1 ceiling."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from dllm.models import create_adapter
from dllm.attention.blasst.core import _prepare_attention_scores, _attention_type, evaluate_blasst_thresholds, blasst_2d_attention_forward
from .config import MODEL, REVISION, condition_map
from .controlled_runner import load_completed, code_hashes, install
from .runner import _request, _set_context, _write

BASE=Path('results/diffusion_gemma_solattn_blasst_multibench_controlled')
OUT=BASE/'lambda1_investigation'

class Probe:
    def __init__(self): self.records=[];self.arrays={};self.counts=defaultdict(int)
    @torch.no_grad()
    def __call__(self,module,q,k,v,mask,**kw):
        layer=int(module.layer_idx);step=self.counts[layer];self.counts[layer]+=1
        # Every layer, first and fourth call: all heads and first physical query tile.
        if step in (0,3):
            ek,ev,s,valid=_prepare_attention_scores(module,q,k,v,mask,scaling=kw.get('scaling'),
                is_causal=kw.get('is_causal'),sliding_window=kw.get('sliding_window'))
            _,_,n,L=s.shape;pad=(-L)%64
            scores=s[0,:,:64].float(); vv=valid.expand_as(s)[0,:,:64]
            maxima=F.pad(scores.masked_fill(~vv,-torch.inf),(0,pad),value=-torch.inf).reshape(q.shape[1],min(n,64),-1,64).amax(-1)
            counts=F.pad(vv,(0,pad)).reshape(q.shape[1],min(n,64),-1,64).sum(-1)
            has=counts>0
            previous=F.pad(maxima.cummax(-1).values[...,:-1],(1,0),value=-torch.inf)
            strict=has&(maxima>previous);ties=has&(maxima==previous);keep=strict|ties
            eligible=has.any(-2);tilekeep=keep.any(-2)
            decisions=evaluate_blasst_thresholds(s,valid,module._blasst_2d_runtime.active_query_mask,[1.],q_tile_size=64,kv_tile_size=64)[1.]
            assert torch.equal(decisions.skip_mask[0,:,0],eligible&~tilekeep)
            probs=torch.softmax(torch.where(vv.any(-1,keepdim=True),scores,0),-1)*vv
            mass=F.pad(probs,(0,pad)).reshape_as(F.pad(vv,(0,pad)).reshape(q.shape[1],min(n,64),-1,64)).sum(-1)
            key=f'l{layer}_c{step}'
            for name,value in dict(maxima=maxima,previous=previous,counts=counts,mass=mass).items():
                self.arrays[key+'_'+name]=value.cpu().numpy()
            def num(t): return int(t.sum().item())
            record=dict(key=key,layer=layer,call=step,attention_type=_attention_type(module,kw.get('sliding_window')),
                prefix_length=L-n,kv_length=L,heads=q.shape[1],query_rows=min(n,64),eligible_tiles=num(eligible),
                skipped_tiles=num(eligible&~tilekeep),row_votes=num(has),row_skip_votes=num(has&~keep),
                strict_record_tiles=num(strict.any(-2)),tie_only_tiles=num(tilekeep&~strict.any(-2)),
                mass_rows=num(vv.any(-1)),physical_mass_sum=float((mass*tilekeep[:,None]).sum()),
                row_mask_mass_sum=float((mass*keep).sum()),
                finalmax_skipped_tiles=num(eligible&~(has&(maxima==maxima.amax(-1,keepdim=True))).any(-2)))
            self.records.append(record)
        return blasst_2d_attention_forward(module,q,k,v,mask,**kw)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    saved=load_completed(BASE,'blasst_length_aware_s90')
    # Two deterministic IDs per benchmark, no accuracy-based selection.
    selected=[]
    for b in ('ruler16k','longbench','aime24','livecodebench_v6'):
        selected+=sorted([r for r in saved if r['benchmark']==b],key=lambda r:r['id'])[:2]
    config=json.loads((BASE/'conditions/blasst_length_aware_s90/run_config.json').read_text())
    changed={k:[v,code_hashes().get(k)] for k,v in config['code_sha256'].items() if code_hashes().get(k)!=v}
    _write(OUT/'selection.json',dict(ids=[r['id'] for r in selected],code_changes_since_run=changed))
    if changed: raise RuntimeError(f'code changed since original: {changed}')
    adapter=create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
    audits=[]
    for i,row in enumerate(selected):
        out=OUT/f'probe_{i}'
        if (out/'audit.json').exists(): audits.append(json.loads((out/'audit.json').read_text()));continue
        out.mkdir(exist_ok=True);probe=Probe()
        binding,stats,_=install(adapter,condition_map()['blasst_length_aware_s90'])
        binding.runtime.attention_override=probe
        try:
            _set_context(binding,row);generated=adapter.generate(_request(row))
        finally: binding.close()
        np.savez_compressed(out/'snapshots.npz',**probe.arrays)
        _write(out/'records.json',probe.records)
        original=row['blasst_summary'];current=stats.summary()
        audit=dict(id=row['id'],benchmark=row['benchmark'],exact_tokens=generated.completion_tokens==row['completion_tokens'],
            tile_counts_match=all(original[k]==current[k] for k in ('eligible_tiles','skipped_tiles')),
            original_tiles={k:original[k] for k in ('eligible_tiles','skipped_tiles')},calls=len(probe.records))
        _write(out/'audit.json',audit);audits.append(audit);print(audit,flush=True)
    _write(OUT/'replay_audit.json',dict(passed=all(r['exact_tokens'] and r['tile_counts_match'] for r in audits),records=audits))

if __name__=='__main__': main()
