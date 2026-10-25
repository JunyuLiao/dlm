"""Three constant all-kept consumers; first-canvas operator-only observation."""
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from operators import score_bank,evaluate_future,WORK


def digest(x):
    return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


class Runtime:
    def __init__(self):
        self.active=False
        self.begin(None,[],False,False)
        self.active=False

    def begin(self,folder,boundaries,capture,enabled=True,treatment=None):
        self.folder=folder;self.boundaries=boundaries;self.capture=capture;self.active=enabled
        self.phase=None;self.first_canvas=True;self.step=0;self.states={};self.previous={};self.prefix_hash={}
        self.physical=[];self.qual=[];self.route_fingerprints=[];self.operator_seconds=0.;self.operator_qk=self.operator_pv=0
        WORK.clear()
        self.treatment=treatment;self.supports={};self.mask_age=0

    def install(self):
        from vllm.model_executor.models import diffusion_gemma as dg
        from vllm.v1.attention.backends import flash_attn as fa
        original_prepare=dg.DiffusionGemmaModelState.prepare_attn
        original_forward=fa.FlashAttentionImpl.forward;rt=self
        def prepare(state,batch,mode,tables,slots,groups,config,for_capture=False,ubatch_idx=0):
            rt.phase=None
            if rt.active and not for_capture and batch.num_reqs==1:
                slot=int(batch.idx_mapping_np[0]);encoder=bool(state.diffusion_states.is_encoder_phase[slot].item())
                rt.phase='encoder' if encoder or batch.num_draft_tokens==0 else 'denoise'
                if rt.phase=='denoise':rt.step+=1
            return original_prepare(state,batch,mode,tables,slots,groups,config,for_capture=for_capture,ubatch_idx=ubatch_idx)
        def forward(impl,layer,query,key,value,cache,meta,output,output_scale=None,output_block_scale=None):
            name=getattr(layer,'layer_name','');location=next((l for l in [5,17,29] if f'.layers.{l}.' in name),None)
            if not(rt.active and rt.first_canvas and rt.phase=='denoise' and location is not None):
                return original_forward(impl,layer,query,key,value,cache,meta,output,output_scale,output_block_scale)
            n=int(meta.num_actual_tokens);length=int(meta.seq_lens[0]);prefix=length-n
            assert n==256 and impl.num_heads==16 and impl.num_kv_heads==2 and impl.head_size==512
            q=query[:n];kc,vc=cache.transpose(1,2).split(impl.head_size,dim=-1)
            rt.route_fingerprints.append(dict(layer=location,call=rt.step,query=digest(q),key=digest(key[:n]),value=digest(value[:n]),prefix=prefix))
            whole=(length+63)//64;keep=torch.ones((1,16,2,whole),device=q.device,dtype=torch.bool)
            need=rt.capture and any(rt.step in [a,a+1,a+2] for a in rt.boundaries)
            if need:
                rng=torch.cuda.get_rng_state().clone();cpu_rng=torch.get_rng_state().clone()
                page=kc.shape[1];pages=meta.block_table[0,:(length+page-1)//page].long()
                k=kc.index_select(0,pages).flatten(0,1)[:length]
                v=vc.index_select(0,pages).flatten(0,1)[:length]
                assert torch.equal(k[prefix:],key[:n]) and torch.equal(v[prefix:],value[:n]),'current cache tail mismatch'
                ph=digest(k[:prefix]);vh=digest(v[:prefix])
                if location in rt.prefix_hash:assert rt.prefix_hash[location]==(ph,vh),'within-canvas prefix changed'
                rt.prefix_hash[location]=(ph,vh)
                start=time.perf_counter()
                for boundary in rt.boundaries:
                    ident=location,boundary
                    if rt.step==boundary:
                        rt.previous[ident]=dict(q=q.detach().cpu().clone(),k_tail=k[prefix:].detach().cpu().clone(),v_tail=v[prefix:].detach().cpu().clone())
                    if rt.step==boundary+1:
                        assert ident in rt.previous
                        previous=rt.previous.pop(ident);bankrows=[];private=[]
                        for head in [0,8]:
                            for qb in [0,1]:
                                lo=qb*128;unit=head*2+qb
                                prev=dict(q=previous['q'][lo:lo+128,head],k_tail=previous['k_tail'][:,head//8],v_tail=previous['v_tail'][:,head//8])
                                drop=max(1,math.floor(.05*(prefix//64)))
                                b,s=score_bank(q[lo:lo+128,head],k[:,head//8],v[:,head//8],prev,prefix,impl.scale,unit,drop)
                                # Qualify the four actual GQA routes against exact current attention.
                                rt.qual.append(dict(layer=location,head=head,kv_head=head//8,qb=qb,
                                    exact_output=s['dense_output'],boundary=boundary))
                                bankrows.append(dict(layer=location,head=head,kv_head=head//8,qb=qb,**b));private.append(s)
                                rt.operator_qk+=(4+len(b['candidates']))*whole
                                rt.operator_pv+=(8+4*len(b['candidates']))*whole
                        base=rt.folder/f'layer{location:02d}_after{boundary:02d}'
                        save(base/'bank_choices.json',dict(layer=location,boundary=boundary,intervention_call=boundary+1,
                            prefix=prefix,units=bankrows,future_outcomes_available=False,bank_source='current/past only'))
                        private_dir=base.parent.parent/'private';private_dir.mkdir(exist_ok=True)
                        def compact(x):
                            if isinstance(x,torch.Tensor):return x.half() if x.is_floating_point() else x
                            if isinstance(x,dict):return {k:compact(v) for k,v in x.items()}
                            if isinstance(x,list):return [compact(v) for v in x]
                            return x
                        selected=[]
                        for b,s in zip(bankrows,private):
                            indices=sorted({i for i in b['choices'].values() if i is not None})
                            selected.append(dict(selected_indices=indices,selected_vectors=compact([s['candidates'][i] for i in indices]),
                                                 all_bias_directions=compact([x['bias_direction_mean'] for x in s['candidates']])))
                        torch.save(selected,str(private_dir/f'layer{location:02d}_after{boundary:02d}_vectors.pt'))
                        rt.states[ident]=(bankrows,private)
                    if rt.step==boundary+2:
                        bankrows,private=rt.states.pop(ident);rows=[]
                        for b,s in zip(bankrows,private):
                            h,qb=b['head'],b['qb'];lo=qb*128
                            future=evaluate_future(b,s,q[lo:lo+128,h],k[:,h//8],v[:,h//8],prefix,impl.scale)
                            rows.append(dict(layer=location,head=h,qb=qb,results=future))
                            rt.operator_qk+=(2+2*len(future))*whole
                            rt.operator_pv+=(2+4*len(future))*whole
                        save(rt.folder/f'layer{location:02d}_after{boundary:02d}'/'future_evaluation.json',dict(units=rows,
                            native_next_call=rt.step,selection_uses_future=False))
                torch.cuda.synchronize();rt.operator_seconds+=time.perf_counter()-start
                assert torch.equal(rng,torch.cuda.get_rng_state()) and torch.equal(cpu_rng,torch.get_rng_state()),'diagnostic RNG changed'
                del k,v
            if rt.treatment and location==rt.treatment['layer']:
                boundary=rt.treatment['boundary'];schedule=rt.treatment['schedule'];offset=rt.step-boundary-1
                sparse=(offset==0 and schedule[0]=='S') or (offset==1 and schedule[1]=='S')
                if sparse:
                    for unit in rt.treatment['units']:
                        assert len(unit['tiles'])==max(1,math.floor(.05*(prefix//64)))
                        keep[0,unit['head'],unit['qb'],unit['tiles']]=False
            from experiments.numerical_qk_reuse import v27_fa4
            from experiments.numerical_qk_reuse.v31_local_sparse import alias64
            kk,vv,table=alias64(kc,vc,meta.block_table[0],length)
            lists=v27_fa4.block_sparse_tensors(keep)
            result=v27_fa4.load()(q[None],kk,vv,softmax_scale=impl.scale,causal=False,
                page_table=table,seqused_k=torch.tensor([length],device=q.device,dtype=torch.int32),block_sparse_tensors=lists)[0]
            # Exact operator comparison is diagnostic only and only where captured.
            if need:
                current=[x for x in rt.qual if x['layer']==location and x['boundary']==rt.step-1]
                for x in current:
                    ref=x.pop('exact_output').to(q.device);obs=result[0,x['qb']*128:(x['qb']+1)*128,x['head']].float()
                    x['relative_operator_error']=float(((obs-ref).norm(dim=-1)/ref.norm(dim=-1).clamp_min(1e-8)).max())
                    assert x['relative_operator_error']<=.03,'GQA/consumer numerical qualification failed'
            output[:n].copy_(result[0].reshape_as(output[:n]))
            rt.physical.append(dict(layer=location,call=rt.step,eligible=keep.numel(),kept=int(keep.sum()),skipped=int((~keep).sum()),
                protected_canvas_skipped=0,LOCAL_skipped=0,current_QKV=True,prefix=prefix,query_heads=16,kv_heads=2))
            return output
        dg.DiffusionGemmaModelState.prepare_attn=prepare;fa.FlashAttentionImpl.forward=forward

    def receipt(self):
        return dict(physical_calls=self.physical,GQA_qualification=self.qual,route_fingerprints=self.route_fingerprints,operator_only_seconds=self.operator_seconds,
                    diagnostic_QK_rectangles=WORK['QK_rectangles'],diagnostic_PV_rectangles=WORK['PV_rectangles'],
                    diagnostic_value_projection_key_rows=WORK['value_projection_key_rows'],
                    operator_work_count='explicit GEMM-equivalent Q128xKV64 rectangles, excludes model attention consumer',
                    extra_model_forwards=0,cached_attention_outputs=False,old_KV_consumed_in_model=False,
                    pending_future=[list(x) for x in self.states],current_canvas_resets=True)
