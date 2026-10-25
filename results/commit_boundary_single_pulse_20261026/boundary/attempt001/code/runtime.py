"""Fixed all-kept lifecycle; one selector-only pulse at one native later anchor."""
import json
import math
from pathlib import Path
import time
import torch
from bank import build
from measure import digest


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)


class Runtime:
    def __init__(self,state):self.state=state;self.active=False

    def begin(self,folder,common,observe,selection=None,expected=None):
        self.folder=folder;self.common=common;self.observe=observe;self.selection=selection;self.expected=expected
        self.active=True;self.prefix=None;self.step=0;self.phase=None;self.anchor=None;self.bank=None;self.pulse_calls=0
        self.records=[];self.qual=[];self.routes=[];self.work={};self.operator_seconds=0.

    def install(self):
        from vllm.model_executor.models import diffusion_gemma as dg
        from vllm.v1.attention.backends import flash_attn as fa
        original_prepare=dg.DiffusionGemmaModelState.prepare_attn;original_forward=fa.FlashAttentionImpl.forward;rt=self
        def prepare(state,batch,mode,tables,slots,groups,config,for_capture=False,ubatch_idx=0):
            rt.phase=None
            if rt.active and not for_capture and batch.num_reqs==1:
                slot=int(batch.idx_mapping_np[0]);enc=bool(state.diffusion_states.is_encoder_phase[slot].item())
                rt.phase='encoder' if enc or batch.num_draft_tokens==0 else 'denoise'
                if rt.phase=='denoise':rt.step+=1
            return original_prepare(state,batch,mode,tables,slots,groups,config,for_capture=for_capture,ubatch_idx=ubatch_idx)
        def forward(impl,layer,query,key,value,cache,meta,output,output_scale=None,output_block_scale=None):
            if not(rt.active and rt.common and rt.phase=='denoise' and '.layers.29.' in getattr(layer,'layer_name','')):
                return original_forward(impl,layer,query,key,value,cache,meta,output,output_scale,output_block_scale)
            n=int(meta.num_actual_tokens);length=int(meta.seq_lens[0]);prefix=length-n
            assert n==256 and impl.num_heads==16 and impl.num_kv_heads==2 and impl.head_size==512
            if rt.prefix!=prefix:rt.prefix=prefix;rt.step=1
            if rt.step==1:return original_forward(impl,layer,query,key,value,cache,meta,output,output_scale,output_block_scale)
            q=query[:n];kc,vc=cache.transpose(1,2).split(impl.head_size,dim=-1);whole=math.ceil(length/64)
            keep=torch.ones((1,16,2,whole),dtype=torch.bool,device=q.device)
            target=rt.anchor is not None and rt.state['C']==rt.anchor['canvas'] and rt.step==rt.anchor['after_completed_calls']+1
            exact=None
            if target:
                cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();start=time.perf_counter()
                page=kc.shape[1];pages=meta.block_table[0,:math.ceil(length/page)].long()
                k=kc.index_select(0,pages).flatten(0,1)[:length];v=vc.index_select(0,pages).flatten(0,1)[:length]
                assert torch.equal(k[prefix:],key[:n]) and torch.equal(v[prefix:],value[:n]),'stale current tail'
                route=dict(prefix=prefix,canvas=rt.state['C'],step=rt.step,query=digest(q),key=digest(k),value=digest(v),
                    cpu_rng=digest(cpu),cuda_rng=digest(gpu),native_N=rt.state['N'],native_C=rt.state['C'],
                    shape=list(q.shape),query_stride=list(q.stride()),cache_stride=list(cache.stride()),live_current_KV=True)
                rt.routes.append(route)
                if rt.expected is not None:
                    semantic=lambda x:{k:v for k,v in x.items() if k!='cache_stride'}
                    assert semantic(route)==semantic(rt.expected),'pre-intervention current operands/RNG/counters differ'
                if rt.observe:
                    bank,private,work=build(q,k,v,prefix,impl.scale);rt.bank=bank;rt.work=work
                    save(rt.folder/'selections.json',dict(anchor=rt.anchor,bank=bank,pre_intervention_route=route,future_logits_used=False))
                    private_path=rt.folder/'private';private_path.mkdir(exist_ok=True)
                    tensor_path=private_path/'all_candidate_FP32_biases.pt'
                    torch.save(private,tensor_path)
                    import hashlib
                    save(rt.folder/'private_bias_vectors_receipt.json',dict(sha256=hashlib.sha256(tensor_path.read_bytes()).hexdigest(),dtype='FP32',complete_candidate_vectors=True,private_payload_not_published=True))
                if rt.selection is not None:
                    for u in rt.selection['units']:
                        assert len(u['tiles'])==max(1,math.floor(.05*(prefix//64)))
                        assert all(0<=i<prefix//64 for i in u['tiles'])
                        keep[0,u['head'],u['qb'],u['tiles']]=False
                    rt.pulse_calls+=1
                exact=[]
                for h in [0,8]:
                    for qb in [0,1]:
                        qu=q[qb*128:(qb+1)*128,h].float();ku=k[:,h//8].float();vu=v[:,h//8].float()
                        logits=qu@ku.T*impl.scale
                        mask=keep[0,h,qb].repeat_interleave(64)[:length]
                        logits.masked_fill_(~mask[None],-torch.inf);reference=logits.softmax(-1)@vu
                        exact.append((h,qb,reference))
                rt.work['qualification_QK_rectangles']=4*whole;rt.work['qualification_PV_rectangles']=4*whole
                torch.cuda.synchronize();rt.operator_seconds+=time.perf_counter()-start
                assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state()),'diagnostic RNG change'
            from experiments.numerical_qk_reuse import v27_fa4
            from experiments.numerical_qk_reuse.v31_local_sparse import alias64
            kk,vv,table=alias64(kc,vc,meta.block_table[0],length)
            result=v27_fa4.load()(q[None],kk,vv,softmax_scale=impl.scale,causal=False,page_table=table,
                seqused_k=torch.tensor([length],device=q.device,dtype=torch.int32),block_sparse_tensors=v27_fa4.block_sparse_tensors(keep))[0]
            if exact is not None:
                for h,qb,ref in exact:
                    observed=result[0,qb*128:(qb+1)*128,h].float()
                    error=float(((observed-ref).norm(dim=-1)/ref.norm(dim=-1).clamp_min(1e-8)).max())
                    rt.qual.append(dict(head=h,kv_head=h//8,qb=qb,error=error,deleting=rt.selection is not None))
                    assert error<=.03,'deleting/current-QKV consumer qualification failed'
            actual_support=[]
            if rt.selection is not None and target:
                actual_support=[dict(head=u['head'],qb=u['qb'],tiles=(~keep[0,u['head'],u['qb'],:prefix//64]).nonzero().flatten().cpu().tolist()) for u in rt.selection['units']]
                assert actual_support==rt.selection['units'],'actual mask differs from selected support'
            skipped=int((~keep).sum());protected=int((~keep[:,:,:,prefix//64:]).sum());assert protected==0
            rt.records.append(dict(canvas=rt.state['C'],step=rt.step,prefix=prefix,eligible=keep.numel(),kept=int(keep.sum()),skipped=skipped,
                protected_skipped=protected,LOCAL_skipped=0,current_QKV=True,layer=29,actual_deleted_support=actual_support,consumer_mask_sha256=digest(keep) if target else None,phase='one_pulse' if skipped else 'allkept'))
            output[:n].copy_(result[0].reshape_as(output[:n]));return output
        dg.DiffusionGemmaModelState.prepare_attn=prepare;fa.FlashAttentionImpl.forward=forward

    def receipt(self):
        return dict(scope='GLOBAL layer29; heads0/8 queryblocks0/1 delete; all16heads all-kept otherwise; Q128xKV64',anchor=self.anchor,pulse_calls=self.pulse_calls,physical_calls=self.records,current_operand_routes=self.routes,GQA_qualification=self.qual,
            diagnostic_work=self.work,operator_seconds=self.operator_seconds,current_QKV=True,extra_model_forwards=0,old_KV_consumed=False,
            cached_attention_outputs=False,schedule='one pulse then common all-kept, native bootstrap1 each canvas')
