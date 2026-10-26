"""Frozen common A5/17/29 consumer; optional selector callback never caches output."""
import math
import torch

LAYERS=(5,17,29)

def fingerprint(t):
    import hashlib
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

def gather_current(q,key,value,kc,vc,meta,prefix):
    length=int(meta.seq_lens[0]);page=kc.shape[1];pages=meta.block_table[0,:math.ceil(length/page)].long()
    k=kc.index_select(0,pages).flatten(0,1)[:length];v=vc.index_select(0,pages).flatten(0,1)[:length]
    assert torch.equal(k[prefix:],key[:len(q)]) and torch.equal(v[prefix:],value[:len(q)]),'current tail mismatch'
    return k,v

def native_attention(q,kc,vc,meta,scale,support=None):
    """Fresh native FA4 current-QKV computation on retained rectangles."""
    from experiments.numerical_qk_reuse import v27_fa4
    from experiments.numerical_qk_reuse.v31_local_sparse import alias64
    length=int(meta.seq_lens[0]);prefix=length-len(q);whole=math.ceil(length/64)
    keep=torch.ones((1,16,2,whole),dtype=torch.bool,device=q.device)
    if support:
        for u in support:
            assert u['head'] in [0,8] and u['qb'] in [0,1]
            assert all(0<=t<prefix//64 for t in u['tiles'])
            keep[0,u['head'],u['qb'],u['tiles']]=False
    assert not bool((~keep[:,:,:,prefix//64:]).any())
    kk,vv,table=alias64(kc,vc,meta.block_table[0],length)
    result=v27_fa4.load()(q[None],kk,vv,softmax_scale=scale,causal=False,page_table=table,
        seqused_k=torch.tensor([length],device=q.device,dtype=torch.int32),block_sparse_tensors=v27_fa4.block_sparse_tensors(keep))[0][0]
    return result,keep

class CommonConsumer:
    def __init__(self,state):self.state=state;self.callback=None;self.after_callback=None;self.model=None
    def begin(self,common=True):
        self.active=True;self.common=common;self.phase=None;self.records=[];self.extra_work={};self.gqa=[];self.gqa_pending=set()
    def install(self):
        from vllm.model_executor.models import diffusion_gemma as dg
        from vllm.v1.attention.backends import flash_attn as fa
        rt=self;prepare0=dg.DiffusionGemmaModelState.prepare_attn;forward0=fa.FlashAttentionImpl.forward
        def prepare(modelstate,batch,mode,tables,slots,groups,config,for_capture=False,ubatch_idx=0):
            rt.phase=None
            if rt.state['bound'] and rt.common and not for_capture and batch.num_reqs==1:
                slot=int(batch.idx_mapping_np[0]);encoder=bool(modelstate.diffusion_states.is_encoder_phase[slot].item())
                rt.phase='encoder' if encoder or batch.num_draft_tokens==0 else 'denoise'
            return prepare0(modelstate,batch,mode,tables,slots,groups,config,for_capture=for_capture,ubatch_idx=ubatch_idx)
        def forward(impl,layer,query,key,value,cache,meta,output,output_scale=None,output_block_scale=None):
            name=getattr(layer,'layer_name','');location=next((i for i in LAYERS if f'.layers.{i}.' in name),None)
            active=rt.state['bound'] and rt.common and rt.phase=='denoise' and location is not None and rt.state['step']+1>1
            if not active:return forward0(impl,layer,query,key,value,cache,meta,output,output_scale,output_block_scale)
            n=int(meta.num_actual_tokens);length=int(meta.seq_lens[0]);prefix=length-n
            assert n==256 and impl.num_heads==16 and impl.num_kv_heads==2 and impl.head_size==512
            q=query[:n];kc,vc=cache.transpose(1,2).split(impl.head_size,dim=-1)
            support=None;context=dict(layer=location,q=q,key=key,value=value,kc=kc,vc=vc,meta=meta,prefix=prefix,scale=impl.scale,step=rt.state['step']+1,canvas=rt.state['C'])
            if rt.callback is not None:support=rt.callback(context)
            # This is always a FRESH consumer operation, including after bank probes.
            result,keep=native_attention(q,kc,vc,meta,impl.scale,support)
            if location in rt.gqa_pending:
                k,v=gather_current(q,key,value,kc,vc,meta,prefix)
                for h in [0,8]:
                    for qb in [0,1]:
                        qu=q[qb*128:(qb+1)*128,h].float();ku=k[:,h//8].float();vu=v[:,h//8].float();p=(qu@ku.T*impl.scale).softmax(-1);ref=p@vu
                        error=float(((result[qb*128:(qb+1)*128,h].float()-ref).norm(dim=-1)/ref.norm(dim=-1).clamp_min(1e-8)).max());assert error<=.03
                        rt.gqa.append(dict(layer=location,head=h,kv_head=h//8,qb=qb,relative_error=error,current_QKV=True))
                whole=math.ceil(length/64);rt.extra_work['GQA_QK_rectangles']=rt.extra_work.get('GQA_QK_rectangles',0)+4*whole;rt.extra_work['GQA_PV_rectangles']=rt.extra_work.get('GQA_PV_rectangles',0)+4*whole
                rt.gqa_pending.remove(location)
            if rt.after_callback is not None:rt.after_callback(context,support,result,keep)
            if rt.state['audit']:
                skipped=int((~keep).sum());rt.records.append(dict(layer=location,canvas=rt.state['C'],native_step=rt.state['step']+1,prefix=prefix,eligible=keep.numel(),kept=keep.numel()-skipped,skipped=skipped,LOCAL_skipped=0,protected_skipped=0,current_QKV=True,actual_support=support))
            output[:n].copy_(result.reshape_as(output[:n]));return output
        dg.DiffusionGemmaModelState.prepare_attn=prepare;fa.FlashAttentionImpl.forward=forward
