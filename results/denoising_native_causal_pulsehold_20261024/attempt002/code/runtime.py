"""Selector-only four-unit pulse/hold adapter; native Q/K/V are always current."""
import torch
from pulsehold import next_branch_action


class Runtime:
    def __init__(self):self.reset()
    def reset(self):
        self.enabled=False;self.phase=None;self.calls=0;self.canvas=0
        self.anchor=None;self.schedule='AA';self.consumer='native'
        self.support=None;self.records=[];self.capture=False;self.frames=[]
        self.gqa_qualified=False;self.gqa_receipt=None
        self.global_eligible=0;self.local_eligible=0
    def install(self):
        from vllm.model_executor.models import diffusion_gemma as dg
        from vllm.v1.attention.backends import flash_attn as fa
        prepare0=dg.DiffusionGemmaModelState.prepare_attn
        forward0=fa.FlashAttentionImpl.forward
        rt=self
        def prepare(state,batch,mode,tables,slots,groups,config,for_capture=False,ubatch_idx=0):
            rt.phase=None
            if rt.enabled and not for_capture and batch.num_reqs==1:
                slot=int(batch.idx_mapping_np[0])
                encoder=bool(state.diffusion_states.is_encoder_phase[slot].item())
                rt.phase='encoder' if encoder or batch.num_draft_tokens==0 else 'denoise'
                if rt.phase=='denoise':rt.calls+=1
                if rt.phase=='denoise':
                    from experiments.numerical_qk_reuse.v31_value_native_audit import rectangles
                    nk=int(batch.seq_lens[0].item());n=int(batch.num_tokens)
                    types=state.model_config.hf_text_config.layer_types
                    global_layers=sum(x=='full_attention' for x in types)
                    heads=state.model_config.hf_text_config.num_attention_heads
                    rt.global_eligible+=global_layers*rectangles(heads,n,nk)
                    rt.local_eligible+=(len(types)-global_layers)*rectangles(heads,n,nk,local=True)
            return prepare0(state,batch,mode,tables,slots,groups,config,for_capture=for_capture,ubatch_idx=ubatch_idx)
        def forward(impl,layer,query,key,value,cache,meta,output,output_scale=None,output_block_scale=None):
            target=rt.enabled and rt.phase=='denoise' and '.layers.29.' in getattr(layer,'layer_name','')
            if not target:return forward0(impl,layer,query,key,value,cache,meta,output,output_scale,output_block_scale)
            if impl.num_heads!=16 or impl.num_kv_heads!=2:raise RuntimeError('unqualified GQA geometry')
            n=int(meta.num_actual_tokens);length=int(meta.seq_lens[0].item());prefix=length-n
            if n!=256:raise RuntimeError('requires native full first canvas256')
            kc,vc=cache.transpose(1,2).split(impl.head_size,dim=-1)
            q=query[:n]
            if rt.capture:
                page=kc.shape[1];pages=meta.block_table[0,:(length+page-1)//page].long()
                k=kc.index_select(0,pages).flatten(0,1)[:length]
                v=vc.index_select(0,pages).flatten(0,1)[:length]
                rt.frames.append((q.detach().cpu().clone(),k.detach().cpu().clone(),v.detach().cpu().clone(),prefix,float(impl.scale)))
                rt.frames=rt.frames[-2:]
            if rt.consumer=='native' or rt.anchor is None or rt.calls<=rt.anchor or rt.canvas>0:
                return forward0(impl,layer,query,key,value,cache,meta,output,output_scale,output_block_scale)
            action=next_branch_action(rt.schedule,rt.calls-rt.anchor-1,False)
            from experiments.numerical_qk_reuse import v27_fa4
            from experiments.numerical_qk_reuse.v31_local_sparse import alias64
            kk,vv,table=alias64(kc,vc,meta.block_table[0],length)
            kept=torch.ones((1,impl.num_heads,2,(length+63)//64),device=q.device,dtype=torch.bool)
            if action=='S':
                if rt.support is None or tuple(rt.support.shape)!=(4,prefix//64):raise RuntimeError('support geometry mismatch')
                for u,(h,b) in enumerate(((0,0),(0,1),(8,0),(8,1))):kept[0,h,b,:prefix//64]=rt.support[u].to(q.device)
            lists=v27_fa4.block_sparse_tensors(kept)
            result=v27_fa4.load()(q[None],kk,vv,softmax_scale=impl.scale,causal=False,page_table=table,
                seqused_k=torch.tensor([length],device=q.device,dtype=torch.int32),block_sparse_tensors=lists)[0]
            if action=='A' and not rt.gqa_qualified:
                page=kc.shape[1];pages=meta.block_table[0,:(length+page-1)//page].long()
                current_k=kc.index_select(0,pages).flatten(0,1)[:length].double()
                current_v=vc.index_select(0,pages).flatten(0,1)[:length].double()
                errors={}
                for head in (0,8):
                    reference=(q[:,head].double()@current_k[:,head//8].T*impl.scale).softmax(-1)@current_v[:,head//8]
                    error=(result[0,:,head].double()-reference).norm(dim=-1)/reference.norm(dim=-1).clamp_min(1e-12)
                    errors[str(head)]=float(error.max())
                if max(errors.values())>.02:raise RuntimeError('GQA independent current-QKV reference mismatch')
                rt.gqa_receipt=dict(passed=True,head_to_kv={'0':0,'8':1},worst_relative_output_error=errors,tolerance=.02)
                rt.gqa_qualified=True
            output[:n].copy_(result[0].reshape_as(output[:n]))
            rt.records.append(dict(call=rt.calls,action=action,layer=29,heads=[0,8],query_blocks=[0,1],
                gqa_mapping={'0':0,'8':1},prefix=prefix,scope_eligible=4*(prefix//64),
                scope_kept=4*(prefix//64)-int((~kept).sum()),global_target_layer_eligible=kept.numel(),
                global_target_layer_kept=int(kept.sum()),skipped=int((~kept).sum()),
                current_canvas_skipped=0,local_skipped=0,kernel='paged current-QKV SM90 FA4 sparse'))
            return output
        dg.DiffusionGemmaModelState.prepare_attn=prepare
        fa.FlashAttentionImpl.forward=forward
