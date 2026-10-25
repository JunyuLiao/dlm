"""Online current-only bank; one opportunity, independently eligible CE/sham."""
import math,time,json,torch
from bank import build
from measure import digest

class Runtime:
 def __init__(self,state):self.state=state;self.active=False
 def begin(self,arm,audit):
  self.active=True;self.arm=arm;self.audit=audit;self.common=arm not in ['corrected_dense_full','dense_piecewise'];self.prefix=None;self.step=0;self.phase=None
  self.anchor=None;self.bank=None;self.selection=None;self.attempted=False;self.pulse_calls=0;self.reason='no_native_anchor';self.records=[];self.qual=[];self.work={};self.selector_seconds=0.;self.route=None
 def install(self):
  from vllm.model_executor.models import diffusion_gemma as dg
  from vllm.v1.attention.backends import flash_attn as fa
  prep=dg.DiffusionGemmaModelState.prepare_attn;forward0=fa.FlashAttentionImpl.forward;rt=self
  def prepare(state,batch,mode,tables,slots,groups,config,for_capture=False,ubatch_idx=0):
   rt.phase=None
   if rt.active and not for_capture and batch.num_reqs==1:
    slot=int(batch.idx_mapping_np[0]);enc=bool(state.diffusion_states.is_encoder_phase[slot].item());rt.phase='encoder' if enc or batch.num_draft_tokens==0 else 'denoise'
    if rt.phase=='denoise':rt.step+=1
   return prep(state,batch,mode,tables,slots,groups,config,for_capture=for_capture,ubatch_idx=ubatch_idx)
  def forward(impl,layer,query,key,value,cache,meta,output,output_scale=None,output_block_scale=None):
   if not(rt.active and rt.common and rt.phase=='denoise' and '.layers.29.' in getattr(layer,'layer_name','')):return forward0(impl,layer,query,key,value,cache,meta,output,output_scale,output_block_scale)
   n=int(meta.num_actual_tokens);length=int(meta.seq_lens[0]);prefix=length-n
   assert n==256 and impl.num_heads==16 and impl.num_kv_heads==2 and impl.head_size==512
   if rt.prefix!=prefix:rt.prefix=prefix;rt.step=1
   if rt.step==1:return forward0(impl,layer,query,key,value,cache,meta,output,output_scale,output_block_scale)
   q=query[:n];kc,vc=cache.transpose(1,2).split(impl.head_size,dim=-1);whole=math.ceil(length/64);keep=torch.ones((1,16,2,whole),dtype=torch.bool,device=q.device)
   target=rt.anchor is not None and rt.state['C']==rt.anchor['canvas'] and rt.step==rt.anchor['after_completed_calls']+1
   deleting=False;exact=None
   if target and rt.arm=='matched_allkept_layer29' and rt.audit:
    page=kc.shape[1];pages=meta.block_table[0,:math.ceil(length/page)].long();k=kc.index_select(0,pages).flatten(0,1)[:length];v=vc.index_select(0,pages).flatten(0,1)[:length]
    assert torch.equal(k[prefix:],key[:n]) and torch.equal(v[prefix:],value[:n])
    rt.route=dict(query=digest(q),key=digest(k),value=digest(v),prefix=prefix,N=rt.state['N'],C=rt.state['C'],cpu_rng=digest(torch.get_rng_state()),cuda_rng=digest(torch.cuda.get_rng_state()))
   if target and rt.arm!='matched_allkept_layer29':
    assert not rt.attempted;rt.attempted=True;start=time.perf_counter();cpu=torch.get_rng_state().clone() if rt.audit else None;gpu=torch.cuda.get_rng_state().clone() if rt.audit else None
    page=kc.shape[1];pages=meta.block_table[0,:math.ceil(length/page)].long();k=kc.index_select(0,pages).flatten(0,1)[:length];v=vc.index_select(0,pages).flatten(0,1)[:length]
    if rt.audit:
     assert torch.equal(k[prefix:],key[:n]) and torch.equal(v[prefix:],value[:n]),'stale current tail'
     rt.route=dict(query=digest(q),key=digest(k),value=digest(v),prefix=prefix,N=rt.state['N'],C=rt.state['C'],cpu_rng=digest(cpu),cuda_rng=digest(gpu))
    if prefix//64<4:rt.reason='fewer_than_four_prefix_blocks'
    else:
     bank,private,work=build(q,k,v,prefix,impl.scale,audit=rt.audit);rt.bank=bank;rt.work=work
     name=rt.arm.removesuffix('_single_pulse');rt.selection=bank['selections'].get(name)
     rt.reason=None if rt.selection else ('no_feasible_four_unit_support' if not bank['feasible_common'] else 'no_matched_direction_pair')
    if rt.selection:
     for u in rt.selection['units']:
      assert len(u['tiles'])==max(1,math.floor(.05*(prefix//64))) and all(0<=i<prefix//64 for i in u['tiles'])
      keep[0,u['head'],u['qb'],u['tiles']]=False
     rt.pulse_calls+=1;deleting=True
    if rt.audit:
     exact=[]
     for h in [0,8]:
      for qb in [0,1]:
       qu=q[qb*128:(qb+1)*128,h].float();ku=k[:,h//8].float();vu=v[:,h//8].float();logits=qu@ku.T*impl.scale
       mask=keep[0,h,qb].repeat_interleave(64)[:length];logits.masked_fill_(~mask[None],-torch.inf);exact.append((h,qb,logits.softmax(-1)@vu))
     rt.work['qualification_QK_rectangles']=4*whole;rt.work['qualification_PV_rectangles']=4*whole
     assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state()),'operator RNG changes'
     torch.cuda.synchronize();rt.selector_seconds=time.perf_counter()-start
   from experiments.numerical_qk_reuse import v27_fa4
   from experiments.numerical_qk_reuse.v31_local_sparse import alias64
   kk,vv,table=alias64(kc,vc,meta.block_table[0],length)
   result=v27_fa4.load()(q[None],kk,vv,softmax_scale=impl.scale,causal=False,page_table=table,seqused_k=torch.tensor([length],device=q.device,dtype=torch.int32),block_sparse_tensors=v27_fa4.block_sparse_tensors(keep))[0]
   if exact is not None:
    for h,qb,ref in exact:
     obs=result[0,qb*128:(qb+1)*128,h].float();error=float(((obs-ref).norm(dim=-1)/ref.norm(dim=-1).clamp_min(1e-8)).max());rt.qual.append(dict(head=h,kv_head=h//8,qb=qb,error=error,deleting=deleting));assert error<=.03
   if rt.audit:
    actual=[dict(head=u['head'],qb=u['qb'],tiles=(~keep[0,u['head'],u['qb'],:prefix//64]).nonzero().flatten().cpu().tolist()) for u in rt.selection['units']] if deleting else []
    assert not deleting or actual==rt.selection['units'];assert not bool((~keep[:,:,:,prefix//64:]).any())
    rt.records.append(dict(canvas=rt.state['C'],step=rt.step,prefix=prefix,eligible=keep.numel(),kept=int(keep.sum()),skipped=int((~keep).sum()),protected_skipped=0,LOCAL_skipped=0,actual_deleted_support=actual,current_QKV=True))
   output[:n].copy_(result[0].reshape_as(output[:n]));return output
  dg.DiffusionGemmaModelState.prepare_attn=prepare;fa.FlashAttentionImpl.forward=forward
 def receipt(self):
  return dict(anchor=self.anchor,attempted=self.attempted,pulse_calls=self.pulse_calls,abstention=self.reason if self.pulse_calls==0 else None,selected_support=self.selection,
   candidate_bank=self.bank if self.audit else None,physical_calls=self.records,current_operand_route=self.route,GQA_qualification=self.qual,diagnostic_work=self.work,selector_audit_seconds=self.selector_seconds,current_QKV=True,extra_model_forwards=0,old_KV_consumed=False,
   eligibility_revision='CE/sham independent of directional pair; one opportunity only',carried_calls=0,held_calls=0)
