"""Exact diagnostic support bank and current-Q/K/V native FA4 consumer."""
import hashlib
import math
import torch


def differential(p, q, k, v, dq, dk, dv, scale):
    o = p @ v
    def cov(ds):
        z = p * ds
        return z @ v - z.sum(-1, keepdim=True) * o
    return cov(scale*dq@k.T) + cov(scale*q@dk.T) + p@dv


def bank(q, k, v, previous, prefix, fraction, scale):
    """Every candidate has the same number of eligible prefix KV64 tiles."""
    blocks = prefix//64
    budget = max(1, min(blocks, math.ceil(blocks*fraction)))
    p = (scale*q@k.T).softmax(-1)
    dense = p@v
    mass = p[:, :blocks*64].reshape(len(q), blocks, 64).sum(-1)
    order = mass.amax(0).argsort(descending=True, stable=True)
    masks = {}
    def chosen(indices):
        mask = torch.zeros(blocks, dtype=torch.bool, device=q.device)
        mask[indices] = True
        return mask
    masks['mass'] = chosen(order[:budget])
    if budget < blocks:
        masks['swap_near'] = chosen(torch.cat((order[:budget-1], order[budget:budget+1])))
        masks['swap_far'] = chosen(torch.cat((order[:budget-1], order[-1:])))
        sham = torch.remainder(torch.arange(blocks,device=q.device)*73+19,max(blocks,1)).argsort(stable=True)
        masks['position_sham'] = chosen(sham[:budget])
    out = []
    seen = set()
    for label, mask in masks.items():
        support = torch.cat((mask.repeat_interleave(64),torch.ones(len(k)-blocks*64,dtype=torch.bool,device=q.device)))
        identity = hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest()
        if identity in seen:
            continue
        seen.add(identity)
        retained = p*support[None]
        mass_kept = retained.sum(-1)
        ps = retained/mass_kept[:,None]
        sparse = ps@v
        bias = sparse-dense
        denominator = dense.norm(dim=-1).clamp_min(1e-12)
        response = []
        if previous is not None:
            oldq, oldk, oldv = [x.to(device=q.device,dtype=q.dtype) for x in previous]
            dq, dk, dv = q-oldq, k-oldk, v-oldv
            for direction in ((dq,torch.zeros_like(dk),torch.zeros_like(dv)),
                    (torch.zeros_like(dq),dk,dv),(dq,dk,dv)):
                d = differential(ps,q,k,v,*direction,scale)-differential(p,q,k,v,*direction,scale)
                response.append(float((d.norm(dim=-1)/denominator).max()))
        deleted = 1-mass_kept
        bias_max = float((bias.norm(dim=-1)/denominator).max())
        whole_deleted = float(deleted.max())
        prefix_total = p[:,:prefix].sum(-1).clamp_min(1e-12)
        record = dict(candidate=label,support_sha256=identity,mask=mask,
            prefix_blocks=blocks,kept_prefix_blocks=budget,
            whole_deleted_mass_max=whole_deleted,prefix_deleted_mass_max=float((deleted/prefix_total).max()),
            current_output_error=bias_max,response=max(response) if response else None,
            canvas_amplification_max=float((1/mass_kept).max()),
            feasible=whole_deleted<=.2 and bias_max<=.1 and float((1/mass_kept).max())<=1.25)
        out.append(record)
    return out


class Runtime:
    def __init__(self):
        self.begin()

    def begin(self):
        self.bound = False
        self.arm = 'native'
        self.phase = None
        self.calls = 0
        self.history = []
        self.captured = None
        self.mask = None
        self.records = []
        self.target_call = None
        self.duration = 1
        self.heads = (0,)
        self.capture_enabled = False

    def install(self):
        from vllm.model_executor.models import diffusion_gemma as dg
        from vllm.v1.attention.backends import flash_attn as fa
        original_prepare = dg.DiffusionGemmaModelState.prepare_attn
        original_forward = fa.FlashAttentionImpl.forward
        rt = self
        def prepare(state,batch,mode,tables,slots,groups,config,for_capture=False,ubatch_idx=0):
            if rt.bound and not for_capture and batch.num_reqs==1:
                slot = int(batch.idx_mapping_np[0])
                encoder = bool(state.diffusion_states.is_encoder_phase[slot].item())
                rt.phase = 'encoder' if encoder or batch.num_draft_tokens==0 else 'denoise'
                if rt.phase=='denoise':
                    rt.calls += 1
            else:
                rt.phase = None
            return original_prepare(state,batch,mode,tables,slots,groups,config,for_capture=for_capture,ubatch_idx=ubatch_idx)
        def forward(impl,layer,query,key,value,cache,meta,output,output_scale=None,output_block_scale=None):
            name = getattr(layer,'layer_name','')
            target = rt.bound and rt.phase=='denoise' and '.layers.29.' in name
            if not target:
                return original_forward(impl,layer,query,key,value,cache,meta,output,output_scale,output_block_scale)
            n = int(meta.num_actual_tokens)
            length = int(meta.seq_lens[0].item())
            prefix = length-n
            kc, vc = cache.transpose(1,2).split(impl.head_size,dim=-1)
            page = kc.shape[1]
            pages = meta.block_table[0, :(length+page-1)//page].long()
            # Gather CURRENT referenced native cache on every call. No stale K/V.
            k = kc.index_select(0,pages).flatten(0,1)[:length]
            v = vc.index_select(0,pages).flatten(0,1)[:length]
            q = query[:n]
            if rt.capture_enabled:
                rt.captured = (q[:128,0].detach().cpu().clone(),k[:,0].detach().cpu().clone(),v[:,0].detach().cpu().clone(),prefix,float(impl.scale))
            if rt.arm=='native' or rt.target_call is None or not (rt.target_call<=rt.calls<rt.target_call+rt.duration):
                return original_forward(impl,layer,query,key,value,cache,meta,output,output_scale,output_block_scale)
            from experiments.numerical_qk_reuse import v27_fa4
            from experiments.numerical_qk_reuse.v31_local_sparse import alias64
            kk,vv,table = alias64(kc,vc,meta.block_table[0],length)
            kept = torch.ones((1,impl.num_heads,(n+127)//128,(length+63)//64),device=q.device,dtype=torch.bool)
            if rt.arm!='allkept':
                if rt.mask is None or rt.mask.shape!=(prefix//64,):
                    raise RuntimeError('frozen support mismatch')
                for head in rt.heads:
                    kept[0,head,0,:prefix//64] = rt.mask.to(q.device)
            lists = v27_fa4.block_sparse_tensors(kept)
            fwd = v27_fa4.load()
            result = fwd(q[None],kk,vv,softmax_scale=impl.scale,causal=False,
                page_table=table,seqused_k=torch.tensor([length],device=q.device,dtype=torch.int32),
                block_sparse_tensors=lists)[0]
            output[:n].copy_(result[0].reshape(n,-1))
            rt.records.append(dict(call=rt.calls,layer=29,heads=list(rt.heads),query_block=0,
                eligible_tiles=kept.numel(),kept_tiles=int(kept.sum()),skipped_tiles=int((~kept).sum()),
                prefix=prefix,n=n,kernel='native SM90 FA4 block-sparse paged current Q/K/V',
                current_canvas_deleted_tiles=0,local_skipped_tiles=0))
            return output
        dg.DiffusionGemmaModelState.prepare_attn = prepare
        fa.FlashAttentionImpl.forward = forward
