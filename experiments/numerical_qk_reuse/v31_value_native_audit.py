"""Audit native dense geometry at actual prepare-attention calls.

Only the separate audit pass installs this recorder. It leaves native FULL/
PIECEWISE graphs and attention consumers intact, and counts a common Q128 x
KV64 rectangle denominator rather than claiming to count CUDA CTAs.
"""
import math


def rectangles(heads, n, nk, local=False):
    if not local:
        return heads*math.ceil(n/128)*math.ceil(nk/64)
    prefix=nk-n
    count=0
    for begin in range(0,n,128):
        lo=max(0,prefix+begin-1023)
        hi=min(nk-1,prefix+min(begin+127,n-1)+1023)
        count+=hi//64-lo//64+1
    return heads*count


class NativeGeometryAudit:
    def __init__(self, layer_types, heads):
        self.global_layers=sum(kind=='full_attention' for kind in layer_types)
        self.local_layers=len(layer_types)-self.global_layers
        self.heads=heads
        self.bound=False
        self.begin_request()
        self.bound=False

    def begin_request(self):
        self.bound=True
        self.calls=self.global_tiles=self.local_tiles=0
        self.nonfinite=[]
        self.decode_modes={}

    def logits(self, scaled):
        if self.bound:
            import torch
            self.nonfinite.append(~torch.isfinite(scaled).all())

    def record(self, encoder, draft, n, nk, mode=None):
        if not self.bound or encoder or not draft:
            return
        self.calls+=1
        if mode is not None:
            self.decode_modes[mode]=self.decode_modes.get(mode,0)+1
        self.global_tiles+=self.global_layers*rectangles(self.heads,n,nk)
        self.local_tiles+=self.local_layers*rectangles(self.heads,n,nk,local=True)

    def receipt(self):
        import torch
        self.bound=False
        return dict(global_calls=self.calls*self.global_layers,
            decode_cudagraph_modes=dict(self.decode_modes),
            global_eligible_tiles=self.global_tiles,global_kept_tiles=self.global_tiles,
            native_local_calls_audited=self.calls*self.local_layers,
            native_local_eligible_rectangles=self.local_tiles,native_local_skipped_tiles=0,
            overall_eligible_rectangles=self.global_tiles+self.local_tiles,
            overall_skipped_rectangles=0,overall_rectangle_sparsity=0,
            value_phase_tiles={'dense':dict(calls=self.calls*self.global_layers,
                eligible_tiles=self.global_tiles,kept_tiles=self.global_tiles,
                skipped_tiles=0,sparsity=0)},
            native_local_phase_rectangles={'dense':dict(calls=self.calls*self.local_layers,
                eligible_rectangles=self.local_tiles,skipped_tiles=0,sparsity=0)},
            value_nonfinite_logits_calls=int(torch.stack(self.nonfinite).sum().item()) if self.nonfinite else 0,
            native_dense_geometry_source='actual prepare_attn decode metadata; Q128 x KV64 rectangles, not CUDA CTA count',
            order_errors=0,value_nonfinite_attention_calls=None)

    def install(self):
        import torch
        import vllm.model_executor.models.diffusion_gemma as dg
        original=dg.DiffusionGemmaModelState.prepare_attn
        audit=self

        def prepare(state,input_batch,cudagraph_mode,block_tables,slot_mappings,attn_groups,
                    kv_cache_config,for_capture=False,ubatch_idx=0):
            if audit.bound and not for_capture and input_batch.num_reqs==1:
                slot=int(input_batch.idx_mapping_np[0])
                encoder,nk=torch.stack((state.diffusion_states.is_encoder_phase[slot].long(),
                                       input_batch.seq_lens[0].long())).tolist()
                audit.record(bool(encoder),input_batch.num_draft_tokens>0,
                             int(input_batch.num_tokens),int(nk),str(cudagraph_mode))
            return original(state,input_batch,cudagraph_mode,block_tables,slot_mappings,attn_groups,
                            kv_cache_config,for_capture=for_capture,ubatch_idx=ubatch_idx)

        dg.DiffusionGemmaModelState.prepare_attn=prepare
