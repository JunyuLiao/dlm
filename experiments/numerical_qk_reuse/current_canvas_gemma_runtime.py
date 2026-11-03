"""DiffusionGemma current-canvas V2 selector in its inherited native lifecycle."""
import types
import torch


def install(adapter,**unused):
    from . import v27_fa4
    from .v31_value_kernels import statistics_cuda
    from .v2_current_canvas_init import current_canvas_cuda
    from experiments.diffusion_gemma_jl_output_aware.projections import Projections
    if adapter.mage_sink or adapter.mage_recent_tiles or adapter.mage_row_weight is not None:
        raise ValueError('unsupported added protections')
    projections=Projections()
    old_begin,old_receipt=adapter.begin_request,adapter._value_receipt
    old_list=v27_fa4.block_sparse_tensors
    state={'keep':None,'lists':None,'calls':0,'shapes':set()}
    def lists(keep):
        return state['lists'] if keep is state['keep'] else old_list(keep)
    v27_fa4.block_sparse_tensors=lists
    def begin(self,*a,**kw):
        result=old_begin(*a,**kw)
        state.update(keep=None,lists=None,calls=0,shapes=set())
        return result
    def select(self,layer,q,b,scale,prefix,n):
        k,v=b['k'],b['v'];h,hk,nk,d=q.shape[1],k.shape[1],k.shape[2],q.shape[-1]
        if d!=512 or nk!=prefix+n:raise ValueError('GLOBAL geometry drift')
        pt=prefix//64;qb=(n+127)//128;jt=(nk+63)//64
        matrix=projections.get(layer,hk,d,'gaussian',32,1729,q.device)
        z=(v[0].float()@matrix).unsqueeze(0).contiguous()
        nu=v[0].float().square().sum(-1).mean(-1).sqrt().clamp_min(1e-12)
        keep=torch.empty((1,h,qb,jt),device=q.device,dtype=torch.bool)
        for qi in range(qb):
            stats=statistics_cuda(q[:,:,qi*128:(qi+1)*128],k,z,nu,scale,pt)
            mandatory=(torch.arange(jt,device=q.device)[None]>=pt).expand(h,-1).contiguous()
            held=None if self._mage_held is None else self._mage_held[:,:,qi].reshape(h,jt)
            selected,*_=current_canvas_cuda(stats,self.value_threshold,mandatory,pt,held,
                0. if held is None else self.mage_sticky)
            keep[:,:,qi]=selected
            self.calls['value_statistics_qk_rectangles']=self.calls.get('value_statistics_qk_rectangles',0)+h*jt
            self.calls['value_peak_summary_bytes']=max(self.calls.get('value_peak_summary_bytes',0),stats.nbytes)
        packed=old_list(keep)
        out=v27_fa4.sparse_lists(q,k,v,packed,scale)
        state.update(keep=keep,lists=packed,calls=state['calls']+1)
        state['shapes'].add((n,nk,pt,jt))
        if not self.value_clean_timing:
            kept=keep[...,:pt].sum().double()
            self._sparse_kept=kept if self._sparse_kept is None else self._sparse_kept+kept
            self._sparse_total+=h*qb*pt
            self._kept_prefix=kept if self._kept_prefix is None else self._kept_prefix+kept
            self._prefix_total+=h*qb*pt
        self.calls['value_selection_calls']=self.calls.get('value_selection_calls',0)+1
        self.calls['mage_prefix_tiles']+=pt*h*qb
        if held is not None:self.calls['mage_sticky_units']=self.calls.get('mage_sticky_units',0)+1
        return out,keep
    def receipt(self):
        r=old_receipt()
        r.update(current_canvas_initialization=True,current_canvas_selector_calls=state['calls'],
                 current_canvas_selector_shapes=sorted(state['shapes']),
                 value_output_path='current_canvas_init_sparse_refresh_current_qkv',
                 value_state_role='router_only; FA4 consumes selected current full V',
                 value_discovery_work='full QK and Gaussian32 PZ; separately counted',
                 value_projection_manifest=projections.manifest,
                 optimized_refresh_extra_model_forwards=0,
                 optimized_refresh_projection_current_v=True,
                 current_canvas_kernel='existing Triton online chunks; mandatory suffix then prefix')
        return r
    adapter.begin_request=types.MethodType(begin,adapter)
    adapter._select_initial_refresh=types.MethodType(select,adapter)
    adapter._value_receipt=types.MethodType(receipt,adapter)
    return state
