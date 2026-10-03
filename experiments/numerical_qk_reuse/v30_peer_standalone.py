"""Junyu Gaussian32 on vLLM GLOBAL calls, independently of M3.

The peer Attention/Kernel classes must come from a separately pinned export.
This adapter reuses paged KV transport only: no v21.install, DP, R6, score cache,
carry0, or alias-FA4 consumer. Official FA4 remains the external dense reference.
"""
from types import SimpleNamespace
import math
import re
import torch
from .vllm_adapter import VllmMethodAdapter, _StubAdapter, _CacheLayer, MAX_DENOISING_STEPS
from .peer_global_scope import GlobalScopeAdapter, GLOBAL_LAYERS

VARIANTS = ('kernel_dense', 'value_allkept', 'value_unit', 'value_confidence', 'value_temporal')


class QueryWeights:
    """Previous completed call only; never approximates native acceptance."""
    def __init__(self, mode):
        if mode not in ('unit','confidence','temporal'):raise ValueError('Unknown query prior')
        self.mode=mode;self.beta=3.;self.gamma=.5
        self.previous_top=self.temporal=self.confidence=None
        self.expected=None;self.pending=False;self.canvas=-1
        self.begins=self.observes=0

    def begin(self, step, shape, device):
        if self.pending:raise RuntimeError('Missing previous sample observation')
        if type(step) is not int or not 0<=step<MAX_DENOISING_STEPS:raise ValueError('Invalid native step')
        shape=tuple(shape)
        if step==0:
            self.previous_top=self.temporal=self.confidence=None;self.canvas+=1;self.expected=shape
        if self.canvas<0 or shape!=self.expected:raise ValueError('Missing canvas start or changed query geometry')
        self.pending=True;self.begins+=1
        if self.mode=='unit':return None
        if self.mode=='confidence':
            if self.confidence is None:return torch.full(shape,4.,device=device,dtype=torch.float32)
            return (1+3*(1-self.confidence).clamp_min(0).sqrt()).clamp(1,4).contiguous()
        return None if self.temporal is None else (1+3*self.temporal).clamp(1,4).contiguous()

    def observe(self, logits):
        if not self.pending:raise RuntimeError('Sample without a matching begin')
        if self.mode!='unit':
            if logits.ndim!=3 or tuple(logits.shape[:2])!=self.expected:raise ValueError('Logit/query geometry mismatch')
            if self.mode=='confidence':
                x=logits.float();self.confidence=(x.amax(-1)-torch.logsumexp(x,-1)).exp().clamp(0,1).detach()
            else:
                top=logits.argmax(-1)
                if self.previous_top is None:self.temporal=torch.zeros_like(top,dtype=torch.float32)
                else:self.temporal=.5*self.temporal+.5*(top!=self.previous_top).float()
                self.previous_top=top.detach()
        self.pending=False;self.observes+=1


def validate_policy(variant, policy):
    if variant not in VARIANTS:raise ValueError('Unknown standalone variant')
    if variant in ('kernel_dense','value_allkept'):
        if policy is not None:raise ValueError('All-kept controls have no finite policy')
        return -float('inf')
    if (not isinstance(policy,dict) or set(policy)!={'global_log_threshold','source_commit','source_file','calibration_scope'}
            or type(policy['global_log_threshold']) not in (int,float)
            or not math.isfinite(policy['global_log_threshold'])
            or any(not isinstance(policy[k],str) or not policy[k] for k in ('source_commit','source_file','calibration_scope'))):
        raise ValueError('Explicit finite peer threshold and calibration provenance required')
    if (not re.fullmatch(r'[0-9a-f]{40}',policy['source_commit'])
            or not re.fullmatch(r'[A-Za-z0-9_./-]+',policy['source_file'])
            or policy['source_file'].startswith('/') or '..' in policy['source_file'].split('/')):
        raise ValueError('Policy provenance requires a full Git commit and repository-relative source')
    return float(policy['global_log_threshold'])


def check_standalone_receipt(receipt, n, variant):
    """Reject fallback/partial coverage; this alone is not numerical qualification."""
    if variant not in VARIANTS or type(n) is not int or n <= 0:
        raise ValueError('Expected a named variant and positive actual forward count')
    peer=receipt.get('standalone_peer') or {}
    adapter=receipt.get('adapter') or {}
    if (peer.get('variant')!=variant or peer.get('consumer')!='Junyu_SM90_C_ABI'
            or peer.get('scope')!='five_GLOBAL_decoder_layers_only'
            or any(peer.get(k) is not False for k in ('m3_installed','dp','carry0','score_cache'))
            or receipt.get('method') is not None):
        raise ValueError('Wrong standalone algorithm/consumer receipt')
    layers=peer.get('calls_by_layer',{})
    if ({str(k):v for k,v in layers.items()}!={str(k):n for k in GLOBAL_LAYERS}
            or peer.get('calls')!=5*n or adapter.get('global_calls')!=5*n
            or any(peer.get(k)!=n for k in ('weight_begins','weight_observes'))
            or any(adapter.get(k)!=n for k in ('begins','observes'))
            or adapter.get('order_errors')!=0):
        raise ValueError('Incomplete GLOBAL/forward/sample coverage')
    mode={'value_confidence':'confidence','value_temporal':'temporal'}.get(variant,'unit')
    if peer.get('query_weights')!=mode or peer.get('kernel_dense_mode')!=(variant=='kernel_dense'):
        raise ValueError('Wrong query prior/dense path')
    projected=peer.get('projected_tokens')
    if type(projected) is not int or projected<0 or (projected==0)!=(variant=='kernel_dense'):
        raise ValueError('Unexpected value projection coverage')
    rows=peer.get('tile_counts')
    if not isinstance(rows,list) or not rows:raise ValueError('Missing physical tile counters')
    seen=set()
    for row in rows:
        if (type(row.get('layer')) is not int or row['layer'] not in GLOBAL_LAYERS
                or any(type(row.get(k)) is not int or row[k]<0 for k in ('skipped','eligible'))
                or row['skipped']>row['eligible']):
            raise ValueError('Invalid physical tile counters')
        seen.add(row['layer'])
    if seen!=set(GLOBAL_LAYERS):raise ValueError('Missing GLOBAL layer tile evidence')
    if variant in ('kernel_dense','value_allkept'):
        if peer.get('threshold') is not None or any(row['skipped'] for row in rows):
            raise ValueError('All-kept control skipped tiles')
    elif type(peer.get('threshold')) not in (int,float) or not math.isfinite(peer['threshold']):
        raise ValueError('Missing finite standalone policy')


class StandalonePeerAdapter(VllmMethodAdapter):
    def __init__(self,layer_types,*,peer_module,library,variant,policy=None,kv_copy_backend='torch'):
        threshold=validate_policy(variant,policy)
        super().__init__(layer_types,arm='allkept',lifecycle='request_clear',canvas_buffers='legacy',
                         kv_copy_backend=kv_copy_backend,merge_backend='torch')
        if tuple(self.global_layers)!=GLOBAL_LAYERS:raise ValueError('Five GLOBAL layers required')
        self.peer_module,self.library,self.variant,self.threshold=peer_module,library,variant,threshold
        self.router=None;self.weights=None;self.peer_calls=0;self.dense_dummy={}
        self.calls_by_layer={layer:0 for layer in GLOBAL_LAYERS}

    def begin_request(self):
        super().begin_request()  # allkept transport setup; never installs M3
        try:
            self.stub=_StubAdapter(self.layer_types)
            self.scoped=GlobalScopeAdapter(self.stub)
            thresholds={kind:dict(log_threshold=self.threshold) for kind in ('local','global')}
            self.router=self.peer_module.Attention(self.scoped,self.library,thresholds,
                precision='tf32x3_register',collect=True,tma=True,mode='value',projection='fused')
            mode={'value_confidence':'confidence','value_temporal':'temporal'}.get(self.variant,'unit')
            self.weights=QueryWeights(mode);self.peer_calls=0;self.dense_dummy={}
            self.calls_by_layer={layer:0 for layer in GLOBAL_LAYERS}
            for module in self.scoped._selected.values():
                module._blasst_2d_runtime=SimpleNamespace(current_denoising_iteration=0)
        except BaseException:
            self.end_request();raise

    def on_prepare(self,phase_encoder,step,seq_len,num_tokens):
        super().on_prepare(phase_encoder,step,seq_len,num_tokens)
        if phase_encoder:
            self.dense_dummy.clear()
            return
        self.router.query_sensitivity=self.weights.begin(int(step),(1,int(num_tokens)),self.canvas.device)
        for module in self.scoped._selected.values():
            module._blasst_2d_runtime.current_denoising_iteration=int(step)+1

    def on_sample(self,scaled_logits):
        super().on_sample(scaled_logits)  # counters only: runtime is always None
        n=self.step_ctx['n']
        if scaled_logits.ndim!=3 or scaled_logits.shape[0]!=1 or not 0<n<=scaled_logits.shape[1]:
            raise ValueError('Official padded logits do not cover real queries')
        self.weights.observe(scaled_logits[:,:n,:])

    def forward(self,impl,layer_idx,query,kv_cache,attn_metadata,output):
        if layer_idx not in GLOBAL_LAYERS or self.step_ctx is None or self.step_ctx['encoder']:
            raise ValueError('Standalone consumer reached an ineligible call')
        # This narrow qualification explicitly rejects unreviewed bias/features.
        if float(impl.scale)!=1.0 or getattr(impl,'alibi_slopes',None) is not None or getattr(impl,'logits_soft_cap',0) not in (None,0):
            raise ValueError('Unsupported official GLOBAL attention parameters')
        n=int(attn_metadata.num_actual_tokens);ctx=self.step_ctx
        if n!=ctx['n'] or ctx['seq_len']<n:raise ValueError('Invalid prepared KV geometry')
        prefix=ctx['seq_len']-n
        k,v=kv_cache.transpose(1,2).split(impl.head_size,dim=-1)
        buf=self._buffers(layer_idx,k,v,attn_metadata.block_table[0],prefix,n)
        q=query[:n].transpose(0,1).unsqueeze(0)
        self.cache.layers[layer_idx]=_CacheLayer(buf['pk'],buf['pv']);self.cache.length=prefix
        module=self.stub.model.layers[layer_idx]
        module(None,None,None,past_key_values=self.cache)
        if self.variant=='kernel_dense':
            shape=(1,buf['k'].shape[1],ctx['seq_len'],32)
            identity=(layer_idx,shape,str(q.device))
            if identity not in self.dense_dummy:
                geom=(1,n,ctx['seq_len'],q.device,False,None)
                if geom not in self.router.geometry:
                    self.router.geometry[geom]=self.peer_module.geometry(1,n,ctx['seq_len'],device=q.device,causal=False,window=0)
                packed,_=self.router.geometry[geom]
                self.dense_dummy[identity]=(torch.empty(shape,device=q.device,dtype=torch.float32),
                    torch.ones(shape[:2],device=q.device,dtype=torch.float32),packed)
            z,ref,packed=self.dense_dummy[identity]
            result=self.router.kernel(q.contiguous(),buf['k'],buf['v'],z,ref,mask=packed,
                scale=1.,log_threshold=-float('inf'),mode='dense',precision='tf32x3_register',tma=True)
            self.router.calls+=1
            self.router.pending.append((layer_idx,ctx['step']+1,'global',prefix,result.skipped,result.eligible))
            out=result.output.transpose(1,2).contiguous()
        else:
            out,_=self.router(module,q,buf['k'],buf['v'],None,scaling=1.,is_causal=False,sliding_window=None)
        output[:n].view(n,-1).copy_(out.reshape(n,-1))
        self.calls['global_calls']+=1;self.peer_calls+=1
        self.calls_by_layer[layer_idx]+=1
        return output

    def end_request(self):
        router=self.router
        peer=None
        try:
            if router is not None:
                if self.runtime is not None:raise RuntimeError('Unexpected M3 runtime on standalone path')
                rows=router.records()  # Called after the normal request boundary synchronization.
                if self.variant in ('kernel_dense','value_allkept') and any(r['skipped'] for r in rows):
                    raise RuntimeError('All-kept peer control skipped tiles')
                peer=dict(variant=self.variant,consumer='Junyu_SM90_C_ABI',scope='five_GLOBAL_decoder_layers_only',
                          m3_installed=False,dp=False,carry0=False,score_cache=False,query_weights=self.weights.mode,
                          calls=self.peer_calls,weight_begins=self.weights.begins,weight_observes=self.weights.observes,
                          calls_by_layer=dict(self.calls_by_layer),
                          projected_tokens=router.cache.projected_tokens,reused_projection_tokens=router.cache.reused_tokens,
                          tile_counts=rows,threshold=None if not math.isfinite(self.threshold) else self.threshold,
                          kernel_dense_mode=self.variant=='kernel_dense')
        finally:
            try:
                receipt=super().end_request()
            finally:
                try:
                    if router is not None:router.cache.close()
                finally:
                    self.router=self.weights=self.scoped=None;self.dense_dummy.clear()
        receipt['standalone_peer']=peer
        return receipt
