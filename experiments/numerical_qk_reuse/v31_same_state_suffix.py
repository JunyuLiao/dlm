"""Offline last-GLOBAL-layer decision interventions inside a native audit.

The native attention, sampler output, RNG stream, and caches are not changed.
Extra last-layer suffix forwards are explicitly counted as diagnostics, not N.
Only layer 29 is intervened on: this cannot qualify earlier-layer deployment.
"""
from __future__ import annotations

import inspect
import json
import copy
import os
import functools
from pathlib import Path

import torch

from .v31_value_decision import OmittedCache, bounded_proposal, c_gate, moments
from .v31_value_reuse_guard import residual_trust


def distribution(scaled, entropy_bound=.1):
    lp = scaled.log_softmax(-1)
    entropy = -(lp.exp() * lp).sum(-1)
    ordered, indices = entropy.sort(-1)
    accepted = torch.zeros_like(entropy, dtype=torch.bool)
    accepted.scatter_(-1, indices, ordered.cumsum(-1) - ordered.cummax(-1).values <= entropy_bound)
    top = scaled.topk(2, dim=-1)
    return dict(logp=lp, entropy=entropy, accepted=accepted, top1=top.indices[..., 0],
                margin=top.values[..., 0] - top.values[..., 1])


def compare(candidate, reference, noise, previous, entropy_bound=.1, confidence=.005, active=None):
    c, r = distribution(candidate, entropy_bound), distribution(reference, entropy_bound)
    ce = c['top1'] != r['top1']
    acceptance = c['accepted'] != r['accepted']
    unstable_c = bool((c['top1'] != previous).any())
    unstable_r = bool((r['top1'] != previous).any())
    stop_c = not unstable_c and bool(c['entropy'].mean() < confidence)
    stop_r = not unstable_r and bool(r['entropy'].mean() < confidence)
    kl = (r['logp'].exp() * (r['logp'] - c['logp'])).sum(-1)
    result=dict(top1_disagreement=float(ce.float().mean()), acceptance_disagreement=float(acceptance.float().mean()),
        common_noise_disagreement=float(((candidate + noise).argmax(-1) != (reference + noise).argmax(-1)).float().mean()),
        kl_mean=float(kl.mean()), kl_max=float(kl.max()), entropy_mean=float(c['entropy'].mean()),
        mean_entropy_delta=float(c['entropy'].mean() - r['entropy'].mean()), stop_flip=stop_c != stop_r,
        reference_stop=stop_r, candidate_stop=stop_c, max_logit_error=float((candidate-reference).abs().max()),
        margin_crossing_rows=float(((candidate-reference).abs().amax(-1) * 2 >= r['margin']).float().mean()))
    if active is not None:
        result['previously_unsettled_rows']=int(active.sum())
        if bool(active.any()):
            result.update(top1_disagreement_unsettled=float(ce[active].float().mean()),
                acceptance_disagreement_unsettled=float(acceptance[active].float().mean()),
                kl_unsettled_mean=float(kl[active].mean()))
    return result


class Controller:
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.model = None; self.capture = None; self.frame = None
        self.states = {}; self.previous_stats = None; self.previous_request = None
        self.suffix_calls = 0; self.replaying = False; self.records = 0
        self.grams = {}; self.native_parity_failures = 0

    def active(self):
        from .vllm_adapter import _ACTIVE
        a = _ACTIVE
        return (a is not None and a.bound and a.step_ctx is not None
                and not a.step_ctx['encoder'] and a.step_ctx['n'] <= 256)

    def record(self, request, dataset, adapter, layer, q, buffers, scale, prefix, n):
        if layer != 29:
            return
        if request != self.previous_request:
            self.states.clear(); self.previous_stats = None; self.previous_request = request
            self.last_sampler = None
            self.first_canvas = adapter.canvas_id
            self.last_canvas = adapter.canvas_id
        if adapter.canvas_id != self.last_canvas:
            self.previous_stats = None; self.last_sampler = None
            self.last_canvas = adapter.canvas_id
        canvas = adapter.canvas_id - self.first_canvas
        if canvas > 1:
            self.frame = None
            return
        st = adapter.mage_state.get(layer)
        if st is None or st['canvas'] != adapter.canvas_id:
            self.frame = None
            return
        identity = (request, adapter.canvas_id, prefix, n)
        old = self.states.get(identity)
        anchor = old is None or old['list_identity'] != id(st['lists'])
        age = 0 if anchor else old['age'] + 1
        self.frame = dict(request=request, dataset=dataset, adapter=adapter, q=q, k=buffers['k'], v=buffers['v'],
            scale=scale, prefix=prefix, n=n, canvas=canvas, anchor=anchor, age=age, identity=identity,
            kept=st['kept'], list_identity=id(st['lists']), call=int(adapter._canvas_step))
        if old is not None:
            old['age'] = age

    def outputs(self, frame):
        from . import v27_fa4
        f = frame; n=f['n']; pt=f['prefix']//64; qb=(n+127)//128
        identity=f['identity']; anchor=f['anchor']; native_mask=f['kept']
        layer=self.capture['layer']
        if not self.grams:
            weight=layer.self_attn.o_proj.weight.float()
            for h in range(16):
                w=weight[:,h*512:(h+1)*512]
                self.grams[h]=w.T@w
        if anchor:
            self.states[identity]=dict(age=0,list_identity=f['list_identity'], caches={},
                masks={label:native_mask.clone() for label in ('mass_anchor','full_swap8','online_swap8','full_cgate_swap8')})
        state=self.states[identity]
        correction={kind:torch.empty((1,n,16,512),device=f['q'].device,dtype=torch.float32) for kind in ('residual','moments')}
        if anchor:
            anchor_dense=torch.empty_like(correction['residual'])
            anchor_sparse=torch.empty_like(anchor_dense)
        for h in range(16):
            kh=h//8
            for qt in range(qb):
                lo,hi=qt*128,min(n,(qt+1)*128)
                m=moments(f['q'][0,h,lo:hi],f['k'][0,kh],f['v'][0,kh],f['prefix'],f['scale'])
                mask=native_mask[0,h,qt,:pt].bool()
                if anchor:
                    for name,online,cgate_on in [('full_swap8',False,False),('online_swap8',True,False),('full_cgate_swap8',False,True)]:
                        weight=None
                        if cgate_on and self.previous_stats is not None:
                            weight=c_gate(*self.previous_stats)[lo:hi]
                        selected,_=bounded_proposal(m,mask,swaps=8,gram=self.grams[h],online=online,weight=weight)
                        state['masks'][name][0,h,qt,:pt]=selected
                    state['caches'][h,qt]=OmittedCache.build(m,mask)
                    anchor_dense[0,lo:hi,h]=m.dense()
                    anchor_sparse[0,lo:hi,h]=m.output(mask)
                for kind in correction:
                    correction[kind][0,lo:hi,h]=state['caches'][h,qt].reconstruct(m,mask,kind)
                del m
        if anchor:
            weight=layer.self_attn.o_proj.weight.float()
            state['anchor_dense_post']=anchor_dense.reshape(n,-1)@weight.T
            state['anchor_sparse_post']=anchor_sparse.reshape(n,-1)@weight.T
            state['anchor_residual']=anchor_dense-anchor_sparse
            return None
        # Offline contiguous FA4 is distinct from the native alias-split path.
        # A same-mask numerical-floor arm measures that implementation gap.
        result={name:v27_fa4.sparse_lists(f['q'],f['k'],f['v'],v27_fa4.block_sparse_tensors(mask),f['scale'])
                for name,mask in state['masks'].items()}
        result['dense_reference']=v27_fa4.dense(f['q'],f['k'],f['v'],f['scale'])
        result.update({'correction_'+kind:o.to(f['q'].dtype) for kind,o in correction.items()})
        current_post=result['mass_anchor'].float().reshape(n,-1)@layer.self_attn.o_proj.weight.float().T
        for kind in ('direction','norm'):
            alpha=residual_trust(current_post,state['anchor_sparse_post'],state['anchor_dense_post'],kind=kind)
            result['correction_'+kind+'_guard']=(result['mass_anchor'].float()
                +alpha[None,:,None,None]*state['anchor_residual']).to(f['q'].dtype)
        # Pure torch full-V reconstruction is also a distinct numerical path;
        # this is a diagnostic, not an unchanged-FA4 production claim.
        return result

    def suffix(self, projected_attention):
        from vllm.forward_context import override_forward_context
        cap=self.capture; layer=cap['layer']; original=layer.self_attn.forward
        layer.self_attn.forward=lambda *a,**kw:projected_attention
        self.replaying=True
        try:
            with override_forward_context(copy.copy(cap['forward_context'])):
                hidden,_=self.original_layer_forward(layer,cap['positions'],cap['hidden'].clone(),None,
                    per_layer_input=cap['per_layer_input'],**cap['kwargs'])
                hidden=self.model.model.norm(hidden)
                logits=self.model.compute_logits(hidden)
            self.suffix_calls+=1
            return logits
        finally:
            layer.self_attn.forward=original; self.replaying=False

    def before_sample(self, bound):
        f=self.frame
        if f is None or self.capture is None or not self.active():
            return
        self.frame=None
        if os.environ.get('SUFFIX_DIAGNOSTICS', '1') == '0':
            return
        if not f['anchor'] and f['age'] not in (1,2,4,8):
            return
        # The suffix does not touch attention/KV. Preserve all random streams
        # and independently check that it did not change adapter counters.
        counters=dict(f['adapter'].calls)
        cpu_rng=torch.get_rng_state().clone(); gpu_rng=torch.cuda.get_rng_state().clone()
        outputs=self.outputs(f)
        if outputs is None:
            assert torch.equal(gpu_rng,torch.cuda.get_rng_state())
            assert torch.equal(cpu_rng,torch.get_rng_state())
            assert counters==f['adapter'].calls
            return
        logits=bound['logits']; slots=bound['decode_slots']; steps=bound['step_tensor'][slots].float()
        temp=bound['t_min']+(bound['t_max']-bound['t_min'])*(bound['max_denoising_steps']-steps).clamp(min=1)/bound['max_denoising_steps']
        n=f['n']; cl=int(bound['CL']); vocab=logits.shape[-1]
        if n!=cl or len(slots)!=1:
            raise RuntimeError('suffix diagnostic currently requires one complete canvas')
        previous=bound['argmax_canvas'][slots].clone().reshape(n)
        eb=float(bound['entropy_bound']); conf=float(bound['confidence_threshold'])
        def scaled(x):return x.reshape(n,vocab).float()/temp[0].clamp_min(1e-10)
        # Identical repeated suffix with the exact native projected-attention
        # tensor must reproduce the native logits, before alternative masks.
        repeated=self.suffix(self.capture['native_attention'].clone())
        parity=float((repeated-logits).abs().max())
        top_mismatch=int((repeated.argmax(-1)!=logits.argmax(-1)).sum())
        if parity>1e-4 or top_mismatch:
            self.native_parity_failures+=1
            (self.root/'suffix_parity_failure.json').write_text(json.dumps(dict(
                max_logit_error=parity,mean_logit_error=float((repeated-logits).abs().mean()),
                top1_mismatches=top_mismatch,rows=n,canvas=f['canvas'],call=f['call'],
                captured_positions_shape=list(self.capture['positions'].shape),
                captured_hidden_shape=list(self.capture['hidden'].shape),
                native_logit_abs_mean=float(logits.abs().mean()),
                repeated_logit_abs_mean=float(repeated.abs().mean())))+'\n')
            raise RuntimeError(f'native suffix parity failed: max={parity}, top1={top_mismatch}')
        def forward(out):
            projected= self.capture['layer'].self_attn.o_proj(out.reshape(n,-1))[0]
            return scaled(self.suffix(projected))
        dense=forward(outputs.pop('dense_reference'))
        # Couple the hypothetical methods with one private diagnostic draw.
        # Eager rand and compiled sampler RNG lowering can differ: this is
        # deliberately not claimed as the actual native Gumbel realization.
        generator=torch.Generator(device=dense.device).manual_seed(1729+f['call'])
        uniform=torch.rand((1,n,vocab),generator=generator,device=dense.device,dtype=dense.dtype)
        noise=-torch.log(-torch.log(uniform.clamp(min=1e-20))).reshape(n,vocab)
        base=scaled(logits)
        active=None
        if self.previous_stats is not None:
            accepted,changed,reversed_=self.previous_stats
            active=(~accepted)|changed|reversed_
        values={'native_control':compare(base,dense,noise,previous,eb,conf,active)}
        values['native_control']['native_suffix_max_abs']=parity
        for name,out in outputs.items():
            alternative=forward(out)
            values[name]=compare(alternative,dense,noise,previous,eb,conf,active)
            if name=='mass_anchor':
                values['same_mask_numerical_floor_vs_native']=compare(alternative,base,noise,previous,eb,conf,active)
        if not torch.equal(gpu_rng,torch.cuda.get_rng_state()) or not torch.equal(cpu_rng,torch.get_rng_state()):
            raise RuntimeError('offline suffix consumed native RNG')
        if counters!=f['adapter'].calls:
            raise RuntimeError('offline suffix changed native attention counters')
        item=dict(schema='last_global_same_state_decision_v1',request=f['request'],dataset=f['dataset'],canvas=f['canvas'],
            call=f['call'],age=f['age'],layer=29,rows=n,metrics=values,offline_suffix_calls=self.suffix_calls,
            common_noise_source='private coupled diagnostic Gumbel; not the compiled native realization',
            intervention_scope='layer29 only; all preceding native states identical',timing='paused')
        with (self.root/'decisions.jsonl').open('a') as stream:
            stream.write(json.dumps(item,allow_nan=False)+'\n')
        self.records+=1
        (self.root/'state.json').write_text(json.dumps(dict(records=self.records,offline_suffix_calls=self.suffix_calls,
            parity_failures=self.native_parity_failures,native_sampler_modified=False,timing='paused'))+'\n')


def install(root):
    from vllm.model_executor.models import gemma4, diffusion_gemma as dg
    from . import v31_stable_trajectory
    controller=Controller(root)
    original_layer=gemma4.Gemma4DecoderLayer.forward
    controller.original_layer_forward=original_layer
    original_attention=gemma4.Gemma4Attention.forward

    # Explicit split custom ops survive vLLM's fullgraph tracing and execute
    # the observer outside CUDA-graph segments on every real call. Their
    # They preserve graph-owned tensor addresses. Returning a newly allocated
    # identity tensor here is invalid when downstream CUDA graphs do not copy
    # their inputs: they would replay against the warmup allocation.
    # The mutation declaration retains observer ordering; observers only read
    # the input and make separate private copies, and return no replacement.
    @torch.library.custom_op('v31_value_diag::capture_input', mutates_args=('hidden',))
    def capture_input(hidden:torch.Tensor,positions:torch.Tensor,per_input:torch.Tensor|None)->None:
        from vllm.forward_context import get_forward_context
        if controller.active() and not controller.replaying:
            controller.capture=dict(layer=controller.model.model.layers[29],positions=positions.clone(),
                hidden=hidden.clone(),per_layer_input=per_input.clone() if per_input is not None else None,
                kwargs={},forward_context=copy.copy(get_forward_context()))
    @capture_input.register_fake
    def fake_input(hidden,positions,per_input):
        return None

    @torch.library.custom_op('v31_value_diag::capture_attention', mutates_args=('output',))
    def capture_attention(output:torch.Tensor)->None:
        if controller.active() and controller.capture is not None and not controller.replaying:
            controller.capture['native_attention']=output.clone()
    @capture_attention.register_fake
    def fake_attention(output):
        return None

    original_init=dg.DiffusionGemmaForConditionalGeneration.__init__
    @functools.wraps(original_init)
    def model_init(self,*args,**kwargs):
        config=kwargs['vllm_config'].compilation_config
        if os.environ.get('SUFFIX_EAGER')!='1':
            if config.splitting_ops is None:
                raise RuntimeError('native attention split inventory must be initialized')
            config.splitting_ops=list(config.splitting_ops)+['v31_value_diag::capture_input','v31_value_diag::capture_attention']
        original_init(self,*args,**kwargs); controller.model=self
        self.model.layers[29].self_attn._value_suffix_last=True
    dg.DiffusionGemmaForConditionalGeneration.__init__=model_init

    @functools.wraps(original_layer)
    def layer_forward(self,positions,hidden_states,residual,per_layer_input=None,**kw):
        if self.layer_idx==29:
            capture_input(hidden_states,positions,per_layer_input)
        return original_layer(self,positions,hidden_states,residual,per_layer_input=per_layer_input,**kw)
    gemma4.Gemma4DecoderLayer.forward=layer_forward
    @functools.wraps(original_attention)
    def attention_forward(self,*a,**kw):
        out=original_attention(self,*a,**kw)
        if getattr(self,'_value_suffix_last',False):
            capture_attention(out)
        return out
    gemma4.Gemma4Attention.forward=attention_forward

    class Recorder:
        def __init__(self,directory,request,dataset):
            self.request,self.dataset=request,dataset
        def record(self,adapter,layer,q,buffers,scale,prefix,n):
            controller.record(self.request,self.dataset,adapter,layer,q,buffers,scale,prefix,n)
    v31_stable_trajectory.TrajectoryRecorder=Recorder
    original_sample=dg._compiled_sample_step
    signature=inspect.signature(original_sample)
    @functools.wraps(original_sample)
    def sample(*a,**kw):
        bound=signature.bind(*a,**kw).arguments
        if controller.active():
            controller.before_sample(bound)
        scaled=original_sample(*a,**kw)
        if controller.active() and scaled.shape[0]==1:
            ds=distribution(scaled[0],float(bound['entropy_bound']))
            old=getattr(controller,'last_sampler',None)
            changed=ds['top1']!=old['top1'] if old is not None and old['top1'].shape==ds['top1'].shape else torch.zeros_like(ds['accepted'])
            reversed_=ds['accepted']!=old['accepted'] if old is not None and old['accepted'].shape==ds['accepted'].shape else torch.zeros_like(ds['accepted'])
            controller.previous_stats=(ds['accepted'].clone(),changed,reversed_)
            controller.last_sampler=dict(top1=ds['top1'].clone(),accepted=ds['accepted'].clone())
        return scaled
    dg._compiled_sample_step=sample
    return controller
