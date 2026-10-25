"""Read-only semantic observation across a branch's own commit boundary."""
import hashlib,math,re
import torch

def tensor_semantics(t):
    x=t.detach().contiguous().cpu();raw=x.view(torch.uint8).numpy().tobytes()
    d=dict(shape=list(x.shape),dtype=str(x.dtype),byte_sha256=hashlib.sha256(raw).hexdigest())
    if x.is_floating_point():
        # Canonicalize a CPU COPY, never the native tensor. Equal numbers with
        # opposite signed zeros have equal numerical hashes. NaNs are explicit.
        z=x.clone();zero=z==0;negative_zero=int((zero & torch.signbit(z)).sum())
        z[zero]=0;nan=torch.isnan(z);z[nan]=float('nan')
        finite=torch.isfinite(x);v=x[finite].double()
        d.update(numerical_sha256=hashlib.sha256(z.view(torch.uint8).numpy().tobytes()).hexdigest(),negative_zero=negative_zero,
                 zeros=int(zero.sum()),nan=int(nan.sum()),positive_inf=int(torch.isposinf(x).sum()),negative_inf=int(torch.isneginf(x).sum()),
                 finite_min=float(v.min()) if v.numel() else None,finite_max=float(v.max()) if v.numel() else None,
                 finite_L2=float(v.norm()),finite_mean=float(v.mean()) if v.numel() else None)
    else:d['numerical_sha256']=d['byte_sha256']
    return d

class Boundary:
    def __init__(self,state,rt):self.state=state;self.rt=rt;self.enabled=False
    def begin(self,enabled):self.enabled=enabled;self.events=[];self.prefix=[];self.latest={};self.pending=None
    def rng(self):return dict(cpu=tensor_semantics(torch.get_rng_state())['byte_sha256'],cuda=tensor_semantics(torch.cuda.get_rng_state())['byte_sha256'])
    def wanted(self):return self.enabled and self.rt.active and self.rt.anchor is not None
    def sample_before(self,a):
        if not self.wanted():return
        c=self.state['C'];anchor=self.rt.anchor['canvas'];commit=bool(a[8][a[1]].item())
        if c in [anchor,anchor+1] and (c==anchor or (not commit and int(a[7][a[1]].item())==0)):
            self.capture_args('commit_sampler_entry' if commit else 'ordinary_sampler_entry',a)
    def capture_args(self,phase,a,result=None):
        slot=a[1];length=int(a[14][slot].item());h=a[13][slot];active=min(length,h.shape[1])
        values={name:tensor_semantics(a[i][slot]) for name,i in [('stochastic_canvas',5),('argmax_canvas',6),('self_condition',10),('step',7),('encoder_phase',8),('confidence',9)]}
        values['history']=tensor_semantics(h[:,:active])
        # history slots are inactive after native reset; don't compare capacity.
        if result is not None:values['scaled_logits']=tensor_semantics(result)
        self.events.append(dict(phase=phase,canvas=self.state['C'],N=self.state['N'],C=self.state['C'],history_length=length,
          valid_canvas_length=tensor_semantics(a[4]),states=values,rng=self.rng()))
    def sample_after(self,a,result,committing,canvas_before):
        if committing:
            idx=a[2];n=int(a[16][idx].reshape(-1)[0].item());tokens=a[15][idx].reshape(-1)[:n].detach().cpu().tolist();self.prefix.extend(tokens)
        if not self.wanted():return
        anchor=self.rt.anchor['canvas'];c=canvas_before
        if c==anchor:
            self.capture_args('commit_sampler_exit_and_next_initialization' if committing else 'ordinary_sampler_exit',a,result)
            if committing:
                self.events[-1].update(emitted_canvas=tensor_semantics(a[15][a[2]].reshape(-1)[:n]),emitted_count=n,
                   cumulative_output_prefix_sha256=hashlib.sha256(str(self.prefix).encode()).hexdigest())
        elif c==anchor+1 and not committing and int(a[7][a[1]].item())==1:self.capture_args('next_canvas_first_sampler_exit',a,result)
    def install(self):
        from vllm.model_executor.models import diffusion_gemma as dg
        from vllm.v1.attention.backends import flash_attn as fa
        orig=dg.DiffusionGemmaModelState.prepare_inputs;b=self
        def prepare(model,batch,req):
            event=None;b.pending=None
            if b.wanted() and batch.num_reqs==1:
                c=b.state['C'];target=b.rt.anchor['canvas'];slot=int(batch.idx_mapping_np[0]);s=model.diffusion_states
                commit=bool(s.is_encoder_phase[slot]);step=int(s.step[slot]);
                if (c==target and commit) or (c==target+1 and not commit and step==0):
                    event=dict(phase='native_commit_model_entry' if commit else 'next_canvas_first_model_input',canvas=c,N=b.state['N'],rng=b.rng(),
                      input_ids=tensor_semantics(batch.input_ids[:batch.num_tokens]),positions=tensor_semantics(batch.positions[:batch.num_tokens]) if hasattr(batch,'positions') else None,
                      self_condition=tensor_semantics(s.self_conditioning_embeds[slot]),stochastic_canvas=tensor_semantics(s.canvas[slot]),
                      argmax_canvas=tensor_semantics(s.argmax_canvas[slot]),history_length=int(s.accepted_canvas_history_len[slot]),layers={})
                    b.pending=event
            result=orig(model,batch,req)
            if event is not None:
                event['inputs_embeds']=tensor_semantics(result['inputs_embeds'][:batch.num_tokens]);b.events.append(event)
            return result
        dg.DiffusionGemmaModelState.prepare_inputs=prepare
        forward0=fa.FlashAttentionImpl.forward
        def forward(impl,layer,query,key,value,cache,meta,output,*args,**kw):
            observe=b.wanted() and b.pending is not None and b.pending['canvas']==b.state['C'] and b.pending['phase'] in ['native_commit_model_entry','next_canvas_first_model_input']
            result=forward0(impl,layer,query,key,value,cache,meta,output,*args,**kw)
            if observe:
                name=getattr(layer,'layer_name','');m=re.search(r'layers\.(\d+)\.',name)
                if m:
                    n=int(meta.num_actual_tokens);length=int(meta.seq_lens[0]);kc,vc=cache.transpose(1,2).split(impl.head_size,dim=-1);page=kc.shape[1]
                    # LOCAL effective window includes only valid nearby positions.
                    begin=0 if impl.head_size==512 else max(0,length-n-1023)
                    first=begin//page;last=math.ceil(length/page);ids=meta.block_table[0,first:last].long();assert bool((ids>=0).all())
                    k=kc.index_select(0,ids).flatten(0,1)[begin-first*page:length-first*page]
                    v=vc.index_select(0,ids).flatten(0,1)[begin-first*page:length-first*page]
                    b.pending['layers'][m[1]]=dict(active_key_start=begin,active_key_end=length,query=tensor_semantics(query[:n]),
                      current_key=tensor_semantics(key[:n]),current_value=tensor_semantics(value[:n]),committed_K=tensor_semantics(k),committed_V=tensor_semantics(v),
                      output=tensor_semantics(output[:n]),num_tokens=n,execution_phase=b.rt.phase)
            return result
        fa.FlashAttentionImpl.forward=forward
