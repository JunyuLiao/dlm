"""Fresh native requests, one current-QKV pulse; saved diagnostics never feed attention."""
from pathlib import Path
from collections import Counter
import hashlib,json,math,sys
import torch
ROOT=Path(__file__).resolve().parents[1];STUDY=ROOT.parents[1];PROFILE=STUDY/'profiling/attempt002'
sys.path.insert(0,str(PROFILE/'code'))
from native_harness import NativeHarness,save,sha
from shared_consumer import gather_current,native_attention,fingerprint
from observations import state_snapshot
from op_bank import build,make_probe,gather_units,bias_metrics,support_hash,UNITS

def tensor_save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:torch.save(value,f)
    return sha(path)

def geometry(t):
    return dict(shape=list(t.shape),stride=list(t.stride()),storage_offset=t.storage_offset(),dtype=str(t.dtype),sha256=fingerprint(t))

def stamp(h,args):
    return dict(state=state_snapshot(args),N=h.state['N'],C=h.state['C'],step=h.state['step'],executions_since_request=h.state['executions']-h.extension.execute_start)

def site_key(anchor,layer):return f"c{anchor['canvas']:03d}_s{anchor['after_completed_calls']+1:02d}_l{layer:02d}"
def active_site(ctx,a):return ctx['canvas']==a['canvas'] and ctx['step']==a['after_completed_calls']+1
def exact_request(a,b):
    keys=['output_sha256','N','C','T','final_cpu_rng','final_cuda_rng','model_forwards','prompt_tokenization_sha256','startup_model_calls_in_request','extra_inside_execute_Python_forwards']
    return {k:a[k]==b[k] for k in keys}

def core_trace_equal(a,b):
    extra={'reference_winner_crossings','actual_native_winner_changes','worst_margin_erosion','mean_absolute_margin_change','signed_margin_quantiles','entropy_by_position','native_margin_by_position','signed_reference_winner_margin_change','reference_pair_margin'}
    return len(a)==len(b) and all({k:v for k,v in x.items() if k not in extra}=={k:v for k,v in y.items() if k not in extra} for x,y in zip(a,b))

def evaluate_fixed(ctx,k,v,units):
    """Same support at another layer, CURRENT operands and actual consumer bias."""
    q=ctx['q'];scale=ctx['scale'];prefix=ctx['prefix'];work=Counter();whole=math.ceil(len(k)/64)
    A,_=native_attention(q,ctx['kc'],ctx['vc'],ctx['meta'],scale)
    Ar,_=native_attention(q,ctx['kc'],ctx['vc'],ctx['meta'],scale);assert torch.equal(A,Ar)
    S,_=native_attention(q,ctx['kc'],ctx['vc'],ctx['meta'],scale,units)
    masses=[];biases=[];dense=[]
    for u in units:
        h=u['head'];qb=u['qb'];p=(q[qb*128:(qb+1)*128,h].float()@k[:,h//8].float().T*scale).softmax(-1);vv=v[:,h//8].float();o=p@vv
        ids=(torch.tensor(u['tiles'],device=q.device)[:,None]*64+torch.arange(64,device=q.device)).flatten();m=p[:,ids].sum(-1)
        b=(m[:,None]*o-p[:,ids]@vv[ids])/(1-m).clamp_min(1e-12)[:,None];masses.append(m);biases.append(b);dense.append(o)
    mb=torch.stack(biases);mass=torch.stack(masses);D=torch.stack(dense);measured,b=bias_metrics(gather_units(S),gather_units(A),mb)
    err=float((mb.norm(dim=-1)/D.norm(dim=-1).clamp_min(1e-8)).max());gqa=float(((gather_units(S).float()-(D+mb)).norm(dim=-1)/(D+mb).norm(dim=-1).clamp_min(1e-8)).max());assert gqa<=.03
    work.update(native_A_QK_rectangles=2*32*whole,native_A_PV_rectangles=2*32*whole,native_S_QK_rectangles=32*whole-sum(len(u['tiles']) for u in units),native_S_PV_rectangles=32*whole-sum(len(u['tiles']) for u in units),QK_rectangles=4*whole,PV_rectangles=4*whole+sum(len(u['tiles']) for u in units))
    rec=dict(units=units,support_sha256=support_hash(units),**measured,deleted_mass=float(mass.max()),math_current_error=err,GQA_error=gqa,feasible=measured['current_error']<=.1 and err<=.1 and float(mass.max())<=.2)
    private=dict(dense=gather_units(A).cpu(),sparse=gather_units(S).cpu(),actual_bias=b.cpu(),FP32_dense=D.cpu(),FP32_bias=mb.cpu(),mass=mass.cpu())
    return rec,private,dict(work)

class PulseStudy:
    needs_prior=True
    def __init__(self,root,mode,config):
        self.root=Path(root);self.mode=mode;self.config=config;self.installed=False
    def begin(self,h,row,seed,label,audit):
        self.h=h;self.row=row;self.seed=seed;self.label=label;self.audit=audit;self.events=[];self.work=Counter();self.pending=None;self.pulse_calls=0;self.before={};self.commits={};self.finished=set();self.execute_start=h.state['executions'];self.generated=[]
        self.directory=self.root/'private/diagnostics'/label;self.public=self.root/'banks'/label
        h.consumer.callback=self.before_attention;h.consumer.after_callback=self.after_attention
        self.anchors=self.config.get('anchors',{});self.references={}
        ref=self.config.get('reference_label')
        if ref:
            p=self.root/'private/generations'/ref;p=p.with_suffix('.pt');self.references=torch.load(p,map_location='cpu',weights_only=False)
        if not self.installed:self.install_commit_observer();self.installed=True
    def install_commit_observer(self):
        from vllm.v1.attention.backends import flash_attn as fa
        original=fa.FlashAttentionImpl.forward;ext=self
        def wrapped(impl,layer,query,key,value,cache,meta,output,output_scale=None,output_block_scale=None):
            result=original(impl,layer,query,key,value,cache,meta,output,output_scale,output_block_scale)
            h=ext.h
            if h.state['bound'] and h.consumer.phase=='encoder' and h.state['C'] in ext.finished:
                name=getattr(layer,'layer_name','');sites=ext.config.get('commit_layers',[29])
                location=next((i for i in sites if f'.layers.{i}.' in name),None)
                keyname=f"c{h.state['C']:03d}_l{location}"
                if location is not None and keyname not in ext.commits:
                    length=int(meta.seq_lens[0]);kc,vc=cache.transpose(1,2).split(impl.head_size,dim=-1);page=kc.shape[1];ids=meta.block_table[0,:math.ceil(length/page)].long();vals=[]
                    for t in [kc,vc]:
                        x=t.index_select(0,ids).flatten(0,1)[:length].contiguous();y=x.float();vals.append(dict(byte_sha256=fingerprint(x),numeric_sha256=fingerprint(torch.where(y==0,torch.zeros_like(y),y)),nonfinite=int((~torch.isfinite(y)).sum()),negative_zero=int(((y==0)&torch.signbit(y)).sum()),shape=list(x.shape)))
                    ext.commits[keyname]=dict(layer=location,canvas=h.state['C'],active_length=length,K=vals[0],V=vals[1],scope='active_logical_cache_at_native_commit')
            return result
        fa.FlashAttentionImpl.forward=wrapped
    def model_entry(self,m,args,kw):pass
    def reference_for(self,canvas,step):
        a=next((a for a in self.anchors.values() if a['canvas']==canvas),None)
        return self.references.get((canvas,step)),bool(a and step>=a['after_completed_calls'])
    def after_sample(self,result,args,kw,committing,row,private):
        h=self.h
        if committing:
            return
        if row is not None:
            for name,a in self.anchors.items():
                if h.state['C']==a['canvas'] and h.state['step']==a['after_completed_calls']:
                    v=stamp(h,args);self.before[name]=v
                    if self.mode in ['branch','control']:
                        expected=self.config['reference_before'][name];assert v==expected,'Pre-intervention native state/RNG/counters differ'
                if h.state['C']==a['canvas'] and row['converged']:self.finished.add(h.state['C'])
                if h.state['C']==a['canvas'] and (h.state['step'] in [a['after_completed_calls']+1,a['after_completed_calls']+2] or row['converged']):
                    self.events.append(dict(stratum=name,canvas=h.state['C'],step=h.state['step'],terminal=row['converged'],native_state=state_snapshot(args)))
    def before_attention(self,ctx):
        self.pending=None;layer=ctx['layer'];target=[(name,a) for name,a in self.anchors.items() if active_site(ctx,a)]
        if not target:return None
        if self.mode in ['branch','control']:
            if self.mode=='control' or layer!=self.config['layer'] or not any(n==self.config['stratum'] for n,a in target):return None
        elif self.mode=='bank_reference' and layer!=29:return None
        elif self.mode=='early_reference' and layer not in [5,17]:return None
        names=[n for n,a in target];anchor=target[0][1];assert ctx['prefix']==anchor['prefix']
        q=ctx['q'];k,v=gather_current(q,ctx['key'],ctx['value'],ctx['kc'],ctx['vc'],ctx['meta'],ctx['prefix'])
        g=dict(q=geometry(q),k=geometry(k),v=geometry(v),prefix=ctx['prefix'],length=len(k),layer=layer,canvas=ctx['canvas'],step=ctx['step'],gqa_map=[h//8 for h in range(16)],page_table_sha256=fingerprint(ctx['meta'].block_table[0,:math.ceil(len(k)/ctx['kc'].shape[1])]),cpu_rng=fingerprint(torch.get_rng_state()),cuda_rng=fingerprint(torch.cuda.get_rng_state()))
        sk=site_key(anchor,layer);path=self.public/(sk+'.json');op=lambda units:native_attention(q,ctx['kc'],ctx['vc'],ctx['meta'],ctx['scale'],units)[0]
        if self.mode=='branch':
            expected=self.config['site'];logical=lambda x:{k:v for k,v in x.items() if k!='page_table_sha256'}
            assert logical(g)==logical(expected['geometry']),'Current QKV/geometry/RNG differ at intervention'
            A=op(None);self.work.update(native_A_QK_rectangles=32*math.ceil(len(k)/64),native_A_PV_rectangles=32*math.ceil(len(k)/64));A4=gather_units(A)
            assert fingerprint(A4)==self.config['dense_sha256'],'Diagnostic A differs at common state'
            self.pending=dict(A=A4,expected=self.config,geometry=g);self.pulse_calls+=1
            assert self.pulse_calls==1
            return self.config['units']
        banks=[];dense_hash=None
        if self.mode=='bank_reference':
            probe,entropy,protected,margin,pwork=make_probe(self.h.consumer.model,29,self.h.state['prior'],ctx['step'])
            self.work.update({key:pwork[key] for key in ['linear_projection_multiply_adds','head_weight_rows']})
            done={}
            for fraction in [.05,.10]:
                drop=max(1,math.floor(fraction*(ctx['prefix']//64)))
                if drop in done:
                    banks.append(dict(fraction=fraction,alias_of=done[drop],drop_count=drop));continue
                identity=f"{self.row['id']}|{self.seed}|{ctx['canvas']}|{ctx['step']}|29"
                bank,priv,work=build(q,k,v,ctx['prefix'],ctx['scale'],fraction,probe,entropy,protected,margin,op,identity);self.work.update(work)
                bank.update(fraction=fraction,readout_projection=pwork)
                pp=self.directory/f'{sk}_d{drop}.pt';bank['private_sha256']=tensor_save(pp,priv);bank['private_relative']=str(pp.relative_to(self.root));done[drop]=len(banks)
                if priv:
                    bank['dense_sha256']=fingerprint(priv['dense']);dense_hash=bank['dense_sha256']
                    for i,r in enumerate(bank['candidates']):r['sparse_sha256']=fingerprint(priv['candidates'][i]['sparse'])
                banks.append(bank)
        else:
            # Maps were already chosen at layer29, before any early-layer outcome.
            for spec in self.config['maps']:
                rec,priv,work=evaluate_fixed(ctx,k,v,spec['units']);self.work.update(work)
                pp=self.directory/f"{sk}_d{spec['drop_count']}_{rec['support_sha256'][:12]}.pt"
                banks.append(dict(**rec,fraction=spec['fraction'],fractions=spec['fractions'],drop_count=spec['drop_count'],families=spec['families'],dense_sha256=fingerprint(priv['dense']),sparse_sha256=fingerprint(priv['sparse']),private_relative=str(pp.relative_to(self.root)),private_sha256=tensor_save(pp,priv)))
                dense_hash=fingerprint(priv['dense'])
        operands=None
        if self.config.get('operand_anchor')==site_key(anchor,29):
            pp=self.directory/(sk+'_operands.pt');operands=dict(relative=str(pp.relative_to(self.root)),sha256=tensor_save(pp,dict(q=q.detach().cpu(),k=k.detach().cpu(),v=v.detach().cpu(),scale=ctx['scale'],prefix=ctx['prefix'])))
        site=dict(geometry=g,strata=names,banks=banks,operands=operands,choices_frozen_before_downstream=True)
        save(path,site);self.generated.append(str(path.relative_to(self.root)));self.pending=dict(dense_sha256=dense_hash,geometry=g)
        return None
    def after_attention(self,ctx,support,result,keep):
        if self.pending is None:return
        p=self.pending;actual=gather_units(result)
        if self.mode=='branch':
            assert support_hash(support)==self.config['support_sha256']
            assert fingerprint(actual)==self.config['sparse_sha256'],'Actual model S differs from fresh diagnostic S'
            measured,bias=bias_metrics(actual,p['A']);assert measured['nonfinite']==0
            assert int((~keep).sum())==4*self.config['drop_count']
            self.events.append(dict(intervention=True,actual_support_sha256=support_hash(support),actual_S_sha256=fingerprint(actual),actual_bias_sha256=fingerprint(bias),measurement=measured,current_QKV=True,LOCAL_deleted=0,protected_deleted=0))
        elif p['dense_sha256'] is not None:assert fingerprint(actual)==p['dense_sha256'],'Actual A differs from fresh diagnostic A'
        self.pending=None
    def finish(self,record):
        if self.mode=='branch':assert self.pulse_calls==1,'One ordinary pulse required'
        endpoint={}
        for name,a in self.anchors.items():
            calls=[r for r in record['native_trace'] if not r['commit'] and r['canvas']==a['canvas']]
            terminal=calls[-1] if calls else None
            commits=[r for r in record['native_trace'] if r['commit'] and r['canvas']==a['canvas']]
            endpoint[name]=dict(remaining_calls=sum(r['native_step']>a['after_completed_calls'] for r in calls),terminal=terminal,commit=commits[0] if commits else None,next_same_canvas_available=any(r['native_step']==a['after_completed_calls']+2 for r in calls))
        return dict(study_mode=self.mode,sites=self.generated,pre_intervention=self.before,pulse_calls=self.pulse_calls,local_endpoints=endpoint,events=self.events,committed_KV_where_measured=self.commits,operator_diagnostic_work=dict(self.work),clean_online_latency_qualified=False)
