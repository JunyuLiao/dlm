"""Fresh-request finite-window reuse. Saved diagnostics never feed model outputs."""
from pathlib import Path
from collections import Counter
import math, json, hashlib
import torch
from native_harness import NativeHarness,save,sha,anchors as find_anchors
from shared_consumer import gather_current,native_attention,fingerprint
from observations import state_snapshot
from op_bank import build,make_probe,gather_units,bias_metrics,support_hash,matched,UNITS
from reuse_selectors import augment,active_schedule,union_window

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT.parent
OLD=STUDY.parent/'native_support_discovery_20261026'
PROFILE=OLD/'profiling/attempt002'

def tensor_save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:torch.save(value,f)
    return sha(p)

def geometry(t):return dict(shape=list(t.shape),stride=list(t.stride()),storage_offset=t.storage_offset(),dtype=str(t.dtype),sha256=fingerprint(t))
def stamp(h,args):return dict(state=state_snapshot(args),N=h.state['N'],C=h.state['C'],step=h.state['step'],executions_since_request=h.state['executions']-h.extension.execute_start)
def logical(g):return {k:v for k,v in g.items() if k!='page_table_sha256'}

def fixed(ctx,k,v,units):
    q=ctx['q'];scale=ctx['scale'];whole=math.ceil(len(k)/64)
    A,_=native_attention(q,ctx['kc'],ctx['vc'],ctx['meta'],scale)
    Ar,_=native_attention(q,ctx['kc'],ctx['vc'],ctx['meta'],scale);assert torch.equal(A,Ar)
    S,_=native_attention(q,ctx['kc'],ctx['vc'],ctx['meta'],scale,units)
    untouched=S.clone();untouched_A=A.clone()
    for u in units:
        untouched[u['qb']*128:(u['qb']+1)*128,u['head']]=0
        untouched_A[u['qb']*128:(u['qb']+1)*128,u['head']]=0
    assert torch.equal(untouched,untouched_A),'unselected routing units must remain all-kept'
    biases=[];masses=[];dense=[];prefix_mass=[]
    for u in units:
        h=u['head'];qb=u['qb'];p=(q[qb*128:(qb+1)*128,h].float()@k[:,h//8].float().T*scale).softmax(-1);vv=v[:,h//8].float();o=p@vv
        ids=(torch.tensor(u['tiles'],device=q.device)[:,None]*64+torch.arange(64,device=q.device)).flatten();m=p[:,ids].sum(-1)
        biases.append((m[:,None]*o-p[:,ids]@vv[ids])/(1-m).clamp_min(1e-12)[:,None]);masses.append(m);dense.append(o);prefix_mass.append(p[:,:ctx['prefix']].sum(-1))
    mb=torch.stack(biases);mass=torch.stack(masses);D=torch.stack(dense);measured,bias=bias_metrics(gather_units(S),gather_units(A),mb)
    err=float((mb.norm(dim=-1)/D.norm(dim=-1).clamp_min(1e-8)).max());gqa=float(((gather_units(S).float()-(D+mb)).norm(dim=-1)/(D+mb).norm(dim=-1).clamp_min(1e-8)).max());assert gqa<=.03
    rec=dict(units=units,support_sha256=support_hash(units),**measured,deleted_mass=float(mass.max()),math_current_error=err,GQA_error=gqa,
             prefix_only_deleted_mass_max=float((mass/torch.stack(prefix_mass).clamp_min(1e-12)).max()),renormalization_max=float((1/(1-mass)).max()),removed_current_canvas_mass=0.,
             feasible=measured['current_error']<=.1 and err<=.1 and float(mass.max())<=.2)
    private=dict(dense=gather_units(A).cpu(),sparse=gather_units(S).cpu(),actual_bias=bias.cpu(),FP32_bias=mb.cpu(),mass=mass.cpu())
    work=dict(native_A_QK_rectangles=2*32*whole,native_A_PV_rectangles=2*32*whole,native_S_QK_rectangles=32*whole-sum(len(u['tiles']) for u in units),native_S_PV_rectangles=32*whole-sum(len(u['tiles']) for u in units),fixed_QK_rectangles=4*whole,fixed_PV_rectangles=4*whole+sum(len(u['tiles']) for u in units))
    return rec,private,work

class ReuseStudy:
    needs_prior=True
    def __init__(self,root,mode='warm',config=None):self.root=Path(root);self.mode=mode;self.config=config or {}
    def begin(self,h,row,seed,label,audit):
        self.h=h;self.row=row;self.seed=seed;self.label=label;self.audit=audit;self.events=[];self.work=Counter();self.pending=None;self.before=None;self.sites=[];self.pulse_calls=0;self.previous_q=None;self.q_canvas=None;self.held=None;self.cert=None;self.last_bias=None;self.execute_start=h.state['executions'];self.states={};self.refreshes=0;self.holds=0
        self.anchor=self.config.get('anchor');self.references={};self.initial_site=None
        self.needs_prior=self.mode not in ['warm','control'] or audit
        if self.config.get('reference_label'):
            self.references=torch.load((self.root/'private/generations'/self.config['reference_label']).with_suffix('.pt'),map_location='cpu',weights_only=False)
        h.consumer.callback=self.before_attention;h.consumer.after_callback=self.after_attention
    def model_entry(self,m,args,kw):pass
    def reference_for(self,canvas,step):
        aa={'primary':self.anchor} if self.anchor else {}
        return self.references.get((canvas,step)),union_window(aa,canvas,step)
    def after_sample(self,result,args,kw,committing,row,private):
        h=self.h
        if committing:
            self.held=None;self.cert=None;self.previous_q=None;self.q_canvas=None
            return
        if not self.anchor:return
        c,t=self.anchor['canvas'],self.anchor['after_completed_calls']
        if h.state['C']==c and h.state['step']==t:
            self.before=stamp(h,args)
            if 'reference_before' in self.config:assert self.before==self.config['reference_before'],'common pre-intervention state/RNG/counts'
        if h.state['C']==c and t<=h.state['step']<=t+5 and self.audit:
            self.states[str(h.state['step'])]=state_snapshot(args)
    def capture_geometry(self,ctx,k,v):
        return dict(q=geometry(ctx['q']),k=geometry(k),v=geometry(v),prefix=ctx['prefix'],length=len(k),layer=29,canvas=ctx['canvas'],step=ctx['step'],gqa_map=[h//8 for h in range(16)],page_table_sha256=fingerprint(ctx['meta'].block_table[0,:math.ceil(len(k)/ctx['kc'].shape[1])]),cpu_rng=fingerprint(torch.get_rng_state()),cuda_rng=fingerprint(torch.cuda.get_rng_state()))
    def bank(self,ctx,k,v,fraction,key,previous_q):
        probe,entropy,protected,margin,pwork=make_probe(self.h.consumer.model,29,self.h.state['prior'],ctx['step']);self.work.update({x:pwork[x] for x in ['linear_projection_multiply_adds','head_weight_rows']})
        op=lambda units:native_attention(ctx['q'],ctx['kc'],ctx['vc'],ctx['meta'],ctx['scale'],units)[0]
        identity=f"{self.row['id']}|{self.seed}|{ctx['canvas']}|{ctx['step']}|29"
        bank,priv,work=build(ctx['q'],k,v,ctx['prefix'],ctx['scale'],fraction,probe,entropy,protected,margin,op,identity);self.work.update(work)
        for i,r in enumerate(bank['candidates']):r['sparse_sha256']=fingerprint(priv['candidates'][i]['sparse'])
        choices,work=augment(bank,priv,ctx['q'],k,v,ctx['scale'],previous_q,self.h.state['prior']);self.work.update(work);bank['all_selections']=choices
        bank['selector_source']='common_bank_l29_adaptations_v1';bank['fraction']=fraction
        full=self.mode=='reference' or (self.config.get('selector')=='current_error' and self.config.get('lifecycle')=='refresh2' and fraction==.05 and self.seed==42 and self.row.get('stratum')=='targeted')
        exported=priv if full else dict(tensor_hashes={n:fingerprint(x) for n,x in priv.items() if torch.is_tensor(x)},candidates=[{n:dict(shape=list(x.shape),dtype=str(x.dtype),sha256=fingerprint(x)) for n,x in c.items() if torch.is_tensor(x)} for c in priv.get('candidates',[])],scope='metadata_only_operator_bank_all_applied_supports_saved_separately')
        pp=self.root/'private/banks'/self.label/(key+'.pt');bank['private_sha256']=tensor_save(pp,exported);bank['private_relative']=str(pp.relative_to(self.root));bank['full_operator_tensors_retained']=full
        path=self.root/'banks'/self.label/(key+'.json');save(path,bank);self.sites.append(str(path.relative_to(self.root)))
        return bank,priv
    def certificate(self,ctx,k,units):
        cs=[];blocks=ctx['prefix']//64
        for u in units:
            h,qb=u['head'],u['qb'];qu=ctx['q'][qb*128:(qb+1)*128,h].float();kp=k[:blocks*64,h//8].float();z=qu@kp.T*ctx['scale'];keep=torch.ones(blocks*64,dtype=torch.bool,device=qu.device)
            for tile in u['tiles']:keep[tile*64:(tile+1)*64]=False
            cs.append(dict(q=qu.detach().clone(),prefix_logZ=z.logsumexp(-1),kept_logZ=z.masked_fill(~keep[None],-torch.inf).logsumexp(-1),key_norm=kp.norm(dim=-1).max(),key_sha256=fingerprint(k[:blocks*64,h//8])))
        self.work['certificate_anchor_QK_rectangles']+=4*blocks
        return cs
    def shadow(self,ctx,k,rec,age):
        if self.cert is None:return None
        bounds=[];blocks=ctx['prefix']//64;drifts=[]
        for u,c in zip(self.config['units'],self.cert):
            h,qb=u['head'],u['qb'];qu=ctx['q'][qb*128:(qb+1)*128,h].float();tail=k[blocks*64:,h//8].float();assert fingerprint(k[:blocks*64,h//8])==c['key_sha256'],'within-canvas immutable-prefix certificate scope'
            z=(qu@tail.T*ctx['scale']).logsumexp(-1);d=(qu-c['q']).norm(dim=-1)*c['key_norm']*ctx['scale'];bound=(torch.logaddexp(c['kept_logZ']-d,z)-torch.logaddexp(c['prefix_logZ']+d,z)).exp().clamp(max=1);bounds.append(float(bound.min()));drifts.append(float((qu-c['q']).norm(dim=-1).max()))
        self.work['certificate_current_tail_QK_rectangles']+=4*math.ceil((len(k)-blocks*64)/64)
        minimum=min(bounds);actual=1-rec['deleted_mass'];assert minimum<=actual+2e-5
        return dict(age=age,lower_bound_min=minimum,actual_retained_mass_min=actual,safe=minimum>=.8,query_drift_max=max(drifts),shadow_only=True,affects_policy=False)
    def before_attention(self,ctx):
        cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
        result=self._before_attention(ctx)
        assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state()),'selector/operator consumed native RNG'
        return result
    def _before_attention(self,ctx):
        self.pending=None
        if ctx['layer']!=29:return None
        if self.q_canvas!=ctx['canvas']:self.previous_q=None;self.q_canvas=ctx['canvas']
        previous=self.previous_q;self.previous_q=ctx['q'].detach().clone()
        if not self.anchor or ctx['canvas']!=self.anchor['canvas']:return None
        offset=ctx['step']-self.anchor['after_completed_calls']-1
        if self.mode=='reference' and offset==0:
            k,v=gather_current(ctx['q'],ctx['key'],ctx['value'],ctx['kc'],ctx['vc'],ctx['meta'],ctx['prefix']);geo=self.capture_geometry(ctx,k,v);done={};banks=[]
            for fraction in [.05,.1]:
                drop=max(1,math.floor(fraction*(ctx['prefix']//64)))
                if drop in done:banks.append(dict(fraction=fraction,drop_count=drop,alias_of=done[drop]));continue
                name=f'initial_d{drop}';bank,priv=self.bank(ctx,k,v,fraction,name,previous);done[drop]=len(banks);banks.append(dict(fraction=fraction,drop_count=drop,relative=self.sites[-1],selections=bank['all_selections']))
            path=self.root/'sites'/self.label/'initial.json';save(path,dict(geometry=geo,banks=banks,choices_written_before_native_outcomes=True));self.initial_site=str(path.relative_to(self.root));return None
        if self.mode!='branch' or not active_schedule(self.config['lifecycle'],offset):return None
        k,v=gather_current(ctx['q'],ctx['key'],ctx['value'],ctx['kc'],ctx['vc'],ctx['meta'],ctx['prefix']);geo=self.capture_geometry(ctx,k,v)
        if offset==0 or (self.config['lifecycle']=='AS' and offset==1):
            if offset==0:assert logical(geo)==logical(self.config['geometry']),'initial CURRENT QKV/RNG/geometry'
            units=self.config['units'];rec,priv,work=fixed(ctx,k,v,units);self.work.update(work)
            if offset==0:
                assert fingerprint(priv['sparse'])==self.config['sparse_sha256']
                self.cert=self.certificate(ctx,k,units)
        else:
            heldrec,heldpriv,work=fixed(ctx,k,v,self.config['units']);self.work.update(work)
            rec,priv,units=heldrec,heldpriv,self.config['units']
            if self.config['lifecycle'].startswith('refresh') or self.config['lifecycle']=='matched_change2':
                bank,bpriv=self.bank(ctx,k,v,self.config['fraction'],f'refresh_s{ctx["step"]:02d}',previous);chosen=bank['all_selections'].get(self.config['selector'])
                if self.config['lifecycle']=='matched_change2':
                    valid=[]
                    for i,r in enumerate(bank['candidates']):
                        b=bpriv['candidates'][i]['actual_bias'].double();hb=heldpriv['actual_bias'].double();sep=float((b-hb).norm())/max(float(b.norm()),float(hb.norm()),1e-12)
                        if r['feasible'] and r['support_sha256']!=heldrec['support_sha256'] and matched(r,heldrec) and min(r['bias_relative_Frobenius'],heldrec['bias_relative_Frobenius'])>=1e-6 and sep>=.25:valid.append(i)
                    chosen=min(valid,key=lambda i:hashlib.sha256(('matched-change-v1|'+bank['candidates'][i]['support_sha256']).encode()).hexdigest()) if valid else None
                    if chosen is None:chosen='held_alias'
                if chosen=='held_alias':pass
                elif chosen is None:
                    self.events.append(dict(offset=offset,refresh_fallback_allkept=True,geometry=geo));return None
                else:
                    units=bank['candidates'][chosen]['units'];rec,priv,work=fixed(ctx,k,v,units);self.work.update(work);self.refreshes+=1
            hb=heldpriv['actual_bias'].double();cb=priv['actual_bias'].double();sep=float((hb-cb).norm())/max(float(hb.norm()),float(cb.norm()),1e-12)
            self.events.append(dict(second_or_later_comparison=True,offset=offset,geometry=geo,held_support_sha256=heldrec['support_sha256'],selected_support_sha256=rec['support_sha256'],identifying=rec['support_sha256']!=heldrec['support_sha256'],matched_current_error_mass=matched(rec,heldrec),actual_bias_separation=sep,meaningful=matched(rec,heldrec) and sep>=.25 and min(rec['bias_relative_Frobenius'],heldrec['bias_relative_Frobenius'])>=1e-6,held_guard_now=heldrec['feasible']))
        if rec['support_sha256']==self.config['support_sha256']:self.holds+=int(offset>0)
        shadow=self.shadow(ctx,k,rec,offset) if units==self.config['units'] else None
        stepkey=f's{ctx["step"]:02d}';pp=self.root/'private/applied'/self.label/(stepkey+'.pt')
        ph=tensor_save(pp,priv);choice=dict(step=ctx['step'],offset=offset,geometry=geo,support_sha256=rec['support_sha256'],units=units,measurement=rec,shadow=shadow,private_sha256=ph,private_relative=str(pp.relative_to(self.root)),selection_before_native_consumer=True)
        save(self.root/'choices'/self.label/(stepkey+'.json'),choice)
        self.pending=dict(choice=choice,sparse_sha256=fingerprint(priv['sparse']),bias=priv['actual_bias']);self.pulse_calls+=1
        return units
    def after_attention(self,ctx,support,result,keep):
        if self.pending is None:return
        p=self.pending;assert support_hash(support)==p['choice']['support_sha256'];assert fingerprint(gather_units(result))==p['sparse_sha256'],'actual consumer vs fresh diagnostic same support'
        assert int((~keep).sum())==4*self.config['drop_count'];b=p['bias'].double();cos=None
        if self.last_bias is not None and float(b.norm())>0 and float(self.last_bias.norm())>0:cos=float((b*self.last_bias).sum()/(b.norm()*self.last_bias.norm()))
        self.last_bias=b
        self.events.append(dict(intervention=True,step=ctx['step'],offset=p['choice']['offset'],actual_support_sha256=support_hash(support),actual_output_sha256=p['sparse_sha256'],measurement=p['choice']['measurement'],shadow=p['choice']['shadow'],cross_call_bias_cosine=cos,current_QKV=True,LOCAL_deleted=0,protected_deleted=0));self.pending=None
    def finish(self,record):
        local=None
        if self.anchor:
            calls=[r for r in record['native_trace'] if not r['commit'] and r['canvas']==self.anchor['canvas']];commits=[r for r in record['native_trace'] if r['commit'] and r['canvas']==self.anchor['canvas']]
            if calls:local=dict(remaining_calls=sum(r['native_step']>self.anchor['after_completed_calls'] for r in calls),terminal=calls[-1],commit=commits[0] if commits else None)
        return dict(study_mode=self.mode,pre_intervention=self.before,initial_site=getattr(self,'initial_site',None) if self.mode=='reference' else None,bank_sites=self.sites,intervention_calls=self.pulse_calls,events=self.events,window_native_states=self.states,local_endpoint=local,operator_diagnostic_work=dict(self.work),refresh_calls=self.refreshes,held_calls=self.holds,carried_calls=0,clean_online_latency_qualified=False)
