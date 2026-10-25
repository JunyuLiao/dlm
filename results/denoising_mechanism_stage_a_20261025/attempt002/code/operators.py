"""Current operator diagnostics. Past operands enter scores only, never attention."""
import hashlib
import itertools
import math
import torch
from collections import Counter
WORK=Counter()


def count(name,rows,keys):
    WORK[name]+=math.ceil(rows/128)*math.ceil(keys/64)


def anchors(length):
    return sorted({x for x in [max(2,length//2),length-2] if 2<=x<length})


def support_bank(p, v, blocks, drop, unit, limit=48):
    """Diverse finite proposals: whole independent sets, multi-swaps and value signs."""
    mass=p[:,:blocks*64].reshape(len(p),blocks,64).amax(0).sum(-1)
    ranking=torch.argsort(mass,stable=True).tolist()
    result=[]
    def add(values):
        values=tuple(sorted(values))
        if len(values)==drop and len(set(values))==drop and values not in result:result.append(values)
    if math.comb(blocks,drop)<=limit:
        return list(itertools.combinations(range(blocks),drop))
    add(ranking[:drop]);add(range(drop));add(range(blocks-drop,blocks))
    for offset in range(4):
        add((offset+int(j*blocks/drop))%blocks for j in range(drop))
    # Attention-weighted signed tile contribution along current output's mean
    # direction. This is current-only proposal information, not a future score.
    count('PV_rectangles',len(p),len(v))
    WORK['value_projection_key_rows']+=blocks*64
    o=p@v; axis=o.mean(0);axis=axis/axis.norm().clamp_min(1e-8)
    signed=(p[:,:blocks*64]*(v[:blocks*64]@axis)[None]).reshape(len(p),blocks,64).sum(-1).mean(0)
    for mass_width in [max(drop*2,16),max(drop*4,32),blocks]:
        pool=ranking[:min(blocks,mass_width)]
        for reverse in [False,True]:
            add(sorted(pool,key=lambda j:float(signed[j]),reverse=reverse)[:drop])
    for index in range(limit*2):
        pool=ranking[:min(blocks,max(drop*4,16))] if index%2==0 else list(range(blocks))
        hashed=sorted(pool,key=lambda j:hashlib.sha256(f'19331|{unit}|{index}|{j}'.encode()).digest())
        add(hashed[:drop])
        # Multi-tile exchange changes half the deletion set, rather than one.
        base=set(ranking[:drop]); remove=max(1,drop//2)
        retained=sorted(base,key=lambda j:hashlib.sha256(f'r|{index}|{j}'.encode()).digest())[remove:]
        extras=[j for j in hashed if j not in base][:remove]
        add(retained+extras)
        if len(result)>=limit:break
    return result[:limit]


def directions(q,k,v,previous,prefix,unit):
    dq=q-previous['q'].to(q.device).float()
    dk=k[prefix:]-previous['k_tail'].to(k.device).float()
    dv=v[prefix:]-previous['v_tail'].to(v.device).float()
    # Orthogonal feature permutation/sign sham, separate for each operand;
    # Frobenius motion scale preserved exactly up to arithmetic roundoff.
    shift=1+(unit*13)%max(1,q.shape[-1]-1)
    sham=tuple(torch.roll(x,shift, -1)*(-1 if unit%2 else 1) for x in (dq,dk,dv))
    return {'q_past':(dq,torch.zeros_like(dk),torch.zeros_like(dv)),
            'qkv_past':(dq,dk,dv),'direction_sham':sham}


def differential(p,q,k,v,direction,prefix,scale):
    dq,dk,dv=direction
    count('QK_rectangles',len(q),len(k));count('QK_rectangles',len(q),len(dk))
    count('PV_rectangles',len(q),len(v));count('PV_rectangles',len(q),len(dv))
    ds=(dq@k.T)*scale
    ds[:,prefix:]+=(q@dk.T)*scale
    dp=p*(ds-(p*ds).sum(-1,keepdim=True))
    return dp@v+p[:,prefix:]@dv


def operator(q,k,v,scale,support=None):
    count('QK_rectangles',len(q),len(k));count('PV_rectangles',len(q),len(v))
    logits=q@k.T*scale
    if support is not None:
        indices=torch.tensor(support,device=q.device)[:,None]*64+torch.arange(64,device=q.device)[None]
        logits[:,indices.flatten()]=-torch.inf
    p=logits.softmax(-1)
    return p,p@v


def score_bank(q,k,v,previous,prefix,scale,unit,drop):
    q,k,v=q.float(),k.float(),v.float()
    p,o=operator(q,k,v,scale)
    denominator=o.norm(dim=-1).clamp_min(1e-8)
    dirs=directions(q,k,v,previous,prefix,unit)
    ja={name:differential(p,q,k,v,d,prefix,scale) for name,d in dirs.items()}
    bank=support_bank(p,v,prefix//64,drop,unit)
    public=[];private=[]
    for support in bank:
        ps=p.clone()
        idx=(torch.tensor(support,device=q.device)[:,None]*64+torch.arange(64,device=q.device)[None]).flatten()
        mass=ps[:,idx].sum(-1);ps[:,idx]=0;ps/=ps.sum(-1,keepdim=True)
        count('PV_rectangles',len(q),len(v))
        output=ps@v;bias=output-o
        responses={name:differential(ps,q,k,v,d,prefix,scale) for name,d in dirs.items()}
        error=float((bias.norm(dim=-1)/denominator).max());deleted=float(mass.max())
        row=dict(tiles=list(support),support_sha256=hashlib.sha256(str(support).encode()).hexdigest(),current_error=error,deleted_mass=deleted,
                 feasible=error<=.1 and deleted<=.2,renormalization=float((1/(1-mass)).max()),
                 scores={name:float((value.norm(dim=-1)/denominator).max()) for name,value in responses.items()},
                 residual_scores={name:float(((value-ja[name]).norm(dim=-1)/denominator).max()) for name,value in responses.items()},
                 position_bias_magnitude=(bias.norm(dim=-1)/denominator).cpu().tolist(),
                 bias_direction_sha256=hashlib.sha256(bias.mean(0).detach().cpu().numpy().tobytes()).hexdigest())
        public.append(row)
        qpart=responses['q_past']
        count('PV_rectangles',len(q),len(dirs['qkv_past'][2]))
        vpart=ps[:,prefix:]@dirs['qkv_past'][2]
        kpart=responses['qkv_past']-qpart-vpart
        row['qkv_response_component_norms']=[float(x.norm()) for x in [qpart,kpart,vpart]]
        private.append(dict(output=output.cpu(),bias=bias.cpu(),responses={n:x.cpu() for n,x in responses.items()},
                            qkv_components=[x.cpu() for x in [qpart,kpart,vpart]],bias_direction_mean=bias.mean(0).cpu()))
    feasible=[i for i,x in enumerate(public) if x['feasible']]
    choices={}
    for name in ['current_error','q_past','qkv_past','direction_sham']:
        choices[name]=min(feasible,key=lambda i:(public[i]['current_error'] if name=='current_error' else public[i]['scores'][name],public[i]['tiles'])) if len(feasible)>=4 else None
    # Sham support is matched to current-error's bias/mass from the same set.
    ce=choices['current_error']
    matched_sham=[i for i in feasible if ce is not None and pair_match(public[ce],public[i])[0]]
    choices['support_sham']=min(matched_sham,key=lambda i:hashlib.sha256(f'19331|{unit}|{public[i]["tiles"]}'.encode()).digest()) if matched_sham else None
    pairs=[]
    for a,b in itertools.combinations(feasible,2):
        match,separation=pair_match(public[a],public[b])
        if match:pairs.append(dict(a=a,b=b,full_response_separated=separation,
            separated_by_score={name:pair_match(public[a],public[b],name)[1] for name in ['q_past','qkv_past','direction_sham']},
            signed_bias_cosine=float(torch.nn.functional.cosine_similarity(private[a]['bias'].flatten(),private[b]['bias'].flatten(),dim=0))))
    # Private tensors carry all signed vectors; no prefix K/V are stored.
    state=dict(q=q.cpu(),k_tail=k[prefix:].cpu(),v_tail=v[prefix:].cpu(),dense_output=o.cpu(),
        denominator=denominator.cpu(),JA={name:x.cpu() for name,x in ja.items()},candidates=private,
        direction_norms={name:[float(t.norm()) for t in d] for name,d in dirs.items()})
    return dict(candidates=public,choices=choices,feasible=len(feasible),pairs=pairs,drop_count=drop,
                prefix_blocks=prefix//64,direction_norms=state['direction_norms']),state


def pair_match(a,b,score='qkv_past'):
    error=abs(a['current_error']-b['current_error'])/max(a['current_error'],b['current_error'],1e-6)
    mass=abs(a['deleted_mass']-b['deleted_mass'])
    x,y=a['scores'][score],b['scores'][score]
    return error<=.05 and mass<=.01,abs(x-y)/max(abs(x),abs(y),1e-6)>=.25


def evaluate_future(bank,state,q,k,v,prefix,scale):
    q,k,v=q.float(),k.float(),v.float();p,o=operator(q,k,v,scale)
    # Actual future motion is evaluation-only. It never enters score_bank.
    oldq=state['q'].to(q.device); oldk=k.clone();oldv=v.clone()
    oldk[prefix:]=state['k_tail'].to(k.device);oldv[prefix:]=state['v_tail'].to(v.device)
    actual=(q-oldq,k[prefix:]-oldk[prefix:],v[prefix:]-oldv[prefix:])
    oldp,_=operator(oldq,oldk,oldv,scale)
    rows=[]
    for index,candidate in enumerate(bank['candidates']):
        ps,os=operator(q,k,v,scale,candidate['tiles']);old=state['candidates'][index]
        motion=os.cpu()-old['output'];bias=os.cpu()-o.cpu();bias_delta=bias-old['bias']
        den=state['denominator'][:,None]
        # Vector RMSE relative to current dense output row norms, averaged over rows.
        def loss(pred,truth):return float((((pred-truth)/den).square().sum(-1)).mean())
        pred=old['responses'];baseline=state['JA']['qkv_past']
        oldps=oldp.clone();idx=(torch.tensor(candidate['tiles'],device=q.device)[:,None]*64+torch.arange(64,device=q.device)[None]).flatten()
        oldps[:,idx]=0;oldps/=oldps.sum(-1,keepdim=True)
        oracle=differential(oldps,oldq,oldk,oldv,actual,prefix,scale).cpu()
        qa=differential(oldp,oldq,oldk,oldv,actual,prefix,scale).cpu()
        rows.append(dict(candidate=index,feasible=candidate['feasible'],actual_motion=float((motion.norm(dim=-1)/state['denominator']).max()),
            actual_bias_change=float((bias_delta.norm(dim=-1)/state['denominator']).max()),
            forecast_loss={name:loss(value,motion) for name,value in pred.items()} |
                dict(bias_carry_dense_forecast=loss(baseline,motion),persistence=loss(torch.zeros_like(motion),motion),future_motion_oracle=loss(oracle,motion)),
            residual_forecast_loss={name:loss(value-state['JA'][name],bias_delta) for name,value in pred.items()} |
                dict(carry_current_bias=loss(torch.zeros_like(bias_delta),bias_delta),future_motion_oracle=loss(oracle-qa,bias_delta)),
            future_current_error=float((bias.norm(dim=-1)/state['denominator']).max())))
    return rows
