"""Named common-bank adaptations. No model calls and no native RNG draws."""
import hashlib, math
from collections import Counter
import torch
from op_bank import UNITS, matched

SELECTORS=['current_error','mass','support_sham','readout_margin','response_qpast','margin_weighted','signed_high','signed_low']

def augment(bank,private,q,k,v,scale,previous_q,prior):
    work=Counter(); records=bank['candidates']
    if not records:return {},dict(work)
    response=[[] for _ in records];residual=[[] for _ in records];ja_all=[]
    for unit,(h,qb) in enumerate(UNITS):
        qu=q[qb*128:(qb+1)*128,h].float();ku=k[:,h//8].float();vu=v[:,h//8].float()
        assert previous_q is not None,'same-canvas previous query required'
        direction=qu-previous_q[qb*128:(qb+1)*128,h].float()
        p=(qu@ku.T*scale).softmax(-1);o=p@vu;ds=direction@ku.T*scale
        dp=p*(ds-(p*ds).sum(-1,keepdim=True));ja=dp@vu;norm=o.norm(dim=-1).clamp_min(1e-8)
        ja_all.append(float((ja.norm(dim=-1)/norm).max()))
        work.update(response_QK_rectangles=2*math.ceil(len(k)/64),response_PV_rectangles=2*math.ceil(len(k)/64))
        for i,r in enumerate(records):
            tiles=r['units'][unit]['tiles'];ids=(torch.tensor(tiles,device=q.device)[:,None]*64+torch.arange(64,device=q.device)).flatten()
            mass=p[:,ids].sum(-1);dm=dp[:,ids].sum(-1);c=p[:,ids]@vu[ids];dc=dp[:,ids]@vu[ids];ret=(1-mass).clamp_min(1e-12)
            db=(dm[:,None]*o+mass[:,None]*ja-dc)/ret[:,None]+(mass[:,None]*o-c)*dm[:,None]/ret[:,None].square()
            response[i].append(float(((ja+db).norm(dim=-1)/norm).max()));residual[i].append(float((db.norm(dim=-1)/norm).max()))
            work.update(response_PV_rectangles=2*len(tiles))
    dense=private['dense'].float();dn=dense.norm(dim=-1).clamp_min(1e-8)
    stable=prior['stable_positions'];margin=prior['margin'].float()
    protected=stable if bool(stable.any()) else torch.ones_like(stable)
    weights=protected.float()/margin.clamp_min(.1)
    w=torch.stack([weights[qb*128:(qb+1)*128] for h,qb in UNITS])
    for i,r in enumerate(records):
        b=private['candidates'][i]['actual_bias'].float()
        r.update(response_J_S=max(response[i]),response_residual=max(residual[i]),response_J_A=max(ja_all),
                 response_direction='q_current_minus_previous_same_canvas_unscaled',margin_weighted=float((w*b.norm(dim=-1)/dn).max()),
                 signed_output_alignment=float((b*dense).sum()/dense.square().sum().clamp_min(1e-16)),previous_protected_empty=not bool(stable.any()))
    ids=[i for i,r in enumerate(records) if r['feasible']]
    if not ids:return {},dict(work)
    ce=bank['selections']['current_error'];pool=[i for i in ids if matched(records[i],records[ce])]
    tie=lambda i:records[i]['support_sha256']
    choices={k:bank['selections'][k] for k in ['current_error','mass','support_sham','readout_margin']}
    choices.update(response_qpast=min(ids,key=lambda i:(records[i]['response_J_S'],tie(i))),
                   margin_weighted=min(ids,key=lambda i:(records[i]['margin_weighted'],tie(i))),
                   signed_high=min(pool,key=lambda i:(-records[i]['signed_output_alignment'],tie(i))),
                   signed_low=min(pool,key=lambda i:(records[i]['signed_output_alignment'],tie(i))))
    return choices,dict(work)

def active_schedule(lifecycle,offset):
    return {'SA':offset==0,'AS':offset==1,'held2':0<=offset<2,'refresh2':0<=offset<2,
            'matched_change2':0<=offset<2,'held4':0<=offset<4,'refresh4':0<=offset<4}[lifecycle]

def union_window(anchors,canvas,step):
    return any(a['canvas']==canvas and a['after_completed_calls']<=step<=a['after_completed_calls']+5 for a in anchors.values())

def canonical_numeric(t):
    x=t.detach().float().clone();return torch.where(x==0,torch.zeros_like(x),x)

def full_policy_regressions(policies,controls):
    """Includes every cell, including a policy's neutral all-kept fallback."""
    return [dict(id=v['id'],seed=v['seed']) for v in policies if controls[(v['id'],v['seed'])]['correct'] and not v['correct']]
