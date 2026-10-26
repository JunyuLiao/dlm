"""Read actual native winners/flags; previous information and fixed-reference margins."""
import torch
from shared_consumer import fingerprint

def observe(result,args,kw,reference=None,positions=False,minimal=False,previous=None):
    slot=args[1];z=result.reshape(-1,result.shape[-1]);winner=args[6][slot].reshape(-1)
    if not minimal:assert torch.equal(z.argmax(-1),winner),'native argmax equality'
    values,indices=z.topk(2,dim=-1);competitor=torch.where(indices[:,0]==winner,indices[:,1],indices[:,0]);competitor_value=z.gather(1,competitor[:,None]).squeeze(1);winner_value=z.gather(1,winner[:,None]).squeeze(1)
    ent=[]
    for chunk in z.split(16):
        lp=chunk.float().log_softmax(-1);ent.append((-(lp.exp()*lp).sum(-1)).cpu())
    entropy=torch.cat(ent);margin=(winner_value-competitor_value).cpu();history=args[13][slot];ready=int(args[14][slot].item())>=kw['ST'];stable_positions=(history==history[:,:1]).all(dim=1).reshape(-1).cpu()
    private=dict(winner=winner.detach().cpu(),competitor=competitor.detach().cpu(),winner_logit=winner_value.detach().cpu(),competitor_logit=competitor_value.detach().cpu(),margin=margin,entropy=entropy,stable_positions=stable_positions)
    if previous is not None:
        pa=previous['winner'].to(z.device);pb=previous['competitor'].to(z.device)
        private['previous_winner_current_logit']=z.gather(1,pa[:,None]).squeeze(1).detach().cpu()
        private['previous_competitor_current_logit']=z.gather(1,pb[:,None]).squeeze(1).detach().cpu()
        private['previous_pair_current_margin']=private['previous_winner_current_logit']-private['previous_competitor_current_logit']
        private['previous_pair_aligned']=(winner==pa).detach().cpu()
    if minimal:return None,private
    row=dict(native_step=int(args[7][slot].item()),history_ready=ready,history_length=int(args[14][slot].item()),stable=bool(stable_positions.all()),confidence=bool(args[9][slot].item()),converged=bool(args[8][slot].item()),cap_reached=int(args[7][slot].item())>=48,
        native_winner_sha256=fingerprint(winner),entropy_mean=float(entropy.mean()),entropy_slack=.005-float(entropy.mean()),margin_mean=float(margin.mean()),margin_low_quantiles=torch.quantile(margin,torch.tensor([0.,.05,.5])).tolist(),stable_position_fraction=float(stable_positions.float().mean()))
    assert row['converged']==((row['history_ready'] and row['stable'] and row['confidence']) or row['cap_reached'])
    if positions:row.update(entropy_by_position=entropy.tolist(),native_margin_by_position=margin.tolist())
    if reference is not None:
        a=reference['winner'].to(z.device);comp=torch.where(indices[:,0]==a,values[:,1],values[:,0]);signed=z.gather(1,a[:,None]).squeeze(1)-comp;delta=signed.cpu()-reference['margin']
        pa=reference['winner'].to(z.device);pb=reference['competitor'].to(z.device);pair=z.gather(1,pa[:,None]).squeeze(1)-z.gather(1,pb[:,None]).squeeze(1)
        row.update(reference_winner_crossings=int((signed<0).sum()),actual_native_winner_changes=int((winner!=a).sum()),worst_margin_erosion=float((-delta).max()),mean_absolute_margin_change=float(delta.abs().mean()),signed_margin_quantiles=torch.quantile(delta,torch.tensor([0.,.05,.5,.95,1.])).tolist())
        if positions:row.update(signed_reference_winner_margin_change=delta.tolist(),reference_pair_margin=pair.cpu().tolist())
    return row,private

def state_snapshot(args):
    slot=args[1];state={}
    for name,index in [('stochastic_canvas',5),('native_winner',6),('step',7),('converged',8),('confidence',9),('self_condition',10),('history',13),('history_length',14)]:
        t=args[index][slot].detach();state[name]=dict(byte_sha256=fingerprint(t),shape=list(t.shape),dtype=str(t.dtype))
        if t.is_floating_point():
            copy=t.float().cpu();state[name].update(numerical_sha256=fingerprint(torch.where(copy==0,torch.zeros_like(copy),copy)),nonfinite=int((~torch.isfinite(copy)).sum()),L2=float(copy.double().norm()))
    state['cpu_rng']=fingerprint(torch.get_rng_state());state['cuda_rng']=fingerprint(torch.cuda.get_rng_state())
    return state
