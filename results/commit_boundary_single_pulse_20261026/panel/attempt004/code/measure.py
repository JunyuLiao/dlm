"""Read actual native state; reference winners are private, numeric deltas public."""
import hashlib
import torch


def digest(t):return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def summarize(result,args,kw):
    slot=args[1];history=args[13][slot];length=int(args[14][slot].item())
    winner=args[6][slot].reshape(-1)
    assert torch.equal(result.argmax(-1).reshape(-1),winner),'actual native winner mismatch'
    entropy=[];margin=[];top_competitor=[]
    for block in result.reshape(-1,result.shape[-1]).split(16):
        logp=block.float().log_softmax(-1);entropy.append((-(logp.exp()*logp).sum(-1)).cpu())
        values=block.topk(2,dim=-1).values; margin.append((values[:,0]-values[:,1]).cpu())
    ent=torch.cat(entropy);marg=torch.cat(margin)
    row=dict(native_step=int(args[7][slot].item()),stable=bool((history==history[:,:1]).all()),history_ready=length>=kw['ST'],
        history_length=length,confidence=bool(args[9][slot].item()),converged=bool(args[8][slot].item()),
        cap_reached=int(args[7][slot].item())>=48,entropy_mean=float(ent.mean()),entropy_slack=.005-float(ent.mean()),
        entropy_by_position=ent.tolist(),margin_by_position=marg.tolist())
    state={name:digest(args[index][slot]) for name,index in [('canvas',5),('native_winner',6),('self_condition',10)]}
    state.update(logits=digest(result),history=digest(history[:,:min(length,history.shape[1])]),cpu_rng=digest(torch.get_rng_state()),cuda_rng=digest(torch.cuda.get_rng_state()),
        native_step=row['native_step'],history_length=length,confidence=row['confidence'],converged=row['converged'])
    private=dict(winner=winner.detach().cpu(),margin=marg,entropy=ent)
    return row,state,private


def compare_to_reference(result,reference):
    z=result.reshape(-1,result.shape[-1]);winner=reference['winner'].to(z.device)
    values,indices=z.topk(2,dim=-1)
    competitor=torch.where(indices[:,0]==winner,values[:,1],values[:,0])
    signed=z.gather(1,winner[:,None]).squeeze(1)-competitor
    delta=signed.cpu()-reference['margin']
    return dict(reference_winner_margin_by_position=signed.cpu().tolist(),signed_margin_change_by_position=delta.tolist(),
        worst_margin_erosion=float((-delta).max()),mean_absolute_margin_change=float(delta.abs().mean()),
        signed_margin_change_quantiles=torch.quantile(delta,torch.tensor([0.,.05,.5,.95,1.])).tolist(),
        reference_winner_crossings=int((signed<0).sum()),reference_agreement_is_task_correctness=False)
