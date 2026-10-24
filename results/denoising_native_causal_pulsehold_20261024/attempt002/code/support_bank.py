"""Bounded four-unit current-Q/K/V support bank for pulse/hold diagnostics."""
from __future__ import annotations

import hashlib
import math
import random

import torch

from pulsehold import build_short_single_deletion_bank


UNIT_HEADS = (0, 8)
UNIT_QUERY_BLOCKS = (0, 1)
GQA_KV_HEAD = {0: 0, 8: 1}


def _hash_support(support) -> str:
    payload = bytes(int(x) for unit in support for x in unit)
    return hashlib.sha256(payload).hexdigest()


def _tile_stats(q, k, v, keep, prefix_blocks, scale):
    """FP64 diagnostic projection for one routing unit; no native RNG use."""
    q = q.double()
    k = k.double()
    v = v.double()
    p = (q @ k.T * scale).softmax(-1)
    dense = p @ v
    support = torch.ones(k.shape[0], dtype=torch.bool, device=k.device)
    support[:prefix_blocks*64] = keep.to(k.device).bool().repeat_interleave(64)
    retained = p*support[None]
    kept_mass = retained.sum(-1).clamp_min(torch.finfo(p.dtype).tiny)
    sparse = retained @ v/kept_mass[:, None]
    error = (sparse-dense).norm(dim=-1)/dense.norm(dim=-1).clamp_min(1e-12)
    deleted = 1-kept_mass
    prefix_mass = p[:, :prefix_blocks*64].sum(-1).clamp_min(1e-12)
    return dict(error_worst=float(error.max().item()),
        whole_deleted_mass_worst=float(deleted.max().item()),
        prefix_deleted_mass_worst=float((deleted/prefix_mass).max().item()),
        whole_renormalization_worst=float((1/kept_mass).max().item()),
        dense_output=dense, probabilities=p)


def _unit_views(q, k, v, prefix):
    if q.ndim != 3 or q.shape[1] < 9 or k.ndim != 3 or k.shape[1] < 2:
        raise ValueError('requires Q[N,heads,D] and GQA K/V[K,2,D]')
    if prefix > k.shape[0]:
        raise ValueError('prefix exceeds current K/V length')
    blocks = prefix//64
    result = []
    for head in UNIT_HEADS:
        kv_head = GQA_KV_HEAD[head]
        for qb in UNIT_QUERY_BLOCKS:
            lo, hi = qb*128, min((qb+1)*128, q.shape[0])
            if hi-lo != 128:
                raise ValueError('query blocks must have exactly128 rows')
            result.append((q[lo:hi, head], k[:, kv_head], v[:, kv_head]))
    return blocks, result


def candidate_bank(q, k, v, prefix, budget, previous_qkv=None,
                   seed=42, proposal_limit=256, max_candidates=12):
    """Return equal-work distinct supports with first-call-only operator scores."""
    blocks, units = _unit_views(q, k, v, prefix)
    if budget['kind'] == 'single_delete':
        rows = build_short_single_deletion_bank((blocks,)*4, max_candidates, seed)
        supports = [row['support'] for row in rows]
    elif budget['kind'] == 'per_unit_fraction':
        keep_each = math.ceil(blocks*float(budget['fraction']))
        if not 0 < keep_each < blocks:
            raise ValueError('long per-unit budget must delete at least one and keep at least one block')
        supports = []
        mass_support = []
        per_unit_mass = []
        for q_u,k_u,v_u in units:
            p = (q_u.float()@k_u.float().T*float(budget['scale'])).softmax(-1)
            tile_mass = p[:, :prefix].reshape(128, blocks, 64).sum(-1).mean(0)
            per_unit_mass.append(tile_mass)
            keep = torch.zeros(blocks, dtype=torch.bool, device=q.device)
            keep[tile_mass.topk(keep_each, largest=True, sorted=True).indices] = True
            mass_support.append(keep.cpu().tolist())
        supports.append(mass_support)
        # Near/far single-unit swaps and balanced four-unit swaps.
        for unit in range(4):
            kept_ids = torch.where(torch.tensor(mass_support[unit]))[0].tolist()
            drop_ids = torch.where(~torch.tensor(mass_support[unit]))[0].tolist()
            add_near = max(drop_ids, key=lambda x: float(per_unit_mass[unit][x]))
            add_far = min(drop_ids, key=lambda x: float(per_unit_mass[unit][x]))
            remove_low = min(kept_ids, key=lambda x: float(per_unit_mass[unit][x]))
            for add_id, label in ((add_near, 'near'), (add_far, 'far')):
                support = [list(x) for x in mass_support]
                support[unit][remove_low] = False
                support[unit][add_id] = True
                supports.append(support)
        balanced = [list(x) for x in mass_support]
        for unit in range(4):
            kept_ids = [i for i,x in enumerate(balanced[unit]) if x]
            drop_ids = [i for i,x in enumerate(balanced[unit]) if not x]
            balanced[unit][min(kept_ids)] = False
            balanced[unit][max(drop_ids, key=lambda x: (x*2654435761+seed)&0xffffffff)] = True
        supports.append(balanced)
        # Independent positional/hash shams. Python RNG is isolated from native RNG.
        rng = random.Random(seed)
        for sham_index in range(2):
            support=[]
            for _ in range(4):
                ids=sorted(rng.sample(range(blocks), keep_each))
                row=[False]*blocks
                for idx in ids: row[idx]=True
                support.append(row)
            supports.append(support)
        supports = supports[:max_candidates]
        if len(supports) < max_candidates:
            # Bounded operator proposals use deterministic swaps within each unit.
            proposal_no=0
            while len(supports)<max_candidates and proposal_no<proposal_limit:
                unit=proposal_no%4
                support=[list(x) for x in mass_support]
                kept=[i for i,x in enumerate(support[unit]) if x]
                dropped=[i for i,x in enumerate(support[unit]) if not x]
                remove=kept[proposal_no%len(kept)]
                add=dropped[(proposal_no*73+seed)%len(dropped)]
                support[unit][remove]=False;support[unit][add]=True
                supports.append(support);proposal_no+=1
    else:
        raise ValueError(budget['kind'])

    dedup={}
    previous = previous_qkv
    for support in supports:
        digest=_hash_support(support)
        if digest in dedup:
            continue
        unit_records=[]
        response_parts=[]
        feasible=True
        for unit,(q_u,k_u,v_u) in enumerate(units):
            q_u,k_u,v_u=[x.double() for x in (q_u,k_u,v_u)]
            keep=torch.tensor(support[unit],dtype=torch.bool,device=q.device)
            stats=_tile_stats(q_u,k_u,v_u,keep,blocks,float(budget['scale']))
            row=dict(unit=unit,head=UNIT_HEADS[unit//2],query_block=UNIT_QUERY_BLOCKS[unit%2],
                kept_blocks=int(keep.sum().item()),eligible_blocks=blocks,
                current_error_worst=stats['error_worst'],
                whole_deleted_mass_worst=stats['whole_deleted_mass_worst'],
                prefix_deleted_mass_worst=stats['prefix_deleted_mass_worst'],
                whole_renormalization_worst=stats['whole_renormalization_worst'])
            unit_records.append(row)
            feasible &= row['whole_deleted_mass_worst'] <= .2 and row['current_error_worst'] <= .1 and row['whole_renormalization_worst'] <= 1.25
            # Response is populated from the frozen prior displacement when available.
            if previous is not None:
                pq,pk,pv=previous[unit]
                dq=q_u-pq.to(q_u.device,dtype=q_u.dtype)
                dk=k_u-pk.to(k_u.device,dtype=k_u.dtype)
                dv=v_u-pv.to(v_u.device,dtype=v_u.dtype)
                detail=_response_residual(stats['probabilities'],q_u,k_u,v_u,
                    keep,blocks,dq,dk,dv,float(budget['scale']))
                row['response_components']=detail
                response_parts.append(detail['residual_worst'])
        response=max(response_parts) if response_parts else None
        labels=budget.get('labels',{})
        row=dict(candidate_index=len(dedup),support_hash=digest,support=support,
            support_family=labels.get(digest,'operator_proposal'),
            physical_kept=sum(x['kept_blocks'] for x in unit_records),
            physical_eligible=sum(x['eligible_blocks'] for x in unit_records),
            units=unit_records,current_error=max(x['current_error_worst'] for x in unit_records),
            whole_deleted_mass=max(x['whole_deleted_mass_worst'] for x in unit_records),
            whole_renormalization=max(x['whole_renormalization_worst'] for x in unit_records),
            response_score=response,feasible=bool(feasible),
            guard_definition=dict(whole_deleted_mass_max=.2,relative_output_bias_worst=.1,
                whole_renormalization_max=1.25),
            feasibility_note='Proxy guards only; not accuracy certification')
        dedup[digest]=row
        if len(dedup)>=max_candidates:
            break
    return list(dedup.values()), dict(scope=dict(heads=list(UNIT_HEADS),
        query_blocks=list(UNIT_QUERY_BLOCKS),q_block=128,kv_block=64,
        gqa_mapping={str(k):v for k,v in GQA_KV_HEAD.items()}),
        requested_candidates=len(supports),distinct_candidates=len(dedup),
        proposal_limit=proposal_limit,proposal_seed=seed)


def _response_residual(p, q, k, v, keep, blocks, dq, dk, dv, scale):
    support=torch.ones(k.shape[0],dtype=torch.bool,device=k.device)
    support[:blocks*64]=keep.repeat_interleave(64)
    retained=p*support[None]
    ps=retained/retained.sum(-1,keepdim=True).clamp_min(torch.finfo(p.dtype).tiny)
    denom=(p@v).norm(dim=-1).clamp_min(1e-12)
    rows={}
    private_vectors={}
    for name,parts in zip(('q','k','v','joint'),((dq,torch.zeros_like(dk),torch.zeros_like(dv)),
            (torch.zeros_like(dq),dk,torch.zeros_like(dv)),
            (torch.zeros_like(dq),torch.zeros_like(dk),dv),(dq,dk,dv))):
        ja=_differential(p,q,k,v,*parts,scale)
        js=_differential(ps,q,k,v,*parts,scale)
        residual=js-ja
        private_vectors[name]=dict(J_A=ja.detach().cpu(),J_S=js.detach().cpu(),residual=residual.detach().cpu())
        # Signed vectors projected on current dense output are retained per row.
        axis=(p@v)/denom[:,None]
        rows[name]=dict(J_A_worst=float((ja.norm(dim=-1)/denom).max()),
            J_S_worst=float((js.norm(dim=-1)/denom).max()),
            residual_worst=float((residual.norm(dim=-1)/denom).max()),
            signed_J_A_along_dense_output=(ja*axis).sum(-1).tolist(),
            signed_J_S_along_dense_output=(js*axis).sum(-1).tolist(),
            signed_residual_along_dense_output=(residual*axis).sum(-1).tolist())
    return dict(residual_worst=max(x['residual_worst'] for x in rows.values()),components=rows,
        _private_signed_vectors=private_vectors,
        direction_relative_sizes=dict(q=float(dq.norm()/q.norm().clamp_min(1e-12)),
            k=float(dk.norm()/k.norm().clamp_min(1e-12)),v=float(dv.norm()/v.norm().clamp_min(1e-12))),
        meaning='Attention bias differential (J_S-J_A)u; not a denoising contraction score')


def _differential(p,q,k,v,dq,dk,dv,scale):
    output=p@v
    def cov(ds):
        weighted=p*ds
        return weighted@v-weighted.sum(-1,keepdim=True)*output
    return cov(scale*dq@k.T)+cov(scale*q@dk.T)+p@dv


def apply_guards(candidates):
    feasible=[x for x in candidates if x['feasible']]
    return dict(feasible_hashes=[x['support_hash'] for x in feasible],
        feasible_count=len(feasible),identifying=len(feasible)>=4,
        abstain=not feasible,ranking_allowed=len(feasible)>=4)


def matched_pairs(candidates, error_relative_tolerance=.05,
                  deleted_mass_tolerance=.01, response_separation=.25,
                  error_floor=1e-6, response_floor=1e-6):
    rows=[]
    feasible=[x for x in candidates if x['feasible']]
    for i,left in enumerate(feasible):
        for right in feasible[i+1:]:
            e=max(left['current_error'],right['current_error'],error_floor)
            err=abs(left['current_error']-right['current_error'])/e
            mass=abs(left['whole_deleted_mass']-right['whole_deleted_mass'])
            a=max(abs(left['response_score'] or 0),abs(right['response_score'] or 0),response_floor)
            sep=abs((left['response_score'] or 0)-(right['response_score'] or 0))/a
            rows.append(dict(left=left['support_hash'],right=right['support_hash'],
                current_error_relative_difference=err,whole_mass_absolute_difference=mass,
                response_relative_separation=sep,
                matched=err<=error_relative_tolerance and mass<=deleted_mass_tolerance,
                response_identifying_pair=err<=error_relative_tolerance and mass<=deleted_mass_tolerance and sep>=response_separation))
    return rows
