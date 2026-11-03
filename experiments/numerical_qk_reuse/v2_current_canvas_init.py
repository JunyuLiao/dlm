"""Current mandatory suffix first, followed by chronological optional tiles.

Generalization of current-tile initialization to a multi-tile native canvas.
The original V2 selector and its historical results are unchanged.
"""
import math
import torch


def current_canvas_cuda(stats, threshold, mandatory, start, held=None, sticky=0.):
    from .v31_value_kernels import _online_chunk
    if math.isnan(threshold) or threshold < 0:
        raise ValueError('invalid threshold')
    u,jt,q=stats.log_mass.shape
    if not 0<=start<jt or q!=128:
        raise ValueError('invalid mandatory suffix geometry')
    expected=torch.arange(jt,device=mandatory.device)[None]>=start
    torch._assert_async((mandatory==expected).all(),'only the current mandatory suffix may be protected')
    keep=torch.empty_like(mandatory)
    state=torch.zeros((u,128,32),device=mandatory.device)
    mass=torch.full((u,128),-math.inf,device=mandatory.device)
    support=torch.zeros((u,128),device=mandatory.device,dtype=torch.bool)
    risk=torch.empty((u,jt),device=mandatory.device)
    held=torch.zeros_like(mandatory) if held is None else held
    weight=torch.ones_like(stats.nu)
    # Both ranges use the original optimized preserve-mass recurrence. Every
    # physical tile is visited once; current tiles cannot be counted twice.
    for lo,hi in ((start,jt),(0,start)):
        for j0 in range(lo,hi,128):
            _online_chunk[(u,)](stats.log_mass,stats.mean,stats.nu,stats.invalid,
                mandatory,held,weight,keep,mass,state,support,risk,jt,j0,
                min(128,hi-j0),math.log(threshold) if threshold else -math.inf,
                True,sticky,num_warps=8)
    keep|=stats.invalid.any((1,2))[:,None]
    return keep,state,mass,risk


def current_canvas_reference(stats,threshold,mandatory,start,held=None,sticky=0.):
    """Independent CPU oracle via a permutation of the trusted V2 recurrence."""
    from .v31_value_selectors import Stats,online_reference
    order=[*range(start,stats.log_mass.shape[1]),*range(start)]
    inverse=torch.argsort(torch.tensor(order,device=stats.log_mass.device))
    shuffled=Stats(stats.log_mass[:,order],stats.mean[:,order],stats.nu,
                   stats.invalid[:,order],stats.prefix_tiles,stats.heads,stats.blocks)
    held=None if held is None else held[:,order]
    keep,state,mass,risk=online_reference(shuffled,threshold,True,mandatory[:,order],held,sticky)
    return keep[:,inverse],state,mass,risk[:,inverse]
