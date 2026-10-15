"""Full-V online preserve-mass risks in one ordered GPU traversal.

The block acceptance vote spans every row of a query tile. Keeping the tile
in one program preserves that vote and physical key order without one Python
dispatch per KV block. This is a proposal scorer, never a consumer mask.
"""
import triton
import triton.language as tl


@triton.jit
def online_risk_kernel(M, C, VRMS, OUT, BLOCKS: tl.constexpr, ROWS: tl.constexpr,
                       DIM: tl.constexpr, THRESHOLD: tl.constexpr,
                       R: tl.constexpr, D: tl.constexpr):
    row=tl.arange(0,R)
    dim=tl.arange(0,D)
    running=tl.full((R,D),0,tl.float32)
    denominator=tl.full((R,),0,tl.float32)
    rms=tl.maximum(tl.load(VRMS),1.e-30)
    for block in range(BLOCKS):
        mass=tl.load(M+block*ROWS+row,row<ROWS,other=0)
        numerator=tl.load(C+(block*ROWS+row[:,None])*DIM+dim[None,:],
                          (row[:,None]<ROWS)&(dim[None,:]<DIM),other=0)
        denominator=denominator+mass
        eta=mass/tl.maximum(denominator,1.e-30)
        mean=numerator/tl.maximum(mass[:,None],1.e-30)
        delta=mean-running
        row_risk=eta*tl.sqrt(tl.sum(delta*delta,axis=1))/rms
        risk=tl.max(row_risk,axis=0)
        take=(risk>=THRESHOLD)|(block==0)
        running=tl.where(take,running+eta[:,None]*delta,running)
        tl.store(OUT+block,risk)


def online_risks_cuda(mass,numerator,value_rms,threshold):
    import torch
    if mass.dtype!=torch.float32 or numerator.dtype!=torch.float32:
        raise ValueError('online kernel requires FP32 normalized moments')
    if not mass.is_cuda or not numerator.is_cuda or not value_rms.is_cuda:
        raise ValueError('online kernel requires CUDA moments and normalization')
    if numerator.shape[:2]!=mass.shape or mass.shape[1]>128 or numerator.shape[2]>512:
        raise ValueError('online kernel supports at most one Q128/full-V512 tile')
    blocks,rows=mass.shape;dim=numerator.shape[2]
    output=torch.empty(blocks,device=mass.device,dtype=torch.float32)
    if not blocks:return output
    r,d=triton.next_power_of_2(rows),triton.next_power_of_2(dim)
    # Large full-V tiles need enough threads to keep the recurrent accumulator
    # distributed across registers. Small oracle geometries use fewer warps.
    warps=32 if r*d>=32768 else (8 if r*d>=4096 else 4)
    online_risk_kernel[(1,)](mass.contiguous(),numerator.contiguous(),value_rms,output,
        BLOCKS=blocks,ROWS=rows,DIM=dim,THRESHOLD=threshold,R=r,D=d,
        num_warps=warps,enable_fp_fusion=False)
    return output
