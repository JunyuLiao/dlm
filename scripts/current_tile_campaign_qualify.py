"""CUDA oracle checks for TraDo4, LLaDA32 and Gemma256 current initialization."""
import argparse
import json
from pathlib import Path
import torch
from experiments.numerical_qk_reuse.v31_value_kernels import statistics_cuda
from experiments.numerical_qk_reuse.v31_value_selectors import Stats
from experiments.numerical_qk_reuse.v2_current_tile_init import current_tile_cuda,current_tile_reference
from experiments.numerical_qk_reuse.v2_current_canvas_init import current_canvas_cuda,current_canvas_reference


def main(out):
    result=[]
    gen=torch.Generator(device='cuda').manual_seed(20261103)
    for n,d in ((4,128),(32,128),(256,512)):
        for prefix in (64,160,512,8256):
            nk=prefix+n;jt=(nk+63)//64;start=prefix//64 if n==256 else jt-1
            q=torch.randn((1,2,n,d),dtype=torch.bfloat16,device='cuda',generator=gen)
            k=torch.randn((1,1,nk,d),dtype=torch.bfloat16,device='cuda',generator=gen)
            z=torch.randn((1,1,nk,32),device='cuda',generator=gen)
            for qi in range((n+127)//128):
                s=statistics_cuda(q[:,:,qi*128:(qi+1)*128],k,z,torch.ones(1,device='cuda'),d**-.5,start)
                cpu=Stats(*(x.cpu() for x in (s.log_mass,s.mean,s.nu,s.invalid)),s.prefix_tiles,s.heads,s.blocks)
                m=(torch.arange(jt,device='cuda')[None]>=start).expand(2,-1).contiguous()
                for tau in (0.,.03,.2,.9):
                    for sticky in ((0.,1.386) if n==256 else (0.,)):
                        held=torch.zeros_like(m);held[:,::2]=True
                        if n==256:
                            a=current_canvas_cuda(s,tau,m,start,held,sticky)
                            b=current_canvas_reference(cpu,tau,m.cpu(),start,held.cpu(),sticky)
                        else:
                            a=current_tile_cuda(s,tau,True,m,held,0.)
                            b=current_tile_reference(cpu,tau,m.cpu())
                        assert torch.equal(a[0].cpu(),b[0]),(n,prefix,qi,tau,sticky)
                        assert torch.allclose(a[1].cpu(),b[1],atol=3e-5,rtol=3e-5)
                        assert torch.allclose(a[2].cpu(),b[2],atol=3e-5,rtol=3e-5)
                        result.append({'query_tokens':n,'head_dim':d,'prefix':prefix,'query_group':qi,
                                       'threshold':tau,'sticky':sticky,'mask_mismatches':0})
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump({'status':'pass','cases':result,'count':len(result)},f,indent=2)
    print('qualified',len(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();main(a.output)
