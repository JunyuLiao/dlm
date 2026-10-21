"""Independent current-state synthetic qualification for real online WGMMA V2."""
import argparse
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tma',action='store_true')
    p.add_argument('--schedules',type=int,nargs='+',default=[0,1,2])
    a=p.parse_args()
    if a.out.exists():raise RuntimeError('new qualification attempt required')
    a.out.mkdir(parents=True)
    from experiments.numerical_qk_reuse.v2_online_sm90 import Kernel
    from experiments.numerical_qk_reuse.v2_sparse_refresh import discover
    from experiments.numerical_qk_reuse import v27_fa4
    options=json.loads((a.library.parent/'build_receipt.json').read_text())['options']
    if options.get('wavefront'):
        from experiments.numerical_qk_reuse.v2_online_pipeline import WavefrontKernel
        kernel=WavefrontKernel(a.library)
    elif options.get('packed'):
        from experiments.numerical_qk_reuse.v2_online_warpspec import PackedKernel
        kernel=PackedKernel(a.library)
    else:kernel=Kernel(a.library)
    rows=[]
    for n,nk,tau,sticky in [(256,449,0.,0.),(137,641,.3,1.386),(256,1025,.08,1.386)]:
        g=torch.Generator(device='cuda').manual_seed(nk)
        q=torch.randn((1,4,n,512),device='cuda',dtype=torch.bfloat16,generator=g)
        k=torch.randn((1,2,nk,512),device='cuda',dtype=torch.bfloat16,generator=g)
        v=torch.randn(k.shape,device='cuda',dtype=torch.bfloat16,generator=g)
        z=torch.randn((1,2,nk,32),device='cuda',generator=g)
        nu=torch.full((1,2),2.,device='cuda')
        pt=max(0,(nk-n)//64)
        held=torch.rand((1,4,(n+127)//128,(nk+63)//64),device='cuda',generator=g)>.5
        ref=discover(q,k,z,nu,512**-.5,pt,tau,held=held,sticky=sticky,ratio=True,trace=True)
        for schedule in a.schedules:
            if a.tma and schedule==0:continue
            print(dict(event='launch',queries=n,keys=nk,schedule=schedule),flush=True)
            out,keep,state,lm,risks=kernel(q,k,v,z,nu,512**-.5,tau,prefix_tiles=pt,held=held,sticky=sticky,schedule=schedule,trace=not kernel.production,tma=a.tma)
            torch.cuda.synchronize()
            mask_errors=int((keep.reshape_as(ref[0])!=ref[0]).sum())
            lists=v27_fa4.block_sparse_tensors(keep)
            expected=v27_fa4.sparse_lists(q,k,v,lists,512**-.5)
            error=out.float()-expected.float()
            rel=float(error.norm()/expected.float().norm())
            got_state=state.reshape(4,n,32) if not kernel.production else None
            ref_state=ref[1].reshape(4,-1,32)[:,:n]
            got_lm=lm.reshape(4,n) if not kernel.production else None;ref_lm=ref[2].reshape(4,-1)[:,:n]
            state_err=float((got_state-ref_state).abs().max()) if not kernel.production else None
            lm_err=float((got_lm-ref_lm).abs().max()) if not kernel.production else None
            row=dict(queries=n,keys=nk,threshold=tau,sticky=sticky,schedule=schedule,
                mask_errors=mask_errors,output_relative_l2=rel,projected_state_max_abs=state_err,logmass_max_abs=lm_err)
            rows.append(row);print(json.dumps(row),flush=True)
            (a.out/'receipt.json').write_text(json.dumps(dict(status='running',rows=rows),indent=2)+'\n')
            if mask_errors or rel>.01 or (state_err is not None and state_err>1.e-4) or (lm_err is not None and lm_err>2.e-5):raise RuntimeError('online kernel qualification failed')
    (a.out/'receipt.json').write_text(json.dumps(dict(status='passed',rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
