"""Run with QUERY_ADAPTIVE_KERNEL and QUERY_ADAPTIVE_BRIDGE on an H100."""
import os

import pytest
import torch

from experiments.value_direction_hopper.cuda import Kernel


@pytest.mark.skipif(not torch.cuda.is_available() or not os.getenv('QUERY_ADAPTIVE_KERNEL'),reason='H100 build not selected')
def test_weighted_kernel_parity_and_protection_on_example_state():
    kernel=Kernel(os.environ['QUERY_ADAPTIVE_KERNEL'],torch_library=os.getenv('QUERY_ADAPTIVE_BRIDGE'))
    torch.manual_seed(17)
    q=torch.randn((1,4,129,256),device='cuda',dtype=torch.bfloat16).contiguous()
    k=torch.randn((1,2,256,256),device='cuda',dtype=torch.bfloat16).contiguous()
    v=torch.randn_like(k)
    matrix=torch.randn((2,256,32),device='cuda')/(32**.5)
    z=torch.einsum('bhkd,hdr->bhkr',v.float(),matrix).contiguous()
    reference=v.float().square().mean((-2,-1)).sqrt().contiguous()
    mask=torch.ones((1,1,129,256),device='cuda',dtype=torch.bool)
    mask[:,:,128:,192:]=False  # structural boundary in the partial Q tile
    options=dict(mask=mask,log_threshold=2.,precision='tf32x3_register')
    dense=kernel(q,k,v,z,reference,**options)
    ones=kernel(q,k,v,z,reference,sensitivity=torch.ones((1,129),device='cuda'),**options)
    protected=kernel(q,k,v,z,reference,sensitivity=torch.full((1,129),4.,device='cuda'),**options)
    torch.cuda.synchronize()
    assert torch.equal(dense.output,ones.output)
    assert torch.equal(dense.skipped,ones.skipped)
    assert torch.equal(dense.eligible,ones.eligible)
    assert bool(torch.isfinite(protected.output).all())
    # Greedy retained-state evolution means no general monotonicity claim is
    # made for later tiles. This particular state must show a routing effect.
    assert not torch.equal(protected.skipped,dense.skipped)
    assert int(protected.skipped.sum())<int(dense.skipped.sum())
    assert int(dense.eligible.sum())>0
    assert int((dense.eligible&~dense.skipped).sum())>0
