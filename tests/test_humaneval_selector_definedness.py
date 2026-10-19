"""Native-dimension scratch poisoning, bounds and growth/reuse checks."""
import math
import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='H100 operator diagnostic')


def test_native_statistics_and_selectors_ignore_poisoned_scratch_and_keep_canaries():
    from experiments.numerical_qk_reuse.v31_value_kernels import statistics_cuda
    from experiments.numerical_qk_reuse.v31_value_workspace import ValueWorkspace
    from experiments.numerical_qk_reuse.v31_value_selectors import select
    generator=torch.Generator(device='cuda').manual_seed(42116)
    q=torch.randn((1,16,256,512),device='cuda',dtype=torch.bfloat16,generator=generator)
    inputs={}
    for prefix in (64*40+17,64*138+63,64*64+3):
        nk=prefix+256
        inputs[prefix]=(torch.randn((1,2,nk,512),device='cuda',dtype=torch.bfloat16,generator=generator),
            torch.randn((1,2,nk,32),device='cuda',generator=generator)/math.sqrt(32))
    nu=torch.ones(2,device='cuda');workspace=ValueWorkspace(9216)
    methods=('value_v2_online_preserve_mass','value_v3a_singleton_delete')
    # Discover all scratch allocations using the largest geometry, then put
    # independent red zones around each retained allocation before poisoning.
    largest=max(inputs);k,z=inputs[largest]
    warm=statistics_cuda(q,k,z,nu,1/math.sqrt(512),largest//64,workspace=workspace)
    for method in methods:select(warm,method,budget=4,threshold=.094485192673943)
    guarded=[]
    for key,value in list(workspace.storage.items()):
        guard=True if value.dtype==torch.bool else 12345.25 if value.is_floating_point() else 12345
        raw=torch.full((value.numel()+512,),guard,
                       dtype=value.dtype,device=value.device)
        workspace.storage[key]=raw[256:-256]
        guarded.append((raw,guard))
    previous=[]
    for prefix,(k,z) in inputs.items():
        reference=statistics_cuda(q,k,z,nu,1/math.sqrt(512),prefix//64)
        expected={method:select(reference,method,budget=4,threshold=.094485192673943)[0].clone()
                  for method in methods}
        for poison in (float('nan'),0.,-123.5):
            for value in workspace.storage.values():
                fill=poison!=0 if value.dtype==torch.bool else poison if value.is_floating_point() else 0 if poison==0 else -123
                value.fill_(fill)
            actual=statistics_cuda(q,k,z,nu,1/math.sqrt(512),prefix//64,workspace=workspace)
            for field in ('log_mass','mean','nu','invalid'):
                assert torch.equal(getattr(actual,field),getattr(reference,field)),(prefix,poison,field)
            for method in methods:
                result,_=select(actual,method,budget=4,threshold=.094485192673943)
                assert torch.equal(result,expected[method]),(prefix,poison,method)
                previous.append((result,result.clone()))
            for raw,guard in guarded:
                assert bool((raw[:256]==guard).all()) and bool((raw[-256:]==guard).all())
            assert all(torch.equal(original,saved) for original,saved in previous)
