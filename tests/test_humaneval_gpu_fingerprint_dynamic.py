import numpy as np
import pytest
import torch
from scripts.v31_humaneval_gpu_fingerprint import fingerprint_cuda as fixed
from scripts.v31_humaneval_gpu_fingerprint_dynamic import fingerprint_cuda
from scripts.v31_humaneval_graph_fingerprint import cpu_fingerprints


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA diagnostic')
@pytest.mark.parametrize('count',[1,4,1025,1048708])
def test_runtime_lengths_match_numpy_fixed_kernel_repeats_and_tail_bit(count):
    generator=torch.Generator().manual_seed(4219)
    source=torch.randint(-(2**31),2**31-1,(count,),generator=generator,dtype=torch.int32)
    values=source.numpy().view(np.uint32).astype(np.uint64)
    positions=np.arange(count,dtype=np.uint64)*np.uint64(0x9e3779b97f4a7c15)
    reference=[]
    with np.errstate(over='ignore'):
        for seed in (0x243f6a8885a308d3,0x13198a2e03707344):
            mixed=values^positions^np.uint64(seed)
            mixed=(mixed^(mixed>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
            mixed=(mixed^(mixed>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
            mixed=mixed^(mixed>>np.uint64(31))
            reference.append(mixed.sum(dtype=np.uint64))
    device=source.cuda();saved=device.clone();digest=fingerprint_cuda(device)
    np.testing.assert_array_equal(digest.cpu().numpy(),np.array(reference,dtype=np.uint64))
    assert torch.equal(digest,fixed(device)) and torch.equal(device,saved)
    assert torch.equal(digest,fingerprint_cuda(device))
    device[-1]^=1
    assert not torch.equal(digest,fingerprint_cuda(device))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA diagnostic')
def test_gpu_checksum_drain_preserves_unsigned_high_bits():
    digest=fingerprint_cuda(torch.tensor([1,-1,3,42],device='cuda',dtype=torch.int32))
    high=torch.tensor([2**64-1,2**63],dtype=torch.uint64,device='cuda')
    actual=cpu_fingerprints(dict(digest=digest,nested=[dict(high=high)]))
    assert actual['digest']==digest.cpu().tolist()
    assert actual['nested']==[dict(high=[2**64-1,2**63])]
