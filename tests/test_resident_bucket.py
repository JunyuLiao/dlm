"""Bucket replay must read current paged values and current physical support."""
import pytest
import torch
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='H100 required')

@pytest.mark.parametrize('splits',[1,2])
def test_padded_bucket_reads_current_values_lengths_and_masks(splits):
 from experiments.numerical_qk_reuse.resident_sparse import GraphPagedBucket,ResidentMask
 from experiments.numerical_qk_reuse import v27_fa4
 g=torch.Generator(device='cuda').manual_seed(712);h=4;hk=2;n=137;d=512;capacity=1024
 cache=torch.empty((capacity//128,128,hk,2*d),device='cuda',dtype=torch.bfloat16);key,value=cache[...,:d],cache[...,d:]
 table=torch.arange(capacity//128,device='cuda',dtype=torch.int32)
 op=GraphPagedBucket(key,value,table,2048,n,h,d**-.5,splits=splits)
 q=torch.empty((1,h,n,d),device='cuda',dtype=cache.dtype);dest=torch.empty((1,n,h,d),device=q.device,dtype=q.dtype)
 pointers=(op.q.data_ptr(),op.output.data_ptr(),op.context.page_table.data_ptr(),op.mask.lists.full_block_idx.data_ptr())
 for i in range(8):
  nk=641 if i%2 else 1024;kt=(nk+63)//64
  cache.copy_(torch.randn(cache.shape,device=cache.device,dtype=cache.dtype,generator=g));q.copy_(torch.randn(q.shape,device=q.device,dtype=q.dtype,generator=g))
  table.copy_(torch.randperm(capacity//128,device=cache.device,generator=g).int())
  keep=torch.rand((1,h,2,kt),device=q.device,generator=g)>.5;keep[...,0]=True
  mask=ResidentMask.create(keep,splits=splits);op.prepare(mask,table,nk)
  out=op(q,out=dest)
  k=key[table.long()].reshape(-1,hk,d)[:nk].transpose(0,1)[None].contiguous();v=value[table.long()].reshape(-1,hk,d)[:nk].transpose(0,1)[None].contiguous()
  expected=v27_fa4.sparse_lists(q,k,v,v27_fa4.block_sparse_tensors(keep),d**-.5)
  assert torch.isfinite(out).all() and float((out.float()-expected.float()).norm()/expected.float().norm())<.004
  assert out.data_ptr()==dest.data_ptr()
  assert pointers==(op.q.data_ptr(),op.output.data_ptr(),op.context.page_table.data_ptr(),op.mask.lists.full_block_idx.data_ptr())
