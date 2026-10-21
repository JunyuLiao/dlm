"""Paged observer must preserve the inherited prefix-normalized selector."""
import pytest
import torch
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='H100 required')

@pytest.mark.parametrize('nk,n',[(641,137),(4096,256)])
def test_paged_refresh_uses_current_cache_and_keeps_exact_budget(tmp_path,nk,n):
 from experiments.numerical_qk_reuse.refresh_v31_fast import Observer
 from experiments.numerical_qk_reuse import v27_fa4
 g=torch.Generator(device='cuda').manual_seed(nk);h=4;hk=2;d=512;pages=(nk+127)//128
 cache=torch.randn((pages,128,hk,2*d),device='cuda',dtype=torch.bfloat16,generator=g);key,value=cache[...,:d],cache[...,d:]
 table=torch.randperm(pages,device='cuda',generator=g).int();q=torch.randn((1,h,n,d),device=key.device,dtype=key.dtype,generator=g)
 op=Observer(tmp_path/'fa4');held=torch.rand((1,h,(n+127)//128,(nk+63)//64),device=q.device,generator=g)>.5;prefix=nk-n
 for i in range(3):
  cache.copy_(torch.randn(cache.shape,device=cache.device,dtype=cache.dtype,generator=g));table.copy_(torch.randperm(pages,device=key.device,generator=g).int())
  if i==2:q[...,0]=1;key[...,0]=-32768;key[table[0].long(),:64,:,0]=32768
  k=key[table.long()].reshape(-1,hk,d)[:nk].transpose(0,1)[None].contiguous();v=value[table.long()].reshape(-1,hk,d)[:nk].transpose(0,1)[None].contiguous()
  ref=op.refresh(q,k,v,d**-.5,prefix,128,held=held,sticky=1.386)
  for splits in [0,1]:
   out,keep,lists=op.refresh_paged(q,key,value,table,nk,d**-.5,prefix,128,held=held,sticky=1.386,num_splits=splits)
   assert torch.equal(keep,ref[1])
   assert (keep[...,:prefix//64].sum(-1)==min(2,prefix//64)).all()
   dense=v27_fa4.dense_plain(q,k,v,d**-.5)
   assert torch.isfinite(out).all() and float((out.float()-dense.float()).norm()/dense.float().norm())<.01
