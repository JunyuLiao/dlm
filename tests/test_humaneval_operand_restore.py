import pytest
import torch
from scripts.v31_humaneval_operand_restore import tensor,paged,allocate_group


def capture(x,group=0):return dict(value=x.clone(memory_format=torch.contiguous_format),shape=list(x.shape),stride=list(x.stride()),
    dtype=str(x.dtype),storage_group=group,storage_offset=x.storage_offset(),alignment=x.data_ptr()%256)


def test_stride0_batch_and_pointer_alignment_are_restored():
    q=torch.arange(7*4*8,dtype=torch.bfloat16).reshape(1,7,4,8).expand(2,-1,-1,-1)
    r=tensor(capture(q),'cpu')
    assert r.stride()==q.stride() and r.data_ptr()%256==q.data_ptr()%256 and torch.equal(r,q)
    raw=capture(q);raw['value'][1,0,0,0]+=1
    with pytest.raises(ValueError,match='stride0'):tensor(raw,'cpu')


def test_paged_aliases_full_padding_and_physical_ids_are_restored():
    storage=torch.arange(5*64*2*16,dtype=torch.float32).reshape(5,64,2,16)
    k,v=storage[...,:8],storage[...,8:];ids=torch.tensor([3,1,3,1])
    raw=dict(layout='paged',k_metadata=capture(k),v_metadata=capture(v),page_ids=capture(ids),
        k_pages=capture(k[ids]),v_pages=capture(v[ids]))
    kr,vr=paged(raw,'cpu',unused_fill=-999)
    assert kr.untyped_storage().data_ptr()==vr.untyped_storage().data_ptr()
    assert vr.storage_offset()-kr.storage_offset()==v.storage_offset()-k.storage_offset()
    assert kr.stride()==k.stride() and vr.stride()==v.stride()
    assert kr.data_ptr()%256==k.data_ptr()%256 and vr.data_ptr()%256==v.data_ptr()%256
    assert torch.equal(kr[ids],k[ids]) and torch.equal(vr[ids],v[ids])
    assert bool((kr[0]==-999).all()) and bool((vr[4]==-999).all())
    raw['k_pages']['value'][2,0,0,0]+=1
    with pytest.raises(ValueError,match='duplicated'):paged(raw,'cpu')


def test_shared_storage_alignment_must_be_consistent():
    x=torch.arange(16,dtype=torch.float32)
    a,b=capture(x[:8]),capture(x[8:]);b['alignment']=(b['alignment']+4)%256
    with pytest.raises(ValueError,match='inconsistent'):allocate_group([a,b],'cpu')


def test_invalid_shape_stride_geometry_is_rejected():
    raw=capture(torch.ones(2,3));raw['stride']=[-3,1]
    with pytest.raises(ValueError,match='nonnegative'):tensor(raw,'cpu')
