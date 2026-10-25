import importlib.util,sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parents[1]/'code/boundary.py';s=importlib.util.spec_from_file_location('boundary_test',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_signed_zero_copies_only():
 x=torch.tensor([0.,-0.]);y=torch.tensor([0.,0.]);before=x.clone().view(torch.uint8).clone()
 a=m.tensor_semantics(x);b=m.tensor_semantics(y)
 assert a['byte_sha256']!=b['byte_sha256'] and a['numerical_sha256']==b['numerical_sha256']
 assert a['negative_zero']==1 and torch.equal(before,x.view(torch.uint8))
def test_nonfinite_explicit():
 x=m.tensor_semantics(torch.tensor([float('nan'),float('inf'),-float('inf'),1.]))
 assert (x['nan'],x['positive_inf'],x['negative_inf'])==(1,1,1) and x['finite_mean']==1.
def test_inactive_capacity_excluded():
 a=torch.tensor([2,3,888]);b=torch.tensor([2,3,999]);assert m.tensor_semantics(a[:2])==m.tensor_semantics(b[:2])
def test_dtype_and_shape_retained():
 a=m.tensor_semantics(torch.zeros(2,3));assert a['shape']==[2,3] and a['dtype']=='torch.float32'
