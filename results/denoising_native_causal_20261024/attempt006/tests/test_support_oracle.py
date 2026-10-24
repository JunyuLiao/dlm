import importlib.util
from pathlib import Path
import torch

p=Path(__file__).resolve().parents[1]/'code/support_oracle.py'
spec=importlib.util.spec_from_file_location('support_oracle_test',p)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_differential_has_moving_qkv_and_matches_finite_difference():
    g=torch.Generator().manual_seed(23)
    q,k,v=[torch.randn(shape,generator=g,dtype=torch.float64) for shape in ((3,4),(7,4),(7,5))]
    dq,dk,dv=[torch.randn(x.shape,generator=g,dtype=x.dtype) for x in (q,k,v)]
    def f(t):return ((q+t*dq)@(k+t*dk).T*.5).softmax(-1)@(v+t*dv)
    p=(q@k.T*.5).softmax(-1)
    torch.testing.assert_close(m.differential(p,q,k,v,dq,dk,dv,.5),(f(1e-5)-f(-1e-5))/2e-5,rtol=1e-6,atol=1e-8)

def test_bank_exact_budget_unique_support_and_no_native_rng_draws():
    g=torch.Generator().manual_seed(42)
    q=torch.randn((4,8),generator=g,dtype=torch.float64)
    k=torch.randn((310,8),generator=g,dtype=torch.float64)
    v=torch.randn((310,6),generator=g,dtype=torch.float64)
    state=torch.get_rng_state()
    rows=m.bank(q,k,v,(q*.99,k*.99,v*.99),256,.5,.3)
    assert len({r['support_sha256'] for r in rows})==len(rows)
    assert all(int(r['mask'].sum())==2 for r in rows)
    assert all(r['response'] is not None for r in rows)
    assert torch.equal(state,torch.get_rng_state())


def test_native_output_shapes_preserve_head_dimension():
    result=torch.arange(2*3*4).reshape(1,2,3,4)
    for shape in ((2,3,4),(2,12)):
        output=torch.zeros(shape,dtype=result.dtype)
        m.copy_attention_output(output,result,2)
        torch.testing.assert_close(output.reshape(2,3,4),result[0])
