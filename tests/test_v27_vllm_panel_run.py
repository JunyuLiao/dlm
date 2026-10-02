import pytest
from scripts.v27_vllm_panel_run import validate_receipts


def receipt():
    return {'adapter':dict(order_errors=0,begins=8,observes=8,global_calls=40,split_fa4_calls=30),
            'method':dict(effective_method={'score_period':64},active_layers=[5,11,17,23,29],
                gated_native_calls=0,layer_native_calls=0,unsupported_mask_refreshes=0,
                fused_observations=5,dp_routes=5)}


def test_dense_has_no_adapter():
    validate_receipts('dense',None,8,{})
    with pytest.raises(ValueError):
        validate_receipts('dense',receipt(),8,{})


def test_real_clock_matches():
    validate_receipts('method',receipt(),8,{'score_period':64})
    with pytest.raises(ValueError,match='clock'):
        validate_receipts('method',receipt(),9,{'score_period':64})


def test_wrong_method_rejected():
    with pytest.raises(ValueError,match='effective'):
        validate_receipts('method',receipt(),8,{'score_period':8})


@pytest.mark.parametrize('field',['gated_native_calls','layer_native_calls','unsupported_mask_refreshes'])
def test_fallback_rejected(field):
    r=receipt();r['method'][field]=1
    with pytest.raises(ValueError,match='fallback'):
        validate_receipts('method',r,8,{})


def test_sparse_path_required():
    r=receipt();r['method']['dp_routes']=0
    with pytest.raises(ValueError,match='sparse path'):
        validate_receipts('method',r,8,{})
