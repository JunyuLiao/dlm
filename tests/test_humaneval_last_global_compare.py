import torch
from scripts.v31_humaneval_last_global_compare import differences,event_contrast,tensor_difference,raw_difference


def test_first_changed_actual_operand_precedes_final_output():
    common=dict(sampler_index=154,context=dict(step=1,seq_len=3584),canvas=12,arm='residual_direction_guard',query={'gpu128':[1,2]})
    a=dict(common,consumers=[dict(logical_k={'gpu128':[2,3]},output={'gpu128':[4,5]})],output={'gpu128':[6,7]})
    b=dict(common,consumers=[dict(logical_k={'gpu128':[2,4]},output={'gpu128':[4,6]})],output={'gpu128':[6,8]})
    result=event_contrast([a],[b]);assert not result['equal']
    assert result['first_difference']['first']['path']=='consumers/0/logical_k/gpu128/1'
    assert all(result['first_difference']['same_geometry'].values())


def test_missing_coverage_and_geometry_do_not_count_as_equality():
    a=dict(sampler_index=1,context={'step':1},canvas=0,arm='full_swap8')
    b=dict(a,sampler_index=2)
    assert not event_contrast([a],[])['equal']
    assert not event_contrast([a],[b])['first_difference']['same_geometry']['sampler_index']
    assert differences({'x':[1,2]},{'x':[1]})[0]['kind']=='coverage'


def test_raw_comparison_is_bitwise_and_ignores_only_declared_alloc_metadata():
    a=torch.tensor([0.,float('nan'),1.],dtype=torch.bfloat16);b=a.clone()
    assert tensor_difference(a,b)['equal']
    b[0]=-0.;r=tensor_difference(a,b);assert not r['equal'] and r['different_bytes']==1
    assert raw_difference({'alignment':0,'value':a},{'alignment':128,'value':a.clone()})==[]
    assert raw_difference({'stride':[1],'value':a},{'stride':[2],'value':a.clone()})[0]['path']=='/stride/0'
    b=a.clone();b[2]=2.;r=tensor_difference(a,b);assert r['max_absolute']==1. and r['nonfinite_differences']==1


def test_logical_inputs_exclude_padding_and_inactive_slots_but_keep_active_order():
    from copy import deepcopy
    from scripts.v31_humaneval_last_global_compare import logical_fa4_inputs
    value=lambda x:dict(value=x)
    a=dict(q=value(torch.ones(1,2,1,2)),layout='contiguous',k=value(torch.ones(1,4,1,2)),v=value(torch.ones(1,4,1,2)),
        scalar_kwargs={'causal':False},tensor_kwargs={},aux=[],block_sparse_tensors=dict(block_size=[128,64],
        full_block_cnt=value(torch.ones(1,1,1,dtype=torch.int32)),full_block_idx=value(torch.tensor([[[[2,3,4]]]])),
        mask_block_cnt=value(torch.zeros(1,1,1,dtype=torch.int32)),mask_block_idx=value(torch.tensor([[[[0]]]]))))
    b=deepcopy(a);b['block_sparse_tensors']['full_block_idx']['value'][0,0,0,1]=99
    assert raw_difference(logical_fa4_inputs(a),logical_fa4_inputs(b))==[]
    b['block_sparse_tensors']['full_block_idx']['value'][0,0,0,0]=1
    assert raw_difference(logical_fa4_inputs(a),logical_fa4_inputs(b))[0]['path'].startswith('/active_lists/full')


def test_physical_translation_change_does_not_hide_later_operator_change():
    common=dict(context=dict(step=0),canvas=0,arm='full_swap8')
    a0=dict(common,sampler_index=0,consumers=[dict(page_table={'gpu128':[1,2]},logical_k={'gpu128':[3,4]})])
    b0=dict(common,sampler_index=0,consumers=[dict(page_table={'gpu128':[5,6]},logical_k={'gpu128':[3,4]})])
    a1=dict(common,sampler_index=1,output={'gpu128':[7,8]});b1=dict(common,sampler_index=1,output={'gpu128':[7,9]})
    r=event_contrast([a0,a1],[b0,b1]);assert r['first_difference']['left_sampler_index']==1
    assert r['physical_translation_or_padding_changes']['events']==1
    assert event_contrast([a0],[b0])['equal']
