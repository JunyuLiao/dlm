import ast
from pathlib import Path


def checker():
    source=Path(__file__).with_name('controller.py')
    if not source.exists():source=Path(__file__).parents[1]/'scripts/v31_humaneval_residual_repro.py'
    tree=ast.parse(source.read_text())
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='observation_policy_errors')
    namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen actual receipt validator>','exec'),namespace)
    return namespace[node.name]


def receipt():return dict(observing_fa4_numerical_policy='no_intra_wg_overlap',observing_fa4_intra_wg_overlap=False,
    observing_fa4_policy_sha256='source',observing_fa4_extra_forwards=0,observing_fa4_plain_arguments_preserved=True,observing_fa4_calls=7)


def test_actual_nested_public_receipt_is_accepted_and_top_level_spoof_rejected():
    validate=checker();a=receipt()
    assert validate(dict(receipts=dict(adapter=a)),'source')==[]
    assert validate(dict(adapter=a),'source')
    assert validate(dict(receipts=dict(adapter=a)),'different-source')


def test_missing_and_invalid_call_counters_are_not_treated_as_measured_zero():
    validate=checker()
    for value in (None,-1,True,'0'):
        a=receipt();a['observing_fa4_calls']=value
        assert validate(dict(receipts=dict(adapter=a)),'source')
    a=receipt();del a['observing_fa4_calls'];assert validate(dict(receipts=dict(adapter=a)),'source')
    a['observing_fa4_calls']=0;assert validate(dict(receipts=dict(adapter=a)),'source')==[]
