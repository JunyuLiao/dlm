from types import SimpleNamespace
import pytest
from scripts.v31_value_interleaved_worker import bind_requests,CONTROL


def test_control_restores_original_path_and_every_request_resets_state():
    runtime=SimpleNamespace(mode='full_swap8',cache=None)
    seen=[]
    def original_mage():seen.append('original');return 'original'
    def refined_mage():seen.append(runtime.mode);runtime.cache='populated';return runtime.mode
    def begin():runtime.cache=None
    adapter=SimpleNamespace(_mage=refined_mage,_value_receipt=lambda:dict(value_refinement=runtime.mode),begin_request=begin)
    schedule=[CONTROL,'residual',CONTROL,'full_swap8_direction_guard',CONTROL]
    state=bind_requests(adapter,runtime,original_mage,lambda:dict(budget=27),schedule)
    for i,arm in enumerate(schedule):
        adapter.begin_request();assert runtime.cache is None
        assert adapter._mage()==('original' if arm==CONTROL else arm)
        receipt=adapter._value_receipt()
        assert receipt['comparison_arm']==arm and receipt['comparison_request_index']==i
        assert receipt.get('value_refinement')==(None if arm==CONTROL else arm)
    assert state['index']==4
    with pytest.raises(RuntimeError):adapter.begin_request()
    assert len(seen)==len(schedule)


@pytest.mark.parametrize('schedule',[[],['residual'],[CONTROL,'full_cgate_swap8'],[CONTROL,'unknown']])
def test_unsupported_or_unwarmed_schedules_fail_before_binding(schedule):
    with pytest.raises(ValueError):bind_requests(None,None,None,None,schedule)
