from types import SimpleNamespace

import pytest

from scripts.v31_refresh_headroom_profile import (
    HeadroomCollector, PREFIX, attention_scope, global_phase, summarize_chrome, union_duration,
)


def event(name, cat, start, duration, *, pid=1, tid=1, **args):
    return dict(name=name, cat=cat, ph='X', ts=start, dur=duration, pid=pid, tid=tid, args=args)


def test_overlapping_gpu_intervals_are_not_summed_as_wall_time():
    trace = [event('opaque', 'kernel', 0, 10, pid=0), event('opaque', 'kernel', 5, 10, pid=0),
             event('opaque', 'kernel', 20, 5, pid=0)]
    result = summarize_chrome(trace, {})
    assert result['devices']['0']['activity_union_ms'] == .020
    assert result['devices']['0']['no_activity_within_envelope_ms'] == .005
    assert result['exclusive_activity_categories']['unattributed_rest']['activity_sum_ms'] == .025


def test_each_activity_uses_deepest_scope_and_correlation_fallback_once():
    trace = [event(PREFIX+'0', 'user_annotation', 0, 100),
             event(PREFIX+'1', 'user_annotation', 10, 20),
             event('aten::x', 'cpu_op', 12, 1, **{'External id': 4}),
             event('cudaLaunchKernel', 'cuda_runtime', 13, 1, correlation=9, **{'External id': 4}),
             event('a', 'kernel', 50, 10, pid=0, **{'External id': 4}),
             event('b', 'kernel', 55, 10, pid=0, correlation=9),
             event('aten::y', 'cpu_op', 40, 1, **{'External id': 5}),
             event('c', 'kernel', 70, 10, pid=0, **{'External id': 5})]
    result = summarize_chrome(trace, {'0':'denoise/global_refresh', '1':'denoise/discovery'})
    assert result['scope_linked_activities'] == 3
    assert result['exclusive_activity_categories']['denoise/discovery']['activities'] == 2
    assert result['exclusive_activity_categories']['denoise/global_refresh']['activities'] == 1
    assert sum(r['activities'] for r in result['exclusive_activity_categories'].values()) == 3


def test_scope_threads_do_not_leak_and_full_graph_unknown_is_retained():
    trace = [event(PREFIX+'0', 'user_annotation', 0, 100, tid=1),
             event('aten::x', 'cpu_op', 10, 1, tid=2, **{'External id': 2}),
             event('cutlass_gemm', 'kernel', 20, 2, pid=0, **{'External id': 2})]
    result = summarize_chrome(trace, {'0':'denoise/global_refresh'})
    assert result['scope_linked_activities'] == 0
    assert result['exclusive_activity_categories']['kernel_name_other_gemm']['activities'] == 1


def test_actual_counter_phase_and_layer_scopes():
    assert global_phase({}, {'mage_reselections':1}) == 'refresh'
    assert global_phase({}, {'mage_reselections':1,'mage_reused_calls':1}) == 'mixed'
    assert attention_scope('model.layers.29.self_attn.attn','denoise',{}, {'mage_carried_calls':1}) == 'denoise/global_carried'
    assert attention_scope('model.layers.4.self_attn.attn','commit') == 'commit/local_attention'
    assert attention_scope('model.layers.5.self_attn.attn','prefill') == 'prefill/global_dense'


def test_phase_and_request_boundary_contract(tmp_path):
    collector = HeadroomCollector(tmp_path, cuda_events=False, warmup_requests=1)
    collector.begin_request()
    assert collector.end_request(boundary_completed=True)['warmup']
    collector.begin_request()
    collector.record_phase(True)
    collector.record_phase(False)
    collector.record_phase(True)
    assert dict(collector.phase_calls) == {'prefill':1,'denoise':1,'commit':1}
    with pytest.raises(RuntimeError, match='boundary sync'):
        collector.end_request(boundary_completed=False)


def test_nested_cpu_exclusion_and_no_event_sync(tmp_path):
    # No CUDA methods exist in this fake: CPU analysis cannot accidentally sync.
    collector = HeadroomCollector(tmp_path, cuda_events=False, warmup_requests=0,
                                  torch_module=SimpleNamespace())
    collector.begin_request()
    with collector.scope('parent'):
        with collector.scope('child'):
            pass
    result = collector.end_request(boundary_completed=True)
    parent, child = result['spans']['parent'], result['spans']['child']
    assert parent['cpu_exclusive_own_scopes_ms'] == pytest.approx(
        parent['cpu_inclusive_ms']-child['cpu_inclusive_ms'])
    assert parent['cuda_inclusive_span_ms'] == 0


def test_invalid_intervals_and_unbounded_profile_rejected(tmp_path):
    with pytest.raises(ValueError):
        union_duration([(4,3)])
    with pytest.raises(ValueError):
        HeadroomCollector(tmp_path, profile_request_indices=(1,2))
    with pytest.raises(ValueError):
        HeadroomCollector(tmp_path, profile_request_indices=(0,))
