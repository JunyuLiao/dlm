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


def test_collector_is_inert_before_first_request(tmp_path):
    # This fake intentionally has no CUDA or profiler attributes. Startup must
    # neither initialize torch nor record events, even when events are enabled.
    collector = HeadroomCollector(tmp_path, torch_module=SimpleNamespace())
    collector.record_phase(False)
    with collector.scope('startup') as span:
        assert span is None
    assert collector.request_index == -1 and collector.phase == 'prefill'
    assert not collector.active and collector.spans == []
    assert dict(collector.phase_calls) == {} and collector.dropped_spans == 0
    assert list(tmp_path.iterdir()) == []


def fake_install_modules(monkeypatch):
    import sys
    from types import ModuleType
    from scripts.v31_refresh_headroom_profile import install

    class FlashAttentionImpl:
        def forward(self, layer, value):
            return value

    class NativeGeometryAudit:
        def record(self, encoder, draft, *args, **kwargs):
            return (encoder, draft, args, kwargs)

    names = ['vllm', 'vllm.v1', 'vllm.v1.attention', 'vllm.v1.attention.backends',
             'vllm.v1.attention.backends.flash_attn']
    for name in names:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    fa = sys.modules[names[-1]]
    fa.FlashAttentionImpl = FlashAttentionImpl
    sys.modules[names[-2]].flash_attn = fa
    audit = ModuleType('experiments.numerical_qk_reuse.v31_value_native_audit')
    audit.NativeGeometryAudit = NativeGeometryAudit
    monkeypatch.setitem(sys.modules, audit.__name__, audit)
    return install, FlashAttentionImpl, NativeGeometryAudit


@pytest.mark.parametrize('native', [False, True])
def test_installed_hooks_pass_through_startup_and_excluded_warmup(monkeypatch, tmp_path, native):
    install, Flash, Native = fake_install_modules(monkeypatch)
    monkeypatch.delenv('VALUE_CLEAN_TIMING', raising=False)

    class Layer:
        @property
        def layer_name(self):
            raise AssertionError('inactive hook inspected layer state')

    class Adapter:
        @property
        def calls(self):
            raise AssertionError('inactive hook read uninitialized adapter counters')

        def begin_request(self):
            return 'original_begin'

        def end_request(self):
            return {'adapter': {'original': True}}

        def on_prepare(self, encoder):
            return encoder

        def _buffers(self, value):
            return value

        sparse_lists = _buffers
        _select_initial_refresh = _buffers
        on_sample = _buffers

    adapter = None if native else Adapter()
    original_forward = Flash.forward
    collector = install(adapter, output_dir=tmp_path, cuda_events=True)
    collector.torch = SimpleNamespace()  # Any accidental CUDA call fails.
    value = object()
    try:
        assert Flash().forward(Layer(), value) is value
        if adapter is None:
            assert Native().record(True, False, 'payload') == (True, False, ('payload',), {})
            collector.begin_request()
        else:
            assert adapter._buffers(value) is value
            assert adapter.sparse_lists(value) is value
            assert adapter._select_initial_refresh(value) is value
            assert adapter.on_sample(value) is value
            assert adapter.on_prepare(True) is True
            assert adapter.begin_request() == 'original_begin'
        # First request is the excluded implicit warmup and must also bypass
        # all state reads/events; startup has not shifted its index.
        assert collector.request_index == 0 and not collector.active
        assert Flash().forward(Layer(), value) is value
        if adapter is not None:
            assert adapter._buffers(value) is value
            assert adapter.end_request()['adapter']['original'] is True
        else:
            assert collector.end_request(boundary_completed=True)['warmup'] is True
        assert collector.spans == [] and dict(collector.phase_calls) == {}
        assert list(tmp_path.iterdir()) == []
    finally:
        collector.close()
    assert Flash.forward is original_forward
