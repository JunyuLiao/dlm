"""Phase receipts use existing CPU scheduler outputs, never token-length guesses."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from scripts.v27_vllm_metrics import PhaseTracker, graph_snapshot


def output(rid='r', tokens=10, draft=None, emitted=0):
    scheduled = SimpleNamespace(num_scheduled_tokens={rid: tokens},
                                scheduled_spec_decode_tokens={} if draft is None else {rid: draft})
    model = SimpleNamespace(req_id_to_index={rid: 0}, sampled_token_ids=[[1] * emitted])
    return scheduled, model


def tracker():
    result = PhaseTracker()
    result.start_request('r')
    return result


def prefill(t, at=2, tokens=10):
    t.observe_scheduler(*output(tokens=tokens), completed_at=at)


def test_real_prefill_chunks_empty_denoise_and_partial_final_commit():
    t = tracker()
    # Thirty prompt tokens actually used three unequal chunks, not ceil(30/16).
    for at, tokens in [(2, 7), (3, 11), (4, 12)]:
        prefill(t, at, tokens)
    t.observe_scheduler(*output(tokens=256, draft=[0] * 256), completed_at=5)
    t.observe_scheduler(*output(tokens=256, draft=[0] * 256), completed_at=6)
    t.observe_scheduler(*output(tokens=7, draft=[0] * 7, emitted=7), completed_at=7)
    rec = t.finalize(1, 8)
    assert rec == dict(prefill_steps=3, prefill_tokens=30, denoising_forwards=2,
                       commit_forwards=1, commit_tokens=7, scheduler_steps=6,
                       prefill_s=3, decode_span_s=4, phase_boundary='scheduler_completion')
    assert t.decode_start == 5
    assert rec['decode_span_s'] == 8 - 4  # Includes the first denoising forward.


def test_start_request_resets_warmup_counts_and_timing():
    t = tracker()
    prefill(t)
    t.observe_scheduler(*output(draft=[1]), completed_at=3)
    t.start_request('run')
    assert t.N == t.C == t.prefill_count == t.prefill_tokens == t.scheduler_steps == 0
    assert t.last_prefill_end is t.decode_start is t.last_completed_at is None
    t.observe_scheduler(*output(rid='r', draft=[1]), completed_at=4)
    assert t.scheduler_steps == 0


def test_non_target_requests_and_empty_scheduling_are_ignored():
    t = tracker()
    t.observe_scheduler(*output(rid='other', draft=[]), completed_at=1)
    t.observe_scheduler(SimpleNamespace(num_scheduled_tokens={}, scheduled_spec_decode_tokens={}),
                        SimpleNamespace(req_id_to_index={}, sampled_token_ids=[]), completed_at=2)
    assert t.scheduler_steps == 0


@pytest.mark.parametrize('mutation,match', [
    ('missing_mapping', 'missing request mapping'),
    ('bad_mapping', 'missing sampled token list'),
    ('multi_scheduled', 'exactly one scheduled'),
    ('multi_mapping', 'exactly one model-runner'),
    ('missing_samples', 'missing sampled token list'),
    ('empty_draft', 'empty scheduled draft'),
    ('zero_tokens', 'scheduled token count'),
    ('negative_index', 'invalid request mapping'),
    ('too_many_emitted', 'emitted tokens exceed'),
])
def test_malformed_phase_receipts_fail(mutation, match):
    t = tracker()
    prefill(t)
    s, m = output(tokens=2, draft=[1, 1])
    if mutation == 'missing_mapping':
        m.req_id_to_index = {}
    elif mutation == 'bad_mapping':
        m.req_id_to_index['r'] = 3
    elif mutation == 'multi_scheduled':
        s.num_scheduled_tokens['other'] = 1
    elif mutation == 'multi_mapping':
        m.req_id_to_index['other'] = 1
    elif mutation == 'missing_samples':
        m.sampled_token_ids = None
    elif mutation == 'empty_draft':
        s.scheduled_spec_decode_tokens['r'] = []
    elif mutation == 'zero_tokens':
        s.num_scheduled_tokens['r'] = 0
    elif mutation == 'negative_index':
        m.req_id_to_index['r'] = -1
    elif mutation == 'too_many_emitted':
        m.sampled_token_ids = [[1, 1, 1]]
    with pytest.raises(ValueError, match=match):
        t.observe_scheduler(s, m, completed_at=3)
    assert t.scheduler_steps == 1


def test_unknown_phase_and_missing_prefill_fail():
    t = tracker()
    with pytest.raises(ValueError, match='without a completed prefill'):
        t.observe_scheduler(*output(draft=[1]), completed_at=2)
    with pytest.raises(ValueError, match='no completed prefill'):
        t.finalize(1, 3)
    with pytest.raises(ValueError, match='without a scheduled draft'):
        t.observe_scheduler(*output(emitted=1), completed_at=2)
    prefill(t)
    t.observe_scheduler(*output(draft=[1]), completed_at=3)
    with pytest.raises(ValueError, match='prefill appeared after decode'):
        prefill(t, at=4)


def test_prefill_without_token_rows_and_boundary_validation():
    t = tracker()
    s, m = output()
    m.sampled_token_ids = []
    t.observe_scheduler(s, m, completed_at=2)
    assert t.finalize(1, 3)['denoising_forwards'] == 0
    with pytest.raises(ValueError, match='outside request boundaries'):
        t.finalize(3, 4)
    with pytest.raises(ValueError, match='went backwards'):
        t.observe_scheduler(*output(draft=[1]), completed_at=1)


def test_hook_records_after_original_and_does_not_double_wrap(monkeypatch):
    events = []

    class Scheduler:
        def update_from_output(self, scheduled, model):
            events.append('original')
            return 'result'

    module = ModuleType('vllm.v1.core.sched.scheduler')
    module.Scheduler = Scheduler
    monkeypatch.setitem(sys.modules, module.__name__, module)
    t = PhaseTracker(clock=lambda: events.append('clock') or 2)
    t.start_request('r')
    t.install_scheduler_hook()
    wrapped = Scheduler.update_from_output
    t.install_scheduler_hook()
    assert Scheduler.update_from_output is wrapped
    assert Scheduler().update_from_output(*output()) == 'result'
    assert events == ['original', 'clock']
    assert t.prefill_count == 1 and t.last_prefill_end == 2


def test_failed_scheduler_update_does_not_record(monkeypatch):
    class Scheduler:
        def update_from_output(self, scheduled, model):
            raise RuntimeError('scheduler failed')

    module = ModuleType('vllm.v1.core.sched.scheduler')
    module.Scheduler = Scheduler
    monkeypatch.setitem(sys.modules, module.__name__, module)
    t = tracker()
    t.install_scheduler_hook()
    with pytest.raises(RuntimeError, match='scheduler failed'):
        Scheduler().update_from_output(*output())
    assert t.prefill_count == t.scheduler_steps == 0


def test_graph_snapshot_reads_installed_counter(monkeypatch):
    names = ['num_cudagraph_captured', 'num_gpu_runner_capture_triggers',
             'num_backend_compilations', 'num_inductor_compiles']
    expected = dict(zip(names, [7, 8, 9, 10]))
    module = ModuleType('vllm.compilation.counter')
    module.compilation_counter = SimpleNamespace(**expected)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert graph_snapshot() == expected
