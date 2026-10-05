"""Phase receipts use existing CPU scheduler outputs, never token-length guesses."""
import sys
import gc
from types import ModuleType, SimpleNamespace
import weakref

import numpy as np
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
    execution(t, tokens=tokens)


def execution(t, tokens=10, draft=0, emitted=0):
    snapshot = np.array([emitted], dtype=np.int32)
    t.observe_execution_snapshot([t.rid], draft, tokens, snapshot)
    return snapshot


def test_real_prefill_chunks_empty_denoise_and_partial_final_commit():
    t = tracker()
    # Thirty prompt tokens actually used three unequal chunks, not ceil(30/16).
    for at, tokens in [(2, 7), (3, 11), (4, 12)]:
        prefill(t, at, tokens)
    t.observe_scheduler(*output(tokens=256, draft=[0] * 256), completed_at=5)
    execution(t, tokens=256, draft=256)
    t.observe_scheduler(*output(tokens=256, draft=[0] * 256), completed_at=6)
    execution(t, tokens=256, draft=256)
    t.observe_scheduler(*output(tokens=7, draft=[0] * 7, emitted=7), completed_at=7)
    execution(t, tokens=7, draft=7, emitted=7)
    rec = t.finalize(1, 8)
    assert rec == dict(prefill_steps=3, prefill_tokens=30, denoising_forwards=2,
                       commit_forwards=1, commit_tokens=7, scheduler_steps=6,
                       scheduler_denoising_forwards=2, scheduler_commit_forwards=1,
                       speculative_unused_denoising=0,
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
    execution(t)
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


def test_async_final_extra_step_counts_actual_and_consumed_separately():
    t = tracker()
    prefill(t)
    t.observe_scheduler(*output(tokens=256, draft=[0] * 256), completed_at=3)
    execution(t, tokens=256, draft=256)
    t.observe_scheduler(*output(tokens=256, draft=[0] * 256, emitted=256), completed_at=4)
    execution(t, tokens=256, draft=256, emitted=256)
    # Already executed, but its queued output was never retired by scheduler.
    execution(t, tokens=256, draft=256)
    result = t.finalize(1, 5)
    assert result['denoising_forwards'] == 2
    assert result['scheduler_denoising_forwards'] == 1
    assert result['speculative_unused_denoising'] == 1
    assert result['commit_forwards'] == result['scheduler_commit_forwards'] == 1
    assert result['scheduler_steps'] == 3


def test_missing_execution_snapshots_fail_closed():
    t = tracker()
    t.observe_scheduler(*output(), completed_at=2)
    with pytest.raises(ValueError, match='no execution CPU snapshots'):
        t.finalize(1, 3)
    execution(t)
    t.observe_scheduler(*output(draft=[1]), completed_at=3)
    with pytest.raises(ValueError, match='missing consumed scheduler work'):
        t.finalize(1, 4)


def test_actual_and_consumed_prefill_counts_and_tokens_must_match():
    t = tracker()
    t.observe_scheduler(*output(tokens=7), completed_at=2)
    execution(t, tokens=8)
    with pytest.raises(ValueError, match='prefill receipts differ'):
        t.finalize(1, 3)
    t.start_request('r')
    t.observe_scheduler(*output(tokens=7), completed_at=2)
    execution(t, tokens=3)
    execution(t, tokens=4)
    with pytest.raises(ValueError, match='prefill receipts differ'):
        t.finalize(1, 3)


def test_execution_multi_request_and_stale_epoch_fail():
    t = tracker()
    with pytest.raises(ValueError, match='exactly one request'):
        t.observe_execution_snapshot(['r', 'other'], 0, 7, np.array([0, 0]))
    with pytest.raises(ValueError, match='stale execution snapshot'):
        t.observe_execution_snapshot(['r'], 0, 7, np.array([0]), request_epoch=t._request_epoch-1)
    t.observe_execution_snapshot(['other'], 0, 7, None)
    assert not t._execution_snapshots


def test_cpu_snapshot_reference_keeps_storage_alive_until_request_reset():
    t = tracker()
    snapshot = execution(t)
    ref = weakref.ref(snapshot)
    del snapshot
    gc.collect()
    assert ref() is not None
    t.start_request('next')
    gc.collect()
    assert ref() is None


def test_reusing_same_pinned_storage_or_array_view_fails_closed():
    t = tracker()
    snapshot = execution(t)
    for reused in (snapshot, snapshot.view()):
        with pytest.raises(ValueError, match='buffer was reused'):
            t.observe_execution_snapshot(['r'], 1, 10, reused)


def test_snapshot_values_are_read_only_at_finalization():
    t = tracker()
    t.observe_scheduler(*output(tokens=10), completed_at=2)
    # Simulate an asynchronous D2H copy not having completed at hook time.
    pending = np.array([-1], dtype=np.int32)
    t.observe_execution_snapshot(['r'], 0, 10, pending)
    pending[0] = 0
    assert t.finalize(1, 3)['prefill_steps'] == 1


def test_execution_hooks_keep_only_existing_cpu_array_and_ignore_other_request(monkeypatch):
    import scripts.v27_vllm_metrics as metrics

    class DiffusionSampler:
        def __call__(self, logits, input_batch):
            return SimpleNamespace(emitted=input_batch.emitted)

    class AsyncOutput:
        def __init__(self, model_runner_output, sampler_output, *args, **kwargs):
            self.num_sampled_tokens_np = np.array([sampler_output.emitted], dtype=np.int32)

    dg = ModuleType('vllm.model_executor.models.diffusion_gemma')
    dg.DiffusionSampler = DiffusionSampler
    au = ModuleType('vllm.v1.worker.gpu.async_utils')
    au.AsyncOutput = AsyncOutput
    monkeypatch.setitem(sys.modules, dg.__name__, dg)
    monkeypatch.setitem(sys.modules, au.__name__, au)
    monkeypatch.setattr(metrics, '_ACTIVE_TRACKER', None)
    t = tracker()
    t.install_execution_hook()
    sampler_hook, output_hook = DiffusionSampler.__call__, AsyncOutput.__init__
    t.install_execution_hook()
    assert DiffusionSampler.__call__ is sampler_hook and AsyncOutput.__init__ is output_hook

    def run(req='r', draft=0, emitted=0):
        batch = SimpleNamespace(req_ids=[req], num_reqs=1, num_draft_tokens=draft,
                                num_tokens=10, emitted=emitted)
        sampled = DiffusionSampler()(None, batch)
        return AsyncOutput(SimpleNamespace(req_ids=[req]), sampled)

    obj = run()
    snapshot = obj.num_sampled_tokens_np
    assert t._execution_snapshots[0][2] is snapshot
    del obj
    t.observe_scheduler(*output(), completed_at=2)
    run(draft=10)
    run(req='other')
    assert t.finalize(1, 4)['denoising_forwards'] == 1
    with pytest.raises(ValueError, match='missing sampler execution phase'):
        AsyncOutput(SimpleNamespace(req_ids=['r']), SimpleNamespace(emitted=0))
