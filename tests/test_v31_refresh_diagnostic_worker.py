"""CPU-only tests of native worker dispatch and prefix-state trace integrity."""
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from scripts.v31_refresh_diagnostic_worker import bind, install_state_trace, validate_schedule


def test_schedule_is_explicit_and_binding_selects_policy_before_native_reset():
    schedule = [dict(policy="control", label="warmup"),
                dict(policy="dense_fresh", label="dense_fresh"),
                dict(policy="control", label="control_after")]
    validate_schedule(schedule)
    events = []
    adapter = SimpleNamespace()
    def begin():
        events.append("native_begin")
    adapter.begin_request = begin
    adapter._value_receipt = lambda: {"inherited": True}
    runtime = SimpleNamespace(set_policy=lambda policy: events.append(("policy", policy)))
    trace = {"post": [{"call": 3}]}
    state = bind(adapter, runtime, schedule, trace)
    for index, entry in enumerate(schedule):
        adapter.begin_request()
        receipt = adapter._value_receipt()
        assert events[-2:] == [("policy", entry["policy"]), "native_begin"]
        assert receipt["refresh_request_index"] == index
        assert receipt["refresh_label"] == entry["label"]
        assert receipt["inherited"] and receipt["refresh_post_event_states"] == trace["post"]
    assert state["index"] == 2
    with pytest.raises(RuntimeError, match="exceeded"):
        adapter.begin_request()
    for invalid in ([], [dict(policy="sparse_hold", label="warmup")],
                    [dict(policy="control", label="warmup"), dict(policy="bad", label="bad")]):
        with pytest.raises(ValueError):
            validate_schedule(invalid)


def test_non_mage_controls_reject_policy_switches():
    adapter = SimpleNamespace(begin_request=lambda: None, _value_receipt=lambda: {})
    bind(adapter, None, [dict(policy="fixed16", label="wrong")])
    with pytest.raises(RuntimeError, match="non-MAGE"):
        adapter.begin_request()


def mock_native(monkeypatch, policy="dense_fresh"):
    packages = {name: ModuleType(name) for name in (
        "vllm", "vllm.model_executor", "vllm.model_executor.models",
        "vllm.model_executor.models.diffusion_gemma")}
    dg = packages["vllm.model_executor.models.diffusion_gemma"]
    packages["vllm.model_executor.models"].diffusion_gemma = dg
    for name, module in packages.items():
        monkeypatch.setitem(sys.modules, name, module)
    calls = []
    def sampler(*args, **kwargs):
        calls.append(args)
        slots = args[1]
        args[5][slots] += 1
        args[6][slots] += 2
        args[7][slots] -= 1
        args[10][slots] += 0.25
        return args[0]
    dg._compiled_sample_step = sampler
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda *args, **kwargs: torch.get_rng_state().clone())
    adapter = SimpleNamespace(bound=False, step_ctx=None, canvas_id=1, _canvas_step=7,
        _trig_at=(1, 6), _mage_count={layer: [1, 1280, 5] for layer in (5, 11, 17, 23, 29)},
        mage_state={layer: {"kept": torch.tensor([True, False])} for layer in (5, 11, 17, 23, 29)})
    def begin():
        adapter.bound = True
        adapter.step_ctx = {"encoder": False}
    adapter.begin_request = begin
    runtime = SimpleNamespace(policy=policy, before_event_callback=None, episode=(1, 6))
    trace = install_state_trace(adapter, runtime)
    adapter.begin_request()
    args = (torch.zeros(1, 3), torch.tensor([0]), None, None, None,
            torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 4, dtype=torch.long),
            torch.tensor([48]), torch.tensor([False]), torch.tensor([False]), torch.zeros(1, 4, 3))
    return dg, adapter, runtime, trace, args, calls


def capture(monkeypatch, policy="dense_fresh", change=None):
    dg, adapter, runtime, trace, args, calls = mock_native(monkeypatch, policy)
    dg._compiled_sample_step(*args)
    q = torch.ones(1, 2, 4, 3)
    b = {"k": torch.zeros(1, 1, 16, 3), "v": torch.ones(1, 1, 16, 3)}
    if change == "q":
        q += 1
    elif change == "self_conditioning":
        args[10].add_(1)
    elif change == "map":
        adapter.mage_state[29]["kept"][0] = False
    metadata = runtime.before_event_callback(adapter, 5, q, b, 12, 4)
    return metadata, (dg, adapter, runtime, trace, args, calls)


def test_prefix_signatures_match_across_control_and_factorial_and_detect_state_changes(monkeypatch):
    rng = torch.get_rng_state().clone()
    reference, _ = capture(monkeypatch, "control")
    for policy in ("dense_fresh", "dense_hold", "sparse_fresh", "sparse_hold"):
        candidate, _ = capture(monkeypatch, policy)
        assert candidate["prefix_signature"] == reference["prefix_signature"]
    for change in ("q", "self_conditioning", "map"):
        candidate, _ = capture(monkeypatch, change=change)
        assert candidate["prefix_signature"] != reference["prefix_signature"]
    assert torch.equal(rng, torch.get_rng_state())


def test_trace_adds_no_sampler_call_or_rng_draw_preserves_metadata_and_bounds_post_history(monkeypatch):
    rng = torch.get_rng_state().clone()
    metadata, (dg, adapter, runtime, trace, args, calls) = capture(monkeypatch)
    assert len(calls) == 1 and metadata["sampler_calls_before_event"] == 1
    assert len(trace["history"]) == 1 and not trace["post"]
    original_count = metadata["state_fingerprints"]["progress"]["counts"][5][2]
    adapter._mage_count[5][2] += 1
    assert metadata["state_fingerprints"]["progress"]["counts"][5][2] == original_count
    for _ in range(10):
        result = dg._compiled_sample_step(*args)
        assert result is args[0]
    assert len(calls) == 11 and trace["sampler_calls"] == 11
    assert len(trace["post"]) == 8 and len(trace["history"]) == 1
    assert "self_conditioning" in trace["post"][0]
    assert all("self_conditioning" not in item for item in trace["post"][1:])
    assert torch.equal(rng, torch.get_rng_state())
    adapter.begin_request()
    assert trace["args"] is None and not trace["history"] and not trace["post"]


def test_prefix_requires_native_sampler_state_and_inactive_policy_passes_through(monkeypatch):
    dg, adapter, runtime, trace, args, calls = mock_native(monkeypatch, "fixed16")
    q = torch.ones(1)
    with pytest.raises(RuntimeError, match="preceding"):
        runtime.before_event_callback(adapter, 5, q, {"k": q, "v": q}, 12, 4)
    dg._compiled_sample_step(*args)
    assert len(calls) == 1 and trace["sampler_calls"] == 0 and trace["args"] is None
