"""CPU lifecycle tests for the bounded native refresh intervention."""
import pytest
import torch

from experiments.numerical_qk_reuse.v31_refresh_diagnostic import (
    FACTORIAL, GLOBAL_LAYERS, fingerprint, install,
)


class Adapter:
    arm = "mage"
    mage_select = "fa4"
    mage_granularity = "qblock_max"
    mage_carry_first = True
    mage_k = 128
    mage_select_step = 1
    mage_reselect = None
    mage_reselect_trigger = (0.15,)
    bound = False

    def __init__(self):
        self.calls = {}
        self.original_invocations = []
        self.begin_request()
        self.bound = False

    def begin_request(self):
        self.bound = True
        self.canvas_id = 1
        self._trig_at = (1, 6)
        self._mage_count = {layer: [1, 1280, 5] for layer in GLOBAL_LAYERS}
        self.calls = dict(mage_reselections=0, mage_reused_calls=0, split_fa4_calls=0)
        self._kept_prefix, self._prefix_total = torch.tensor(9.), 17
        self._sparse_kept, self._sparse_total = torch.tensor(8.), 16
        self.mage_state = {}
        for layer in GLOBAL_LAYERS:
            kept = torch.zeros(1, 2, 2, 20, dtype=torch.bool)
            kept[..., :2] = True
            kept[..., 16:] = True
            self.mage_state[layer] = dict(canvas=1, nk=1280, prefix=1024, kept=kept, lists=kept.clone())

    def _mage(self, layer, q, b, scale, prefix, n):
        count = self._mage_count[layer]
        count[2] += 1
        self.original_invocations.append((layer, count[2]))
        state = self.mage_state[layer]
        due = (count[2] in self.mage_reselect if self.mage_reselect is not None
               else self._trig_at == (self.canvas_id, count[2]))
        if due:
            fresh = state["kept"].clone()
            fresh[..., :16] = False
            fresh[..., 2:4] = True
            state.update(kept=fresh, lists=fresh.clone())
            self.calls["mage_reselections"] += 1
            return q.transpose(1, 2) + 20 + layer
        self.calls["mage_reused_calls"] += 1
        return self.sparse(q, b["k"], b["v"], state["lists"], scale)

    def sparse(self, q, k, v, lists, scale):
        self.calls["split_fa4_calls"] += 1
        self._sparse_kept = self._sparse_kept + 2
        self._sparse_total += 16
        self._kept_prefix = self._kept_prefix + 2
        self._prefix_total += 16
        position_sum = (lists[..., :16] * torch.arange(16)).sum()
        return q.transpose(1, 2) + position_sum

    def _value_receipt(self):
        return {"inherited": True}


def inputs():
    q = torch.arange(1 * 2 * 256 * 3, dtype=torch.float32).reshape(1, 2, 256, 3) / 1000
    b = {"k": torch.zeros(1, 1, 1280, 3), "v": torch.ones(1, 1, 1280, 3)}
    return q, b


def setup(policy, scope=GLOBAL_LAYERS, audit=True):
    adapter = Adapter()
    runtime = install(adapter, scope=scope, audit=audit, sparse_fn=adapter.sparse)
    runtime.set_policy(policy)
    adapter.begin_request()
    return adapter, runtime


@pytest.mark.parametrize("policy", FACTORIAL)
def test_factorial_separates_returned_output_from_next_support_and_preserves_clock(policy):
    adapter, runtime = setup(policy)
    q, b = inputs()
    old = {layer: state["kept"].clone() for layer, state in adapter.mage_state.items()}
    old_lists = {layer: state["lists"] for layer, state in adapter.mage_state.items()}
    accounting_before = [getattr(adapter, name).clone() if torch.is_tensor(getattr(adapter, name))
                         else getattr(adapter, name)
                         for name in ("_kept_prefix", "_prefix_total", "_sparse_kept", "_sparse_total")]
    for layer in GLOBAL_LAYERS:
        output = adapter._mage(layer, q, b, 0.1, 1024, 256)
        expected = q.transpose(1, 2) + 20 + layer if policy.startswith("dense_") else q.transpose(1, 2) + 4
        assert torch.equal(output, expected)
        assert adapter._mage_count[layer][2] == 6
        if policy.endswith("_hold"):
            assert adapter.mage_state[layer]["lists"] is old_lists[layer]
            assert torch.equal(adapter.mage_state[layer]["kept"], old[layer])
        else:
            assert not torch.equal(adapter.mage_state[layer]["kept"], old[layer])
    assert adapter.calls == dict(mage_reselections=5, mage_reused_calls=0, split_fa4_calls=0)
    for name, value in zip(("_kept_prefix", "_prefix_total", "_sparse_kept", "_sparse_total"), accounting_before):
        assert getattr(adapter, name) == value
    assert adapter._trig_at == (1, 6)  # The episode consumes native clocks, never rolls them back.
    receipt = runtime.receipt()["refresh_diagnostic"]
    assert receipt["exposed"] and receipt["complete_scope"] and not receipt["errors"]
    assert receipt["event"] == dict(canvas=1, step=6)
    assert receipt["layers"] == list(GLOBAL_LAYERS)
    assert receipt["extra_diagnostic_work"]["sparse_attention_calls"] == 5
    assert receipt["extra_diagnostic_work"]["kept_qk_rectangles"] == 5 * 4 * 6
    assert receipt["extra_diagnostic_work"]["counter_deltas"] == {"split_fa4_calls": 5}
    for layer in GLOBAL_LAYERS:
        adapter._mage(layer, q, b, 0.1, 1024, 256)
    receipt = adapter._value_receipt()["refresh_diagnostic"]
    assert all(row["held_consumer"] for row in receipt["first_post_event_call"].values())
    assert len(adapter.original_invocations) == 10  # Exactly one original call per real attention.
    assert len(receipt["events"]) == 5  # No second intervention on a subsequent call.


def test_dense_pair_and_sparse_pair_are_identical_on_event_before_support_consumption():
    q, b = inputs()
    outputs, supports = {}, {}
    for policy in FACTORIAL:
        adapter, runtime = setup(policy, scope=(29,))
        outputs[policy] = adapter._mage(29, q, b, 0.1, 1024, 256)
        supports[policy] = adapter.mage_state[29]["kept"]
    assert torch.equal(outputs["dense_fresh"], outputs["dense_hold"])
    assert torch.equal(outputs["sparse_fresh"], outputs["sparse_hold"])
    assert torch.equal(supports["dense_fresh"], supports["sparse_fresh"])
    assert torch.equal(supports["dense_hold"], supports["sparse_hold"])


def test_control_is_untouched_and_fixed16_restores_original_schedule_between_requests():
    adapter, runtime = setup("control", audit=False)
    q, b = inputs()
    expected = q.transpose(1, 2) + 20 + 29
    assert torch.equal(adapter._mage(29, q, b, 0.1, 1024, 256), expected)
    assert runtime.receipt()["refresh_diagnostic"]["extra_diagnostic_work"]["sparse_attention_calls"] == 0
    adapter.bound = False
    runtime.set_policy("fixed16")
    adapter.begin_request()
    assert adapter.mage_reselect == (16, 32) and adapter.mage_reselect_trigger is None
    adapter._mage(29, q, b, 0.1, 1024, 256)
    assert adapter.calls["mage_reselections"] == 0
    adapter._mage_count[29][2] = 15
    adapter._mage(29, q, b, 0.1, 1024, 256)
    assert adapter.calls["mage_reselections"] == 1
    adapter.bound = False
    runtime.set_policy("control")
    assert adapter.mage_reselect is None and adapter.mage_reselect_trigger == (0.15,)
    adapter.begin_request()
    assert not runtime.receipt()["refresh_diagnostic"]["exposed"]


def test_callback_precedes_first_treatment_once_and_missing_scope_is_detected():
    adapter, runtime = setup("sparse_hold")
    q, b = inputs()
    calls = []
    def callback(a, layer, query, buffers, prefix, n):
        calls.append(layer)
        assert not a.original_invocations
        assert a.calls["mage_reselections"] == 0
        return {"prefix_signature": "same-prefix", "q": fingerprint(query)}
    runtime.before_event_callback = callback
    adapter._mage(5, q, b, 0.1, 1024, 256)
    receipt = runtime.receipt()["refresh_diagnostic"]
    assert calls == [5] and receipt["prefix_signature"] == "same-prefix"
    assert not receipt["complete_scope"] and receipt["errors"]
    for layer in GLOBAL_LAYERS[1:]:
        adapter._mage(layer, q, b, 0.1, 1024, 256)
    assert calls == [5] and runtime.receipt()["refresh_diagnostic"]["complete_scope"]


def test_audit_control_captures_same_anchor_without_extra_sparse_or_scope_requirement():
    adapter, runtime = setup("control")
    q, b = inputs()
    callbacks = []
    def callback(*args):
        callbacks.append(args[1])
        return {"prefix_signature": "control-prefix"}
    runtime.before_event_callback = callback
    output = adapter._mage(5, q, b, 0.1, 1024, 256)
    assert torch.equal(output, q.transpose(1, 2) + 20 + 5)
    for layer in GLOBAL_LAYERS[1:]:
        adapter._mage(layer, q, b, 0.1, 1024, 256)
    receipt = runtime.receipt()["refresh_diagnostic"]
    assert callbacks == [5] and receipt["exposed"] and receipt["complete_scope"]
    assert receipt["prefix_signature"] == "control-prefix" and not receipt["errors"]
    assert receipt["control_anchor"]["sparse_oracle_calls"] == 0
    assert receipt["extra_diagnostic_work"]["sparse_attention_calls"] == 0
    assert adapter.calls["mage_reselections"] == 5 and adapter.calls["split_fa4_calls"] == 0


def test_no_event_when_not_due_or_not_midcanvas_and_no_reselection_is_fabricated():
    adapter, runtime = setup("dense_hold", scope=(29,))
    q, b = inputs()
    adapter._trig_at = None
    adapter._mage(29, q, b, 0.1, 1024, 256)
    assert not runtime.receipt()["refresh_diagnostic"]["exposed"]
    adapter.bound = False
    adapter.begin_request()
    adapter._mage_count[29][2] = 0
    adapter._trig_at = (1, 1)
    adapter._mage(29, q, b, 0.1, 1024, 256)
    assert not runtime.receipt()["refresh_diagnostic"]["exposed"]


def test_budget_failure_and_missing_layer_refresh_fail_loudly():
    adapter, runtime = setup("dense_fresh")
    q, b = inputs()
    adapter.mage_state[5]["kept"][..., 2] = True
    with pytest.raises(RuntimeError, match="budget"):
        adapter._mage(5, q, b, 0.1, 1024, 256)
    adapter, runtime = setup("dense_fresh")
    adapter._mage(5, q, b, 0.1, 1024, 256)
    adapter.mage_state[11]["nk"] = 999
    with pytest.raises(RuntimeError, match="same-geometry"):
        adapter._mage(11, q, b, 0.1, 1024, 256)


def test_policy_changes_require_request_boundary_and_factorial_requires_audit():
    adapter, runtime = setup("control", audit=False)
    with pytest.raises(RuntimeError, match="before"):
        runtime.set_policy("fixed16")
    adapter.bound = False
    with pytest.raises(ValueError, match="audit"):
        runtime.set_policy("dense_fresh")


def test_sparse_oracle_rng_consumption_is_rejected():
    adapter = Adapter()
    def bad_sparse(*args):
        torch.rand(1)
        return adapter.sparse(*args)
    runtime = install(adapter, scope=(29,), sparse_fn=bad_sparse)
    runtime.set_policy("dense_fresh")
    adapter.begin_request()
    q, b = inputs()
    with pytest.raises(RuntimeError, match="RNG"):
        adapter._mage(29, q, b, 0.1, 1024, 256)


def test_real_adapter_refresh_branch_keeps_old_lists_and_dispatches_current_sparse_function(monkeypatch):
    """Exercise the production lifecycle with CPU stand-ins only for kernels."""
    from experiments.numerical_qk_reuse import v27_fa4
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    layer_types = ["full_attention" if i in GLOBAL_LAYERS else "sliding_attention" for i in range(30)]
    adapter = VllmMethodAdapter(layer_types, arm="mage", mage_k=128,
        mage_select="fa4", mage_select_step=1, mage_carry_first=True,
        mage_granularity="qblock_max", mage_reselect_trigger=[0.15], mage_sticky=1.386)
    runtime = install(adapter, scope=(29,))
    runtime.set_policy("sparse_hold")
    adapter.begin_request()
    adapter.canvas_id = 1
    adapter._mage_count[29] = [1, 1280, 5]
    adapter._trig_at = (1, 6)
    kept = torch.zeros(1, 2, 2, 20, dtype=torch.bool)
    kept[..., :2] = kept[..., 16:] = True
    old_lists = kept.clone()
    adapter.mage_state[29] = dict(canvas=1, nk=1280, prefix=1024, kept=kept, lists=old_lists, pool=None)
    q, b = inputs()
    observed = []
    def select(layer, query, buffers, scale, prefix, n):
        observed.append((layer, adapter._mage_held))
        fresh = kept.clone()
        fresh[..., :16] = False
        fresh[..., 2:4] = True
        return query.transpose(1, 2) + 100, fresh
    adapter._select_initial_refresh = select
    monkeypatch.setattr(v27_fa4, "block_sparse_tensors", lambda mask: mask.clone())
    paged = object()
    adapter.paged = paged
    dispatches = []
    def sparse(query, key, value, lists, scale):
        dispatches.append(lists)
        assert adapter.paged is paged
        assert lists is old_lists
        return query.transpose(1, 2) + 7
    monkeypatch.setattr(v27_fa4, "sparse_lists", sparse)
    output = adapter._mage(29, q, b, 0.1, 1024, 256)
    assert torch.equal(output, q.transpose(1, 2) + 7)
    assert observed == [(29, kept)] and dispatches == [old_lists]
    assert adapter.mage_state[29]["lists"] is old_lists
    assert torch.equal(adapter.mage_state[29]["kept"], kept)
    assert adapter.calls["mage_reselections"] == 1 and adapter._mage_count[29][2] == 6
    assert adapter._mage_held is None and adapter._k_override is None
    adapter._mage(29, q, b, 0.1, 1024, 256)
    assert len(dispatches) == 2 and len(observed) == 1
    assert runtime.receipt()["refresh_diagnostic"]["first_post_event_call"]["29"]["held_consumer"]
