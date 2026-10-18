"""One native refresh episode: current output x subsequent support diagnostic.

This is an instrumented causal diagnostic, not a cheap refresh implementation.
Every factorial arm executes the inherited dense observation and one extra
native sparse attention operation at each treated layer. No model forward or
sampler invocation is added. Extra sparse work is receipted separately from
the inherited observation counters; diagnostic latency is never clean timing.
"""
from __future__ import annotations

import hashlib
import types
from typing import Callable

import torch


GLOBAL_LAYERS = (5, 11, 17, 23, 29)
FACTORIAL = ("dense_fresh", "dense_hold", "sparse_fresh", "sparse_hold")
POLICIES = ("control", *FACTORIAL, "fixed16")
_ACCOUNT_FIELDS = ("_kept_prefix", "_prefix_total", "_sparse_kept", "_sparse_total")


def fingerprint(tensor):
    if tensor is None:
        return None
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _summary(tensor):
    value = tensor.detach().float()
    return dict(sha256=fingerprint(tensor), shape=list(tensor.shape), dtype=str(tensor.dtype),
                finite=bool(torch.isfinite(value).all()),
                rms=float(value.square().mean().sqrt()), max_abs=float(value.abs().max()))


def _rng(device):
    return (torch.get_rng_state().clone(),
            torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None)


def _same_rng(before, device):
    after = _rng(device)
    return torch.equal(before[0], after[0]) and (
        before[1] is None or torch.equal(before[1], after[1]))


class Runtime:
    """Instance wrapper installed after the normal native adapter patches.

    Call ``set_policy`` before ``adapter.begin_request``. ``control`` is an
    untouched original _mage call. ``fixed16`` changes only the refresh clock
    to zero-based canvas calls 16 and 32; native call-1 selection/carry remain.
    All-five-layer treatment admits downstream Q/K/V differences as mediators;
    the common-state qualification belongs before the first treated layer.
    """

    def __init__(self, adapter, scope=GLOBAL_LAYERS, audit=True, sparse_fn: Callable | None = None):
        self.adapter = adapter
        self.scope = tuple(scope)
        if self.scope not in (GLOBAL_LAYERS, (29,)):
            raise ValueError("diagnostic scope must be all five GLOBAL layers or layer29")
        actual_layers = getattr(adapter, "global_layers", self.scope)
        if not set(self.scope).issubset(actual_layers):
            raise ValueError("diagnostic scope is not present in the model GLOBAL layers")
        if getattr(adapter, "_refresh_diagnostic_runtime", None) is not None:
            raise ValueError("refresh diagnostic already installed")
        if (adapter.arm != "mage" or adapter.mage_select != "fa4"
                or adapter.mage_granularity != "qblock_max" or not adapter.mage_carry_first):
            raise ValueError("requires inherited fixed-budget FA4 qblock_max with carried support")
        for name in ("cached_refresh_policy", "stable_refresh_config", "value_selector", "mage_pool",
                     "mage_keep_frac", "mage_row_weight", "mage_reselect_kmin", "residual", "drop_guard"):
            if getattr(adapter, name, None) is not None:
                raise ValueError(f"unsupported diagnostic interaction: {name}")
        if getattr(adapter, "mage_always_refresh", False):
            raise ValueError("always-refresh is not the inherited lifecycle")
        if getattr(adapter, "mage_reselect_k", None) not in (None, adapter.mage_k):
            raise ValueError("refresh must preserve the initial fixed K")
        self.audit = bool(audit)
        self.sparse_fn = sparse_fn
        self.original_mage = adapter._mage
        self.original_begin = adapter.begin_request
        self.original_receipt = adapter._value_receipt
        self.original_schedule = (adapter.mage_reselect, adapter.mage_reselect_trigger)
        self.policy = "control"
        self.before_event_callback = None
        self.reset()

    def reset(self):
        self.episode = None
        self.events = []
        self.next_calls = {}
        self.extra_work = dict(sparse_attention_calls=0, eligible_qk_rectangles=0,
                               kept_qk_rectangles=0, kept_pv_rectangles=0,
                               counter_deltas={})
        self.errors = []
        self.prefix_metadata = None
        self.control_anchor = None

    def set_policy(self, label):
        if label not in POLICIES:
            raise ValueError(f"unsupported refresh policy: {label}")
        if getattr(self.adapter, "bound", False):
            raise RuntimeError("choose the policy before the native request begins")
        if label in FACTORIAL and (self.original_schedule[0] is not None
                                  or self.original_schedule[1] is None):
            raise ValueError("factorial episode requires the inherited progress refresh clock")
        if label in FACTORIAL and not self.audit:
            raise ValueError("factorial attention oracles require explicitly instrumented audit mode")
        self.policy = label
        if label == "fixed16":
            self.adapter.mage_reselect = (16, 32)
            self.adapter.mage_reselect_trigger = None
        else:
            self.adapter.mage_reselect, self.adapter.mage_reselect_trigger = self.original_schedule

    def begin_request(self, *args, **kwargs):
        self.reset()
        return self.original_begin(*args, **kwargs)

    def _call(self, layer, nk):
        count = self.adapter._mage_count.get(layer)
        return count[2] + 1 if count is not None and count[:2] == [self.adapter.canvas_id, nk] else 0

    def _support(self, state, prefix, n):
        kept = state.get("kept") if state is not None else None
        if kept is None or kept.ndim != 4 or kept.shape[0] != 1:
            raise RuntimeError("missing native per-head/Q128 support")
        pt = prefix // 64
        if kept.shape[2:] != ((n + 127) // 128, (prefix + n + 63) // 64):
            raise RuntimeError("support geometry differs from the current attention")
        budget = min(self.adapter.mage_k // 64, pt)
        if not bool((kept[..., :pt].sum(-1) == budget).all()):
            raise RuntimeError("fixed eligible-prefix budget violation")
        if not bool(kept[..., pt:].all()):
            raise RuntimeError("mandatory boundary/canvas support was dropped")
        return kept, budget

    def _extra_sparse(self, q, b, lists, scale, kept):
        """Run the same patched paged sparse consumer, isolating oracle counters.

        The ordinary phase ledger retains the physically executed dense
        observation. This separate ledger adds the sparse oracle actually run
        in every factorial arm, independently of which output is returned.
        """
        adapter = self.adapter
        counters = dict(adapter.calls)
        accounting = {key: getattr(adapter, key, None) for key in _ACCOUNT_FIELDS}
        try:
            if self.sparse_fn is None:
                from . import v27_fa4
                output = v27_fa4.sparse_lists(q, b["k"], b["v"], lists, scale)
            else:
                output = self.sparse_fn(q, b["k"], b["v"], lists, scale)
        finally:
            for key, value in adapter.calls.items():
                delta = value - counters.get(key, 0)
                if delta:
                    totals = self.extra_work["counter_deltas"]
                    totals[key] = totals.get(key, 0) + delta
            adapter.calls.clear()
            adapter.calls.update(counters)
            for key, value in accounting.items():
                setattr(adapter, key, value)
        self.extra_work["sparse_attention_calls"] += 1
        self.extra_work["eligible_qk_rectangles"] += kept.numel()
        work = int(kept.sum())
        self.extra_work["kept_qk_rectangles"] += work
        self.extra_work["kept_pv_rectangles"] += work
        return output

    def mage(self, layer, q, b, scale, prefix, n):
        adapter = self.adapter
        if self.policy not in FACTORIAL:
            # Audit bookends need the same event-entry proof, without running
            # either treatment or the extra sparse oracle. Clean control never
            # enters this path and remains a direct inherited call.
            if (self.policy == "control" and self.audit and self.before_event_callback is not None
                    and self.episode is None and layer == self.scope[0]):
                call = self._call(layer, prefix + n)
                identity = (adapter.canvas_id, call)
                old = adapter.mage_state.get(layer)
                same = (old is not None and old.get("canvas") == adapter.canvas_id
                        and old.get("nk") == prefix + n and old.get("prefix") == prefix)
                if (same and adapter._trig_at == identity and call > int(adapter.mage_select_step)
                        and prefix // 64 > adapter.mage_k // 64):
                    kept, budget = self._support(old, prefix, n)
                    rng_before = _rng(q.device)
                    self.episode = identity
                    metadata = self.before_event_callback(adapter, layer, q, b, prefix, n)
                    if not isinstance(metadata, dict) or "prefix_signature" not in metadata:
                        raise RuntimeError("event callback must return metadata with prefix_signature")
                    self.prefix_metadata = metadata
                    old_hash = fingerprint(kept)
                    count = adapter.calls.get("mage_reselections", 0)
                    output = self.original_mage(layer, q, b, scale, prefix, n)
                    if adapter.calls.get("mage_reselections", 0) != count + 1:
                        raise RuntimeError("control anchor did not execute its expected native refresh")
                    if not _same_rng(rng_before, q.device):
                        raise RuntimeError("control event observer consumed native RNG")
                    self.control_anchor = dict(canvas=identity[0], call=call, layer=layer,
                        prefix_budget=budget, old_support_sha256=old_hash,
                        next_support_sha256=fingerprint(adapter.mage_state[layer]["kept"]),
                        returned_output_sha256=fingerprint(output), original_mage_calls=1,
                        rng_unchanged=True, sparse_oracle_calls=0)
                    return output
            return self.original_mage(layer, q, b, scale, prefix, n)
        call = self._call(layer, prefix + n)
        identity = (adapter.canvas_id, call)
        old = adapter.mage_state.get(layer)
        same = (old is not None and old.get("canvas") == adapter.canvas_id
                and old.get("nk") == prefix + n and old.get("prefix") == prefix)
        due = adapter._trig_at == identity
        rng_before = None
        if (self.episode is None and layer == self.scope[0] and same and due
                and call > int(adapter.mage_select_step)
                and prefix // 64 > adapter.mage_k // 64):
            self._support(old, prefix, n)
            self.episode = identity
            rng_before = _rng(q.device)
            if self.before_event_callback is not None:
                metadata = self.before_event_callback(adapter, layer, q, b, prefix, n)
                if not isinstance(metadata, dict) or "prefix_signature" not in metadata:
                    raise RuntimeError("event callback must return metadata with prefix_signature")
                self.prefix_metadata = metadata
        treated = self.episode == identity and layer in self.scope
        after_event = (self.episode is not None and layer in self.scope
                       and layer not in self.next_calls and identity != self.episode
                       and any(event["layer"] == layer for event in self.events))
        post_support = fingerprint(old.get("kept")) if after_event and old is not None else None
        counts_before = dict(adapter.calls)
        if treated:
            if not same or not due or any(event["layer"] == layer for event in self.events):
                raise RuntimeError("episode must be exactly one same-geometry refresh in every scoped layer")
            old_kept, budget = self._support(old, prefix, n)
            old_kept = old_kept.clone()
            old_lists = old["lists"]
            input_hashes = dict(q=fingerprint(q))
            if layer == self.scope[0]:
                input_hashes.update(k=fingerprint(b["k"]), v=fingerprint(b["v"]))
            if rng_before is None:
                rng_before = _rng(q.device)
        # Exactly one inherited call, including its clock and refresh mutation.
        dense = self.original_mage(layer, q, b, scale, prefix, n)
        if after_event:
            self.next_calls[layer] = dict(canvas=identity[0], call=call,
                support_at_entry_sha256=post_support,
                held_consumer=adapter.calls.get("mage_reused_calls", 0) > counts_before.get("mage_reused_calls", 0),
                refresh_before_consumption=adapter.calls.get("mage_reselections", 0) > counts_before.get("mage_reselections", 0))
        if not treated:
            return dense
        if adapter.calls.get("mage_reselections", 0) != counts_before.get("mage_reselections", 0) + 1:
            raise RuntimeError("scheduled episode did not execute one native refresh")
        current = adapter.mage_state[layer]
        fresh, fresh_budget = self._support(current, prefix, n)
        if fresh_budget != budget:
            raise RuntimeError("fresh and held budgets differ")
        fresh = fresh.clone()
        sparse = self._extra_sparse(q, b, old_lists, scale, old_kept)
        if sparse.shape != dense.shape or sparse.dtype != dense.dtype:
            raise RuntimeError("dense and native sparse output layouts differ")
        if not bool(torch.isfinite(dense).all() & torch.isfinite(sparse).all()):
            raise RuntimeError("nonfinite attention alternative in the factorial diagnostic")
        output = dense if self.policy.startswith("dense_") else sparse
        if self.policy.endswith("_hold"):
            current["lists"] = old_lists
            current["kept"] = old_kept
        after, _ = self._support(current, prefix, n)
        rng_equal = _same_rng(rng_before, q.device)
        if not rng_equal:
            raise RuntimeError("attention diagnostic consumed native RNG")
        event = dict(canvas=identity[0], call=call, layer=layer, prefix=prefix, query_rows=n,
            prefix_budget=budget, input_sha256=input_hashes,
            old_support_sha256=fingerprint(old_kept), fresh_support_sha256=fingerprint(fresh),
            next_support_sha256=fingerprint(after), changed_prefix_blocks=int((old_kept ^ fresh)[..., :prefix // 64].sum()),
            current_output="dense" if self.policy.startswith("dense_") else "sparse_old_support",
            next_support="held" if self.policy.endswith("_hold") else "fresh",
            dense_output=_summary(dense), sparse_old_output=_summary(sparse), returned_output_sha256=fingerprint(output),
            dense_sparse_difference_l2=float((dense.float() - sparse.float()).norm()),
            rng_unchanged=True, original_mage_calls=1)
        self.events.append(event)
        return output

    def receipt(self):
        layers = [event["layer"] for event in self.events]
        if self.control_anchor is not None:
            layers = [self.control_anchor["layer"]]
        complete = self.policy not in FACTORIAL or self.episode is None or layers == list(self.scope)
        return dict(refresh_diagnostic=dict(policy=self.policy, scope=list(self.scope), layers=layers,
            exposed=self.episode is not None,
            prefix_signature=None if self.prefix_metadata is None else self.prefix_metadata["prefix_signature"],
            prefix_metadata=self.prefix_metadata,
            control_anchor=self.control_anchor,
            event=None if self.episode is None else dict(canvas=self.episode[0], step=self.episode[1]),
            episode=None if self.episode is None else dict(canvas=self.episode[0], call=self.episode[1]),
            event_occurred=self.episode is not None, complete_scope=complete,
            errors=[] if complete else ["incomplete scoped refresh episode"], events=list(self.events),
            first_post_event_call={str(k): v for k, v in self.next_calls.items()},
            extra_diagnostic_work=self.extra_work,
            physical_accounting="inherited dense observation ledger plus separately counted sparse oracle; returned output recorded independently",
            comparison_scope="common state before first treated layer; downstream states are treatment mediators",
            added_model_forwards=0, added_sampler_calls=0,
            timing_valid=not self.audit and self.policy in ("control", "fixed16"),
            actual_fixed_refresh_calls=[16, 32] if self.policy == "fixed16" else None))


def install(adapter, scope=GLOBAL_LAYERS, audit=True, sparse_fn=None):
    """Wrap only this adapter instance; no shared source/class mutation."""
    runtime = Runtime(adapter, scope=scope, audit=audit, sparse_fn=sparse_fn)
    adapter._refresh_diagnostic_runtime = runtime
    adapter._mage = types.MethodType(lambda self, *a, **kw: runtime.mage(*a, **kw), adapter)
    adapter.begin_request = types.MethodType(lambda self, *a, **kw: runtime.begin_request(*a, **kw), adapter)
    adapter._value_receipt = types.MethodType(
        lambda self: dict(runtime.original_receipt(), **runtime.receipt()), adapter)
    return runtime
