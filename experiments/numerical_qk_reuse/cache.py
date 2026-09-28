"""Host-static score/decision lifecycle; no tensor reads or GPU synchronization.

The caller supplies identity from actual native cache/position metadata. A plan
is not an observation: publication occurs only after the score producer has
been enqueued on the consuming stream (or its dependency event has been waited).
"""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Identity:
    request: int
    canvas: int
    encoder_epoch: int
    layer: int
    batch: int
    query_heads: int
    kv_heads: int
    queries: int
    keys: int
    width: int
    query_start: int
    key_start: int
    source_keys: int
    scale: float
    dtype: str
    device: str
    mask_signature: tuple
    prefix_owner: int

    @property
    def storage_bytes(self):
        return self.batch * self.query_heads * self.queries * self.keys * 4


@dataclass(frozen=True)
class Plan:
    score_refresh: bool
    decision_refresh: bool
    reason: str
    score_age: int | None
    decision_age: int | None


@dataclass
class Entry:
    identity: Identity
    scores: Any
    score_step: int
    decision: Any = None
    decision_step: int | None = None


class ScoreCache:
    def __init__(self, score_period=8, decision_interval=1, max_bytes=2 * 1024**3, origin=0):
        if min(score_period, decision_interval, max_bytes) < 1:
            raise ValueError("Positive periods and storage budget required")
        if origin not in (0, 1):
            raise ValueError("score clock origin must be 0 (native M3) or 1 (bootstrap observation)")
        # origin=1: the first true observation is canvas call 1, so later
        # anchors fall at 9, 17, ... and never at a call index divisible by 8.
        self.origin = origin
        self.score_period = score_period
        self.decision_interval = decision_interval
        self.max_bytes = max_bytes
        self.entries = {}

    def clear(self):
        self.entries.clear()

    @property
    def storage_bytes(self):
        return sum(x.identity.storage_bytes for x in self.entries.values())

    def plan(self, identity, step, *, force_refresh=False):
        if step < 0:
            raise ValueError("Canvas-local decoder call index must be nonnegative")
        old = self.entries.get(identity.layer)
        if old is None or old.identity != identity:
            return Plan(True, True, "initial_or_identity_change", None, None)
        age = step - old.score_step
        decision_age = None if old.decision_step is None else step - old.decision_step
        if age < 0 or (decision_age is not None and decision_age < 0):
            raise ValueError("Decoder iteration went backwards without reset")
        fresh = (force_refresh or (step - self.origin) % self.score_period == 0
                 or age >= self.score_period)
        decision = fresh or old.decision is None or decision_age >= self.decision_interval
        return Plan(fresh, decision, "forced_mask_refresh" if force_refresh else
                    "score_schedule" if fresh else "decision_schedule" if decision else "held_decision", age, decision_age)

    def reserve(self, identity):
        old = self.entries.get(identity.layer)
        total = self.storage_bytes - (old.identity.storage_bytes if old else 0) + identity.storage_bytes
        if total > self.max_bytes:
            raise MemoryError(f"Score cache would need {total} bytes; bound is {self.max_bytes}")

    def publish_scores(self, identity, step, scores):
        self.reserve(identity)
        self.entries[identity.layer] = Entry(identity, scores, step)

    def publish_decision(self, identity, step, decision):
        old = self.entries.get(identity.layer)
        if old is None or old.identity != identity or step < old.score_step:
            raise ValueError("No compatible observed scores for decision")
        old.decision = decision
        old.decision_step = step

    def get(self, identity):
        old = self.entries[identity.layer]
        if old.identity != identity:
            raise ValueError("Stale cache identity")
        return old
