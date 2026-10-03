"""CPU-only token/position invariants before a new-model sparse port.

These are supplied-snapshot clocks, not measured model-forward counters. Nothing
imports torch, reads device tensors, installs hooks, or decides attention support.
Matching identities establish token/layout provenance only: unchanged token IDs
do not imply unchanged Q/K/V, hidden states, attention masks, or safe score reuse.

Integration boundaries (not implemented here):
* SGLang 0.5.21 JointThreshold.step: snapshot actual input positions/IDs before
  its owning run_batch forward, observe subsequent M2T/T2T edits, and promote a
  prefix only after the native block commit. A non-mask row is not a commit:
  joint_threshold.py:289-313 permits T2T edits and post-edit rounds.
* I-DLM a23c1a12ef997c7f3ad616b25bcfb62db39ded68: snapshot decode IDs after
  pending/spec/cold sampling and before run_batch's causal model_runner.forward;
  ingest only the output tokens the scheduler actually commits. KV trim/rejected
  drafts explicitly invalidate the next snapshot, even if committed prefix IDs
  do not change. idlm_blockN.py uses variable advances (1+rejection index,
  1+accepted specs, or 1), not a constant four-token advance.

The caller must supply a coherent snapshot, unique per-request ownership and a
monotonic event number from its future native adapter. Ragged prefill, duplicate
logical positions, physical paged-KV ownership, layer/projection identity, causal
mask identity and asynchronous stream ordering remain outside this prototype.
Real token arrays/identities must not be published in experiment reports.
"""
from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ActiveToken:
    position: int
    token_id: int


@dataclass(frozen=True)
class TokenIdentity:
    owner: object = field(repr=False)
    request_generation: int
    prefix_epoch: int
    layout_epoch: int
    invalidation_epoch: int
    position: int
    revision: int


@dataclass(frozen=True)
class ModelEvent:
    event: int
    prefix_epoch: int
    layout_epoch: int
    invalidation_epoch: int
    active_positions: tuple[int, ...]
    identities: tuple[TokenIdentity, ...]
    new_positions: frozenset[int]
    changed_positions: frozenset[int]
    removed_positions: frozenset[int]
    protected_positions: frozenset[int]
    invalidated_positions: frozenset[int]
    requires_refresh: bool
    rollback: bool


class _TokenEventClock:
    """Conservative CPU provenance tracking; one instance owns one request.

    Prefix mutation/retraction requires an explicit rollback. Prefix extension
    also changes its epoch; this intentionally avoids claiming append-safe reuse.
    A position leaving/reentering the layout gets a new revision, as does A->B->A.
    Validation is transactional: rejected snapshots do not advance any state.
    """

    def __init__(self):
        self._owner = object()
        self.request_generation = 0
        self.reset()

    def reset(self):
        """New request incarnation; even identical tokens cannot match old IDs."""
        self.request_generation += 1
        self.prefix_epoch = self.layout_epoch = self.invalidation_epoch = 0
        self._prefix = ()
        self._positions = ()
        self._boundaries = frozenset()
        self._active = {}
        self._revisions = {}
        self._serial = 0
        self._last_event = -1

    def _validate_layout(self, rows):
        raise NotImplementedError

    def observe(
        self,
        accepted_prefix: Sequence[int],
        active: Sequence[ActiveToken],
        *,
        event: int,
        rollback: bool = False,
        boundary_positions: Iterable[int] = (),
    ) -> ModelEvent:
        """Observe actual pre-forward layout after any previous native commit.

        accepted_prefix covers absolute positions [0, len(prefix)); active rows
        may include immutable prompt/prefix rows, which must match those tokens.
        rollback denotes *any* rejected/trimmed speculative state, not just an
        accepted-prefix retraction. Boundary protection is row-level metadata;
        no tile geometry or sparse consumer is implemented.
        """
        prefix, rows = tuple(accepted_prefix), tuple(active)
        if type(rollback) is not bool:
            raise ValueError("rollback must explicitly be a boolean")
        if type(event) is not int or event <= self._last_event:
            raise ValueError("event must advance monotonically within this request")
        if any(type(t) is not int or t < 0 for t in prefix):
            raise ValueError("prefix token IDs must be nonnegative integers")
        if any(not isinstance(r, ActiveToken) or type(r.position) is not int
               or r.position < 0 or type(r.token_id) is not int or r.token_id < 0
               for r in rows):
            raise ValueError("active rows require nonnegative integer positions/IDs")
        positions = tuple(r.position for r in rows)
        if len(set(positions)) != len(positions):
            raise ValueError("duplicate logical positions need a separate proven adapter")
        self._validate_layout(rows)
        current = {r.position: r.token_id for r in rows}
        if any(r.position < len(prefix) and r.token_id != prefix[r.position]
               for r in rows):
            raise ValueError("active prefix rows disagree with the committed prefix")
        if not rollback and prefix[:len(self._prefix)] != self._prefix:
            raise ValueError("prefix mutation/retraction requires explicit rollback")
        boundaries = frozenset(boundary_positions)
        if any(type(p) is not int for p in boundaries):
            raise ValueError("boundary positions must be integers")
        if not boundaries <= current.keys():
            raise ValueError("boundary positions must belong to the active layout")

        previous_positions, current_positions = set(self._active), set(current)
        new = current_positions - previous_positions
        removed = previous_positions - current_positions
        changed = {p for p in current_positions & previous_positions
                   if current[p] != self._active[p]}
        prefix_changed = prefix != self._prefix
        layout_changed = (positions != self._positions
                          or boundaries != self._boundaries)
        prefix_epoch = self.prefix_epoch + int(prefix_changed or rollback)
        layout_epoch = self.layout_epoch + int(layout_changed or rollback)
        invalidation_epoch = self.invalidation_epoch + int(rollback)
        invalidated = new | removed | changed
        if prefix_changed or layout_changed or rollback:
            invalidated |= current_positions | previous_positions
        protected = new | changed | set(boundaries)
        protected.update(p for p in current_positions if p < len(prefix))
        protected.update(p for p in (positions[0], positions[-1],
                                    len(prefix) - 1, len(prefix))
                         if p in current_positions)
        if rollback:
            protected |= current_positions

        serial, revisions = self._serial, {}
        for p in positions:
            if p in new or p in changed or rollback:
                serial += 1
                revisions[p] = serial
            else:
                revisions[p] = self._revisions[p]
        identities = tuple(TokenIdentity(self._owner, self.request_generation,
                           prefix_epoch, layout_epoch, invalidation_epoch, p,
                           revisions[p]) for p in positions)
        result = ModelEvent(event, prefix_epoch, layout_epoch, invalidation_epoch,
                            positions, identities, frozenset(new), frozenset(changed),
                            frozenset(removed), frozenset(protected),
                            frozenset(invalidated), bool(invalidated), rollback)
        self._prefix, self._positions, self._active = prefix, positions, current
        self._boundaries = boundaries
        self._serial, self._revisions, self._last_event = serial, revisions, event
        self.prefix_epoch, self.layout_epoch = prefix_epoch, layout_epoch
        self.invalidation_epoch = invalidation_epoch
        return result


class LLaDA32EventClock(_TokenEventClock):
    """Native 32-row JointThreshold block; token refinement is not a commit."""

    def _validate_layout(self, rows):
        if len(rows) != 32:
            raise ValueError("LLaDA32 decode snapshot must contain exactly 32 rows")
        start = rows[0].position
        if tuple(r.position for r in rows) != tuple(range(start, start + 32)):
            raise ValueError("LLaDA32 requires ordered contiguous absolute positions")


class IDLMEventClock(_TokenEventClock):
    """Causal verify/draft decode layout; never infers advance from block size.

    The official N=4 configuration uses 2*N-1=7 forward rows. Positions must be
    supplied from native metadata; they may be strided. Prefill is unsupported.
    """

    def __init__(self, gen_block_size: int = 4):
        if type(gen_block_size) is not int or gen_block_size < 1:
            raise ValueError("gen_block_size must be a positive integer")
        self.gen_block_size = gen_block_size
        super().__init__()

    def _validate_layout(self, rows):
        if len(rows) != 2 * self.gen_block_size - 1:
            raise ValueError("I-DLM decode requires 2*N-1 actual rows; prefill unsupported")
