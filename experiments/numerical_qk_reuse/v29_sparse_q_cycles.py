"""Opt-in sparse in-place Q permutation on an exclusively owned component clone.

No model integration. Disjoint cycles and dimension shards are the only legal
program partitions. Every CTA completes all loads before its first store.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np


def amortization_receipt(construction_seconds, natural_ms, candidate_ms, valid_reuses=None):
    """Optimistic bound, never treats an unmeasured lifetime as free reuse."""
    if (any(not math.isfinite(x) or x < 0 for x in
            (construction_seconds, natural_ms, candidate_ms))
            or valid_reuses is not None and (type(valid_reuses) is not int or valid_reuses < 1)):
        raise ValueError('Finite nonnegative costs and positive measured reuse count required')
    saving = (natural_ms-candidate_ms)/1000
    calls = None if saving <= 0 else math.ceil(construction_seconds/saving)
    return dict(total_component_saving_seconds_per_call=saving,
                optimistic_break_even_calls=calls, measured_valid_order_reuses=valid_reuses,
                breaks_even_within_measured_lifetime=None if valid_reuses is None
                else calls is not None and calls <= valid_reuses)


@dataclass(frozen=True)
class CyclePlan:
    heads: int
    rows: int
    cycles: tuple

    @property
    def moved_rows(self):
        return sum(len(rows) for _, rows in self.cycles)

    @property
    def max_cycle_length(self):
        return max((len(rows) for _, rows in self.cycles), default=0)

    def statistics(self, head_dim=512, itemsize=2):
        return dict(moved_rows=self.moved_rows, total_rows=self.heads*self.rows,
                    nonidentity_cycles=len(self.cycles), max_cycle_length=self.max_cycle_length,
                    cycle_lengths=[len(rows) for _, rows in self.cycles],
                    one_direction_logical_read_write_bytes=2*self.moved_rows*head_dim*itemsize,
                    restoration_inclusive_logical_read_write_bytes=4*self.moved_rows*head_dim*itemsize,
                    permutation_launches_per_restored_call=2 if self.cycles else 0,
                    byte_scope='Logical Q payload only; metadata/cache/dispatch not estimated as speed')


def plan_cycles(order):
    """order[h,slot] names the original row; exact bijection, no support changes."""
    order = np.asarray(order)
    if order.ndim != 2 or min(order.shape) < 1 or order.dtype.kind not in 'iu':
        raise ValueError('Nonempty per-head integer bijection required')
    h, q = order.shape
    if not np.array_equal(np.sort(order, axis=1), np.broadcast_to(np.arange(q), order.shape)):
        raise ValueError('Every head must be a bijection')
    cycles = []
    for head in range(h):
        seen = set()
        for first in range(q):
            if first in seen:
                continue
            rows, row = [], first
            while row not in seen:
                seen.add(row); rows.append(row); row = int(order[head, row])
            if row != first:
                raise ValueError('Cycle partition is not closed')
            if len(rows) > 1:
                cycles.append((head, tuple(rows)))
    return CyclePlan(h, q, tuple(cycles))


def apply_reference(query, plan, inverse=False):
    """Independent CPU reference [H,Q,D], returning a copy even for identity."""
    query = np.asarray(query)
    if query.ndim != 3 or query.shape[:2] != (plan.heads, plan.rows):
        raise ValueError('Head-major query geometry differs')
    result = query.copy()
    for head, rows in plan.cycles:
        source = np.roll(rows, 1 if inverse else -1)
        result[head, list(rows)] = query[head, source]
    return result


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def rotate(Q, Rows, Heads, Lengths, D: tl.constexpr,
               SH: tl.constexpr, SQ: tl.constexpr, SD: tl.constexpr,
               CAP: tl.constexpr, COLS: tl.constexpr, INVERSE: tl.constexpr):
        cycle = tl.program_id(0)
        head = tl.load(Heads+cycle)
        length = tl.load(Lengths+cycle)
        row = tl.arange(0, CAP)
        col = tl.program_id(1)*COLS+tl.arange(0, COLS)
        next_row = (row+length-1) % length if INVERSE else (row+1) % length
        dest = tl.load(Rows+cycle*CAP+row, mask=row < length, other=0)
        source = tl.load(Rows+cycle*CAP+next_row, mask=row < length, other=0)
        valid = (row[:, None] < length) & (col[None, :] < D)
        values = tl.load(Q+head*SH+source[:, None]*SQ+col[None, :]*SD,
                         mask=valid, other=0)
        # No implicit cross-warp synchronization: CTA load completion precedes
        # all stores. Distinct CTAs own disjoint cycles or disjoint D columns.
        tl.debug_barrier()
        tl.store(Q+head*SH+dest[:, None]*SQ+col[None, :]*SD, values, mask=valid)
    return rotate


class SparseQCycles:
    """Mutates only the explicitly supplied, independent query clone.

    Callers own failure handling: discard the clone after any interrupted CUDA
    operation. This does not assert a model Q tensor is safe to modify in place.
    """
    def __init__(self, order, device):
        import torch
        import triton
        self.plan = plan_cycles(order)
        self.device = torch.device(device)
        self.cap = triton.next_power_of_2(max(2, self.plan.max_cycle_length))
        rows = np.zeros((len(self.plan.cycles), self.cap), dtype=np.int32)
        heads, lengths = [], []
        for index, (head, cycle) in enumerate(self.plan.cycles):
            rows[index, :len(cycle)] = cycle
            heads.append(head); lengths.append(len(cycle))
        self.rows = torch.from_numpy(rows).to(self.device)
        self.heads = torch.tensor(heads, device=self.device, dtype=torch.int32)
        self.lengths = torch.tensor(lengths, device=self.device, dtype=torch.int32)
        self.kernel = _kernel()

    def __call__(self, query, inverse=False):
        import torch
        import triton
        if (query.shape[:3] != (1, self.plan.heads, self.plan.rows) or query.ndim != 4
                or query.dtype != torch.bfloat16 or not query.is_cuda
                or query.device != self.device or query.shape[-1] < 1
                or any(s <= 0 for s in query.stride())):
            raise ValueError('Owned CUDA BF16 query [1,H,Q,D] required')
        # Require the real native token-major view; arbitrary overlapping strides
        # are unsafe for concurrent cycles. No hidden copy or D2H validation.
        h, q, d = self.plan.heads, self.plan.rows, query.shape[-1]
        if query.stride()[1:] != (d, h*d, 1):
            raise ValueError('Exact nonoverlapping native token-major Q view required')
        if self.plan.cycles:
            self.kernel[(len(self.plan.cycles), triton.cdiv(d, 128))](
                query, self.rows, self.heads, self.lengths, d,
                *query.stride()[1:], self.cap, 128, bool(inverse), num_warps=4)
        return query
