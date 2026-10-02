"""CPU-only phase receipts for batch-one vLLM DiffusionGemma requests.

The scheduler receives sampled token lists after the existing D2H transfer.
An actual scheduled draft canvas with no emitted tokens is a denoising forward;
one with emitted tokens is a commit forward (including a partial final canvas).
Timing boundaries are scheduler completion times, not GPU event timestamps.
This observer inserts no tensor operations or CUDA synchronizations.
"""
from __future__ import annotations

import math
import time


_ACTIVE_TRACKER = None


class PhaseTracker:
    def __init__(self, clock=None):
        self.clock = clock or time.perf_counter
        self.rid = None
        self.start_request(None)

    def start_request(self, rid):
        """Reset all state, including warm-up observations, for one request."""
        self.rid = rid
        self.prefill_count = self.prefill_tokens = 0
        self.N = self.C = self.commit_tokens = 0
        self.scheduler_steps = 0
        self.last_prefill_end = self.decode_start = self.last_completed_at = None

    def _classify(self, scheduler_output, model_runner_output):
        scheduled = scheduler_output.num_scheduled_tokens
        if self.rid is None or self.rid not in scheduled:
            return None
        if len(scheduled) != 1:
            raise ValueError('phase tracker requires exactly one scheduled request')
        tokens = scheduled[self.rid]
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
            raise ValueError('unknown phase: scheduled token count must be positive')
        mapping = model_runner_output.req_id_to_index
        if self.rid not in mapping:
            raise ValueError('missing request mapping for phase receipt')
        if len(mapping) != 1:
            raise ValueError('phase tracker requires exactly one model-runner request')
        index = mapping[self.rid]
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError('invalid request mapping for phase receipt')
        sampled = model_runner_output.sampled_token_ids
        spec = scheduler_output.scheduled_spec_decode_tokens
        if self.rid in spec:
            if not spec[self.rid]:
                raise ValueError('unknown phase: empty scheduled draft canvas')
            if sampled is None or index >= len(sampled):
                raise ValueError('missing sampled token list for draft canvas')
            emitted = len(sampled[index])
            if self.last_prefill_end is None:
                raise ValueError('decode observed without a completed prefill')
            if emitted > tokens:
                raise ValueError('unknown phase: emitted tokens exceed scheduled tokens')
            return ('commit' if emitted else 'denoise', tokens, emitted)
        # The prefill sampler may return no token rows, or one empty row.
        if sampled and (index >= len(sampled) or sampled[index]):
            raise ValueError('unknown phase: output tokens without a scheduled draft canvas')
        if self.decode_start is not None:
            raise ValueError('unknown phase: prefill appeared after decode began')
        return ('prefill', tokens, 0)

    def _record(self, phase, completed_at):
        if phase is None:
            return
        completed_at = float(completed_at)
        if not math.isfinite(completed_at):
            raise ValueError('invalid scheduler completion timestamp')
        if self.last_completed_at is not None and completed_at < self.last_completed_at:
            raise ValueError('scheduler completion timestamps went backwards')
        kind, tokens, emitted = phase
        self.scheduler_steps += 1
        self.last_completed_at = completed_at
        if kind == 'prefill':
            self.prefill_count += 1
            self.prefill_tokens += tokens
            self.last_prefill_end = completed_at
        else:
            if self.decode_start is None:
                # This is the first decoder completion, not its start time.
                self.decode_start = completed_at
            if kind == 'denoise':
                self.N += 1
            else:
                self.C += 1
                self.commit_tokens += emitted

    def observe_scheduler(self, scheduler_output, model_runner_output, completed_at=None):
        """Record a completed scheduler update; useful with CPU fixture outputs."""
        phase = self._classify(scheduler_output, model_runner_output)
        if phase is not None:
            self._record(phase, self.clock() if completed_at is None else completed_at)

    def install_scheduler_hook(self):
        """Install once, then select this tracker; only the active request is read."""
        global _ACTIVE_TRACKER
        from vllm.v1.core.sched.scheduler import Scheduler
        _ACTIVE_TRACKER = self
        if getattr(Scheduler.update_from_output, '_v27_phase_tracker_hook', False):
            return
        original = Scheduler.update_from_output

        def update_from_output(scheduler, scheduler_output, model_runner_output):
            tracker = _ACTIVE_TRACKER
            phase = None if tracker is None else tracker._classify(scheduler_output, model_runner_output)
            result = original(scheduler, scheduler_output, model_runner_output)
            if phase is not None:
                tracker._record(phase, tracker.clock())
            return result

        update_from_output._v27_phase_tracker_hook = True
        update_from_output._v27_original = original
        Scheduler.update_from_output = update_from_output

    def finalize(self, start, end):
        """Return phase counts and spans after the caller's final boundary sync."""
        if self.last_prefill_end is None:
            raise ValueError('no completed prefill observed')
        start, end = float(start), float(end)
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError('invalid request boundary timestamps')
        if not start <= self.last_prefill_end <= self.last_completed_at <= end:
            raise ValueError('scheduler completion lies outside request boundaries')
        return dict(prefill_steps=self.prefill_count, prefill_tokens=self.prefill_tokens,
                    denoising_forwards=self.N, commit_forwards=self.C,
                    commit_tokens=self.commit_tokens, scheduler_steps=self.scheduler_steps,
                    prefill_s=self.last_prefill_end - start,
                    decode_span_s=end - self.last_prefill_end,
                    phase_boundary='scheduler_completion')


def graph_snapshot():
    """Read real vLLM compilation counters without touching CUDA."""
    from vllm.compilation.counter import compilation_counter
    names = ('num_cudagraph_captured', 'num_gpu_runner_capture_triggers',
             'num_backend_compilations', 'num_inductor_compiles')
    return {name: int(getattr(compilation_counter, name)) for name in names}
