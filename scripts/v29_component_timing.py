"""Independent component diagnostic timing with saved samples and warm/JIT gates.

No CUDA imports at module import. This is not request timing instrumentation.
CUDA event durations do not provide a Chrome timeline. Nested/side-stream spans
must not be summed into wall time. Host-call time can include implicit waits.
"""
from contextlib import contextmanager
import math
import statistics
import time


def rotate(names, ordinal):
    names = list(names)
    if not names:
        raise ValueError('nonempty measurement family required')
    offset = ordinal % len(names)
    return names[offset:]+names[:offset]


def zero_counter_deltas(before, after):
    if (not before or set(before) != set(after)
            or any(type(v) is not int or v < 0 for v in (*before.values(), *after.values()))):
        raise RuntimeError('complete nonnegative integer compilation/capture snapshots required')
    deltas = {key: after[key]-before[key] for key in before}
    if any(deltas.values()):
        raise RuntimeError('new monitored compilation/capture during timed controls')
    return deltas


def activate_monitor(monitor, triton_hook):
    """Official vLLM0.30.0 API; call before importing FA4/consumer modules.

    activate is a no-op if already active, so warn mode must already match.
    Installation flags supplement actual counter deltas, not prove all coverage.
    """
    monitor.activate(mode='warn', verbose=False)
    if (not monitor.is_active() or getattr(monitor, '_mode', None) != 'warn'
            or getattr(monitor, '_cutedsl_hook_installed', None) is not True
            or not callable(triton_hook())):
        raise RuntimeError('active warn-mode CuTe/Triton JIT monitoring required')
    return dict(active=True, mode='warn', cute_hook_installed=True, triton_hook_callable=True,
                activated_before_component_imports=True,
                coverage='installed handler hooks; previously saved compile aliases and unsupported compilers can escape coverage')


@contextmanager
def vllm_compile_guard():
    """New standalone process only; no installed package or shared file edits."""
    from vllm.utils import jit_monitor
    from triton import knobs
    from scripts.v28_jit_receipts import jit_receipts
    from scripts.v27_vllm_metrics import graph_snapshot
    receipt = activate_monitor(jit_monitor, lambda: knobs.runtime.jit_post_compile_hook)
    with jit_receipts(jit_monitor) as receipts:
        def snapshot():
            return dict(**graph_snapshot(), **{'jit_monitor_'+k:v for k,v in receipts.snapshot().items()})
        yield snapshot, receipt


def measure_components(torch, functions, *, repeats, warm, snapshot, clock=time.perf_counter):
    """Oracle-qualified callables -> all raw samples plus descriptive medians.

    Caller must run all numerical oracles before entry. Warm each callable on
    the exact same shape/support/backend. Snapshot AFTER warm synchronization.
    Samples retain repetition/order/arm and three different timing boundaries.
    This routine deliberately synchronizes between independent controls; never
    use it in a formal request path or claim it models asynchronous overlap.
    """
    if (not functions or any(not isinstance(k,str) or not callable(fn) for k,fn in functions.items())
            or type(repeats) is not int or repeats < 4 or type(warm) is not int or warm < 1
            or not callable(snapshot)):
        raise ValueError('named callables, positive warm, >=4 repeats and snapshot callback required')
    for fn in functions.values():
        for _ in range(warm):
            fn()
    torch.cuda.synchronize()
    before = snapshot()
    zero_counter_deltas(before, before)           # reject missing/broken monitoring
    samples = {name: [] for name in functions}
    rows = []
    host_origin = None
    for rep in range(repeats):
        for position, name in enumerate(rotate(functions, rep)):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall = clock()
            if host_origin is None:
                host_origin = wall
            begin.record()
            result = functions[name]()
            end.record()
            submitted = clock()
            end.synchronize()
            complete = clock()
            row = dict(repeat=rep, order_position=position, sample_ordinal=len(rows), arm=name,
                       cuda_event_ms=begin.elapsed_time(end), synchronized_host_ms=(complete-wall)*1000,
                       host_call_ms=(submitted-wall)*1000,
                       host_start_relative_s=wall-host_origin,
                       host_submitted_relative_s=submitted-host_origin,
                       host_completed_relative_s=complete-host_origin)
            if (not all(math.isfinite(row[k]) and row[k] >= 0
                        for k in ('cuda_event_ms','synchronized_host_ms','host_call_ms'))
                    or row['host_call_ms'] > row['synchronized_host_ms']):
                raise RuntimeError('invalid timing sample; stop without a speed claim')
            rows.append(row)
            samples[name].append(row)
            del result
    after = snapshot()
    deltas = zero_counter_deltas(before, after)
    medians = {name: {key: statistics.median(row[key] for row in values)
                     for key in ('cuda_event_ms','synchronized_host_ms','host_call_ms')}
               for name, values in samples.items()}
    return dict(sample_rows=rows, timings={name:dict(samples=values, medians=medians[name])
                                         for name,values in samples.items()},
                compilation_deltas=deltas, counter_before=before, counter_after=after,
                warm_calls_per_arm=warm, repeats=repeats, rotated_order=True,
                raw_event_timestamps_saved=False, chrome_timeline_saved=False)
