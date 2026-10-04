"""Per-step / per-call time breakdown of one seed-paired arm (diagnostic run, not a panel measurement).

Wraps scripts/v31_vllm_paired_bench.py without changing it. Every intercepted GLOBAL attention call, the sampler hook,
the split FA4 sparse call and the contiguous K/V refresh are bracketed by torch.cuda.synchronize(), so async overlap
is removed and the absolute step times are inflated; the record is for attribution only:
  per GLOBAL call: kind (bootstrap / observe / route / held / dense / mage_select / mage_reuse, from counter deltas),
                   total ms, the FA4 sparse call ms inside it, the K/V refresh ms, kept fraction of 64-key tiles
  per denoising step: wall ms, sum of GLOBAL ms, sampler-hook ms (accepted mask, T / C-gate state update)
Usage: same arguments and env as v31_vllm_paired_bench.py, plus PROFILE_OUT=<jsonl>. Use LIMIT / DATASETS to keep it
short. Writes one JSON line per request with per-kind aggregates and the step list (times and kinds only, no text).

PROFILE_MODE=events (v31): no synchronization inside a request. Every bracketed region records a pair of CUDA events on
the current stream (GPU time of that region, async overlap and side-stream waits included); each engine step records
its CPU wall time (the step ends with the token read-back, so this is the real step latency). Events are resolved once
per request. The per-call GPU times are valid; the per-step CPU wall clock is NOT a latency under vLLM's async
scheduling (the CPU runs ahead; its sum under-counts the request's decode time by ~1/3 in practice) -- use the bench
record's decode_s for latency. The first record of every process is the warm-up request (JIT compiles, first-call
allocations; e.g. 200+ ms dense calls): it is flagged warmup=true and must be excluded from aggregates.
"""
import collections
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v31_vllm_paired_bench as bench  # noqa: E402

KINDS = ('bootstrap_dense_calls', 'fused_observations', 'dp_routes', 'held_decision_calls', 'carried_first_calls',
         'async_observation_routes', 'mage_selections', 'mage_reused_calls')


def install_profiler():
    import torch
    import experiments.numerical_qk_reuse as pkg
    if os.environ.get('V27_ADAPTER_DIR') and os.environ['V27_ADAPTER_DIR'] not in pkg.__path__:
        pkg.__path__.append(os.environ['V27_ADAPTER_DIR'])
    from experiments.numerical_qk_reuse import vllm_adapter as va
    prof = dict(calls=[], steps=[], hook=[], pending=[])
    events = os.environ.get('PROFILE_MODE', 'sync') == 'events'

    def sync_time(fn, *a, **k):
        if events:                                     # an unresolved [start, end] event pair, resolved per request
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            r = fn(*a, **k)
            e1.record()
            return r, [e0, e1]
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = fn(*a, **k)
        torch.cuda.synchronize()
        return r, 1000 * (time.perf_counter() - t)

    def wall_time(fn, *a, **k):                        # step latency: CPU wall clock, no added synchronization
        if not events:
            return sync_time(fn, *a, **k)
        t = time.perf_counter()
        r = fn(*a, **k)
        return r, 1000 * (time.perf_counter() - t)
    prof['events'] = events

    def counters(a):
        c = dict(a.calls)
        if a.runtime is not None:
            c.update({k: v for k, v in a.runtime['counters']().items() if isinstance(v, int)})
        return c

    inner_sparse = va.VllmMethodAdapter.sparse_lists

    def sparse_lists(self, original, q, k, v, lists, scale):
        r, ms = sync_time(inner_sparse, self, original, q, k, v, lists, scale)
        cnt = lists.full_block_cnt
        kept = (cnt.sum() / (cnt.numel() * lists.full_block_idx.shape[-1])) if events else             float(cnt.sum()) / (cnt.numel() * lists.full_block_idx.shape[-1])
        prof['pending'].append(('sparse', ms, kept))
        return r
    va.VllmMethodAdapter.sparse_lists = sparse_lists

    inner_buffers = va.VllmMethodAdapter._buffers

    def buffers(self, *a):
        r, ms = sync_time(inner_buffers, self, *a)
        prof['pending'].append(('kv', ms, None))
        return r
    va.VllmMethodAdapter._buffers = buffers

    inner_sample = va.VllmMethodAdapter.on_sample

    def on_sample(self, *a, **k):
        r, ms = sync_time(inner_sample, self, *a, **k)
        prof['hook'].append(ms)
        return r
    va.VllmMethodAdapter.on_sample = on_sample

    from experiments.numerical_qk_reuse import v31_logit_stats as ls
    for name in ('row_stats', 'accepted_from_entropy'):
        def timed(*a, _inner=getattr(ls, name), **k):
            r, ms = sync_time(_inner, *a, **k)
            prof['hook'].append(ms)
            return r
        setattr(ls, name, timed)

    inner_mask = va.accepted_mask

    def accepted_mask(scaled, bound):
        r, ms = sync_time(inner_mask, scaled, bound)
        prof['hook'].append(ms)
        return r
    va.accepted_mask = accepted_mask

    inner_install = va.install_vllm_patches

    def install(adapter):
        inner_install(adapter)
        from vllm.v1.attention.backends import flash_attn as fa
        patched = fa.FlashAttentionImpl.forward

        def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None,
                    output_block_scale=None):
            a = va._ACTIVE
            layer_idx = None
            if a is not None and a.bound and attn_metadata is not None and output_scale is None:
                layer_idx = a.active_for(getattr(layer, 'layer_name', ''))
            if layer_idx is None:
                return patched(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale,
                               output_block_scale)
            before = counters(a)
            prof['pending'].clear()
            r, ms = sync_time(patched, self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale,
                              output_block_scale)
            after = counters(a)
            kind = '+'.join(k for k in KINDS if after.get(k, 0) != before.get(k, 0)) or (
                'dense' if a.arm == 'native' else 'other')
            sparse = [p for p in prof['pending'] if p[0] == 'sparse']
            prof['calls'].append(dict(layer=layer_idx, kind=kind, ms=ms, sparse_ms=[p[1] for p in sparse],
                                      kv_ms=[p[1] for p in prof['pending'] if p[0] == 'kv'],
                                      kept=None if not sparse else sparse[-1][2]))
            return r
        fa.FlashAttentionImpl.forward = forward
    va.install_vllm_patches = install

    from vllm.v1.engine.llm_engine import LLMEngine
    inner_step = LLMEngine.step

    def step(self):
        n0, h0 = len(prof['calls']), len(prof['hook'])
        r, ms = wall_time(inner_step, self)
        prof['steps'].append(dict(ms=ms, calls=(n0, len(prof['calls'])), hook=(h0, len(prof['hook']))))
        return r
    LLMEngine.step = step
    return prof


def _ms(x):
    """A float (sync mode) or an unresolved [start, end] CUDA event pair (events mode) -> milliseconds."""
    return x[0].elapsed_time(x[1]) if isinstance(x, list) else x


def summarize(prof, out):
    steps, calls = prof['steps'], prof['calls']
    if not steps:
        return
    import torch
    if prof.get('events'):
        torch.cuda.synchronize()                       # once per request: resolve every recorded event pair
    for c in calls:
        c['ms'] = round(_ms(c['ms']), 4)
        c['sparse_ms'] = round(sum(_ms(x) for x in c['sparse_ms']), 4)
        c['kv_ms'] = round(sum(_ms(x) for x in c['kv_ms']), 4)
        if c['kept'] is not None:
            c['kept'] = round(float(c['kept']), 4)
    hook = [round(_ms(x), 4) for x in prof['hook']]
    for s in steps:
        s['ms'] = round(_ms(s['ms']), 3)
        mine = calls[s['calls'][0]:s['calls'][1]]
        s['global_ms'] = round(sum(c['ms'] for c in mine), 3)
        s['hook_ms'] = round(sum(hook[s['hook'][0]:s['hook'][1]]), 3)
        s['kinds'] = sorted({c['kind'] for c in mine})
    by_kind = collections.defaultdict(list)
    for c in calls:
        by_kind[c['kind']].append(c)
    agg = {k: dict(n=len(v), ms_mean=round(sum(c['ms'] for c in v) / len(v), 4),
                   sparse_ms_mean=round(sum(c['sparse_ms'] for c in v) / len(v), 4),
                   kv_ms_mean=round(sum(c['kv_ms'] for c in v) / len(v), 4),
                   kept_mean=(round(sum(c['kept'] for c in v if c['kept'] is not None)
                                    / max(1, sum(c['kept'] is not None for c in v)), 4)))
           for k, v in sorted(by_kind.items())}
    prof['written'] = prof.get('written', 0) + 1
    rec = dict(mode='events' if prof.get('events') else 'sync', warmup=prof['written'] == 1, steps=len(steps),
               step_ms_total=round(sum(s['ms'] for s in steps), 2),
               global_ms_total=round(sum(s['global_ms'] for s in steps), 2),
               hook_ms_total=round(sum(s['hook_ms'] for s in steps), 2), calls_by_kind=agg,
               step_list=[[s['ms'], s['global_ms'], s['hook_ms'], s['kinds']] for s in steps])
    out.write(json.dumps(rec) + '\n')
    out.flush()
    print(json.dumps({k: rec[k] for k in ('steps', 'step_ms_total', 'global_ms_total', 'hook_ms_total')}),
          json.dumps(agg), flush=True)


def main():
    prof = install_profiler()
    out = open(os.environ['PROFILE_OUT'], 'a', encoding='utf-8')
    from vllm.v1.engine.llm_engine import LLMEngine
    inner_add = LLMEngine.add_request

    def add_request(self, *a, **k):                      # one record per request: flush the previous one
        summarize(prof, out)
        for key in ('calls', 'steps', 'hook'):
            prof[key].clear()
        return inner_add(self, *a, **k)
    LLMEngine.add_request = add_request
    try:
        bench.main()
    finally:
        summarize(prof, out)


if __name__ == '__main__':
    main()
