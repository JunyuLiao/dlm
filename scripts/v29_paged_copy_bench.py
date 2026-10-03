"""Synthetic native-stride KV-copy qualification; no request speed/accuracy claim."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('new_report', type=Path)
    p.add_argument('--repeats', type=int, default=100)
    args = p.parse_args(argv)
    if args.new_report.exists() or args.repeats < 20:
        raise ValueError('new report path and at least20 rotated repeats required')
    import torch
    from experiments.numerical_qk_reuse.v29_paged_copy import copy_paged_kv
    started = time.monotonic()
    torch.manual_seed(2901)
    gen = torch.Generator(device='cuda').manual_seed(2901)
    records = []
    # Native BF16 K/V are split views of [physical page, KV head, slot, K+V].
    for prefix, n in [(0, 1), (65, 256), (32768, 256), (65536, 256), (98304, 256)]:
        page, h, d, nk = 64, 2, 512, prefix + n
        npages = (nk + page - 1) // page
        storage = torch.randn(npages + 3, h, page, 2*d, device='cuda', dtype=torch.bfloat16, generator=gen)
        key, value = storage.transpose(1, 2).split(d, dim=-1)
        table = torch.randperm(npages+3, device='cuda', generator=gen, dtype=torch.int64)[:npages].to(torch.int32)
        for mode in ('full', 'tail'):
            start, count = (0, nk) if mode == 'full' else (prefix, n)
            outputs = {name: (torch.full((1,h,nk,d), 13., device='cuda', dtype=key.dtype),
                              torch.full((1,h,nk,d), -7., device='cuda', dtype=key.dtype))
                       for name in ('torch', 'triton')}
            def original():
                k, v = outputs['torch']
                first, last = start // page, (start+count-1) // page
                pages = table[first:last+1].long()
                off = start - first*page
                k[0,:,start:start+count].copy_(key[pages].reshape(-1,h,d)[off:off+count].transpose(0,1))
                v[0,:,start:start+count].copy_(value[pages].reshape(-1,h,d)[off:off+count].transpose(0,1))
            def fused():
                copy_paged_kv(key, value, table, *outputs['triton'], start, count)
            funcs = dict(torch=original, triton=fused)
            original(); fused()
            if not all(torch.equal(a,b) for a,b in zip(outputs['torch'], outputs['triton'])):
                raise RuntimeError('full-buffer exact oracle failed, including untouched prefix')
            for fn in funcs.values():
                for _ in range(8): fn()
            torch.cuda.synchronize()
            times = {name: [] for name in funcs}
            for rep in range(args.repeats):
                order = ('torch','triton') if rep % 2 == 0 else ('triton','torch')
                for name in order:
                    a,b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record(); funcs[name](); b.record(); b.synchronize()
                    times[name].append(a.elapsed_time(b))
            records.append(dict(prefix_tokens=prefix, canvas_tokens=n, mode=mode,
                exact_gpu_oracle=True, native_cache_strides=list(key.stride()), repeats=args.repeats,
                torch_median_ms=statistics.median(times['torch']),
                triton_median_ms=statistics.median(times['triton']),
                triton_over_torch=statistics.median(times['triton'])/statistics.median(times['torch'])))
        del outputs, key, value, storage, table
    result = dict(schema='v29_paged_copy_component_v1', records=records,
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        body_wall_seconds=time.monotonic()-started,
        scope='Synthetic native-stride exact copy only; preallocated destination in both arms. Includes original intermediate allocation and host dispatch gaps. No request or accuracy claim.')
    args.new_report.parent.mkdir(parents=True,exist_ok=True)
    args.new_report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(cases=len(records),exact_gpu_oracle=True,
                         geometric_ratio=math.exp(sum(math.log(r['triton_over_torch']) for r in records)/len(records)))))


if __name__ == '__main__': main()
