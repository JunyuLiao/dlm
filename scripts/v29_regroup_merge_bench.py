"""Alias2 merge/writeback diagnostic; no adapter patch or request claim.

Natural identity-map fusion is the incremental reference. Held orders and search
parameters are unchanged. GPU oracles precede every timed arm. Private inputs;
anonymous report only. CPU import does not import torch or Triton.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import time
import numpy as np
from scripts import v28_regroup_held_bench as held
from scripts.v28_regroup_triton_bench import _time_functions
from experiments.numerical_qk_reuse.v29_lse_merge import Alias2MappedMerge


def interpretation():
    return 'incremental regroup compares held_fused only to natural_fused; no request or quality claim'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('private_need_dir')
    p.add_argument('new_report')
    p.add_argument('--repeats',type=int,default=32)
    args=p.parse_args(argv)
    dest=Path(args.new_report)
    if dest.exists() or args.repeats < 4:raise ValueError('New report and at least four repeats required')
    select_start=time.monotonic()
    selected=held.select_accepted(args.private_need_dir)
    offline_selection_s=time.monotonic()-select_start
    if len(selected)!=3 or {x[0] for x in selected}!={32768,65536,98304}:
        raise ValueError('Three frozen historical held bins required')
    import torch
    import triton
    from experiments.numerical_qk_reuse import v27_fa4 as fa
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    if hasattr(torch.backends.cuda.matmul,'fp32_precision'):
        torch.backends.cuda.matmul.fp32_precision='ieee'
    else:torch.backends.cuda.matmul.allow_tf32=False
    fwd=fa.load()
    generator=torch.Generator(device='cuda').manual_seed(2903)
    records=[]
    for length,need,screened in selected:
        h,q,pt=need.shape
        if (h,q)!=(16,256):raise ValueError('Gemma GLOBAL shape required')
        start=time.monotonic()
        rebuilt=held.screen_need(need,held.SEARCH)
        screen_compute_s=time.monotonic()-start
        if not rebuilt['accepted'] or not np.array_equal(rebuilt['orders']['gated'],screened['orders']['gated']):
            raise RuntimeError('Frozen held order reproduction failed')
        d,hk,page=512,2,64
        nk=pt*page+q
        cache=torch.randn(nk//page,hk,page,2*d,device='cuda',dtype=torch.bfloat16,generator=generator)
        kc,vc=cache.transpose(1,2).split(d,-1)
        table=torch.randperm(nk//page,device='cuda',generator=generator).to(torch.int32)
        k=kc[table.long()].reshape(nk,hk,d)
        v=vc[table.long()].reshape(nk,hk,d)
        query=torch.randn(q,h,d,device='cuda',dtype=torch.bfloat16,generator=generator).transpose(0,1)[None]
        orders={'natural':held.natural_order(need),'held':screened['orders']['gated']}
        construction_start=time.monotonic()
        maps={name:torch.from_numpy(order).to(device='cuda',dtype=torch.long) for name,order in orders.items()}
        merge={name:Alias2MappedMerge(maps[name]) for name in orders}
        lists={name:fa.block_sparse_tensors(torch.from_numpy(held.support_for_order(need,order))[None].to('cuda'),q_block=64) for name,order in orders.items()}
        adapter=VllmMethodAdapter(['full_attention'],arm='allkept',lifecycle='request_clear')
        if adapter.splits!=2:raise RuntimeError('Unchanged alias2 required')
        build_start=time.monotonic()
        splits={name:adapter._split(value) for name,value in lists.items()}
        torch.cuda.synchronize()
        first_split_build_wall_s=time.monotonic()-build_start
        first_map_list_split_build_wall_s=time.monotonic()-construction_start

        def partials(name):
            # The same table/used allocations as the existing generic adapter.
            x=query if name=='natural' else held.gather_query(query,maps['held'])
            qs=x.transpose(1,2).expand(2,-1,-1,-1)
            tables=table[None].expand(2,-1).contiguous()
            used=torch.full((2,),nk,device=query.device,dtype=torch.int32)
            return fwd(qs,kc,vc,softmax_scale=d**-.5,causal=False,page_table=tables,
                seqused_k=used,block_sparse_tensors=splits[name],num_splits=1,return_lse=True)[:2]

        def standard(name):
            o,l=partials(name)
            weights=torch.softmax(l,dim=0).permute(0,2,1)[...,None]
            result=(o.float()*weights).sum(0,keepdim=True).to(o.dtype)
            return result if name=='natural' else held.scatter_output(result,maps[name])

        def fused(name):
            o,l=partials(name)
            return merge[name](o,l)

        functions={'natural_standard':lambda:standard('natural'),'natural_fused':lambda:fused('natural'),
            'held_standard_scatter':lambda:standard('held'),'held_fused':lambda:fused('held')}
        errors={}
        qualification_start=time.monotonic()
        for name in orders:
            o,l=partials(name)
            if not torch.isfinite(o).all().item() or torch.isnan(l).any().item() or torch.isposinf(l).any().item() or (~torch.isfinite(l)).all(0).any().item():
                raise RuntimeError('Invalid alias2 partial or LSE before timing')
            oracle=(o.float()*torch.softmax(l,dim=0).permute(0,2,1)[...,None]).sum(0,keepdim=True).to(o.dtype)
            if name=='held':oracle=held.scatter_output(oracle,maps[name])
            got=merge[name](o,l)
            if not torch.isfinite(got).all().item():raise RuntimeError('Nonfinite mapped merge')
            errors[name+'_merge']=held.qualified_error((got.float()-oracle.float()).abs().max().item(),oracle.float().abs().max().item(),tolerance=.01)
        for name,fn in functions.items():
            got=fn()
            if not torch.isfinite(got).all().item():raise RuntimeError('Nonfinite complete consumer')
            frame='held' if name.startswith('held') else 'natural'
            mask=torch.from_numpy(held.original_row_support(need,orders[frame])).to('cuda').repeat_interleave(page,-1)
            max_abs,max_ref=0.,0.
            for head in range(h):
                scores=query[0,head].float() @ k[:,head//(h//hk)].float().T*d**-.5
                reference=scores.masked_fill(~mask[head],-torch.inf).softmax(-1) @ v[:,head//(h//hk)].float()
                if not torch.isfinite(reference).all().item():raise RuntimeError('Nonfinite masked IEEE FP32 oracle')
                max_abs=max(max_abs,(got[0,:,head].float()-reference).abs().max().item())
                max_ref=max(max_ref,reference.abs().max().item())
            errors[name]=held.qualified_error(max_abs,max_ref)
        torch.cuda.synchronize()
        first_use_and_oracles_wall_s=time.monotonic()-qualification_start
        builds_before=adapter.calls['split_list_builds']
        medians=_time_functions(torch,functions,args.repeats)
        if adapter.calls['split_list_builds']!=builds_before:raise RuntimeError('Timed split builds')
        records.append(dict(nominal_length_bin=length,samples=args.repeats,gpu_ms=medians,errors=errors,
            held_fused_over_natural_fused=medians['held_fused']/medians['natural_fused'],
            fusion_only_natural_over_standard=medians['natural_fused']/medians['natural_standard'],
            cpu_order_screen_compute_seconds=screen_compute_s,first_split_build_wall_seconds=first_split_build_wall_s,
            first_map_list_split_build_wall_seconds=first_map_list_split_build_wall_s,
            first_use_and_oracles_wall_seconds=first_use_and_oracles_wall_s,
            construction_note='CPU screen on already loaded need; excludes signal generation, transfer and input I/O. Split build is for both arms, includes synchronization. Neither is hidden as online free work.',
            cpu_reproduced_same_frozen_order=True,required_support_covered=True,alias_splits=2,
            no_timed_oracle_or_d2h=True,no_timed_list_build=True,
            query_stride=list(query.stride()),k_stride=list(kc.stride())))
        del cache,kc,vc,k,v,query,table,maps,merge,lists,splits,adapter,o,l,oracle,got,mask,scores,reference
    report=dict(schema='v29_lse_merge_component_v1',records=records,offline_selection_wall_seconds=offline_selection_s,
        software=dict(torch=torch.__version__,torch_cuda=torch.version.cuda,triton=triton.__version__),
        device=dict(name=torch.cuda.get_device_name(),compute_capability=list(torch.cuda.get_device_capability())),
        numerical_scope='FP32 stable softmax weights followed by weighted multiply/add; Triton FP fusion disabled. Tolerance-qualified merge, not bit-exact equality, sampler trajectory equality or model accuracy.',
        merge_fp_fusion=False,
        held_fused_over_natural_fused_geomean=math.exp(statistics.mean(math.log(r['held_fused_over_natural_fused']) for r in records)),
        interpretation=interpretation(),quality_evaluated=False,request_run=False,
        effective_support='unchanged historical held gate; full canvas kept; random physical pages',
        online_build_cost_measured=False,online_cost_limit='CPU reconstruction and split-build costs recorded; no online signal/validity lifetime measured. No free online construction or reuse assumption.',
        allocation_policy='same Q gather as historical; same table/used allocations; all routes allocate output; fused routes remove separate merged output and held O scatter',
        timing_parameters=dict(repeats=args.repeats,warm=8,rotated_order=True),
        timing_semantics='CUDA events around complete Python component calls including Q gather, FA4 and merge/writeback; possible host dispatch waits. CPU selection and first construction separate; JIT/oracles before timing.',
        native_adapter_changed=False,fa4_q_loader_changed=False)
    with dest.open('x',encoding='utf-8') as out:json.dump(report,out,indent=2,allow_nan=False);out.write('\n')
    return report


if __name__=='__main__':main()
