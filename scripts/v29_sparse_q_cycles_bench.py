"""Sparse in-place cycle Q diagnostic on an independent clone; no model integration.

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
from scripts.v29_component_timing import measure_components, vllm_compile_guard
from experiments.numerical_qk_reuse.v29_lse_merge import Alias2MappedMerge
from experiments.numerical_qk_reuse.v29_sparse_q_cycles import SparseQCycles, plan_cycles, amortization_receipt


def interpretation():
    return 'Sparse-cycle consumer potential only: same held order/support, natural fused reference, forward and inverse included; no online/request gain claim'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('private_need_dir')
    p.add_argument('new_report')
    p.add_argument('--repeats',type=int,default=32)
    p.add_argument('--softmax-scale',type=float,default=1.0,
                   help='Current Gemma GLOBAL scale is 1.0; historical scale is a separately labelled diagnostic')
    args=p.parse_args(argv)
    dest=Path(args.new_report)
    if dest.exists() or args.repeats < 4:raise ValueError('New report and at least four repeats required')
    if not math.isfinite(args.softmax_scale) or args.softmax_scale <= 0:
        raise ValueError('Finite positive softmax scale required')
    select_start=time.monotonic()
    selected=held.select_accepted(args.private_need_dir)
    offline_selection_s=time.monotonic()-select_start
    if len(selected)!=3 or {x[0] for x in selected}!={32768,65536,98304}:
        raise ValueError('Three frozen historical held bins required')
    with vllm_compile_guard() as (snapshot, activation):
        return run_components(args, selected, offline_selection_s, dest, snapshot, activation)


def run_components(args, selected, offline_selection_s, dest, snapshot, activation):
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
        owned_query=query.clone(memory_format=torch.preserve_format)
        orders={'natural':held.natural_order(need),'held':screened['orders']['gated']}
        cycle_pack_start=time.monotonic()
        cycle_plan=plan_cycles(orders['held'])
        cycle_cpu_pack_s=time.monotonic()-cycle_pack_start
        cycle_upload_start=time.monotonic()
        cycles=SparseQCycles(orders['held'],query.device)
        torch.cuda.synchronize()
        cycle_packing_and_upload_s=time.monotonic()-cycle_upload_start
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

        def partials(name, override=None):
            # The same table/used allocations as the existing generic adapter.
            x=override if override is not None else (query if name=='natural' else held.gather_query(query,maps['held']))
            qs=x.transpose(1,2).expand(2,-1,-1,-1)
            tables=table[None].expand(2,-1).contiguous()
            used=torch.full((2,),nk,device=query.device,dtype=torch.int32)
            return fwd(qs,kc,vc,softmax_scale=args.softmax_scale,causal=False,page_table=tables,
                seqused_k=used,block_sparse_tensors=splits[name],num_splits=1,return_lse=True)[:2]

        def standard(name):
            o,l=partials(name)
            weights=torch.softmax(l,dim=0).permute(0,2,1)[...,None]
            result=(o.float()*weights).sum(0,keepdim=True).to(o.dtype)
            return result if name=='natural' else held.scatter_output(result,maps[name])

        def fused(name):
            o,l=partials(name)
            return merge[name](o,l)

        def cycle_fused():
            # Owned clone only. On any exception this benchmark exits and discards
            # it; no model tensor is shared, and no corrupted clone is retried.
            cycles(owned_query)
            try:
                o,l=partials('held',owned_query)
                return merge['held'](o,l)
            finally:
                cycles(owned_query,inverse=True)

        functions={'natural_standard':lambda:standard('natural'),'natural_fused':lambda:fused('natural'),
            'held_fused_gather':lambda:fused('held'),'held_sparse_cycles':cycle_fused}
        errors={}
        qualification_start=time.monotonic()
        if not torch.isfinite(query).all().item():raise RuntimeError('Nonfinite input Q')
        cycles(owned_query)
        if not torch.equal(owned_query,held.gather_query(query,maps['held'])):raise RuntimeError('Cycle gather oracle failed; discard owned clone')
        cycles(owned_query,inverse=True)
        if not torch.equal(owned_query,query):raise RuntimeError('Cycle inverse oracle failed; discard owned clone')
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
                scores=query[0,head].float() @ k[:,head//(h//hk)].float().T*args.softmax_scale
                reference=scores.masked_fill(~mask[head],-torch.inf).softmax(-1) @ v[:,head//(h//hk)].float()
                if not torch.isfinite(reference).all().item():raise RuntimeError('Nonfinite masked IEEE FP32 oracle')
                max_abs=max(max_abs,(got[0,:,head].float()-reference).abs().max().item())
                max_ref=max(max_ref,reference.abs().max().item())
            errors[name]=held.qualified_error(max_abs,max_ref)
        torch.cuda.synchronize()
        first_use_and_oracles_wall_s=time.monotonic()-qualification_start
        if not torch.equal(owned_query,query):raise RuntimeError('Input restore failed before timing')
        builds_before=adapter.calls['split_list_builds']
        timing_evidence=measure_components(torch,functions,repeats=args.repeats,warm=8,snapshot=snapshot)
        medians={name:row['medians']['cuda_event_ms'] for name,row in timing_evidence['timings'].items()}
        if adapter.calls['split_list_builds']!=builds_before:raise RuntimeError('Timed split builds')
        torch.cuda.synchronize()
        if not torch.equal(owned_query,query):raise RuntimeError('Input restore failed after timing')
        saving_s=(medians['natural_fused']-medians['held_sparse_cycles'])/1000
        order_and_cycles_s=screen_compute_s+cycle_cpu_pack_s+cycle_packing_and_upload_s
        break_even_calls=None if saving_s<=0 else math.ceil(order_and_cycles_s/saving_s)
        amortization=amortization_receipt(order_and_cycles_s,medians['natural_fused'],medians['held_sparse_cycles'])
        records.append(dict(nominal_length_bin=length,samples=args.repeats,gpu_ms=medians,errors=errors,
            timing_evidence=timing_evidence,
            held_cycles_over_natural_fused=medians['held_sparse_cycles']/medians['natural_fused'],
            held_gather_over_natural_fused=medians['held_fused_gather']/medians['natural_fused'],
            cycle_statistics=cycle_plan.statistics(d,query.element_size()),
            cycle_cpu_pack_seconds=cycle_cpu_pack_s,cycle_packing_and_upload_seconds=cycle_packing_and_upload_s,
            input_exactly_restored_before_and_after_timing=True,
            order_and_cycle_construction_seconds=order_and_cycles_s,
            total_component_saving_seconds_per_call=saving_s,optimistic_break_even_calls=break_even_calls,
            amortization=amortization,
            construction_amortization_scope='Includes held CPU screen plus cycle packing/upload; excludes signal/transfer/map-list and uses optimistic constant saving. No valid online reuse lifetime measured.',
            fusion_only_natural_over_standard=medians['natural_fused']/medians['natural_standard'],
            cpu_order_screen_compute_seconds=screen_compute_s,first_split_build_wall_seconds=first_split_build_wall_s,
            first_map_list_split_build_wall_seconds=first_map_list_split_build_wall_s,
            first_use_and_oracles_wall_seconds=first_use_and_oracles_wall_s,
            construction_note='CPU screen on already loaded need; excludes signal generation, transfer and input I/O. Split build is for both arms, includes synchronization. Neither is hidden as online free work.',
            cpu_reproduced_same_frozen_order=True,required_support_covered=True,alias_splits=2,
            no_timed_oracle_or_d2h=True,no_timed_list_build=True,
            query_stride=list(query.stride()),k_stride=list(kc.stride())))
        del owned_query,cycles,cache,kc,vc,k,v,query,table,maps,merge,lists,splits,adapter,o,l,oracle,got,mask,scores,reference
    report=dict(schema='v29_sparse_q_cycles_component_v1',records=records,offline_selection_wall_seconds=offline_selection_s,
        monitor_activation=activation,
        software=dict(torch=torch.__version__,torch_cuda=torch.version.cuda,triton=triton.__version__),
        device=dict(name=torch.cuda.get_device_name(),compute_capability=list(torch.cuda.get_device_capability())),
        numerical_scope='FP32 stable softmax weights followed by weighted multiply/add; Triton FP fusion disabled. Tolerance-qualified merge, not bit-exact equality, sampler trajectory equality or model accuracy.',
        merge_fp_fusion=False,
        softmax_scale=args.softmax_scale,
        query_key_value_source='Synthetic BF16 tensors; actual held support only. Not a replay of model QKV or a request-causal cost explanation.',
        held_cycles_over_natural_fused_geomean=math.exp(statistics.mean(math.log(r['held_cycles_over_natural_fused']) for r in records)),
        interpretation=interpretation(),quality_evaluated=False,request_run=False,
        effective_support='unchanged historical held gate; full canvas kept; random physical pages',
        online_build_cost_measured=False,online_cost_limit='CPU reconstruction and split-build costs recorded; no online signal/validity lifetime measured. No free online construction or reuse assumption.',
        allocation_policy='Independent owned Q clone created before qualification; cycle arm includes forward+inverse in-place operations every call, unchanged alias2/table/used/output allocations and mapped LSE merge. Gather control unchanged.',
        timing_parameters=dict(repeats=args.repeats,warm=8,rotated_order=True),
        timing_semantics='CUDA events around complete Python component calls including cycle forward+inverse or Q gather, FA4 and matched merge/writeback; possible host dispatch waits. CPU selection and first construction separate; JIT/oracles before timing.',
        native_adapter_changed=False,fa4_q_loader_changed=False,
        kernel_status='CPU prepared only unless this explicit GPU script completes all oracles; no GPU qualification claimed by implementation',
        ownership_contract='Only benchmark-owned nonoverlapping clone; arbitrary model Q alias/lifetime unqualified',
        reuse_contract='R6 is method risk configuration, not proof of six valid order reuses. Changed Q does not license stale support/order. Actual same-order canvas lifetime must be independently measured.',
        amortization_gate='No positive total saving => no break-even. Positive consumer result still must pay online construction within measured valid reuse lifetime. Existing approximately one-second CPU search is not free.')
    with dest.open('x',encoding='utf-8') as out:json.dump(report,out,indent=2,allow_nan=False);out.write('\n')
    return report


if __name__=='__main__':main()
