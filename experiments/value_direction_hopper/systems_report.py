"""Consolidate the systems evidence without turning partial tests into claims."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def csv_write(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)


def verify_final(root):
    """Independent score/settings audit, in addition to the shard-count audit."""
    from .experiment import shard_path,validate_shard
    from experiments.diffusion_gemma_ruler8k_jl import score
    contract=load(root/'configuration.json');manifest=load(root/'manifest.json')
    violations=[];missing=[];count=0;metadata=[];layercounts=defaultdict(lambda:dict(eligible=0,skipped=0,calls=0))
    expected_local={i for i in range(30) if i%6!=5}
    for row in manifest:
        for method in contract['methods']:
            path=shard_path(root,method,row)
            if not path.exists():missing.append(f'{method}/{row["id"]}');continue
            record=load(path)
            try:
                validate_shard(record,row,contract,method)
                rescored=score(row,record['prediction'])
                if abs(rescored-record['score'])>1e-10:raise ValueError('Official rescoring mismatch')
                meta=record['metadata']
                metadata.append((meta['denoising_configuration'],meta['sampling'],meta['thinking'],meta['native_canvas_length']))
                if meta['thinking'] or meta['native_canvas_length']!=256:raise ValueError('Changed native decoding configuration')
                if len(record['completion_tokens'])>row['generation_budget']:raise ValueError('Generation budget exceeded')
                if method!='native_dense':
                    raw=Path(record['routing_path'])
                    if hashlib.sha256(raw.read_bytes()).hexdigest()!=record['routing_sha256']:raise ValueError('Routing hash mismatch')
                    with gzip.open(raw,'rt') as f:routing=json.load(f)
                    coverage=defaultdict(set)
                    for s in routing:
                        coverage[s['step']].add((s['layer'],s['head']))
                        expected='local' if s['layer'] in expected_local else 'global'
                        if s['attention_type']!=expected:raise ValueError('Incorrect local/global classification')
                        if sum(s[p+'_skipped'] for p in ('prefix','canvas','boundary'))!=s['skipped']:raise ValueError('Skipped-region accounting mismatch')
                        key=(method,s['layer'],s['head'],s['attention_type'])
                        for field in ('eligible','skipped'):layercounts[key][field]+=s[field]
                        layercounts[key]['calls']+=1
                    wanted={(layer,head) for layer in range(30) for head in range(16)}
                    if any(x!=wanted for x in coverage.values()):raise ValueError('Missing layer/head coverage in a denoising step')
                    if not coverage:raise ValueError('No routing coverage')
                count+=1
            except Exception as exc:violations.append(f'{path}: {exc!r}')
    if metadata and any(item!=metadata[0] for item in metadata):violations.append('Unrelated decoding settings differ across methods/samples')
    layerrows=[dict(method=k[0],layer=k[1],head=k[2],attention_type=k[3],**v,
               physical_sparsity=v['skipped']/v['eligible'] if v['eligible'] else None) for k,v in sorted(layercounts.items())]
    return dict(passed=not missing and not violations,completed=count,expected=len(manifest)*len(contract['methods']),
        missing=missing,violations=violations,checks=['official rescoring','identical denoising settings','budgets','every step layer/head coverage','local/global classification','region counts','routing SHA256']),layerrows


def run(bundle,final,output):
    output.mkdir(parents=True,exist_ok=True)
    summary=load(final/'summary.json')
    if summary is None:raise ValueError('Generate the raw final report first')
    audit,layers=verify_final(final)
    tests=[]
    for name in ('kernel_tests_frozen.xml','kernel_tests_aten.xml'):
        path=bundle/name
        if path.exists():
            doc=ET.parse(path).getroot();suites=[doc] if doc.tag=='testsuite' else list(doc.iter('testsuite'))
            tests.append(dict(file=name,**{key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ('tests','errors','failures','skipped')}))
    micro=load(bundle/'microbench_native_v1.json')
    qualification=load(bundle/'qualification_inline_v1.json')
    resources=load(bundle/'resource_limits_v1.json')
    projection=load(bundle/'projection_fused_v1.json')
    variant=load(bundle/'float_wgmma_comparison_v1.json')
    fair_controls=load(bundle/'microbench_tma_controls_v2.json')
    barrier_variant=load(bundle/'ptx_cluster_comparison_v1.json')
    mass=[]
    if qualification:
        for kind in ('whole','local','global'):
            cases=[r for r in qualification['cases'] if r['threshold']!='-inf' and (kind=='whole' or r['kind']==kind)]
            n=sum(r['valid_query_rows'] for r in cases);e=sum(r['dense_output_sq'] for r in cases)
            mass.append(dict(group=kind,retained_mass=sum(r['retained_mass_sum'] for r in cases)/n if n else None,
                valid_query_rows=n,operator_relative_l2=math.sqrt(sum(r['operator_error_sq'] for r in cases)/e) if e else None,
                eligible=sum(r['eligible'] for r in cases),skipped=sum(r['skipped'] for r in cases),
                scope=qualification['coverage']))
    sanitized={name:(bundle/name).read_text().splitlines()[-1] for name in ('memcheck_inline_v1.log','racecheck_inline_aten_v3.log','synccheck_frozen_v1.log','racecheck_ptx_cluster_v1.log') if (bundle/name).exists()}
    evidence=dict(final=str(final.resolve()),final_summary=summary['summary'],comparisons=summary['comparisons'],
        final_completion_audit=summary['audit'],independent_audit=audit,tests=tests,sanitizers=sanitized,
        microbenchmark=micro,resource_limits=resources,shared_state=qualification,shared_state_aggregates=mass,
        projection=projection,float_wgmma_variant=variant,matched_tma_controls=fair_controls,cluster_barrier_variant=barrier_variant,
        trt_static=load(bundle/'trt_engine_smoke_v3.json'),trt_dynamic=load(bundle/'trt_dynamic_v5.json'),
        production_ready=False,
        remaining=['no established speedup over matched FlashAttention','native local dense/reference mask mismatch',
                   'full DiffusionGemma TensorRT-LLM model conversion not implemented; attention plugin only',
                   'H100 performance counters unavailable (ERR_NVGPUCTRPERM)',
                   'same-logit FP32 state close but independent BF16 QK/online PV are not bitwise identical'])
    (output/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    csv_write(output/'layer_head_counts.csv',layers);csv_write(output/'shared_state_metrics.csv',mass)
    if micro:csv_write(output/'kernel_latencies.csv',micro['cases'])
    if projection:csv_write(output/'projection_latencies.csv',projection['cases'])
    if resources:csv_write(output/'resource_limits.csv',resources['cases'])
    if variant:csv_write(output/'float_wgmma_variant.csv',variant['cases'])
    if fair_controls:csv_write(output/'matched_tma_controls.csv',fair_controls['cases'])
    if barrier_variant:csv_write(output/'cluster_barrier_variant.csv',barrier_variant['cases'])
    def percent(x):return '—' if x is None else f'{100*x:.2f}'
    lines=['# H100 value-direction-aware attention: systems evaluation','',
        '**Status: engineering implementation and measured evaluation, not a production-readiness or FlashAttention-speedup claim.**','',
        '## End-to-end RULER4K130','',
        f"Completed outputs: {summary['audit']['completed']}/{summary['audit']['expected']}; independent score/settings audit: {audit['passed']}.",'',
        '|Method|Official accuracy %|Whole sparsity %|Global %|Local %|Generation wall s|Dense/sparse wall ratio|Denoising steps|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['summary']:
        lines.append(f"|{r['method']}|{percent(r['accuracy'])}|{percent(r['whole_sparsity'])}|{percent(r['global_sparsity'])}|{percent(r['local_sparsity'])}|{r['generation_seconds']:.2f}|{r.get('native_dense_wall_speedup',float('nan')):.3f}×|{r['denoising_steps']}|")
    lines+=['','Same130 cached prompts,13tasks×10; official budgets, seed42, pinned model/tokenizer, BF16, native256-token canvas and native denoising/temperature schedule. Gaussian32 matrices use seed1729. Frozen logτ(local/global)=−0.18442977964878082/−2.2409701347351074. BLASST uses λ=exp(log_scale)/valid_KV_length with frozen log_scale(local/global)=9.403833801578752/8.18658980427153; the prior calibration allowed λ>1. No final-score tuning.','',
        'Sparsity is Σskipped eligible128×64 tiles/Σeligible tiles. Denoising length can change after numerical or sparse output divergence, so equal per-step speed does not imply equal generation latency. Warmup, bindings and post-generation metric reduction are excluded; generation includes projection/cache refresh and host scheduling. Historical cached timings are not used as contemporary speed baselines.','',
        '## Kernel comparisons','']
    if micro:
        lines+=['CUDA-event graph timing, identical warmup, cached sketches. Measurements include neither dense diagnostic replay nor the profiler.','',
            '|Layer/type|Method|Actual sparsity %|Kernel µs|','|---|---|---:|---:|']
        for r in micro['cases']:lines.append(f"|{r['layer']}/{r['kind']}|{r['method']}|{percent(r.get('physical_sparsity'))}|{r['us']:.2f}|")
        if micro['failures']:lines+=['','Unavailable controls are preserved in evidence.json: '+json.dumps(micro['failures'])]
    else:lines+=['Microbenchmark sweep is not yet complete; do not infer performance from physical sparsity.']
    if variant:
        lines+=['','### Compiler scheduling ablation (not substituted into the final run)','',
            'Float-typed WGMMA accumulator constraints plus operand fences eliminate20 compiler serialization warnings. All six sampled native cases have bitwise-identical masks, retained state and outputs to the frozen binary. The local sparse kernel improves approximately8%, while global kernels regress approximately1%; the final frozen binary/results remain unchanged.','',
            '|Layer|logτ|Frozen µs|Float-operand µs|Ratio|','|---:|---:|---:|---:|---:|']
        for r in variant['cases']:lines.append(f"|{r['layer']}|{r['threshold']}|{r['base_us']:.2f}|{r['variant_us']:.2f}|{r['speedup']:.3f}×|")
    if fair_controls:
        lines+=['','### Matched-loader controls (separate post-final experiment)','',
            'The initial BLASST control used cp.async while value-aware used TMA. A separately tested extension gives dense and BLASST the same TMA loader and float-operand instructions. These are our BLASST-compatible kernels, not the unmodified upstream artifact. Thresholds and physical tile rules are unchanged.','',
            '|Layer|Dense TMA µs|BLASST TMA µs|Value TMA µs|BLASST sparsity %|Value sparsity %|','|---:|---:|---:|---:|---:|---:|']
        for layer in (0,5,29):
            group={r['method']:r for r in fair_controls['cases'] if r['layer']==layer}
            dense,bl,val=(group[n] for n in ('our_dense_tma','calibrated_blasst_tma','value_overlap_tma'))
            lines.append(f"|{layer}|{dense['us']:.2f}|{bl['us']:.2f}|{val['us']:.2f}|{percent(bl['physical_sparsity'])}|{percent(val['physical_sparsity'])}|")
    if barrier_variant:
        lines+=['','An explicit PTX release/acquire cluster barrier was also tested under racecheck. It preserves all six outputs bitwise on the sampled states but changes sparse kernel latency by less than0.2%; it is retained only as an experimental control, not selected as a meaningful improvement.']
    lines+=['','## Scheduling and work actually omitted','',
        'Two Hopper CTAs each own64 query rows and vote together for one128×64 physical tile. The router warpgroup computes QK, block softmax and FP32 Gaussian32 PZ; PV consumer warpgroups operate on original BF16 V. Double-buffered probabilities and rescaling factors allow QK(next) and PZ/routing(next) to overlap retained PV(previous). The next vote rendezvous protects buffer lifetime. Skips do not commit retained LSE/projected state, load original V or execute full-dimensional PV. K and Z are still loaded, and QK, softmax, PZ and tile voting still run.','',
        'This dependency is stricter than BLASST: BLASST can decide from block maxima before softmax; the centered value router cannot. Full PV from the previous retained tile can overlap, but the current centered decision depends on the previous retained projected output. Delaying or batching those decisions would change the algorithm and is not done.','',
        'TMA loads Q/K; cp.async loads sketches and retained V. Register-fed TF32x3 performs the rank32 projected product, with FP32 retained state. The main implementation fixes rank32; the separately labeled Triton control supports other ranks. No dense full-dimensional PV is used for routing.','',
        '## Numerical and integration qualification','',
        f"Tests: {json.dumps(tests)}. Sanitizers: {json.dumps(sanitized)}.",'',
        'Independent native BF16 dot reductions differ rarely from Torch/cuBLAS. The original strict absolute-state gate remains failed in the saved native profiles. A separate same-logit audit isolates router arithmetic: zero physical-mask disagreements on sampled native states; state/LSE tolerances are explicit. Online BF16 PV is mathematically the same renormalized operator but not bitwise the historical final-normalized-probability GEMM. Complete generation comparisons, not local error alone, determine downstream agreement.','',
        'TensorRT-LLM1.3.0rc6/TensorRT10.14 tests build and serialize the V3 attention plugin, reload it, and check all six outputs against the CUDA harness. Static tests cover nondefault streams/CUDA graphs; dynamic tests cover D256/D512, batch1/2 and partial tiles. This is real attention-plugin support, not a claim that the full diffusion model runs through TensorRT-LLM. The installed container Transformers version does not implement DiffusionGemma. Native end-to-end runs use the repository HF adapter and the C++/ATen bridge.','',
        '## Baseline and profiling limitations','',
        'The installed original SDPA path ignores its sliding-window keyword when no explicit mask is passed (its encoder cache is already clipped). The frozen experimental sparse reference adds a per-query local window. Both semantics are preserved and labeled; unpruned parity with original native local attention cannot be claimed. Matched-mask SDPA and matched-window Hopper FlashAttention controls isolate this difference. Installed Hopper FlashAttention rejects the global D512 head dimension. A global SDPA fallback win is not a demonstrated FA3 win.','',
        'Nsight Compute hardware counters are permission-blocked. CUDA timestamps report stage spans; overlapping spans must not be summed as total kernel latency. CUDA occupancy queries report resource-limited upper bounds, not achieved occupancy, bandwidth, Tensor Core utilization or barrier stall counters. Nsight Systems traces, where present, measure launch/scheduling behavior only.','',
        'The captured native workload launches32 two-CTA clusters (64CTAs on132SMs). Resource queries allow only one CTA/SM: D256 consumes255 registers/thread and131,712 shared bytes/CTA; D512 consumes168 registers/thread,230,016 shared bytes/CTA and432 local bytes/thread. The four-CTA split-PV alternative admits only30 active clusters, so its32-cluster native grid needs at least two waves. These are measured resource limits, not achieved-occupancy counters.','',
        'Nsight Systems confirms native global SDPA uses FP32 GEMMs/softmax rather than FlashAttention. In the profiled30-call batches, local host ranges were8.28ms through ctypes versus5.24ms through ATen, with approximately the same165µs GPU kernel. Projection refresh kernels themselves average~8.4µs plus~1.7µs for reference reduction; host dispatch and graph reuse therefore matter. These traced numbers are diagnostic, not substituted for uninstrumented benchmark latency.','',
        '## Shared-state retained mass and operator error','',
        '|Group|Retained mass %|Relative operator L2|Valid query rows|','|---|---:|---:|---:|']
    for r in mass:lines.append(f"|{r['group']}|{percent(r['retained_mass'])}|{r['operator_relative_l2']:.6f}|{r['valid_query_rows']}|")
    lines+=['','Coverage: one previously examined development prompt, first denoising step, layers0/5/29 and all16 heads. Aggregates use summed probability mass/valid rows and sqrt(Σsquared operator error/Σsquared dense output). These are not all-layer or final-generation-wide mass estimates.','',
        '## Bottom line','']
    value=next((r for r in summary['summary'] if r['method']=='kernel_gaussian32'),None)
    if value:
        speed=value.get('native_dense_wall_speedup')
        lines += [f"Measured native dense/value-aware generation wall ratio: {speed:.3f}× at {100*value['whole_sparsity']:.2f}% physical sparsity. "+
                  ('This does not establish a production or matched-FA win; baseline/mask and numerical qualifications above still apply.' if speed>=1 else 'The present implementation is slower end to end. Skipping PV has not hidden enough router/launch work, and changed denoising trajectories add another cost. A positive systems-speedup claim is unsupported.')]
    lines+=['','All unsuccessful variants, compiler warnings, failed numerical gates and prior outputs are retained. Final paper claims should not describe this kernel as production-ready until the remaining gates in evidence.json are resolved.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if micro:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for axis,kind in zip(axes,('local','global')):
            cases=[r for r in micro['cases'] if r['kind']==kind and r['layer'] in (0,5)]
            axis.barh([r['method'] for r in cases],[r['us'] for r in cases])
            axis.set(xlabel='Kernel latency (µs)',title=kind,xlim=(0,None))
            axis.tick_params(axis='y',labelsize=7)
        fig.tight_layout();fig.savefig(output/'native_kernel_latencies.png',dpi=160);plt.close(fig)
    sweep=load(bundle/'length_sweep_v1.json')
    if sweep:
        csv_write(output/'length_sparsity_latency.csv',sweep['cases'])
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for axis,width in zip(axes,(256,512)):
            for length in sorted({r['keys'] for r in sweep['cases']}):
                cases=[r for r in sweep['cases'] if r['width']==width and r['keys']==length]
                axis.plot([100*r['physical_sparsity'] for r in cases],[r['kernel_us'] for r in cases],marker='o',label=f'K={length}')
            axis.set(xlabel='Measured physical sparsity (%)',ylabel='Kernel µs',title=f'Synthetic D={width}',yscale='log')
            axis.legend(fontsize=7)
        fig.tight_layout();fig.savefig(output/'synthetic_sparsity_latency.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(independent_audit=audit['passed'],completed=audit['completed'],report=str(output/'report.md'))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--final',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.bundle,a.final,a.output)
