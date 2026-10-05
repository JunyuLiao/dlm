"""Independent dense graph/hook/RNG boundary diagnostic; never formal timing.

Three aliases use separately frozen V28-compatible bindings. The original
worker, sampler and phase tracker are reused unchanged. No CUDA imports at
module import. RNG states/fingerprints stay in an explicitly private file.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import copy
from functools import wraps
import hashlib
import json
import re
from numbers import Integral
from pathlib import Path
from unittest.mock import patch

VARIANTS = {
    'default_dense': ('dense', 'default'),
    'piecewise_dense_nohook': ('dense', 'PIECEWISE'),
    'piecewise_native_hook': ('native', 'PIECEWISE'),
}
DIAGNOSTIC = dict(schema='v29_dense_reference_diagnostic_v1',
                  variants={key: dict(arm=value[0], requested_cudagraph_mode=value[1])
                            for key,value in VARIANTS.items()},
                  fresh_engine_runs_per_seed=2, one_timed_repeat=True,
                  runtime_graph_modes=True, rng_boundary_reads=True,
                  reseed_or_restore_rng=False, profiler_enabled=False,
                  equal_seed_implies_equal_random_trajectory=False,
                  performance_claim_allowed=False, quality_evaluated=False)


def make_spec(base, variant, indices, seeds=(28001, 28002)):
    """Return a new spec; never alter the caller's existing/frozen protocol."""
    if variant not in VARIANTS:
        raise ValueError('unknown dense reference variant')
    if not indices or any(type(x) is not int or x < 0 for x in indices) or len(set(indices)) != len(indices):
        raise ValueError('distinct nonnegative 32K indices required')
    if len(seeds) < 2 or any(type(x) is not int or x < 0 for x in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('at least two distinct engine seeds required')
    spec = copy.deepcopy(base)
    for key in ('profile_ordinal','cuda_event_leaf_scopes','cuda_event_interpretation','dense_reference','trace_export'):
        spec.pop(key,None)
    spec['cuda_events']=False
    spec.update(protocol_id='v28_v29_dense_reference_diag_20261003_'+variant,
                name='Independent graph/hook/RNG diagnostic; not formal timing',
                datasets={'longbench_v2_32k': dict(indices=list(indices), repeats=[0])},
                control_indices={'longbench_v2_32k': list(indices)},
                blocks=[dict(engine_seed=seed, repeats=[0]) for seed in seeds],
                dense_reference_variant=variant, dense_reference_diagnostic=copy.deepcopy(DIAGNOSTIC),
                diagnostic_only=True, performance_claim_allowed=False,
                launch_order=[dict(block=block, engine_repeat=repeat, variant=alias)
                              for block in range(len(seeds)) for repeat in range(2) for alias in VARIANTS])
    spec['arm_settings']['dense'] = dict(compilation_config=VARIANTS[variant][1],
                                        cudagraph_mode=VARIANTS[variant][1])
    spec['arm_settings']['native'] = dict(compilation_config='PIECEWISE', cudagraph_mode='PIECEWISE')
    return spec


def validate_spec(spec, variant, block, engine_repeat):
    if variant not in VARIANTS or spec.get('dense_reference_variant') != variant:
        raise ValueError('variant differs from frozen diagnostic')
    if spec.get('dense_reference_diagnostic') != DIAGNOSTIC or spec.get('diagnostic_only') is not True:
        raise ValueError('independent diagnostic settings must be frozen exactly')
    if spec.get('cuda_events') is not False or any(key in spec for key in ('profile_ordinal','cuda_event_leaf_scopes','trace_export')):
        raise ValueError('do not retain old profiler/trace settings')
    if spec.get('performance_claim_allowed') is not False:
        raise ValueError('diagnostic cannot authorize a formal speed claim')
    if not str(spec.get('protocol_id','')).startswith('v28_v29_dense_reference_diag_20261003_'):
        raise ValueError('new diagnostic protocol required')
    if type(engine_repeat) is not int or not 0 <= engine_repeat < 2:
        raise ValueError('fresh engine repeat outside frozen diagnostic')
    blocks = spec.get('blocks', [])
    seeds = [item.get('engine_seed') for item in blocks]
    if len(seeds) < 2 or any(type(x) is not int or x < 0 for x in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('at least two distinct engine seed blocks required')
    if type(block) is not int or not 0 <= block < len(blocks) or any(item.get('repeats') != [0] for item in blocks):
        raise ValueError('one timed repeat per frozen block required')
    datasets=spec.get('datasets', {})
    if set(datasets) != {'longbench_v2_32k'} or datasets['longbench_v2_32k'].get('repeats') != [0]:
        raise ValueError('diagnostic requires the frozen 32K inventory')
    indices=datasets['longbench_v2_32k'].get('indices', [])
    if not indices or len(set(indices)) != len(indices) or any(type(x) is not int or x < 0 for x in indices):
        raise ValueError('invalid frozen 32K inventory')
    if spec.get('control_indices') != {'longbench_v2_32k': indices}:
        raise ValueError('all references must warm and time the same complete inventory')
    arm, mode=VARIANTS[variant]
    expected=dict(compilation_config=mode,cudagraph_mode=mode)
    if spec.get('arm_settings', {}).get(arm) != expected:
        raise ValueError('requested graph mode differs from frozen variant')
    if spec.get('adapter_settings') != dict(lifecycle='request_clear',alias_splits=2):
        raise ValueError('keep the existing request_clear alias2 protocol')
    if spec.get('primary_receipt_method', {}).get('q_block') != 128 or spec.get('jit_event_receipts') is not True:
        raise ValueError('q128 and timed JIT receipts required')
    if spec.get('settings', {}).get('cpu_threads') != 1:
        raise ValueError('common OMP1 setting required')
    return arm


def mode_name(value):
    name=getattr(value, 'name', None)
    if name is None and type(value) is str:
        name=value
    if name not in ('NONE','FULL','PIECEWISE','FULL_AND_PIECEWISE','FULL_DECODE_ONLY'):
        raise ValueError('unrecognized resolved graph mode')
    return name


def classify_modes(preparations, snapshots, counts):
    """Pair submitted metadata with actual immutable async CPU output receipts.

    Caller must have completed the existing request device synchronization.
    No scheduler-retired ordinal, GPU phase read or early pinned-buffer read.
    """
    if not preparations or len(preparations) != len(snapshots):
        raise ValueError('runtime graph and actual execution coverage mismatch')
    rows=[]
    for ordinal,(prepared,snapshot) in enumerate(zip(preparations,snapshots)):
        draft,tokens,array=snapshot
        if prepared['draft'] != bool(draft) or prepared['tokens'] != tokens:
            raise ValueError('runtime graph and snapshot execution identities differ')
        values=array.tolist()
        if len(values) != 1 or not isinstance(values[0], Integral) or isinstance(values[0], bool) or not 0 <= values[0] <= tokens:
            raise ValueError('invalid actual sample-count snapshot')
        emitted=int(values[0])
        if not draft and emitted:
            raise ValueError('prefill unexpectedly emitted tokens')
        phase='prefill' if not draft else ('commit' if emitted else 'denoise')
        mode=mode_name(prepared['mode'])
        if mode not in ('NONE','FULL','PIECEWISE'):
            raise ValueError('initialization mode is not a runtime graph receipt')
        rows.append(dict(execution_ordinal=ordinal,phase=phase,cudagraph_mode=mode,
                         scheduled_tokens=int(tokens),emitted_token_count=emitted))
    total=Counter(row['phase'] for row in rows)
    expected=dict(prefill=counts['prefill_steps'],denoise=counts['denoising_forwards'],commit=counts['commit_forwards'])
    if dict(total) != {key:value for key,value in expected.items() if value}:
        raise ValueError('runtime phase counts differ from actual execution accounting')
    return rows


class BoundarySession:
    def __init__(self, read_rng):
        self.read_rng=read_rng
        self.tracker=None
        self.preparations=[]
        self.rows=[]
        self.private_rows=[]
        self.initialized_modes=[]
        self.before=None
        self.previous_after=None
        self.final_output_count=None
        self.final_finish_reason=None

    def start(self):
        if self.before is not None:
            raise ValueError('request boundary was not finalized')
        self.before=self.read_rng()  # Private digest; no reseed, restore or random draw.
        self.preparations=[]
        self.final_output_count=None
        self.final_finish_reason=None

    def output(self, outputs):
        if self.before is None:
            return
        for output in outputs:
            if output.finished:
                if self.final_output_count is not None or len(output.outputs)!=1:
                    raise ValueError('ambiguous final output count')
                self.final_output_count=len(output.outputs[0].token_ids)
                self.final_finish_reason=output.outputs[0].finish_reason

    def prepare(self, batch, mode, for_capture):
        tracker=self.tracker
        if self.before is None or tracker is None or tracker.rid is None or tracker.rid not in batch.req_ids or for_capture:
            return
        if batch.num_reqs != 1 or len(batch.req_ids) != 1:
            raise ValueError('batch-one runtime graph diagnostic required')
        if not isinstance(batch.num_tokens,Integral) or isinstance(batch.num_tokens,bool) or batch.num_tokens <= 0:
            raise ValueError('invalid runtime graph token count')
        self.preparations.append(dict(mode=mode_name(mode),draft=batch.num_draft_tokens > 0,tokens=batch.num_tokens))

    def finish(self, tracker, counts, start=None, end=None):
        if self.before is None or tracker is not self.tracker:
            raise ValueError('missing request/RNG boundary identity')
        if self.final_output_count is None:
            raise ValueError('missing final output length')
        execution=classify_modes(self.preparations,tracker._execution_snapshots,counts)
        after=self.read_rng()
        ordinal=len(self.rows)
        histogram={phase:dict(Counter(row['cudagraph_mode'] for row in execution if row['phase']==phase))
                   for phase in ('prefill','denoise','commit')}
        row=dict(request_ordinal=ordinal,runtime_graph_by_phase=histogram,execution=execution,
                 output_tokens=self.final_output_count,finish_reason=self.final_finish_reason,
                 prefill_s=counts.get('prefill_s'),decode_span_s=counts.get('decode_span_s'),
                 diagnostic_wall_s=None if start is None or end is None else end-start,
                 rng_before_equals_previous_after=None if self.previous_after is None else self.before==self.previous_after,
                 rng_before_equals_after=self.before==after,
                 actual_counts={key:counts[key] for key in ('denoising_forwards','commit_forwards','prefill_steps',
                       'scheduler_denoising_forwards','scheduler_commit_forwards','speculative_unused_denoising')})
        self.private_rows.append(dict(request_ordinal=ordinal,cuda_rng_before_private=self.before,
                                      cuda_rng_after_private=after))
        self.rows.append(row)
        self.before=None
        self.previous_after=after


@contextmanager
def instrument(session,panel,metrics,dg,vllm):
    """Process-local metadata/boundary wrappers; original functions always restored."""
    with ExitStack() as stack:
        original_prepare=dg.DiffusionGemmaModelState.prepare_attn
        @wraps(original_prepare)
        def prepare(state,input_batch,cudagraph_mode,*args,**kwargs):
            # The pinned runner calls for_capture by keyword. Reject ambiguous
            # positional capture options rather than accidentally counting capture.
            if len(args)>4:
                raise ValueError('unsupported positional prepare_attn capture signature')
            session.prepare(input_batch,cudagraph_mode,kwargs.get('for_capture',False))
            return original_prepare(state,input_batch,cudagraph_mode,*args,**kwargs)
        stack.enter_context(patch.object(dg.DiffusionGemmaModelState,'prepare_attn',prepare))
        original_add=panel.add_tracked_request
        @wraps(original_add)
        def add(engine,tracker,*args,**kwargs):
            session.start()
            result=original_add(engine,tracker,*args,**kwargs)
            session.tracker=tracker
            return result
        stack.enter_context(patch.object(panel,'add_tracked_request',add))
        original_finalize=metrics.PhaseTracker.finalize
        @wraps(original_finalize)
        def finalize(tracker,*args,**kwargs):
            result=original_finalize(tracker,*args,**kwargs)
            session.finish(tracker,result,*args[:2])  # Existing boundary sync already completed.
            return result
        stack.enter_context(patch.object(metrics.PhaseTracker,'finalize',finalize))
        original_llm=vllm.LLM
        @wraps(original_llm)
        def llm(*args,**kwargs):
            result=original_llm(*args,**kwargs)
            resolved=result.llm_engine.vllm_config.compilation_config.cudagraph_mode
            session.initialized_modes.append(mode_name(resolved))
            engine=result.llm_engine
            original_step=engine.step
            @wraps(original_step)
            def step(*step_args,**step_kwargs):
                outputs=original_step(*step_args,**step_kwargs)
                session.output(outputs)
                return outputs
            stack.enter_context(patch.object(engine,'step',step))
            return result
        stack.enter_context(patch.object(vllm,'LLM',llm))
        yield


def write_new(path,payload):
    with Path(path).open('x',encoding='utf-8') as file:
        json.dump(payload,file,indent=2,allow_nan=False)
        file.write('\n')


def validate_run(run_dir,binding,variant,block,engine_repeat):
    """Fail closed on CPU; return only a safe qualification status."""
    from scripts import v27_vllm_panel_run as panel,v28_vllm_qualify as worker
    root=Path(run_dir)
    spec=panel.read(binding['spec'])
    arm=validate_spec(spec,variant,block,engine_repeat)
    report=panel.read(root/'dense_reference_diagnostic.json')
    terminal=panel.read(root/'terminal.json')
    private=panel.read(root/'rng_boundaries.private.json')
    records=[json.loads(line) for line in (root/'records.jsonl').read_text().splitlines()]
    count=len(spec['datasets']['longbench_v2_32k']['indices'])
    worker.validate_jit_deltas(records,True)
    if (report.get('complete') is not True or report.get('diagnostic_validation_passed') is not True or
        report.get('source_commit')!=binding['deploy_commit'] or report.get('protocol_id')!=spec['protocol_id'] or
        report.get('variant')!=variant or report.get('engine_repeat')!=engine_repeat or
        report.get('engine_seed')!=spec['blocks'][block]['engine_seed'] or
        report.get('performance_claim_allowed') is not False or report.get('profiler_enabled') is not False or
        report.get('rng_reset_or_restore') is not False):
        raise ValueError('incomplete or mismatched diagnostic report')
    if terminal.get('complete') is not True or terminal.get('arm')!=arm or terminal.get('block')!=block or terminal.get('protocol_id')!=spec['protocol_id']:
        raise ValueError('original worker did not close successfully')
    monitor=report.get('jit_monitor_activation',{})
    required_monitor=dict(active=True,mode='warn',cute_hook_installed=True,
                          triton_hook_callable=True,activated_before_component_imports=True)
    if any(type(monitor.get(key)) is not type(expected) or monitor[key]!=expected
           for key,expected in required_monitor.items()):
        raise ValueError('active warn-mode pre-import CuTe/Triton monitor receipt required')
    modes=report.get('initialized_graph_modes',[])
    if len(modes)!=1 or mode_name(modes[0])!=modes[0] or variant!='default_dense' and modes[0]!='PIECEWISE':
        raise ValueError('missing or unexpected resolved engine graph configuration')
    rows=report.get('requests',[])
    secrets=private.get('requests',[])
    if len(rows)!=2*count or len(records)!=count or len(secrets)!=len(rows):
        raise ValueError('warm/timed/RNG receipt coverage mismatch')
    if private.get('private') is not True or private.get('not_for_publication') is not True:
        raise ValueError('RNG receipt must remain private')
    for index,(row,secret) in enumerate(zip(rows,secrets)):
        if row.get('request_ordinal')!=index or row.get('warm')!=(index<count) or secret.get('request_ordinal')!=index:
            raise ValueError('request ordinal/warm identity mismatch')
        if type(row.get('output_tokens')) is not int or row['output_tokens']<0:
            raise ValueError('missing warm/timed output length')
        if any(type(secret.get(key)) is not str or re.fullmatch('[0-9a-f]{64}',secret[key]) is None
               for key in ('cuda_rng_before_private','cuda_rng_after_private')):
            raise ValueError('missing private default-generator boundary state')
        expected_equal=None if index==0 else secret['cuda_rng_before_private']==secrets[index-1]['cuda_rng_after_private']
        if row.get('rng_before_equals_previous_after') is not expected_equal:
            raise ValueError('private/public boundary equality mismatch')
        if row.get('rng_before_equals_after') is not (secret['cuda_rng_before_private']==secret['cuda_rng_after_private']):
            raise ValueError('private/public RNG change receipt mismatch')
        execution=row.get('execution',[])
        actual=row['actual_counts']
        histogram={phase:dict(Counter(item['cudagraph_mode'] for item in execution if item['phase']==phase))
                   for phase in ('prefill','denoise','commit')}
        if row.get('runtime_graph_by_phase')!=histogram:
            raise ValueError('runtime graph histogram mismatch')
        if not execution or any(item.get('execution_ordinal')!=ordinal or item.get('phase') not in histogram or
                                item.get('cudagraph_mode') not in ('NONE','FULL','PIECEWISE')
                                for ordinal,item in enumerate(execution)):
            raise ValueError('incomplete actual execution mode receipt')
        expected_counts=dict(prefill=actual['prefill_steps'],denoise=actual['denoising_forwards'],commit=actual['commit_forwards'])
        if any(sum(histogram[phase].values())!=number for phase,number in expected_counts.items()):
            raise ValueError('runtime graph actual phase accounting mismatch')
        if variant!='default_dense' and any(item['cudagraph_mode']=='FULL' for item in execution):
            raise ValueError('explicit PIECEWISE reference unexpectedly executed FULL')
        if index>=count:
            rec=records[index-count]
            if (rec.get('denoise_forward_count')!=actual['denoising_forwards'] or
                rec.get('commit_forward_count')!=actual['commit_forwards'] or
                rec.get('prefill_steps')!=actual['prefill_steps'] or rec.get('output_tokens')!=row['output_tokens'] or
                rec.get('graph_captures_timed')!=0 or not rec.get('compilation_deltas') or
                any(type(value) is not int or value!=0 for value in rec['compilation_deltas'].values())):
                raise ValueError('original timed receipts disagree or timed compilation occurred')
    return dict(diagnostic_validation_passed=True,variant=variant,engine_seed=report['engine_seed'],
                engine_repeat=engine_repeat,warm_requests=count,timed_requests=count,
                initialized_graph_modes=modes,performance_claim_allowed=False,
                rng_scope='current-device default CUDA generator only; compiled/graph internal RNG not established')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding')
    parser.add_argument('--variant',choices=tuple(VARIANTS),required=True)
    parser.add_argument('--block',type=int,default=0)
    parser.add_argument('--engine-repeat',type=int,choices=(0,1),default=0)
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--print-spec',action='store_true')
    parser.add_argument('--validate-run',action='store_true',help='CPU-only closed receipt validation; no CUDA import')
    parser.add_argument('--index',type=int,nargs='+')
    parser.add_argument('--engine-seeds',type=int,nargs='+',default=None)
    args=parser.parse_args(argv)
    from scripts import v27_vllm_panel_run as panel
    if args.print_spec:
        base=panel.read(Path(__file__).resolve().parents[1]/'results/v28_20261002/specs/v28_q128_request_clear_seed4.json')
        if args.index is None:
            parser.error('--print-spec requires explicit selected public indices')
        spec=make_spec(base,args.variant,args.index,args.engine_seeds or [28001,28002])
        validate_spec(spec,args.variant,args.block,args.engine_repeat)
        print(json.dumps(spec,indent=2,allow_nan=False))
        return spec
    if args.validate_run:
        if not args.binding or args.run_dir is None:
            raise ValueError('closed diagnostic run and immutable binding required')
        binding=panel.read(args.binding); spec=panel.read(binding['spec'])
        panel.validate_binding(binding,spec)
        result=validate_run(args.run_dir,binding,args.variant,args.block,args.engine_repeat)
        print(json.dumps(result,allow_nan=False));return result
    if not args.binding or args.run_dir is None or args.run_dir.exists():
        raise ValueError('frozen binding and new diagnostic run directory required')
    if args.index is not None or args.engine_seeds is not None:
        raise ValueError('runtime inventory/seeds come only from the frozen binding')
    from scripts import v27_vllm_metrics as metrics,v28_vllm_qualify as worker
    binding=panel.read(args.binding)
    spec=panel.read(binding['spec'])
    panel.validate_binding(binding,spec)
    arm=validate_spec(spec,args.variant,args.block,args.engine_repeat)
    pinned={Path(path).resolve() for path in binding['files']}
    root=Path(__file__).resolve().parents[1]
    required=[Path(__file__).resolve(),root/'scripts/v29_vllm_cost_profile.py',root/'scripts/v29_component_timing.py']
    if any(path not in pinned for path in required):
        raise ValueError('binding must pin independent diagnostic and its validation/monitor helper sources')
    config=panel.read(binding['config'])
    worker.validate_frozen_inputs(binding,spec,config,128,'request_clear','legacy')
    cells=panel.read(binding['cells'])
    inventory=sorted((cell['dataset'],cell['index']) for cell in cells)
    expected=sorted(('longbench_v2_32k',index) for index in spec['datasets']['longbench_v2_32k']['indices'])
    if inventory!=expected:
        raise ValueError('binding cells differ from complete frozen warm/timed inventory')
    from scripts.v29_vllm_cost_profile import validate_32k
    validate_32k(binding,panel.read)
    # GPU/library imports only after immutable inputs and selected variant checks.
    import torch
    import vllm
    from vllm.utils import jit_monitor
    from triton import knobs
    from scripts.v29_component_timing import activate_monitor
    monitor_receipt=activate_monitor(jit_monitor,lambda:knobs.runtime.jit_post_compile_hook)
    import vllm.model_executor.models.diffusion_gemma as dg
    def rng_digest():
        return hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()
    session=BoundarySession(rng_digest)
    failure=None
    try:
        with instrument(session,panel,metrics,dg,vllm):
            worker.main(['--binding',args.binding,'--arm',arm,'--block',str(args.block),
                         '--run-dir',str(args.run_dir),'--query-block','128','--canvas-buffers','legacy','--mode','benchmark'])
        if len(session.initialized_modes)!=1 or len(session.rows)!=2*len(cells):
            raise ValueError('incomplete engine/request diagnostic coverage')
    except BaseException as error:
        failure=type(error).__name__
        raise
    finally:
        if args.run_dir.exists():
            report=dict(schema=DIAGNOSTIC['schema'],diagnostic_only=True,performance_claim_allowed=False,
                        quality_evaluated=False,complete=failure is None,diagnostic_validation_passed=failure is None,failure_type=failure,
                        source_commit=binding['deploy_commit'],protocol_id=spec['protocol_id'],variant=args.variant,
                        engine_seed=spec['blocks'][args.block]['engine_seed'],engine_repeat=args.engine_repeat,
                        initialized_graph_modes=session.initialized_modes,jit_monitor_activation=monitor_receipt,
                        requests=[dict(row,warm=index<len(cells)) for index,row in enumerate(session.rows)],
                        warm_request_count=len(cells),timed_request_count=len(cells),
                        rng_state_values_saved_private=True,rng_reset_or_restore=False,profiler_enabled=False,
                        rng_read_scope='torch.cuda.get_rng_state() default generator on current CUDA device only; not complete compiled/CUDAgraph internal RNG state',
                        instrumentation_overhead_unknown=True,
                        caveat='Same seed does not establish the same RNG state or trajectory; diagnostic W is not formal speed evidence.')
            write_new(args.run_dir/'dense_reference_diagnostic.json',report)
            write_new(args.run_dir/'rng_boundaries.private.json',dict(private=True,not_for_publication=True,requests=session.private_rows))
    return report


if __name__=='__main__':
    main()
