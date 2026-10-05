"""Separate synchronization control and opt-in first-sample trace diagnostics.

Reuses the frozen V29 benchmark/phase accounting. Never repairs old runs or
changes native sampling. Trace values, tokens, RNG bytes and their hashes are
strictly private; public receipts contain only counts. Tracing is an invasive
diagnostic, not a performance panel or an unchanged-asynchrony experiment.
"""
from contextlib import ExitStack, contextmanager
import copy
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch

from scripts import v29_dense_reference_diag as base

VARIANTS = dict(base.VARIANTS, piecewise_sync_only=('dense', 'PIECEWISE'))
STATE_FIELDS = ('canvas', 'argmax_canvas', 'step_tensor', 'is_encoder_phase',
                'confident_tensor', 'sc_embeds', 'history', 'history_len_tensor')


def contract(trace_first_sample):
    value = copy.deepcopy(base.DIAGNOSTIC)
    value.update(schema='v30_dense_first_divergence_v1',
                 variants={k:dict(arm=a,requested_cudagraph_mode=m) for k,(a,m) in VARIANTS.items()},
                 sync_only_control=True, first_sample_trace=bool(trace_first_sample),
                 trace_scope='first canvas initialization and first compiled sample of each warm/timed request' if trace_first_sample else 'request boundaries only; no extra per-step tracing',
                 extra_synchronization_from_trace=bool(trace_first_sample),
                 trace_proves_complete_compiled_rng_state=False)
    return value


class TraceState:
    def __init__(self, session, save_tensor, rng_digest, enabled):
        self.session, self.save_tensor, self.rng_digest = session, save_tensor, rng_digest
        self.enabled = enabled
        self.requests = {}
        self.sync_only_reads = 0

    def active(self):
        return self.session.before is not None

    def request(self):
        ordinal = len(self.session.rows)
        return self.requests.setdefault(ordinal, dict(request_ordinal=ordinal, init=[], sample=[]))

    def save(self, label, tensors):
        row = self.request()
        return dict(rng=self.rng_digest(), tensors={name:self.save_tensor(row['request_ordinal'],label,name,x) for name,x in tensors.items()})


def wrap_init(original, trace):
    @wraps(original)
    def initialize(states, slots, *args, **kwargs):
        enabled = trace.enabled and trace.active()
        if enabled:
            row = trace.request()
            if row['init']:
                raise ValueError('More than one request canvas initialization')
            before = trace.save('init_before', {'slots':slots})
        result = original(states, slots, *args, **kwargs)
        if enabled:
            row['init'].append(dict(before=before,after=trace.save('init_after', {'canvas':states.canvas[slots]})))
        return result
    return initialize


def wrap_sample(original, trace):
    signature = inspect.signature(original)
    @wraps(original)
    def sample(*args, **kwargs):
        enabled = trace.enabled and trace.active() and not trace.request()['sample']
        if enabled:
            values = signature.bind(*args, **kwargs).arguments
            required = {'logits','decode_slots',*STATE_FIELDS}
            if not required <= values.keys():
                raise ValueError('Installed sampler signature differs from reviewed source')
            slots = values['decode_slots']
            tensors = {name:values[name][slots] for name in STATE_FIELDS}
            tensors.update(logits=values['logits'], decode_slots=slots)
            tensors.update({name:values[name] for name in ('decode_idx','all_slots','valid_canvas_len') if name in values})
            before = trace.save('sample_before', tensors)
        result = original(*args, **kwargs)
        if enabled:
            after = trace.save('sample_after', {name:values[name][slots] for name in STATE_FIELDS})
            trace.request()['sample'].append(dict(before=before,after=after))
        return result
    return sample


def sync_only_prepare(original, trace, torch_module):
    @wraps(original)
    def prepare(state,input_batch,cudagraph_mode,block_tables,slot_mappings,attn_groups,kv_cache_config,for_capture=False,ubatch_idx=0):
        if trace.active() and not for_capture and input_batch.num_reqs == 1:
            slot = int(input_batch.idx_mapping_np[0])
            st = state.diffusion_states
            # Exactly the adapter's metadata D2H operation, with its values
            # discarded. No observer, buffers, routing or attention replacement.
            torch_module.stack([st.is_encoder_phase[slot].long(),st.step[slot].long(),input_batch.seq_lens[0].long()]).tolist()
            trace.sync_only_reads += 1
        return original(state,input_batch,cudagraph_mode,block_tables,slot_mappings,attn_groups,kv_cache_config,for_capture=for_capture,ubatch_idx=ubatch_idx)
    return prepare


@contextmanager
def base_contract(trace_first_sample):
    with patch.object(base,'VARIANTS',VARIANTS), patch.object(base,'DIAGNOSTIC',contract(trace_first_sample)):
        yield


def make_spec(original, variant, indices, seeds=(28001,28002), trace_first_sample=False):
    with base_contract(trace_first_sample):
        spec = base.make_spec(original,variant,indices,seeds)
    suffix = '_v30_first_sample' if trace_first_sample else '_v30_sync_control'
    spec['protocol_id'] += suffix
    spec['name'] = 'Separate dense synchronization/first-divergence diagnostic; never formal timing'
    return spec


def validate_private_trace(trace, expected_requests):
    if set(trace.requests) != set(range(expected_requests)):
        raise ValueError('Missing request trace')
    for row in trace.requests.values():
        if len(row['init']) != 1 or len(row['sample']) != 1:
            raise ValueError('Need exactly one init and first sample per request')
    return True


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=list(VARIANTS),required=True)
    p.add_argument('--trace-first-sample',action='store_true')
    p.add_argument('--validate-run',action='store_true')
    p.add_argument('--binding',required=True);p.add_argument('--block',type=int,required=True)
    p.add_argument('--engine-repeat',type=int,required=True);p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args(argv)
    binding=json.loads(Path(args.binding).read_text());spec=json.loads(Path(binding['spec']).read_text())
    if spec.get('dense_reference_diagnostic') != contract(args.trace_first_sample):
        raise ValueError('Trace mode differs from frozen independent protocol')
    suffix='_v30_first_sample' if args.trace_first_sample else '_v30_sync_control'
    if not spec.get('protocol_id','').endswith(suffix):raise ValueError('Distinct V30 protocol required')
    if str(Path(__file__).resolve()) not in {str(Path(x).resolve()) for x in binding['files']}:
        raise ValueError('Bind the V30 diagnostic source')
    if args.validate_run:
        with base_contract(args.trace_first_sample):
            proof=base.main(['--binding',args.binding,'--variant',args.variant,'--block',str(args.block),
                            '--engine-repeat',str(args.engine_repeat),'--run-dir',str(args.run_dir),'--validate-run'])
        receipt=json.loads((args.run_dir/'first_divergence_receipt.json').read_text())
        if (receipt.get('variant')!=args.variant or receipt.get('source_commit')!=binding['deploy_commit']
                or receipt.get('first_sample_trace') is not args.trace_first_sample
                or receipt.get('requests')!=proof['warm_requests']+proof['timed_requests']
                or receipt.get('performance_claim_allowed') is not False):
            raise ValueError('New diagnostic receipt identity mismatch')
        if args.trace_first_sample:
            manifest=json.loads((args.run_dir/'first_sample.private/manifest.private.json').read_text())
            if manifest.get('complete') is not True or len(manifest['requests'])!=receipt['requests']:
                raise ValueError('Incomplete first-sample trace')
            for row in manifest['requests']:
                if len(row['init'])!=1 or len(row['sample'])!=1:raise ValueError('Missing first sample/init')
        if args.variant=='piecewise_sync_only' and receipt.get('sync_only_metadata_reads',0)<=0:
            raise ValueError('Missing synchronization reads')
        return proof
    if args.run_dir.exists():raise FileExistsError('New run directory required')
    original_instrument=base.instrument
    traces=[]

    @contextmanager
    def instrument(session,panel,metrics,dg,vllm):
        import torch
        private=args.run_dir/'first_sample.private'
        def save_tensor(ordinal,label,name,tensor):
            private.mkdir(parents=True,exist_ok=True)
            cpu=tensor.detach().contiguous().cpu()
            dest=private/f'request{ordinal}_{label}_{name}.pt'
            if dest.exists():raise FileExistsError('Never replace captured tensor')
            torch.save(cpu,dest)
            return dict(file=dest.name,shape=list(cpu.shape),dtype=str(cpu.dtype),sha256=hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest())
        trace=TraceState(session,save_tensor,session.read_rng,args.trace_first_sample);traces.append(trace)
        with original_instrument(session,panel,metrics,dg,vllm),ExitStack() as stack:
            if args.variant=='piecewise_sync_only':
                stack.enter_context(patch.object(dg.DiffusionGemmaModelState,'prepare_attn',sync_only_prepare(dg.DiffusionGemmaModelState.prepare_attn,trace,torch)))
            if args.trace_first_sample:
                stack.enter_context(patch.object(dg.DiffusionGemmaRequestStates,'init_canvas',wrap_init(dg.DiffusionGemmaRequestStates.init_canvas,trace)))
                stack.enter_context(patch.object(dg,'_compiled_sample_step',wrap_sample(dg._compiled_sample_step,trace)))
            yield

    complete = False
    try:
        with base_contract(args.trace_first_sample),patch.object(base,'instrument',instrument):
            result=base.main(['--binding',args.binding,'--variant',args.variant,'--block',str(args.block),'--engine-repeat',str(args.engine_repeat),'--run-dir',str(args.run_dir)])
        complete = True
    finally:
        # Preserve partial trace provenance on an exception without replacing
        # tensors or claiming a successfully closed diagnostic.
        if args.trace_first_sample and traces:
            private=args.run_dir/'first_sample.private';private.mkdir(parents=True,exist_ok=True)
            base.write_new(private/'manifest.private.json',dict(private=True,never_publish=True,complete=complete,
                           requests=list(traces[0].requests.values())))
    if len(traces)!=1:raise ValueError('One diagnostic session required')
    trace=traces[0]
    if args.trace_first_sample:
        validate_private_trace(trace,len(result['requests']))
    if args.variant=='piecewise_sync_only' and trace.sync_only_reads<=0:
        raise ValueError('Synchronization control did not execute')
    receipt=dict(schema='v30_first_divergence_receipt_v1',source_commit=binding['deploy_commit'],variant=args.variant,
                 first_sample_trace=args.trace_first_sample,requests=len(result['requests']),
                 private_first_samples_saved=len(trace.requests) if args.trace_first_sample else 0,
                 sync_only_metadata_reads=trace.sync_only_reads,
                 sampler_modified=False,rng_reseeded_or_restored=False,performance_claim_allowed=False,
                 trace_changes_synchronization=args.trace_first_sample)
    base.write_new(args.run_dir/'first_divergence_receipt.json',receipt)
    return receipt


if __name__=='__main__':main()
