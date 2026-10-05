from types import SimpleNamespace as NS
import unittest

from scripts import v29_dense_reference_diag as base
from scripts.v30_dense_first_divergence import TraceState, STATE_FIELDS, wrap_init, wrap_sample, sync_only_prepare, make_spec, contract, base_contract, validate_private_trace


class Tensor:
    def __init__(self,value):self.value=value
    def __getitem__(self,index):return self
    def long(self):return self


class DivergenceTests(unittest.TestCase):
    def trace(self,enabled=True):
        session=NS(before='private_rng',rows=[]);saved=[]
        def save(ordinal,label,name,tensor):saved.append((ordinal,label,name,tensor.value));return tensor.value
        return TraceState(session,save,lambda:'rng',enabled),saved

    def test_initialization_original_arguments_return_and_no_rng_override(self):
        t,saved=self.trace();states=NS(canvas=Tensor('old'));slots=Tensor('slots');result=object();calls=[]
        def original(state,slot):
            calls.append((state,slot));state.canvas=Tensor('new');return result
        self.assertIs(wrap_init(original,t)(states,slots),result)
        self.assertEqual(calls,[(states,slots)])
        self.assertEqual(saved,[(0,'init_before','slots','slots'),(0,'init_after','canvas','new')])
        with self.assertRaises(ValueError):wrap_init(original,t)(states,slots)

    def test_sampler_passes_through_once_and_only_traces_first_call(self):
        t,saved=self.trace();result=object();calls=[]
        def original(logits,decode_slots,canvas,argmax_canvas,step_tensor,is_encoder_phase,confident_tensor,sc_embeds,history,history_len_tensor):
            calls.append(logits);canvas.value='after';return result
        args={k:Tensor(k) for k in ('logits','decode_slots',*STATE_FIELDS)}
        wrapped=wrap_sample(original,t)
        self.assertIs(wrapped(**args),result);n=len(saved)
        self.assertIs(wrapped(**args),result);self.assertEqual(len(saved),n);self.assertEqual(len(calls),2)
        self.assertEqual(t.requests[0]['sample'][0]['before']['tensors']['canvas'],'canvas')
        self.assertEqual(t.requests[0]['sample'][0]['after']['tensors']['canvas'],'after')

    def test_unbound_or_disabled_trace_never_touches_tensor(self):
        for enabled,before in ((False,'rng'),(True,None)):
            t,saved=self.trace(enabled);t.session.before=before
            obj=object();self.assertIs(wrap_sample(lambda x:x,t)(obj),obj)
            self.assertEqual(saved,[])

    def test_sync_only_matches_metadata_read_but_forwards_every_argument(self):
        t,_=self.trace(False);read=[];calls=[];result=object()
        torch=NS(stack=lambda values:NS(tolist=lambda:read.append([x.value for x in values])))
        state=NS(diffusion_states=NS(is_encoder_phase=Tensor('phase'),step=Tensor('step')))
        batch=NS(num_reqs=1,idx_mapping_np=[0],seq_lens=Tensor('length'))
        def original(*a,**kw):calls.append((a,kw));return result
        wrapped=sync_only_prepare(original,t,torch)
        self.assertIs(wrapped(state,batch,'mode',1,2,3,4,ubatch_idx=7),result)
        self.assertEqual(read,[['phase','step','length']]);self.assertEqual(t.sync_only_reads,1)
        self.assertEqual(calls[0],((state,batch,'mode',1,2,3,4),dict(for_capture=False,ubatch_idx=7)))
        wrapped(state,batch,'mode',1,2,3,4,for_capture=True)
        self.assertEqual(t.sync_only_reads,1)

    def test_independent_trace_protocol_and_contract_restored(self):
        old=base.DIAGNOSTIC;variants=base.VARIANTS
        s=make_spec({'arm_settings':{}},'piecewise_sync_only',[0],trace_first_sample=False)
        t=make_spec({'arm_settings':{}},'piecewise_sync_only',[0],trace_first_sample=True)
        self.assertNotEqual(s['protocol_id'],t['protocol_id'])
        self.assertEqual(s['dense_reference_diagnostic'],contract(False))
        self.assertEqual(t['dense_reference_diagnostic'],contract(True))
        self.assertEqual(len(s['launch_order']),16)
        self.assertIs(base.DIAGNOSTIC,old);self.assertIs(base.VARIANTS,variants)

    def test_incomplete_trace_rejected(self):
        t,_=self.trace()
        with self.assertRaises(ValueError):validate_private_trace(t,1)
        t.request()
        with self.assertRaises(ValueError):validate_private_trace(t,1)
        t.requests[0].update(init=[{}],sample=[{}]);self.assertTrue(validate_private_trace(t,1))


if __name__=='__main__':unittest.main()
