from contextlib import contextmanager
from types import SimpleNamespace as NS
import unittest

from experiments.numerical_qk_reuse.peer_global_scope import GLOBAL_LAYERS, GlobalScopeAdapter, ScopeCounter, install_global_value


def adapter():
    modules = [NS(layer_idx=i, is_sliding=i not in GLOBAL_LAYERS, decoder=True) for i in range(30)]
    # Encoder also has layer 5: identity/class selection must exclude it.
    modules.append(NS(layer_idx=5, is_sliding=False, decoder=False))
    return NS(model=NS(named_modules=lambda: [(str(i), m) for i,m in enumerate(modules)]),
              is_blasst_attention_module=lambda name,m: m.decoder,
              blasst_call_is_eligible=object(), mask_token_id=42), modules


class GlobalScopeTests(unittest.TestCase):
    def test_only_original_five_decoder_objects_selected(self):
        base, modules = adapter(); scoped = GlobalScopeAdapter(base)
        self.assertEqual([m.layer_idx for m in modules if scoped.is_blasst_attention_module("",m)], list(GLOBAL_LAYERS))
        self.assertIs(scoped.blasst_call_is_eligible, base.blasst_call_is_eligible)
        self.assertEqual(scoped.mask_token_id, 42)
        self.assertFalse(scoped.is_blasst_attention_module("",NS(layer_idx=5,decoder=True)))

    def test_missing_duplicate_sliding_and_existing_binding_rejected(self):
        for kind in ("missing", "duplicate", "sliding", "bound"):
            base, modules = adapter()
            if kind == "missing": modules[5].decoder = False
            elif kind == "duplicate": modules[4].layer_idx=5
            elif kind == "sliding": modules[5].is_sliding=True
            else: modules[5]._blasst_2d_runtime = object()
            with self.assertRaises((ValueError,RuntimeError)): GlobalScopeAdapter(base)

    def test_call_arguments_and_result_identity_and_failure_counter(self):
        base, modules = adapter(); seen=[]; result=object()
        def route(*args,**kwargs): seen.append((args,kwargs)); return result
        counter=ScopeCounter(route,GlobalScopeAdapter(base)); q,k,v,mask=[object() for _ in range(4)]
        for i in GLOBAL_LAYERS:
            self.assertIs(counter(modules[i],q,k,v,mask,scaling=.125,is_causal=False),result)
            args,kwargs=seen[-1];self.assertEqual(args,(modules[i],q,k,v,mask));self.assertEqual(kwargs,dict(scaling=.125,is_causal=False))
        self.assertTrue(counter.receipt()['scope_qualified'])
        with self.assertRaises(RuntimeError): counter(modules[0],q,k,v,mask)
        self.assertFalse(counter.receipt()['scope_qualified']);self.assertEqual(len(seen),5)

    def test_context_cleanup_on_error_and_peer_receives_scoped_selector(self):
        base, modules=adapter(); closed=[]
        @contextmanager
        def peer(scoped,library,thresholds,**kwargs):
            self.assertEqual(sum(scoped.is_blasst_attention_module("",m) for m in modules),5)
            binding=NS(runtime=NS(attention_override=None));router=lambda *a,**kw: None
            try: yield binding,router
            finally: closed.append(True)
        with self.assertRaisesRegex(RuntimeError,"deliberate"):
            with install_global_value(base,peer,"unused",{},collect=True) as (binding,router,counter):
                self.assertIs(binding.runtime.attention_override,counter)
                raise RuntimeError("deliberate")
        self.assertEqual(closed,[True])


if __name__ == "__main__": unittest.main()
