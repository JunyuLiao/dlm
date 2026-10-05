"""Exercise actual adapter wrappers with CPU-only official-module stand-ins.

These check argument/return passthrough, not GPU numerical/RNG equivalence.
"""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


def modules_and_adapter():
    torch = ModuleType('torch')
    torch.nn = SimpleNamespace(Module=object)
    torch.cuda = SimpleNamespace(Event=lambda **_: (_ for _ in ()).throw(AssertionError('unexpected CUDA event')))
    spec = importlib.util.spec_from_file_location('_v29_native_contract', Path(__file__).resolve().parents[1] /
                                                  'experiments/numerical_qk_reuse/vllm_adapter.py')
    adapter = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'torch': torch}):
        spec.loader.exec_module(adapter)
    dg = ModuleType('vllm.model_executor.models.diffusion_gemma')
    fa = ModuleType('vllm.v1.attention.backends.flash_attn')
    sparse = ModuleType('experiments.numerical_qk_reuse.v27_fa4')
    sparse.sparse_lists = lambda *args: ('original_sparse', args)
    calls = []
    result = object()
    def forward(*args):
        calls.append(('forward', args))
        return result
    def sample(*args, **kwargs):
        calls.append(('sample', args, kwargs))
        return result
    def prepare(*args, **kwargs):
        calls.append(('prepare', args, kwargs))
        return result
    dg.DiffusionGemmaModelState = type('State', (), {'prepare_attn': prepare})
    dg._compiled_sample_step = sample
    fa.FlashAttentionImpl = type('FA', (), {'forward': forward})
    tree = {}
    for name in ('vllm', 'vllm.model_executor', 'vllm.model_executor.models', 'vllm.v1',
                 'vllm.v1.attention', 'vllm.v1.attention.backends'):
        tree[name] = ModuleType(name)
        tree[name].__path__ = []
    tree.update({dg.__name__: dg, fa.__name__: fa, sparse.__name__: sparse})
    for name, mod in tuple(tree.items()):
        parent, _, child = name.rpartition('.')
        if parent in tree:
            setattr(tree[parent], child, mod)
    return adapter, dg, fa, sparse, tree, calls, result


class NativePassthrough(unittest.TestCase):
    def setUp(self):
        self.va, self.dg, self.fa, self.sparse, self.mods, self.calls, self.result = modules_and_adapter()
        self.modules = patch.dict(sys.modules, self.mods)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        # Import the real package before masking its consumer attribute.
        import experiments.numerical_qk_reuse as package
        self.consumer = patch.object(package, 'v27_fa4', self.sparse, create=True)
        self.consumer.start()
        self.addCleanup(self.consumer.stop)
        self.adapter = self.va.VllmMethodAdapter(['full_attention'], arm='native', profile=False)
        self.adapter.bound = True
        self.adapter.step_ctx = {'encoder': False, 'step': 1, 'seq_len': 512}
        self.adapter.active_for = lambda _: 0
        self.adapter.forward = lambda *args: self.fail('native entered mathematical attention')
        self.va.install_vllm_patches(self.adapter)

    def test_global_forward_preserves_every_argument_and_return_identity(self):
        args = tuple(object() for _ in range(7))
        # self, layer, Q, K, V, cache, metadata, output, scale, block scale.
        layer = SimpleNamespace(layer_name='model.layers.5.self_attn.attn')
        impl = self.fa.FlashAttentionImpl()
        expected = (impl, layer, *args[:6], None, args[6])
        returned = impl.forward(layer, *args[:6], output_scale=None, output_block_scale=args[6])
        self.assertIs(returned, self.result)
        received = self.calls[-1][1]
        self.assertEqual(len(received), len(expected))
        self.assertTrue(all(a is b for a, b in zip(received, expected)))
        self.assertEqual(self.adapter.calls['global_calls'], 1)
        self.assertIsNone(self.adapter.runtime)

    def test_non_global_unbound_and_scaled_paths_preserve_arguments(self):
        for bound, active, scale in ((True, None, None), (False, 0, None), (True, 0, object())):
            with self.subTest(bound=bound, active=active):
                self.adapter.bound = bound
                self.adapter.active_for = lambda _, selected=active: selected
                args = tuple(object() for _ in range(7))
                impl = self.fa.FlashAttentionImpl()
                expected = (impl, *args, scale, None)
                self.assertIs(impl.forward(*args, output_scale=scale), self.result)
                self.assertTrue(all(a is b for a, b in zip(self.calls[-1][1], expected)))
        self.assertEqual(self.adapter.calls['global_calls'], 0)

    def test_original_sampler_is_called_once_and_exact_result_observed_returned(self):
        observed = []
        self.adapter.on_sample = observed.append
        arg, keyword = object(), object()
        returned = self.dg._compiled_sample_step(arg, native_keyword=keyword)
        self.assertIs(returned, self.result)
        self.assertEqual(self.calls, [('sample', (arg,), {'native_keyword': keyword})])
        self.assertEqual(observed, [self.result])

    def test_native_observation_does_not_read_or_mutate_logits(self):
        class ForbiddenLogits:
            def __getattribute__(self, name):
                raise AssertionError('native accessed logits: '+name)
        self.adapter.on_sample(ForbiddenLogits())
        self.assertEqual(self.adapter.calls['observes'], 1)
        self.assertIsNone(self.adapter.runtime)

    def test_encoder_sample_returns_original_without_observation(self):
        self.adapter.step_ctx = {'encoder': True}
        self.adapter.on_sample = lambda _: self.fail('encoder logits observed')
        self.assertIs(self.dg._compiled_sample_step(object()), self.result)
        self.assertEqual(len(self.calls), 1)

    def test_unbound_preparation_preserves_positional_and_keyword_identity(self):
        self.adapter.bound = False
        args = tuple(object() for _ in range(6))
        state = self.dg.DiffusionGemmaModelState()
        self.assertIs(state.prepare_attn(*args, for_capture=True, ubatch_idx=3), self.result)
        received = self.calls[-1]
        self.assertTrue(all(a is b for a, b in zip(received[1], (state, *args))))
        self.assertEqual(received[2], {'for_capture': True, 'ubatch_idx': 3})


if __name__ == '__main__':
    unittest.main()
