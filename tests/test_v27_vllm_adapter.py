"""Constructor guards run without torch, vLLM, model binding or GPU work."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


def adapter_class():
    # Only nn.Module is needed to define the adapter's HF-shaped stub classes.
    # Keep constructor tests runnable with the standard library alone; any
    # unexpected tensor/GPU access fails instead of invoking an installed torch.
    fake_torch = ModuleType('torch')
    fake_torch.__path__ = []
    fake_torch.nn = ModuleType('torch.nn')
    fake_torch.nn.Module = object

    def unexpected_access(name):
        raise AssertionError(f'constructor accessed torch.{name}')

    fake_torch.__getattr__ = unexpected_access
    path = (Path(__file__).resolve().parents[1] / 'experiments' /
            'numerical_qk_reuse' / 'vllm_adapter.py')
    spec = importlib.util.spec_from_file_location('_v27_adapter_constructor_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'torch': fake_torch, 'torch.nn': fake_torch.nn}):
        spec.loader.exec_module(module)
    return module.VllmMethodAdapter


class ConstructorGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = adapter_class()

    def main_config(self):
        return dict(score_period=64, decision_interval=6, risk_state='dense_prefix',
                    threshold_shift='minus_ln2', carry_first=True, fused_observe=True,
                    async_route=True, fa4_consumer=True)

    def test_main_constructor_passes_without_torch_access_or_binding(self):
        config = self.main_config()
        before = copy.deepcopy(config)
        adapter = self.adapter(['sliding_attention', 'full_attention'], config=config,
                               condition='main', arm='method')
        self.assertEqual(adapter.global_layers, [1])
        self.assertIs(adapter.config, config)
        self.assertFalse(adapter.bound)
        self.assertIsNone(adapter.runtime)
        self.assertIsNone(adapter._stack)
        self.assertEqual(config, before)

    def test_cgate_fails_before_layer_inventory_or_binding(self):
        config = dict(self.main_config(), sensitivity='cgate')
        before = copy.deepcopy(config)
        with self.assertRaisesRegex(ValueError, 'C gate.*accepted-token mask'):
            self.adapter(None, config=config, condition='main', arm='method')
        self.assertEqual(config, before)

    def test_density_gate_fails_for_every_named_preset(self):
        for preset in ('ent0.05', 'ent0.02', 'cap12', 'cap24', 'stall2',
                       'ent0.05_stall2', 'stable1'):
            with self.subTest(preset=preset):
                config = dict(self.main_config(), density_gate=preset)
                before = copy.deepcopy(config)
                with self.assertRaisesRegex(ValueError, 'density_gate.*accepted-token mask'):
                    self.adapter(None, config=config, condition='main', arm='method')
                self.assertEqual(config, before)

    def test_explicit_none_gate_values_remain_compatible(self):
        self.adapter(['full_attention'], config=dict(self.main_config(), sensitivity=None,
                     density_gate=None), condition='main', arm='method')

    def test_control_arms_do_not_install_method_config(self):
        for arm in ('native', 'allkept'):
            with self.subTest(arm=arm):
                adapter = self.adapter(['full_attention'], arm=arm)
                self.assertIsNone(adapter.runtime)
                self.assertFalse(adapter.bound)

    def test_existing_missing_method_config_guard_is_preserved(self):
        with self.assertRaisesRegex(ValueError, 'frozen v21 effective config'):
            self.adapter(None, arm='method')


if __name__ == '__main__':
    unittest.main()
