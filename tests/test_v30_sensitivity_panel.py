import copy
import unittest
from scripts import v29_expanded_panel as base
from scripts import v30_sensitivity_panel as p


class Tests(unittest.TestCase):
    def test_all_suites_modes_preserve_full_inventory_seed_and_substrate(self):
        old = base.METHOD
        for suite in base.SUITES:
            for mode in p.MODES:
                spec = p.build_spec(suite, mode, ['dlm2'])
                with p.protocol(mode):base.validate_spec(spec)
                self.assertEqual(spec['primary_receipt_method'], dict(old, sensitivity=mode))
                self.assertEqual([b['engine_seed'] for b in spec['blocks']], list(range(30001,30009)))
                self.assertEqual(spec['adapter_settings']['kv_copy_backend'], 'torch')
                self.assertEqual(spec['adapter_settings']['merge_backend'], 'torch')
        self.assertIs(base.METHOD, old)

    def test_receipt_requires_real_calls_not_only_named_config(self):
        r = dict(mode='unit_v30', begin=3, observe=3, unit_steps=3, confidence_steps=0,
                 protected_first_steps=0, canvas_resets=1, accepted_mask_used=False,
                 temporal_argmax_observed=False, router_and_carry_modified=False,
                 weighting_state='previous_completed_call_only')
        p.check_sensitivity_receipt({'method':{'v30_sensitivity':r}},3,'unit_v30')
        for key, value in [('observe',2),('accepted_mask_used',True),('unit_steps',2),('canvas_resets',0)]:
            with self.assertRaises(ValueError):
                p.check_sensitivity_receipt({'method':{'v30_sensitivity':dict(r,**{key:value})}},3,'unit_v30')
        r.update(mode='confidence_v30',unit_steps=0,confidence_steps=2,protected_first_steps=1)
        p.check_sensitivity_receipt({'method':{'v30_sensitivity':r}},3,'confidence_v30')
        with self.assertRaises(ValueError):p.check_sensitivity_receipt({'method':{'v30_sensitivity':r}},3,'temporal_T')

    def test_context_restored_after_error(self):
        before=(base.METHOD,base.SOURCES,base.validate_receipt)
        with self.assertRaises(RuntimeError):
            with p.protocol('confidence_v30'):raise RuntimeError('test')
        self.assertEqual(before,(base.METHOD,base.SOURCES,base.validate_receipt))


if __name__=='__main__':unittest.main()
