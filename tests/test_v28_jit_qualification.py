import unittest
from scripts.v28_vllm_qualify import validate_jit_deltas
from scripts.v28_jit_receipts import COUNT_FIELDS


class JitQualificationTests(unittest.TestCase):
    def test_missing_or_nonzero_cannot_be_called_warm(self):
        zero = {'jit_monitor_'+key:0 for key in COUNT_FIELDS}
        self.assertIn('zero_observed', validate_jit_deltas([{'compilation_deltas':zero}],True))
        with self.assertRaises(ValueError):
            validate_jit_deltas([{'compilation_deltas':{}}],True)
        for key in zero:
            for value in (1,-1,None,False):
                with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                    validate_jit_deltas([{'compilation_deltas':dict(zero,**{key:value})}],True)

    def test_legacy_coverage_is_unknown(self):
        self.assertEqual(validate_jit_deltas([{}],False),'not_instrumented')


if __name__ == '__main__':
    unittest.main()
