import unittest
from unittest.mock import patch
from types import SimpleNamespace
from tests.test_v28_adapter_lifecycle import load_adapter


class AdapterMergeTests(unittest.TestCase):
    def test_unknown_backend(self):
        cls,_ = load_adapter()
        with self.assertRaises(ValueError): cls(['full_attention'], arm='allkept', merge_backend='guess')

    def test_identity_cache_and_request_cleanup(self):
        cls,_ = load_adapter()
        adapter = cls(['full_attention'], arm='allkept', lifecycle='request_clear', merge_backend='triton')
        built, invoked = [], []
        output = object()
        def identity(h,q,device):
            built.append((h,q,device))
            def call(o,l):
                invoked.append((o,l));return output
            return call
        module=SimpleNamespace(Alias2MappedMerge=SimpleNamespace(identity=identity))
        o=SimpleNamespace(shape=(2,256,16,512),device='cuda:0');l=object()
        adapter.begin_request()
        with patch.dict('sys.modules', {'experiments.numerical_qk_reuse.v29_lse_merge':module}):
            self.assertIs(adapter._merge_alias2(o,l),output)
            self.assertIs(adapter._merge_alias2(o,l),output)
        self.assertEqual(built,[(16,256,'cuda:0')]);self.assertEqual(len(invoked),2)
        self.assertEqual(adapter.calls['triton_lse_merge_calls'],2)
        self.assertEqual(adapter.calls['triton_lse_identity_builds'],1)
        receipt=adapter.end_request()
        self.assertEqual(receipt['adapter']['triton_lse_merge_calls'],2)
        self.assertEqual(adapter._merge_cache,{})

    def test_alias4_rejected_before_kernel(self):
        cls,_ = load_adapter()
        a=cls(['full_attention'],arm='allkept',merge_backend='triton');a.splits=4
        with patch.dict('sys.modules', {'experiments.numerical_qk_reuse.v29_lse_merge':SimpleNamespace(Alias2MappedMerge=object)}):
            with self.assertRaises(ValueError):a._merge_alias2(None,None)


if __name__=='__main__':unittest.main()
