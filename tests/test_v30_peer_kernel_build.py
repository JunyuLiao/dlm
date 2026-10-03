import tempfile
import unittest
from pathlib import Path

from scripts.v30_peer_kernel_build import contained_new_directory, patch_debug_abi


class PeerBuildTests(unittest.TestCase):
    def test_debug_v4_accepted_with_v3_and_legacy_rejected(self):
        # Execute the actual patched guard expression against a diagnostic buffer
        # contract. Both supported ABI layouts contain debug_scores; ABI4 appends
        # sensitivity. This does not assert a CUDA result.
        source = "self.abi_version!=3 or debug_scores.shape!=(b,h,nq,nk) or debug_scores.dtype!=torch.float32 or debug_scores.device!=q.device or not debug_scores.is_contiguous()"
        from types import SimpleNamespace as NS
        buf = NS(shape=(1,16,256,512), dtype="float32", device="cuda:0", is_contiguous=lambda: True)
        scope = dict(debug_scores=buf, b=1,h=16,nq=256,nk=512,torch=NS(float32="float32"),q=NS(device="cuda:0"))
        for abi in (1,2,3,4,5):
            scope["self"] = NS(abi_version=abi)
            self.assertEqual(eval(patch_debug_abi(source), {}, scope), abi not in (3,4))
        for field, wrong in (("shape", (1,16,256,511)), ("dtype", "float16"), ("device", "cuda:1"), ("is_contiguous", lambda: False)):
            saved = getattr(buf, field); setattr(buf, field, wrong)
            scope["self"] = NS(abi_version=4)
            self.assertTrue(eval(patch_debug_abi(source), {}, scope))
            setattr(buf, field, saved)

    def test_patch_rejects_unknown_or_repeated_source(self):
        for source in ("", "self.abi_version not in (3,4) or debug_scores.shape", "self.abi_version!=3 or debug_scores.shape " * 2):
            with self.assertRaises(ValueError): patch_debug_abi(source)

    def test_output_cannot_escape_or_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertEqual(contained_new_directory(root / "new", root), root.resolve() / "new")
            with self.assertRaises(ValueError): contained_new_directory(root, root)
            with self.assertRaises(ValueError): contained_new_directory(root / ".." / "other", root)
            (root / "old").mkdir()
            with self.assertRaises(FileExistsError): contained_new_directory(root / "old", root)


if __name__ == "__main__": unittest.main()
