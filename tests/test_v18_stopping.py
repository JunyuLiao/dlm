"""CPU checks against the imported native DiffusionGemma stop implementation."""
import hashlib
import unittest
from pathlib import Path

try:
    import torch
    from transformers.models.diffusion_gemma import generation_diffusion_gemma as native
except ImportError:
    torch = native = None


@unittest.skipIf(native is None, 'native Transformers runtime unavailable')
class NativeStoppingTests(unittest.TestCase):
    EXPECTED_SHA = 'b814a6fc41492794c1f50ba71c750206684e25dfdd53932e0842675871fef0de'

    def setUp(self):
        self.assertEqual(hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(), self.EXPECTED_SHA)

    def test_stable_and_confident_requires_previous_argmax(self):
        criterion = native.StableAndConfidentStoppingCriteria(stability_threshold=1, confidence_threshold=.005)
        logits = torch.tensor([[[20.0, 0.0], [20.0, 0.0]]])
        canvas = torch.tensor([[0, 0]])
        self.assertFalse(bool(criterion(canvas, logits).item()))
        self.assertTrue(bool(criterion(canvas, logits).item()))
        self.assertFalse(bool(criterion(torch.tensor([[1, 0]]), logits).item()))

    def test_low_entropy_alone_does_not_stop_and_threshold_is_strict(self):
        logits = torch.tensor([[[4.0, 0.0], [4.0, 0.0]]])
        entropy = float(torch.distributions.Categorical(logits=logits).entropy().mean())
        criterion = native.StableAndConfidentStoppingCriteria(stability_threshold=0, confidence_threshold=entropy)
        self.assertFalse(bool(criterion(torch.tensor([[0, 0]]), logits).item()))
        criterion.confidence_threshold = entropy + .01
        self.assertTrue(bool(criterion(torch.tensor([[0, 0]]), logits).item()))

    def test_reset_clears_history(self):
        criterion = native.StableAndConfidentStoppingCriteria(stability_threshold=1, confidence_threshold=.005)
        logits = torch.tensor([[[20.0, 0.0]]])
        canvas = torch.tensor([[0]])
        criterion(canvas, logits)
        self.assertTrue(bool(criterion(canvas, logits).item()))
        criterion.reset()
        self.assertFalse(bool(criterion(canvas, logits).item()))


if __name__ == '__main__':
    unittest.main()
