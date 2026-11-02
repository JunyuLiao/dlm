import pytest
import torch

from scripts.llada22_eos_confidence_eval import zero_eos_confidence


def test_eos_confidence_is_zero_without_changing_other_positions():
    tokens = torch.tensor([[7, 156892, 11, 156892]])
    confidence = torch.tensor([[.2, .8, .6, .1]])
    original = confidence.clone()
    result = zero_eos_confidence(confidence, tokens, 156892)
    expected = confidence.masked_fill(tokens == 156892, 0.)
    assert torch.equal(result, expected)
    assert torch.equal(confidence, original)
    assert torch.equal(tokens, torch.tensor([[7, 156892, 11, 156892]]))


def test_eos_mask_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="identical shape"):
        zero_eos_confidence(torch.ones((2,)), torch.ones((1, 2), dtype=torch.long), 1)
