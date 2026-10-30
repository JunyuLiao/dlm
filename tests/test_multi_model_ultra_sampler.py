from types import SimpleNamespace

import torch

from experiments.numerical_qk_reuse.multi_model_runtime import ultra_fixed_generate


class MaskPredictor:
    def __init__(self):
        self.calls = 0
        self.start = 0
        self.end = None

    def __call__(self, x):
        logits = torch.zeros((*x.shape, 8))
        logits[..., 3] = 1.
        first = (x[0] == 7).nonzero()[0, 0]
        logits[0, first] = 0.
        logits[0, first, 7 if self.calls < 64 else 3] = 12.
        self.calls += 1
        return SimpleNamespace(logits=logits[:, self.start:self.end])


def test_unresolved_previous_masks_remain_native_selection_candidates():
    model = MaskPredictor()
    blocks = []
    runtime = SimpleNamespace(block=lambda start, end: (blocks.append((start, end)), setattr(model, "end", end)))
    sampler = SimpleNamespace(get_num_transfer_tokens=lambda mask, steps: torch.ones((1, steps), dtype=torch.long))
    result = ultra_fixed_generate(model, torch.tensor([[1, 2]]), 128, runtime, sampler,
                                  lambda lo: setattr(model, "start", lo), mask_id=7)
    # First 64 calls keep predicting MASK at the first live row. The second
    # block's schedule must revisit and fill the unresolved earlier block.
    assert result[0, :2].tolist() == [1, 2]
    assert (result[0, 2:66] == 3).all()
    assert (result[0, 66:] == 7).all()
    assert model.calls == 128
    assert blocks == [(2, 66), (66, 130)]
