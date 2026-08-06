import torch

from dinfer.decoding.parallel_strategy import EditableThresholdParallelDecoder
from dinfer.decoding.utils import TokenArray


def _logits(tokens, vocab_size=16):
    logits = torch.full((1, len(tokens), vocab_size), -10.0)
    for index, token in enumerate(tokens):
        logits[0, index, token] = 10.0
    return logits


def test_editable_decoder_fills_masks_and_runs_stability_step():
    x = TokenArray(
        torch.tensor([[1, 2, 3]]),
        gen_length=2,
        mask_id=9,
        eos_id=8,
        device="cpu",
    )
    decoder = EditableThresholdParallelDecoder(
        temperature=0,
        threshold=0.5,
        editing_threshold=0.8,
        max_post_steps=2,
        mask_id=9,
        eos_id=8,
    )
    decoder.block_init(x[:, 3:5], 0)
    decoder.decode(_logits([4, 5]), 3, 5, x)
    assert x.data.tolist() == [[1, 2, 3, 4, 5]]
    assert decoder.should_continue(x[:, 3:5])

    decoder.decode(_logits([4, 5]), 3, 5, x)
    assert not decoder.should_continue(x[:, 3:5])


def test_editable_decoder_never_changes_prompt_tokens():
    x = TokenArray(
        torch.tensor([[1, 2, 3]]),
        gen_length=1,
        mask_id=9,
        eos_id=8,
        device="cpu",
    )
    decoder = EditableThresholdParallelDecoder(
        temperature=0,
        threshold=0.5,
        editing_threshold=0.0,
        max_post_steps=1,
        mask_id=9,
        eos_id=8,
    )
    decoder.block_init(x[:, 0:4], 0)
    decoder.decode(_logits([6, 6, 6, 7]), 0, 4, x)
    assert x.data[0, :3].tolist() == [1, 2, 3]
    assert x.data[0, 3].item() == 7
