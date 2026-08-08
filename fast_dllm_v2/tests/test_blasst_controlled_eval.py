from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

from generation_functions import Fast_dLLM_QwenForCausalLM  # noqa: E402
from scripts.eval_blasst_mixed_tasks import _execute_humaneval  # noqa: E402
from scripts.sweep_blasst_controlled import (  # noqa: E402
    _aggregate,
    _masked_queries,
)


def test_generation_validates_subblocks_and_supports_dual_cache() -> None:
    kwargs = {
        "input_ids": torch.ones((1, 16), dtype=torch.long),
        "tokenizer": SimpleNamespace(pad_token_id=0),
        "block_size": 16,
        "max_new_tokens": 16,
        "min_len": 16,
        "seq_len": torch.tensor([16]),
    }
    with pytest.raises(ValueError, match="positive divisor"):
        Fast_dLLM_QwenForCausalLM.batch_sample(
            SimpleNamespace(), small_block_size=6, **kwargs
        )

    calls: list[dict[str, object]] = []
    cache = object()

    class DummyModel:
        device = torch.device("cpu")

        def forward(self, input_ids, **forward_kwargs):
            calls.append(
                {
                    "query_length": input_ids.shape[1],
                    "use_block_cache": forward_kwargs.get(
                        "use_block_cache", False
                    ),
                    "has_block_cache": (
                        forward_kwargs.get("block_past_key_values") is not None
                    ),
                }
            )
            logits = torch.zeros((*input_ids.shape, 4))
            return SimpleNamespace(
                logits=logits,
                past_key_values=cache,
                block_past_key_values=cache,
            )

        @staticmethod
        def sample_with_top_p(logits, **_kwargs):
            tokens = torch.ones(logits.shape[:-1], dtype=torch.long)
            probabilities = torch.full_like(logits, 0.1)
            probabilities[..., 1] = 0.5
            return tokens, probabilities

    Fast_dLLM_QwenForCausalLM.batch_sample(
        DummyModel(),
        input_ids=torch.ones((1, 16), dtype=torch.long),
        tokenizer=SimpleNamespace(pad_token_id=0),
        block_size=16,
        max_new_tokens=16,
        small_block_size=8,
        min_len=16,
        seq_len=torch.tensor([16]),
        mask_id=3,
        stop_token=99,
        use_block_cache=True,
        threshold=0.9,
    )
    dual_calls = [row for row in calls if row["use_block_cache"]]
    assert {row["query_length"] for row in dual_calls} == {8, 16}
    assert any(row["has_block_cache"] for row in dual_calls)

    visualization = Fast_dLLM_QwenForCausalLM.mdm_sample_with_visualization(
        SimpleNamespace(),
        input_ids=kwargs["input_ids"],
        tokenizer=kwargs["tokenizer"],
        block_size=16,
        small_block_size=8,
        max_new_tokens=16,
    )
    with pytest.raises(ValueError, match="Sub-block"):
        next(visualization)


@pytest.mark.parametrize("query_length", [1, 2, 4, 8, 16, 32, 64])
def test_controlled_masks_have_requested_rounded_counts(
    query_length: int,
) -> None:
    ratios = [0.9, 0.7, 0.5, 0.3, 0.15]
    states = _masked_queries(
        torch.arange(64),
        query_length,
        ratios,
        mask_token_id=-1,
        seed=42,
    )
    previous: set[int] | None = None
    for (_, ratio, state) in states:
        masked = set((state == -1).nonzero().flatten().tolist())
        assert len(masked) == round(query_length * ratio)
        if previous is not None:
            assert masked <= previous
        previous = masked


def test_aggregation_uses_total_counts_not_mean_percentages() -> None:
    rows = [
        {
            "group": "x",
            "eligible_tiles": 1,
            "skipped_tiles": 1,
            "retained_tiles": 0,
            "structurally_masked_tiles": 0,
            "skippable_row_votes": 1,
            "valid_row_votes": 1,
            "skipped_valid_elements": 1,
            "valid_elements": 1,
        },
        {
            "group": "x",
            "eligible_tiles": 9,
            "skipped_tiles": 0,
            "retained_tiles": 9,
            "structurally_masked_tiles": 0,
            "skippable_row_votes": 0,
            "valid_row_votes": 9,
            "skipped_valid_elements": 0,
            "valid_elements": 9,
        },
    ]
    result = _aggregate(rows, ("group",))[0]
    assert result["physical_tile_sparsity"] == pytest.approx(0.1)
    assert result["row_vote_sparsity"] == pytest.approx(0.1)


def test_humaneval_execution_passes_safe_code_and_rejects_imports() -> None:
    label = {
        "entry_point": "add",
        "prompt": "def add(a, b):\n",
        "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
    }
    passed = _execute_humaneval(
        "def add(a, b):\n    return a + b\n",
        label,
        timeout=2.0,
    )
    assert passed["passed"]
    rejected = _execute_humaneval(
        "import os\ndef add(a, b):\n    return a + b\n",
        label,
        timeout=2.0,
    )
    assert not rejected["passed"]
    assert rejected["status"].startswith("forbidden_import")
