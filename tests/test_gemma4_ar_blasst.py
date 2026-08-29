from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "ruler" / "blasst"
sys.path.insert(0, str(SCRIPT_DIR))

from run_gemma4_ar_blasst import (  # noqa: E402
    CausalBlasstRuntime,
    algorithm1_skip_mask,
    make_attention_interface,
)


def test_algorithm1_uses_chronological_running_maximum() -> None:
    scores = torch.tensor([[[[10.0, 9.0, 9.0, 8.5, 8.0, 7.0]]]])
    valid = torch.ones_like(scores, dtype=torch.bool)
    skip, eligible = algorithm1_skip_mask(scores, valid, 0.3, kv_tile_size=2)
    assert eligible.tolist() == [[[True, True, True]]]
    assert skip.tolist() == [[[False, False, True]]]


def test_algorithm1_never_skips_a_new_maximum() -> None:
    scores = torch.tensor([[[[1.0, 0.0, 4.0, 3.0, 2.0, 1.0]]]])
    valid = torch.ones_like(scores, dtype=torch.bool)
    skip, _ = algorithm1_skip_mask(scores, valid, 0.5, kv_tile_size=2)
    assert skip.tolist() == [[[False, False, True]]]


def test_algorithm1_excludes_structural_tiles() -> None:
    scores = torch.tensor([[[[99.0, 98.0, 5.0, 4.0, 0.0, -1.0]]]])
    valid = torch.tensor([[[[False, False, True, True, True, True]]]])
    skip, eligible = algorithm1_skip_mask(scores, valid, 0.1, kv_tile_size=2)
    assert eligible.tolist() == [[[False, True, True]]]
    assert skip.tolist() == [[[False, False, True]]]


def test_higher_lambda_is_monotonically_more_sparse() -> None:
    scores = torch.tensor([[[[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]]])
    valid = torch.ones_like(scores, dtype=torch.bool)
    low, _ = algorithm1_skip_mask(scores, valid, 0.01, kv_tile_size=2)
    high, _ = algorithm1_skip_mask(scores, valid, 0.5, kv_tile_size=2)
    assert torch.all(low <= high)


def test_decode_interface_masks_only_algorithm1_skips_and_records_counts() -> None:
    runtime = CausalBlasstRuntime(kv_tile_size=2, num_layers=1)
    runtime.configure(0.3, apply_mask=True)
    runtime.reset_sample("sample")
    module = SimpleNamespace(
        layer_idx=0,
        is_sliding=False,
        head_dim=1,
        num_key_value_groups=1,
        training=False,
    )
    query = torch.ones((1, 1, 1, 1))
    key = torch.tensor([[[[10.0], [9.0], [9.0], [8.5], [8.0], [7.0]]]])
    value = torch.tensor([[[[1.0], [1.0], [1.0], [1.0], [1000.0], [1000.0]]]])

    def should_not_prefill(*args, **kwargs):
        raise AssertionError("decode unexpectedly delegated to prefill")

    output, _ = make_attention_interface(runtime, should_not_prefill)(
        module, query, key, value, None, scaling=1.0
    )
    stats = runtime.sample_summary()
    assert output.item() < 2.0
    assert stats["overall"]["eligible_tiles"] == 3
    assert stats["overall"]["skipped_tiles"] == 1
    assert stats["overall"]["retained_tiles"] == 2
    assert stats["overall"]["structurally_masked_tiles"] == 0
