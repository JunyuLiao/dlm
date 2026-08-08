from __future__ import annotations

import torch
import torch.nn as nn

from dllm.attention.blasst import (
    Blasst2DConfig,
    Blasst2DStats,
    dispatch_scaled_dot_product_attention,
    install_blasst,
)


class FakeAttention(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        q = hidden[:, None]
        return dispatch_scaled_dot_product_attention(
            self, q, q, q, attention_mask=None, is_causal=False
        )[:, 0]


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = FakeAttention()

    def forward(self, input_ids=None, hidden=None):
        if hidden is None:
            hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
        return self.attention(hidden)


def _fake_registry_dense(module, query, key, value, attention_mask, **kwargs):
    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask
    )
    return output.transpose(1, 2).contiguous(), None


ALL_ATTENTION_FUNCTIONS = {"sdpa": _fake_registry_dense}


class FakeRegistryAttention(nn.Module):
    num_key_value_groups = 1

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        q = hidden[:, None]
        output, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](self, q, q, q, None)
        return output


class FakeRegistryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = FakeRegistryAttention()

    def forward(self, input_ids=None, hidden=None):
        if hidden is None:
            hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
        return self.attention(hidden)


def test_disabled_binding_is_exact_and_cleanup_is_scoped() -> None:
    model = FakeModel()
    hidden = torch.randn(1, 4, 4)
    baseline = model(hidden=hidden)
    binding = install_blasst(
        model,
        Blasst2DConfig(enable_blasst_2d=False),
        attention_class_names=("FakeAttention",),
        integration="direct",
    )
    assert torch.equal(model(hidden=hidden), baseline)
    binding.close()
    assert not hasattr(model.attention, "_dllm_attention_runtime")
    assert torch.equal(model(hidden=hidden), baseline)


def test_direct_binding_collects_count_consistent_stats() -> None:
    model = FakeModel()
    stats = Blasst2DStats(record_layers=False, record_heads=False)
    with install_blasst(
        model,
        Blasst2DConfig(
            enable_blasst_2d=True,
            blasst_lambda=0.5,
            q_tile_size=2,
            kv_tile_size=2,
            collect_blasst_stats=True,
            collect_blasst_layer_stats=False,
            collect_blasst_head_stats=False,
        ),
        stats,
        attention_class_names=("FakeAttention",),
        integration="direct",
    ):
        output = model(input_ids=torch.tensor([[1, 2, 3, 4]]))
        assert torch.isfinite(output).all()
    summary = stats.summary()
    assert summary["eligible_tiles"] > 0
    assert summary["eligible_tiles"] == (
        summary["skipped_tiles"] + summary["retained_tiles"]
    )


def test_registry_binding_is_reference_counted_and_restored() -> None:
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    first = FakeRegistryModel()
    second = FakeRegistryModel()
    config = Blasst2DConfig(enable_blasst_2d=True, q_tile_size=2, kv_tile_size=2)
    first_binding = install_blasst(
        first, config, attention_class_names=("FakeRegistryAttention",)
    )
    dispatcher = ALL_ATTENTION_FUNCTIONS["sdpa"]
    assert dispatcher is not original
    second_binding = install_blasst(
        second, config, attention_class_names=("FakeRegistryAttention",)
    )
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is dispatcher
    first_binding.close()
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is dispatcher
    output = second(input_ids=torch.tensor([[1, 2, 3, 4]]))
    assert torch.isfinite(output).all()
    second_binding.close()
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is original
