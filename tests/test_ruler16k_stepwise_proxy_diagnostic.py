from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from experiments.ruler16k_stepwise_proxy_diagnostic.collector import (
    DiagnosticConfig,
    StepwisePrefixCollector,
)


def test_stepwise_collector_records_raw_rows_and_metadata(tmp_path):
    collector = StepwisePrefixCollector(DiagnosticConfig(reservoir_per_row=8))
    collector.begin_prompt(request_id="ruler-test", corpus="ruler16k")
    module = nn.Module()
    module.layer_idx = 0
    module.num_key_value_groups = 1
    module._blasst_2d_runtime = SimpleNamespace(metadata={"denoising_step": 0}, dense_kv_prefix_extractor=None)
    query = torch.ones(1, 1, 64, 1)
    key = torch.cat((torch.full((1, 1, 128, 1), 2.0), torch.full((1, 1, 64, 1), 50.0)), dim=-2)
    collector(module, query, key, key, None, scaling=1.0, is_causal=False)
    path = collector.export(tmp_path / "diagnostic.npz")
    with np.load(path) as payload:
        assert payload["denoising_call"].tolist() == [0]
        assert payload["prefix_tiles"].tolist() == [2]
        assert payload["prefix_mean"].tolist() == [2.0]
        assert payload["prefix_reservoir"].shape == (1, 8)


def test_stepwise_collector_uses_model_state_fallback():
    collector = StepwisePrefixCollector(DiagnosticConfig(reservoir_per_row=8))
    collector.begin_prompt(request_id="ruler-state", corpus="ruler16k")
    collector.set_forward_state(masked_tokens=3, active_tokens=10, mask_ratio=0.3)
    module = nn.Module()
    module.layer_idx = 0
    module.num_key_value_groups = 1
    module._blasst_2d_runtime = SimpleNamespace(
        metadata={"denoising_step": 2},
        dense_kv_prefix_extractor=None,
        active_query_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    query = torch.ones(1, 1, 64, 1)
    key = torch.cat((torch.full((1, 1, 128, 1), 2.0), torch.full((1, 1, 64, 1), 50.0)), dim=-2)
    collector(module, query, key, key, None, scaling=1.0, is_causal=False)
    assert collector.rows[0]["mask_ratio"] == 0.3
    assert collector.rows[0]["masked_tokens"] == 3
    assert collector.rows[0]["active_tokens"] == 1
