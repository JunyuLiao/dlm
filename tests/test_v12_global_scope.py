"""v12 CP1: GLOBAL-only scope adapter -- selection, effective-config round trips,
explicit failure of unsupported paths. Real-model dispatch is proven separately
by scripts/v12_scope_probe.py (bounded GPU probe)."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

BUILD = os.environ.get('V11_SUPPORT_BUILD')


class DiffusionGemmaEncoderModel(torch.nn.Module):
    def forward(self, x):
        return x


class Attn(torch.nn.Module):
    def __init__(self, idx, sliding):
        super().__init__()
        self.layer_idx, self.is_sliding = idx, sliding
        self.sliding_window = 1024 if sliding else None


def fake_adapter():
    model = torch.nn.Module()
    model.add_module('encoder', DiffusionGemmaEncoderModel())
    layers = torch.nn.ModuleList(Attn(i, i % 6 != 5) for i in range(12))
    model.add_module('layers', layers)
    return SimpleNamespace(model=model, is_blasst_attention_module=lambda n, m: isinstance(m, Attn),
                           mask_token_id=4, pad_token_id=0, attention_class_names=('Attn',),
                           blasst_query_ids=lambda *a: None, blasst_filter_special_query_ids=False,
                           blasst_call_is_eligible=lambda *a: True, attention_integration='direct')


def test_proxy_selects_exactly_global_modules():
    from experiments.numerical_qk_reuse.global_scope import GlobalScopeAdapter
    scoped = GlobalScopeAdapter(fake_adapter())
    assert scoped.active_layers == [5, 11] and len(scoped.local_layers) == 10
    assert scoped.mask_token_id == 4                                    # delegation


def arm_cfg(arm):
    from scripts.v10_request_runs import arm_config
    root = Path(__file__).resolve().parents[1]
    args = SimpleNamespace(phase='v12test', ids=['aime26/2'],
                           manifest=Path('/home/exouser/dyh/numerical_qk_reuse_native_20260924/code_cp2/results/numerical_qk_reuse_20260924/private/smoke_manifest.json'),
                           policy=root / 'results/query_adaptive_v3/configs/frozen_policies.json',
                           model=Path('/home/exouser/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b'),
                           revision='f7f5b7f5fa82ffc52addd066915886d497f5517b')
    return arm_config(args, arm)


PLUGIN = 'experiments.numerical_qk_reuse.global_scope:install'


def g_arm(name, condition, interval, selector, **extra):
    return dict(name=name, condition=condition, decision_interval=interval, selector=selector,
                selector_layers='all', kernel_variant='generic', telemetry='minimal', guard_mode='fused',
                consumer='hopper', support_build=BUILD, plugin=PLUGIN, **extra)


@pytest.mark.skipif(not (torch.cuda.is_available() and BUILD), reason='CUDA + V11_SUPPORT_BUILD required')
@pytest.mark.parametrize('name,condition,interval,selector', [
    ('G1', 'global_M1', 1, 'prefix_block_summary'), ('G3', 'global_M3', 2, 'prefix_block_summary'),
    ('B8', 'global_B8', 8, 'legacy_recompute')])
def test_effective_config_round_trip(name, condition, interval, selector):
    from experiments.numerical_qk_reuse.global_scope import install
    config = arm_cfg(g_arm(name, condition, interval, selector))
    for key, value in dict(condition=condition, decision_interval=interval, selector=selector, selector_layers='all',
                           consumer='hopper', guard_mode='fused', telemetry='minimal', kernel_variant='generic',
                           score_refresh_period=8, plugin=PLUGIN).items():
        assert config[key] == value, key
    assert config['support_binary']['key'] == json.loads(Path(BUILD).read_text())['key']
    with install(fake_adapter(), config, condition) as runtime:
        c = runtime['counters']()
        router = runtime['router']
        assert (c['scope'], c['active_layers'], c['bound_modules']) == ('global_only_native_local', [5, 11], [5, 11])
        assert (c['decision_interval'], c['score_period'], c['selector'], c['selector_layers']) == (interval, 8, selector, 'all')
        assert (c['consumer'], c['guard_mode'], c['telemetry'], c['kernel_variant']) == ('hopper', 'fused', 'minimal', 'generic')
        assert router.support == 'legacy_junyu_mask'
        hooked = {int(m.layer_idx) for m in runtime['binding'].modules}
        assert hooked == {5, 11}
        with pytest.raises(ValueError):
            runtime['binding'].runtime.attention_override(Attn(5, False), None, None, None, torch.zeros(1))
        with pytest.raises(ValueError):
            runtime['binding'].runtime.attention_override(Attn(4, True), None, None, None, None)


def test_misconfigured_arms_fail_explicitly():
    from experiments.numerical_qk_reuse.global_scope import install
    bad = dict(decision_interval=1, selector='prefix_block_summary', score_refresh_period=8, policy={}, m_ref=1.,
               beta=3., gamma=.5, diagnostic=False)
    with pytest.raises(ValueError):
        with install(fake_adapter(), dict(bad, decision_interval=2), 'global_M1'):
            pass
    with pytest.raises(ValueError):
        with install(fake_adapter(), dict(bad, decision_interval=8), 'global_B8'):   # B8 must not build summaries
            pass
    with pytest.raises(ValueError):
        with install(fake_adapter(), bad, 'M1'):
            pass


def test_global_T_config_round_trip():
    config = arm_cfg(dict(name='T_G', condition='global_T', plugin=PLUGIN, collect=False))
    assert config['condition'] == 'global_T' and config['collect'] is False and config['library'] and config['torch_library']
