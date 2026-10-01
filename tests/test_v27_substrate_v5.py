"""piecewise_v5: the encoder layer's post-attention tail, run as a separate function, is exactly the HF layer."""
import pytest
import torch

from experiments.numerical_qk_reuse.v27_substrate import (SUBSTRATES, encoder_layer_tail, graphed_encoder_append,
                                                         prefill_kernel)


def test_v5_is_registered_with_fa4_prefill():
    assert 'piecewise_v5' in SUBSTRATES and prefill_kernel('piecewise_v5') == 'fa4'


def _tiny_layer():
    modeling = pytest.importorskip('transformers.models.diffusion_gemma.modeling_diffusion_gemma')
    configuration = pytest.importorskip('transformers.models.diffusion_gemma.configuration_diffusion_gemma')
    config = configuration.DiffusionGemmaTextConfig(
        hidden_size=32, intermediate_size=48, moe_intermediate_size=16, num_experts=4, top_k_experts=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, global_head_dim=16, num_global_key_value_heads=1,
        num_hidden_layers=2, layer_types=['sliding_attention', 'full_attention'], sliding_window=8,
        vocab_size=64, max_position_embeddings=128)
    config._attn_implementation = 'sdpa'
    torch.manual_seed(0)
    layer = modeling.DiffusionGemmaEncoderTextLayer(config, layer_idx=0).float().eval()
    for p in layer.parameters():
        torch.nn.init.normal_(p, std=0.2)
    rotary = modeling.DiffusionGemmaTextRotaryEmbedding(config)
    return layer, rotary, config


def test_split_encoder_layer_equals_hf_layer_bitwise():
    layer, rotary, config = _tiny_layer()
    hidden = torch.randn(1, 6, config.hidden_size)
    position_ids = torch.arange(6).unsqueeze(0)
    pe = rotary(hidden, position_ids, config.layer_types[0])
    with torch.no_grad():
        expected = layer(hidden, position_embeddings=pe, attention_mask=None, position_ids=position_ids)
        split = graphed_encoder_append(layer, encoder_layer_tail, canvas=6)
        got = split(hidden, position_embeddings=pe, attention_mask=None, position_ids=position_ids)
        other = split(hidden[:, :5], position_embeddings=tuple(t[:, :5] for t in pe), attention_mask=None,
                      position_ids=position_ids[:, :5])     # non-canvas length: the original forward
    assert torch.equal(got, expected)
    assert other.shape == (1, 5, config.hidden_size)
