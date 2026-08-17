from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from dllm.attention.blasst import Blasst2DConfig, apply_blasst_2d, slow_blasst_2d
from dllm.models import GenerationRequest, create_adapter
from dllm.models.adapters import diffusion_gemma as diffusion_module


class FakeTokenizer:
    mask_token_id = 4
    pad_token_id = 0
    eos_token_id = 1
    name_or_path = "fake-processor"

    def decode(self, tokens, skip_special_tokens=True):
        return "decoded:" + ",".join(str(value) for value in tokens)


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        suffix = 99 if kwargs.get("enable_thinking") else 13
        return torch.tensor([[11, 12, suffix]])


class FakeProcessorLoader:
    processor = FakeProcessor()
    calls = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        return cls.processor


class FakeSamplerConfig:
    def __init__(self, entropy_bound):
        self.entropy_bound = entropy_bound


class FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(
            canvas_length=256,
            eos_token_id=[1, 106],
            text_config=SimpleNamespace(eos_token_id=1),
            image_token_id=258880,
            _commit_hash="resolved-revision",
        )
        self.generation_config = SimpleNamespace(
            eos_token_id=[1, 106, 50],
            pad_token_id=0,
            max_denoising_steps=48,
            t_max=0.8,
            t_min=0.4,
            confidence_threshold=0.005,
            stability_threshold=1,
            sampler_config=FakeSamplerConfig(0.1),
        )
        self.generate_calls = []
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, input_ids, **kwargs):
        self.generate_calls.append(kwargs)
        completion = torch.tensor([[8, 106, 0, 0]])
        return SimpleNamespace(
            sequences=torch.cat([input_ids, completion], dim=-1),
            tokens_per_forward=torch.tensor([2.0]),
        )


class FakeModelLoader:
    model = FakeModel()
    calls = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        return cls.model


@pytest.fixture
def fake_stack(monkeypatch):
    FakeProcessorLoader.calls.clear()
    FakeProcessorLoader.processor = FakeProcessor()
    FakeModelLoader.calls.clear()
    FakeModelLoader.model = FakeModel()
    monkeypatch.setattr(
        diffusion_module,
        "_require_diffusion_gemma",
        lambda: (FakeProcessorLoader, FakeModelLoader, FakeSamplerConfig),
    )


def test_lazy_registry_creation_does_not_load_optional_stack() -> None:
    adapter = create_adapter("diffusion_gemma", "checkpoint", device="cpu")
    assert adapter.name == "diffusion_gemma"
    assert adapter.model is None


def test_processor_only_load_and_official_chat_template_encoding(fake_stack) -> None:
    adapter = diffusion_module.DiffusionGemmaAdapter(
        "checkpoint", device="cpu", revision="revision-a"
    ).load_tokenizer()
    assert FakeModelLoader.calls == []
    assert FakeProcessorLoader.calls == [
        ("checkpoint", {"revision": "revision-a"})
    ]
    assert adapter.encode_prompt("hello") == [11, 12, 13]
    assert adapter.encode_prompt("hello", {"thinking": True}) == [11, 12, 99]
    messages, kwargs = adapter.processor.calls[-1]
    assert messages == [{"role": "user", "content": "hello"}]
    assert kwargs["add_generation_prompt"] is True
    assert kwargs["enable_thinking"] is True
    assert kwargs["tokenize"] is True


def test_load_forwards_revision_and_uses_single_device(fake_stack) -> None:
    adapter = diffusion_module.DiffusionGemmaAdapter(
        "checkpoint", device="cpu", precision="float32", revision="revision-b"
    ).load()
    assert adapter.model.eval_called
    _, kwargs = FakeModelLoader.calls[-1]
    assert kwargs["revision"] == "revision-b"
    assert kwargs["dtype"] is torch.float32
    assert kwargs["device_map"] == {"": "cpu"}
    assert kwargs["attn_implementation"] == "sdpa"


def test_native_defaults_are_preserved_and_completion_uses_actual_prompt_length(fake_stack) -> None:
    adapter = diffusion_module.DiffusionGemmaAdapter(
        "checkpoint", device="cpu", precision="float32"
    ).load()
    result = adapter.generate(
        GenerationRequest(
            "hello",
            max_new_tokens=4,
            block_size=32,
            threshold=0.9,
            temperature=0.0,
        )
    )
    kwargs = adapter.model.generate_calls[-1]
    assert kwargs == {"max_new_tokens": 4}
    assert result.prompt_tokens == [11, 12, 13]
    assert result.completion_tokens == [8, 106]
    assert result.termination_reason == "eos"
    assert result.model_evaluations == 1
    assert result.metadata["native_canvas_length"] == 256
    assert result.metadata["requested_block_size"] == 32
    assert result.metadata["block_size_applied"] is False
    assert result.metadata["denoising_configuration"] == {
        "max_denoising_steps": 48,
        "t_max": 0.8,
        "t_min": 0.4,
        "entropy_bound": 0.1,
        "confidence_threshold": 0.005,
        "stability_threshold": 1,
    }


def test_steps_and_sampler_extras_map_only_to_native_parameters(fake_stack) -> None:
    adapter = diffusion_module.DiffusionGemmaAdapter("checkpoint", device="cpu")
    kwargs = adapter._generation_kwargs(
        GenerationRequest(
            "hello",
            8,
            steps=7,
            extra={
                "t_max": 0.7,
                "t_min": 0.2,
                "entropy_bound": 0.3,
                "confidence_threshold": 0.01,
                "stability_threshold": 0,
                "thinking": True,
            },
        )
    )
    assert kwargs["max_new_tokens"] == 8
    assert kwargs["max_denoising_steps"] == 7
    assert kwargs["t_max"] == 0.7
    assert kwargs["t_min"] == 0.2
    assert kwargs["sampler_config"].entropy_bound == 0.3
    assert "thinking" not in kwargs
    assert "temperature" not in kwargs
    assert "threshold" not in kwargs
    assert "block_size" not in kwargs


def test_scalar_temperature_and_top_p_disable_native_schedule(fake_stack) -> None:
    adapter = diffusion_module.DiffusionGemmaAdapter("checkpoint", device="cpu")
    adapter.model = FakeModel()
    kwargs = adapter._generation_kwargs(
        GenerationRequest(
            "hello",
            max_new_tokens=4,
            temperature=0.6,
            extra={"top_p": 0.95, "thinking": True},
        )
    )
    assert kwargs["generation_config"].t_min is None
    assert kwargs["generation_config"].t_max is None
    assert len(kwargs["logits_processor"]) == 2
    logits = torch.tensor([[[4.0, 3.0, 2.0, 1.0]]])
    processed = kwargs["logits_processor"](
        torch.tensor([[1]]), logits, cur_step=torch.tensor(1)
    )
    assert processed.shape == logits.shape
    assert torch.isfinite(processed).any(dim=-1).all()
    assert adapter.model.generation_config.t_min == 0.4
    assert adapter.model.generation_config.t_max == 0.8


@pytest.mark.parametrize(
    ("temperature", "extra", "message"),
    [
        (0.0, {"top_p": 0.95}, "positive temperature"),
        (0.6, {"top_p": 1.1}, "at most 1"),
    ],
)
def test_invalid_sampling_controls_are_rejected(
    fake_stack, temperature, extra, message
) -> None:
    adapter = diffusion_module.DiffusionGemmaAdapter("checkpoint", device="cpu")
    adapter.model = FakeModel()
    with pytest.raises(ValueError, match=message):
        adapter._generation_kwargs(
            GenerationRequest(
                "hello", max_new_tokens=4, temperature=temperature, extra=extra
            )
        )


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"max_denoising_steps": 0}, "positive integer"),
        ({"entropy_bound": 0.0}, "positive"),
        ({"confidence_threshold": "bad"}, "real number"),
        ({"confidence_threshold": float("nan")}, "finite"),
        ({"stability_threshold": -1}, "non-negative"),
        ({"t_min": 0.8, "t_max": 0.4}, "greater"),
        ({"thinking": 1}, "boolean"),
        ({"unknown": 1}, "unsupported"),
    ],
)
def test_invalid_sampler_overrides_have_useful_errors(fake_stack, extra, match) -> None:
    adapter = diffusion_module.DiffusionGemmaAdapter("checkpoint", device="cpu")
    with pytest.raises(ValueError, match=match):
        adapter._generation_kwargs(GenerationRequest("hello", 8, extra=extra))


def test_module_selection_excludes_encoder_cross_role_and_vision_modules() -> None:
    decoder = type("DiffusionGemmaDecoderTextAttention", (nn.Module,), {})()
    encoder = type("DiffusionGemmaEncoderTextAttention", (nn.Module,), {})()
    vision = type("Gemma4VisionAttention", (nn.Module,), {})()
    cross = type("CrossAttention", (nn.Module,), {})()
    adapter = diffusion_module.DiffusionGemmaAdapter("checkpoint")
    assert adapter.is_blasst_attention_module("model.decoder.layers.0.self_attn", decoder)
    assert not adapter.is_blasst_attention_module("model.encoder.layers.0.self_attn", encoder)
    assert not adapter.is_blasst_attention_module("model.vision.attn", vision)
    assert not adapter.is_blasst_attention_module("model.decoder.cross_attn", cross)


def test_dense_prompt_prefix_is_never_blasst_eligible() -> None:
    scores = torch.tensor([[[[10.0, 9.0, -10.0, -11.0]]]])
    masked, decisions = apply_blasst_2d(
        scores,
        None,
        None,
        Blasst2DConfig(blasst_lambda=0.5, q_tile_size=1, kv_tile_size=2),
        sparse_kv_start=2,
    )
    assert not decisions.eligible_mask[..., 0].any()
    assert decisions.eligible_mask[..., 1].all()
    assert torch.equal(masked[..., :2], scores[..., :2])


def test_masked_empty_cache_tile_is_counted_as_physical_work() -> None:
    scores = torch.tensor(
        [[[[3.0, 2.0, -torch.inf, -torch.inf], [4.0, 1.0, -torch.inf, -torch.inf]]]]
    )
    valid = torch.tensor(
        [[[[True, True, False, False], [True, True, False, False]]]]
    )
    config = Blasst2DConfig(
        blasst_lambda=0.5,
        q_tile_size=2,
        kv_tile_size=2,
        include_masked_kv_tiles_in_physical_stats=True,
    )
    masked, decisions = apply_blasst_2d(scores, valid, None, config)
    slow_masked, slow_decisions = slow_blasst_2d(scores, valid, None, config)

    assert decisions.eligible_mask.sum().item() == 2
    assert decisions.skip_mask[..., 1].all()
    assert decisions.structural_mask.sum().item() == 0
    assert torch.equal(masked, scores)
    assert torch.equal(masked, slow_masked)
    assert torch.equal(decisions.eligible_mask, slow_decisions.eligible_mask)
    assert torch.equal(decisions.skip_mask, slow_decisions.skip_mask)


def test_masked_empty_cache_tile_is_structural_by_default() -> None:
    scores = torch.tensor([[[[3.0, 2.0, -torch.inf, -torch.inf]]]])
    valid = torch.tensor([[[[True, True, False, False]]]])
    _, decisions = apply_blasst_2d(
        scores,
        valid,
        None,
        Blasst2DConfig(blasst_lambda=0.5, q_tile_size=1, kv_tile_size=2),
    )
    assert decisions.eligible_mask.sum().item() == 1
    assert decisions.structural_mask[..., 1].all()


def test_encoder_cache_is_blasst_eligible() -> None:
    adapter = diffusion_module.DiffusionGemmaAdapter("checkpoint")
    query = torch.empty(1, 16, 256, 64)
    key = torch.empty(1, 8, 768, 64)
    assert adapter.blasst_dense_kv_prefix(None, query, key, key, None, {}) == 0
