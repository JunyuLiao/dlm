from __future__ import annotations

import pytest

from dllm.models import GenerationRequest, adapter_names, create_adapter


def test_all_current_text_models_are_registered() -> None:
    assert adapter_names() == (
        "diffusion_gemma",
        "fast_dllm_v1_dream",
        "fast_dllm_v1_llada",
        "fast_dllm_v2",
        "llada2_1_mini",
    )
    for name in adapter_names():
        assert create_adapter(name, "checkpoint", device="cpu").name == name


def test_generation_request_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        GenerationRequest("prompt", 0)
    with pytest.raises(ValueError, match="block_size"):
        GenerationRequest("prompt", 1, block_size=0)
    with pytest.raises(ValueError, match="steps"):
        GenerationRequest("prompt", 1, steps=0)
