"""Shared model and metric helpers for the BLASST experiment scripts."""

from __future__ import annotations

import json
import random
import sys
import types
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_model(model_path: str, device: str, precision: str):
    from generation_functions import Fast_dLLM_QwenForCausalLM

    # The checkpoint decorates its training-only FlexAttention helper with
    # torch.compile at module import. PyTorch explicitly rejects Dynamo on
    # Python 3.13, even though evaluation never calls that helper.
    original_compile = torch.compile
    if sys.version_info >= (3, 13):
        def eager_compile(function=None, **kwargs):
            if function is None:
                return lambda target: target
            return function

        torch.compile = eager_compile
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=resolve_dtype(precision),
        ).to(device)
    finally:
        torch.compile = original_compile
    model.eval()
    model.mdm_sample = types.MethodType(
        Fast_dLLM_QwenForCausalLM.batch_sample, model
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


def load_examples(dataset_path: str | Path, limit: int) -> list[dict[str, str]]:
    payload = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    instances = payload.get("instances", payload)
    examples = []
    for item in instances[:limit]:
        examples.append({"input": str(item["input"]), "output": str(item["output"])})
    return examples


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    *,
    block_size: int,
    max_new_tokens: int,
    threshold: float,
    small_block_size: int | None = None,
    use_block_cache: bool = False,
) -> dict[str, Any]:
    if small_block_size is None:
        small_block_size = block_size
    encoded = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    sequence_length = torch.tensor([encoded.shape[1]], device=model.device)
    generated = model.mdm_sample(
        encoded.clone(),
        tokenizer=tokenizer,
        block_size=block_size,
        max_new_tokens=max_new_tokens,
        small_block_size=small_block_size,
        min_len=encoded.shape[1],
        seq_len=sequence_length,
        mask_id=getattr(model.config, "mask_token_id", 151665),
        stop_token=getattr(model.config, "eos_token_id", 151645),
        use_block_cache=use_block_cache,
        threshold=threshold,
        temperature=0.0,
    )[0]
    completion_ids = generated[encoded.shape[1] :]
    return {
        "prompt": prompt,
        "prompt_tokens": encoded[0].detach().cpu().tolist(),
        "completion_tokens": completion_ids.detach().cpu().tolist(),
        "completion": tokenizer.decode(completion_ids, skip_special_tokens=True),
    }


def relative_l2(reference: torch.Tensor, approximation: torch.Tensor) -> float:
    reference = reference.float().flatten()
    approximation = approximation.float().flatten()
    denominator = torch.linalg.vector_norm(reference).clamp_min(1.0e-12)
    return float(torch.linalg.vector_norm(reference - approximation) / denominator)


def cosine(reference: torch.Tensor, approximation: torch.Tensor) -> float:
    reference = reference.float().flatten()
    approximation = approximation.float().flatten()
    return float(F.cosine_similarity(reference, approximation, dim=0))


def normalized_token_difference(left: Iterable[int], right: Iterable[int]) -> float:
    left = list(left)
    right = list(right)
    if not left and not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, start=1):
        current = [i]
        for j, right_token in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1] / max(len(left), len(right), 1)
