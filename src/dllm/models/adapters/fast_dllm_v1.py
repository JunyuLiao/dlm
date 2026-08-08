from __future__ import annotations

import sys
import time
import types

import torch
from transformers import AutoTokenizer

from ..base import GenerationRequest, GenerationResult, ModelAdapter
from .common import REPO_ROOT, dtype, round_generation_length, seed_everything


def _add_source(path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


class LLaDAV1Adapter(ModelAdapter):
    name = "fast_dllm_v1_llada"
    attention_integration = "direct"
    attention_class_names = (
        "LLaDASequentialBlock",
        "LLaDALlamaBlock",
        "LLaDABlockDiffBlock",
    )

    def load(self) -> "LLaDAV1Adapter":
        source = REPO_ROOT / "fast_dllm_v1" / "llada"
        _add_source(source)
        original_compile = torch.compile
        if sys.version_info >= (3, 13):
            torch.compile = lambda function=None, **_: (
                (lambda target: target) if function is None else function
            )
        try:
            from model.modeling_llada import LLaDAModelLM
        finally:
            torch.compile = original_compile

        self.model = LLaDAModelLM.from_pretrained(
            self.model_path,
            revision=self.revision,
            trust_remote_code=True,
            torch_dtype=dtype(self.precision),
        ).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, revision=self.revision, trust_remote_code=True
        )
        return self

    def encode_prompt(self, prompt: str, extra=None) -> list[int]:
        if getattr(self.tokenizer, "chat_template", None):
            ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
            )
            return list(ids)
        return super().encode_prompt(prompt, extra)

    def _prompt_ids(self, prompt: str) -> torch.Tensor:
        return torch.tensor(
            [self.encode_prompt(prompt)], dtype=torch.long, device=self.model.device
        )

    @torch.no_grad()
    def generate(self, request: GenerationRequest) -> GenerationResult:
        from generate import (
            generate,
            generate_with_dual_cache,
            generate_with_prefix_cache,
        )

        seed_everything(request.seed)
        prompt = self._prompt_ids(request.prompt)
        generation_length = round_generation_length(
            request.max_new_tokens, request.block_size
        )
        steps = request.steps or generation_length
        blocks = generation_length // request.block_size
        if steps % blocks:
            steps = ((steps + blocks - 1) // blocks) * blocks
        cache_mode = str(request.extra.get("cache_mode", "none"))
        function = {
            "none": generate,
            "prefix": generate_with_prefix_cache,
            "dual": generate_with_dual_cache,
        }.get(cache_mode)
        if function is None:
            raise ValueError("cache_mode must be one of: none, prefix, dual")
        started = time.perf_counter()
        output, nfe = function(
            self.model,
            prompt,
            steps=steps,
            gen_length=generation_length,
            block_length=request.block_size,
            temperature=request.temperature,
            threshold=request.threshold,
        )
        elapsed = time.perf_counter() - started
        completion = output[0, prompt.shape[1] : prompt.shape[1] + request.max_new_tokens]
        tokens = completion.detach().cpu().tolist()
        return GenerationResult(
            prompt=request.prompt,
            prompt_tokens=prompt[0].detach().cpu().tolist(),
            completion_tokens=tokens,
            text=self.tokenizer.decode(tokens, skip_special_tokens=True),
            elapsed_seconds=elapsed,
            model_evaluations=int(nfe),
            metadata={"cache_mode": cache_mode, "native_generation_length": generation_length},
        )


class DreamV1Adapter(ModelAdapter):
    name = "fast_dllm_v1_dream"
    attention_integration = "direct"
    attention_class_names = ("DreamSdpaAttention",)

    def load(self) -> "DreamV1Adapter":
        source = REPO_ROOT / "fast_dllm_v1" / "dream"
        _add_source(source)
        from model.modeling_dream import DreamModel

        self.model = DreamModel.from_pretrained(
            self.model_path,
            revision=self.revision,
            trust_remote_code=True,
            torch_dtype=dtype(self.precision),
        ).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, revision=self.revision, trust_remote_code=True
        )
        return self

    def _prompt_ids(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self.tokenizer, "chat_template", None):
            encoded = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            encoded = self.tokenizer(prompt, return_tensors="pt")
        ids = encoded["input_ids"].to(self.model.device)
        mask = encoded.get("attention_mask", torch.ones_like(ids)).to(self.model.device)
        return ids, mask

    @torch.no_grad()
    def generate(self, request: GenerationRequest) -> GenerationResult:
        source = REPO_ROOT / "fast_dllm_v1" / "dream"
        _add_source(source)
        if bool(request.extra.get("use_cache", True)):
            from model.generation_utils_block import DreamGenerationMixin

            self.model.diffusion_generate = types.MethodType(
                DreamGenerationMixin.diffusion_generate, self.model
            )
            self.model._sample = types.MethodType(DreamGenerationMixin._sample, self.model)
        seed_everything(request.seed)
        ids, attention_mask = self._prompt_ids(request.prompt)
        started = time.perf_counter()
        output = self.model.diffusion_generate(
            ids,
            attention_mask=attention_mask,
            max_new_tokens=request.max_new_tokens,
            output_history=False,
            return_dict_in_generate=True,
            steps=request.steps or request.max_new_tokens,
            temperature=request.temperature,
            alg="confidence_threshold",
            threshold=request.threshold,
            block_length=request.block_size,
            dual_cache=bool(request.extra.get("dual_cache", False)),
        )
        elapsed = time.perf_counter() - started
        completion = output.sequences[0, ids.shape[1] :]
        tokens = completion.detach().cpu().tolist()
        return GenerationResult(
            prompt=request.prompt,
            prompt_tokens=ids[0].detach().cpu().tolist(),
            completion_tokens=tokens,
            text=self.tokenizer.decode(tokens, skip_special_tokens=True),
            elapsed_seconds=elapsed,
        )
