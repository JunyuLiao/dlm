from __future__ import annotations

import sys
import time
import types

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..base import GenerationRequest, GenerationResult, ModelAdapter
from .common import REPO_ROOT, dtype, round_generation_length, seed_everything


class FastDLLMV2Adapter(ModelAdapter):
    name = "fast_dllm_v2"
    attention_class_names = ("Fast_dLLM_QwenAttention",)

    def load(self) -> "FastDLLMV2Adapter":
        source = REPO_ROOT / "fast_dllm_v2"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from generation_functions import Fast_dLLM_QwenForCausalLM

        original_compile = torch.compile
        if sys.version_info >= (3, 13):
            torch.compile = lambda function=None, **_: (
                (lambda target: target) if function is None else function
            )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                revision=self.revision,
                trust_remote_code=True,
                torch_dtype=dtype(self.precision),
            ).to(self.device)
        finally:
            torch.compile = original_compile
        self.model.eval()
        self.model.mdm_sample = types.MethodType(
            Fast_dLLM_QwenForCausalLM.batch_sample, self.model
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            revision=self.revision,
            trust_remote_code=True,
        )
        return self

    @torch.no_grad()
    def generate(self, request: GenerationRequest) -> GenerationResult:
        seed_everything(request.seed)
        encoded = self.tokenizer(request.prompt, return_tensors="pt")[
            "input_ids"
        ].to(self.model.device)
        prompt_length = encoded.shape[1]
        generation_length = round_generation_length(
            request.max_new_tokens, request.block_size
        )
        started = time.perf_counter()
        generated = self.model.mdm_sample(
            encoded.clone(),
            tokenizer=self.tokenizer,
            block_size=request.block_size,
            max_new_tokens=generation_length,
            small_block_size=int(request.extra.get("small_block_size", request.block_size)),
            min_len=prompt_length,
            seq_len=torch.tensor([prompt_length], device=self.model.device),
            mask_id=self.mask_token_id or 151665,
            stop_token=int(
                getattr(self.model.config, "eos_token_id", 151645)
            ),
            use_block_cache=bool(request.extra.get("dual_cache", False)),
            threshold=request.threshold,
            temperature=request.temperature,
        )[0]
        elapsed = time.perf_counter() - started
        completion = generated[prompt_length : prompt_length + request.max_new_tokens]
        completion_tokens = completion.detach().cpu().tolist()
        return GenerationResult(
            prompt=request.prompt,
            prompt_tokens=encoded[0].detach().cpu().tolist(),
            completion_tokens=completion_tokens,
            text=self.tokenizer.decode(completion_tokens, skip_special_tokens=True),
            elapsed_seconds=elapsed,
            termination_reason=(
                "eos"
                if getattr(self.model.config, "eos_token_id", None)
                in completion_tokens
                else "length"
            ),
            metadata={"native_generation_length": generation_length},
        )

