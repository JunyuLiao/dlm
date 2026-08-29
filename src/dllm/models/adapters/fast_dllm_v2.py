from __future__ import annotations

import sys
import time
import types
from typing import Any, Mapping

import torch
import transformers
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..base import GenerationRequest, GenerationResult, ModelAdapter
from .common import REPO_ROOT, dtype, round_generation_length, seed_everything

REQUIRED_TRANSFORMERS_VERSION = "4.53.1"


class FastDLLMV2Adapter(ModelAdapter):
    name = "fast_dllm_v2"
    attention_class_names = ("Fast_dLLM_QwenAttention",)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dense_query_prefix_by_block: dict[int, int] = {}

    def blasst_call_is_eligible(
        self, module: Any, args: tuple, kwargs: Mapping[str, Any]
    ) -> bool:
        """Limit BLASST to ordinary cached denoising, matching Fast-dLLM-v2."""
        metadata = kwargs.get("blasst_metadata")
        eligible = bool(
            isinstance(metadata, Mapping)
            and not kwargs.get("update_past_key_values", False)
            and not kwargs.get("use_block_cache", False)
        )
        if eligible:
            block = int(metadata.get("generation_block_index", -1))
            first_iteration = int(metadata.get("denoising_iteration", -1)) == 0
            if first_iteration or block not in self._dense_query_prefix_by_block:
                input_ids = kwargs.get("input_ids")
                if input_ids is None and args:
                    input_ids = args[0]
                if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2:
                    mask = input_ids[0].eq(int(self.mask_token_id or 151665))
                    indices = mask.nonzero(as_tuple=False)
                    self._dense_query_prefix_by_block[block] = (
                        int(indices[0].item()) if len(indices) else int(input_ids.shape[1])
                    )
        return eligible

    def blasst_dense_kv_prefix(
        self,
        module: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Any,
        kwargs: Mapping[str, Any],
    ) -> int:
        """Keep cached prompt blocks and the prompt remainder as dense sinks."""
        del module, value, attention_mask
        metadata = kwargs.get("blasst_metadata") or {}
        block = int(metadata.get("generation_block_index", -1))
        cached_prefix = max(0, int(key.shape[-2] - query.shape[-2]))
        return min(
            int(key.shape[-2]),
            cached_prefix + self._dense_query_prefix_by_block.get(block, 0),
        )

    def load(self) -> "FastDLLMV2Adapter":
        if Version(transformers.__version__) != Version(REQUIRED_TRANSFORMERS_VERSION):
            raise ImportError(
                "Fast-dLLM-v2 requires transformers=="
                f"{REQUIRED_TRANSFORMERS_VERSION}; found {transformers.__version__}. "
                "Newer cache/attention APIs can silently produce NaN outputs."
            )

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

    def runtime_metadata(self) -> dict[str, Any]:
        metadata = super().runtime_metadata()
        metadata["fast_dllm_v2_required_transformers"] = REQUIRED_TRANSFORMERS_VERSION
        return metadata

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
