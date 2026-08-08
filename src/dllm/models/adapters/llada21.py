from __future__ import annotations

import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ..base import GenerationRequest, GenerationResult, ModelAdapter
from .common import dtype, round_generation_length, seed_everything


class LLaDA21Adapter(ModelAdapter):
    name = "llada2_1_mini"
    attention_class_names = ("LLaDA2MoeAttention",)

    def load(self) -> "LLaDA21Adapter":
        import transformers.masking_utils as masking_utils

        if not hasattr(masking_utils, "create_bidirectional_mask"):
            masking_utils.create_bidirectional_mask = (
                lambda *, attention_mask=None, **_: attention_mask
            )
        config = AutoConfig.from_pretrained(
            self.model_path, revision=self.revision, trust_remote_code=True
        )
        if not hasattr(config, "rope_parameters"):
            config.rope_parameters = {
                "rope_type": "default",
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": config.partial_rotary_factor,
            }
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            revision=self.revision,
            config=config,
            trust_remote_code=True,
            torch_dtype=dtype(self.precision),
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, revision=self.revision, trust_remote_code=True
        )
        return self

    def encode_prompt(self, prompt: str, extra=None) -> list[int]:
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
        return list(ids)

    @torch.no_grad()
    def generate(self, request: GenerationRequest) -> GenerationResult:
        seed_everything(request.seed)
        inputs = torch.tensor(
            [self.encode_prompt(request.prompt)],
            dtype=torch.long,
            device=self.model.device,
        )
        length = round_generation_length(request.max_new_tokens, request.block_size)
        started = time.perf_counter()
        generated = self.model.generate(
            inputs=inputs,
            gen_length=length,
            block_length=request.block_size,
            threshold=request.threshold,
            editing_threshold=float(request.extra.get("editing_threshold", 0.0)),
            max_post_steps=int(request.extra.get("max_post_steps", 16)),
            temperature=request.temperature,
            eos_early_stop=True,
            mask_id=self.mask_token_id,
            eos_id=self.tokenizer.eos_token_id,
        )[0]
        elapsed = time.perf_counter() - started
        # Native LLaDA2.1 returns the completion rather than prompt+completion.
        tokens = generated[: request.max_new_tokens].detach().cpu().tolist()
        return GenerationResult(
            prompt=request.prompt,
            prompt_tokens=inputs[0].detach().cpu().tolist(),
            completion_tokens=tokens,
            text=self.tokenizer.decode(tokens, skip_special_tokens=True),
            elapsed_seconds=elapsed,
            termination_reason=(
                "eos" if self.tokenizer.eos_token_id in tokens else "length"
            ),
            metadata={"native_generation_length": length},
        )
