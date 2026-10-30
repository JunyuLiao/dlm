"""Model-specific execution plumbing for the approved three-model campaign.

Sampler functions are the pinned publisher source. The LLaDA2.2 execution
adapter caches block-causal prefix K/V and omits unused prefix logits; this
requires independent small-input qualification against the publisher forward.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import inspect
import math
from pathlib import Path
import textwrap
import types

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from .multi_model_attention import AttentionRuntime, dense_block


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BlockMaskProxy:
    """The block-causal visibility is applied in attention, without an NxN mask."""
    def __getitem__(self, index):
        return None


class LogitWindow:
    def __init__(self, tensor, start):
        self.tensor, self.start = tensor, start

    def __getitem__(self, index):
        batch, rows, vocab = index
        begin = 0 if rows.start is None else rows.start - self.start
        end = self.tensor.shape[1] if rows.stop is None else rows.stop - self.start
        if begin < 0 or end > self.tensor.shape[1]:
            raise IndexError("sampler requested logits outside its active block")
        return self.tensor[batch, slice(begin, end, rows.step), vocab]


class ModelHarness:
    def __init__(self, root, model_key, arm="dense_native", threshold=.02):
        self.root = Path(root).resolve()
        self.key = model_key
        self.runtime = AttentionRuntime(arm, threshold)
        self.cache = {}
        self.store_cache = False
        self.query_offset = 0
        self.current_x = None
        self.original_forward = None
        self.in_joint = False
        self.source_changes = {}

    def load(self, install=True):
        path = self.root / "private/models" / self.key
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        if self.key == "ultrallada":
            # Publisher context extension changes theta but leaves this legacy
            # preallocation field at 4096; RoPE buffers grow to the real extent.
            config.flash_attention = False
        cls = AutoModel if self.key == "ultrallada" else AutoModelForCausalLM
        self.model = cls.from_pretrained(path, config=config, trust_remote_code=True,
                                        torch_dtype=torch.bfloat16,
                                        attn_implementation="sdpa" if self.key != "ultrallada" else "eager",
                                        device_map="cuda")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.original_forward = self.model.forward
        if install:
            self.install()
        return self

    def install(self):
        if self.key == "llada22":
            self._install_llada22()
        elif self.key == "trado":
            self._install_trado()
        else:
            self._install_ultra()
        return self

    def _llada_attention(self, module, hidden_states, attention_mask=None,
                         position_embeddings=None, **kwargs):
        batch, length, _ = hidden_states.shape
        nq, nkv, dim = module.num_heads, module.num_key_value_heads, module.head_dim
        x = module.query_key_value(hidden_states).view(batch, length, nq + 2 * nkv, dim)
        q, k, v = [z.transpose(1, 2) for z in x.split((nq, nkv, nkv), dim=-2)]
        if module.config.use_qk_norm:
            q, k = module.query_layernorm(q), module.key_layernorm(k)
        source = inspect.getmodule(type(module))
        q, k = source.apply_rotary_pos_emb(q, k, *position_embeddings)
        layer = module.layer_idx
        if layer in self.cache:
            old_k, old_v = self.cache[layer]
            k, v = torch.cat((old_k, k), 2), torch.cat((old_v, v), 2)
        if self.store_cache:
            self.cache[layer] = (k, v)
        out = self.runtime.attention(q, k, v, layer, module.scaling,
                                     offset=self.query_offset, block_size=32)
        return module.dense(out.reshape(batch, length, -1)), None, None

    def _llada_forward(self, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        if input_ids is None:
            input_ids = kwargs.pop("inputs", None)
        if self.in_joint:
            start, end = self.runtime.block_start, self.runtime.block_end
            ids = input_ids[:, start:end]
            self.runtime.begin_forward()
            self.current_x = input_ids
            self.query_offset = start
            self.store_cache = False
        else:
            start = self.query_offset
            ids = input_ids
        positions = torch.arange(start, start + ids.shape[1], device=ids.device)[None]
        result = self.model.model(input_ids=ids, position_ids=positions,
                                  attention_mask=None, use_cache=False,
                                  return_dict=True, output_router_logits=False)
        if self.store_cache:
            return types.SimpleNamespace(logits=None)
        logits = self.model.lm_head(result.last_hidden_state).float()
        return types.SimpleNamespace(logits=LogitWindow(logits, start) if self.in_joint else logits)

    def _install_llada22(self):
        model = self.model
        source = inspect.getmodule(type(model))
        self.native_generate = model.generate
        self.native_joint = model._joint_decode_block
        for module in model.modules():
            if type(module).__name__ == "LLaDA2MoeAttention":
                module.forward = types.MethodType(
                    lambda m, *a, **kw: self._llada_attention(m, *a, **kw), module)
        model.forward = types.MethodType(lambda m, *a, **kw: self._llada_forward(*a, **kw), model)
        original = self.native_joint

        def joint(m, x, block_start, block_end, *args, **kwargs):
            # All previously committed blocks have exact causal independence
            # from the new block. Prefill once; subsequent commits append K/V.
            cached = next(iter(self.cache.values()))[0].shape[2] if self.cache else 0
            if cached < block_start:
                self.runtime.phase = "prefill"
                self.query_offset = cached
                self.store_cache = True
                # Bound prefill temporaries at long context.
                for lo in range(cached, block_start, 1024):
                    hi = min(block_start, lo + 1024)
                    self.query_offset = lo
                    self._llada_forward(x[:, lo:hi])
                self.store_cache = False
            self.runtime.block(block_start, block_end)
            self.in_joint = True
            try:
                result = original(x, block_start, block_end, *args, **kwargs)
            finally:
                self.in_joint = False
            self.runtime.phase = "commit"
            self.query_offset = block_start
            self.store_cache = True
            self._llada_forward(x[:, block_start:block_end])
            self.store_cache = False
            return result

        model._joint_decode_block = types.MethodType(joint, model)
        # Keep the native editing loop and all its random draws. Only replace
        # the quadratic mask allocation by a visibility descriptor.
        text = textwrap.dedent(inspect.getsource(type(model).generate))
        begin = text.index("    block_mask = torch.tril(")
        end = text.index("    position_ids =", begin)
        replacement = "    block_diffusion_attention_mask = BlockMaskProxy()\n\n"
        modified = text[:begin] + replacement + text[end:]
        namespace = dict(vars(source))
        namespace["BlockMaskProxy"] = BlockMaskProxy
        exec(compile(modified, "<qualified_llada22_mask_descriptor>", "exec"), namespace)
        model.generate = types.MethodType(namespace["generate"], model)
        self.source_changes["mask_descriptor"] = {
            "original_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "adapted_sha256": hashlib.sha256(modified.encode()).hexdigest()}
        # Native block MoE routing remains unchanged on every active block.

    def _install_ultra(self):
        for layer, module in enumerate(m for m in self.model.modules()
                                       if type(m).__name__ in ("LLaDALlamaBlock", "LLaDASequentialBlock")):
            def attend(m, q, k, v, attn_mask=None, dropout_p=0., is_causal=False, layer=layer):
                return self.runtime.attention(q, k, v, layer, q.shape[-1] ** -.5).transpose(1, 2)
            module._scaled_dot_product_attention = types.MethodType(attend, module)
        # The native full-sequence sampler inspects every position's logits.
        # Chunk only the vocabulary projection, retaining exact output tensor.
        original = self.model.forward
        def forward(m, *args, **kwargs):
            self.runtime.begin_forward()
            return original(*args, **kwargs)
        self.model.forward = types.MethodType(forward, self.model)
        self.sampler = import_file(self.root / "private/source/ultra_llada_generate.py", "pinned_ultra_sampler")
        head = self.model.model.transformer.ff_out
        original_head = head.forward
        def head_forward(m, hidden):
            # The publisher's greedy sampler only uses the active block's
            # independent row logits. Prefix/future logits cannot be selected.
            return original_head(hidden[:, self.runtime.block_start:self.runtime.block_end])
        head.forward = types.MethodType(head_forward, head)

    def _install_trado(self):
        source = inspect.getmodule(type(self.model))
        for module in self.model.modules():
            if type(module).__name__ != "SDARAttention":
                continue
            def attention(m, hidden_states, position_embeddings, attention_mask=None,
                          past_key_value=None, store_kv=False, **kwargs):
                batch, length, _ = hidden_states.shape
                dim = m.head_dim
                q = m.q_norm(m.q_proj(hidden_states).view(batch, length, -1, dim)).transpose(1, 2)
                k = m.k_norm(m.k_proj(hidden_states).view(batch, length, -1, dim)).transpose(1, 2)
                v = m.v_proj(hidden_states).view(batch, length, -1, dim).transpose(1, 2)
                q, k = source.apply_rotary_pos_emb(q, k, *position_embeddings)
                if past_key_value is not None and store_kv:
                    k, v = past_key_value.update(k, v, m.layer_idx)
                elif past_key_value is not None and len(past_key_value) > m.layer_idx:
                    k = torch.cat((past_key_value[m.layer_idx][0], k), 2)
                    v = torch.cat((past_key_value[m.layer_idx][1], v), 2)
                offset = k.shape[2] - length
                out = self.runtime.attention(q, k, v, m.layer_idx, m.scaling,
                                             offset=offset, block_size=4)
                return m.o_proj(out.reshape(batch, length, -1)), None
            module.forward = types.MethodType(attention, module)
        self.sampler = import_file(self.root / "private/source/SDAR_generate.py", "pinned_sdar_sampler")
        original = self.model.forward
        def forward(m, input_ids, **kwargs):
            if kwargs.get("store_kv", False):
                self.runtime.phase = "prefill" if input_ids.shape[1] > 4 else "commit"
            else:
                end = int(kwargs["position_ids"][0, -1]) + 1
                start = end - input_ids.shape[1]
                if self.runtime.block_start != start or self.runtime.phase != "denoise":
                    self.runtime.block(start, end)
                self.runtime.begin_forward()
            return original(input_ids, **kwargs)
        self.model.forward = types.MethodType(forward, self.model)

    def encode(self, text, *, completion=False):
        if self.key == "ultrallada":
            return self.tokenizer(text, add_special_tokens=True).input_ids
        encoded = self.tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                     add_generation_prompt=True, tokenize=True)
        if hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return list(encoded)

    @torch.inference_mode()
    def generate(self, text, cap, seed, *, small=False):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.runtime.reset()
        self.runtime.phase = "prefill"
        self.cache.clear()
        ids = torch.tensor([self.encode(text)], device="cuda", dtype=torch.long)
        prompt_length = ids.shape[1]
        if self.key == "llada22":
            output = self.model.generate(inputs=ids, gen_length=cap, block_length=32,
                                         steps=32, threshold=.5, editing_threshold=0.,
                                         max_post_steps=16, temperature=0., eos_early_stop=True)
            tokens = output[0].tolist()
        elif self.key == "trado":
            output = self.sampler.block_diffusion_generate(
                self.model, {"input_ids": ids}, mask_id=151669, gen_length=cap,
                block_length=4, denoising_steps=4, temperature=1., top_k=0, top_p=1.,
                confidence_threshold=.9, stopping_criteria_idx=[self.tokenizer.eos_token_id])
            tokens = output[0, prompt_length:prompt_length + cap].tolist()
        else:
            block = min(64, cap)
            extent = math_ceil(cap, block) * block
            x = torch.full((1, prompt_length + extent), 126336, device="cuda", dtype=torch.long)
            x[:, :prompt_length] = ids
            # Keep the full future-mask horizon and publisher's fixed schedule.
            # Only omit logits on rows whose selection confidence is -inf.
            for b in range(extent // block):
                start, end = prompt_length + b * block, prompt_length + (b + 1) * block
                self.runtime.block(start, end)
                count = self.sampler.get_num_transfer_tokens(x[:, start:end] == 126336, block)
                for step in range(block):
                    logits = self.model(x).logits
                    x0 = logits.argmax(-1)
                    prob = torch.nn.functional.softmax(logits, dim=-1)
                    confidence = prob.gather(-1, x0[..., None]).squeeze(-1)
                    mask = x[:, start:end] == 126336
                    confidence = torch.where(mask, confidence, -math.inf)
                    chosen = torch.topk(confidence[0], k=count[0, step]).indices
                    x[:, start:end][:, chosen] = x0[:, chosen]
            tokens = x[0, prompt_length:prompt_length + cap].tolist()
        eos = self.tokenizer.eos_token_id
        finished = eos in tokens
        if finished:
            tokens = tokens[:tokens.index(eos) + 1]
        return {"text": self.tokenizer.decode(tokens, skip_special_tokens=True),
                "tokens": tokens, "prompt_tokens": prompt_length,
                "finished": finished, "counts": dict(self.runtime.counts)}


def math_ceil(a, b):
    return (a + b - 1) // b
