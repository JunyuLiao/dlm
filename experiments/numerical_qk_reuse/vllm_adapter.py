"""Thin vLLM adapter: the UNCHANGED method core inside vLLM 0.30.0's native DiffusionGemma.

The core (v21.install -> v20.install -> global_scope.install -> integration.Attention + NativeReuseState) was written
against HF-shaped hooks. This adapter feeds it the same inputs from vLLM and touches nothing else:

- Modules. A stub model exposes one module whose class is named ``DiffusionGemmaEncoderModel`` (the core registers its
  ``invalidate`` pre-hook there) and one stub attention module per decoder layer (``layer_idx``, ``is_sliding``), so the
  core's own GLOBAL-only scope selection, identify hooks and sketch leases run unmodified. The HF binding
  (``runner._install_dense``) is replaced by a stub binding for the duration of the install; the core sets its
  ``attention_override`` as usual and the adapter calls it.
- Clock. Before every forward, the patched ``DiffusionGemmaModelState.prepare_attn`` reads the request's phase and
  denoising step (one host read per step, charged to the method arm):
    * encoder phase (prefill chunk or canvas commit): the encoder stub is called, firing the core's invalidate;
    * denoising step k of a canvas: ``State.begin(48 - k, canvas)``, exactly as the HF loop passes remaining steps.
  After every denoising sample, the wrapped ``_compiled_sample_step`` hands its temperature-scaled logits to
  ``State.observe_logits`` (the fast-T path only takes their argmax).
- K/V. vLLM keeps K/V in 64-token pages. Per GLOBAL layer and canvas the adapter builds one contiguous
  [1, hk, prefix + canvas, d] buffer: the prefix is copied once per canvas (gathered from the pages), the canvas region
  is refreshed from the pages every step. The core sees the prefix as the encoder cache (a stable view per canvas) and
  the whole buffer as the decoder's K/V. These copies are overhead the method arm pays and vLLM's dense arm does not.
- Output. The core returns model-major [1, Q, H, D]; it is written into vLLM's output buffer.

LOCAL layers, prefill, commits, the sampler and everything else stay vLLM's own code. The method arm must run with
``cudagraph_mode=PIECEWISE`` so the attention op executes eagerly every step (a FULL decode graph would freeze the
routing logic at capture time); the dense reference is measured in vLLM's default mode and in PIECEWISE.

Arms:
  'method'  : the core with a frozen v21 effective config (e.g. the main arm M3 R6 DP -ln2 + carry_first).
  'allkept' : the same K/V buffers and FA4 all-kept call (v27_fa4.dense) on every GLOBAL decoder call -- the
              adapter's own cost with no skipping (the analogue of D_fa4_allkept).
"""
from __future__ import annotations

import re
import sys
from contextlib import ExitStack
from types import ModuleType, SimpleNamespace

import torch
from torch import nn

MAX_DENOISING_STEPS = 48          # the HF loop's remaining-step clock starts each canvas at 48
_LAYER_RE = re.compile(r'layers\.(\d+)\.')
_RUNNER = 'experiments.diffusion_gemma_solattn_blasst_multibench.runner'
_BINDING = None


def _stub_runner():
    """The core imports ``_install_dense`` (the HF module binding) from the HF benchmark runner, whose import pulls in
    datasets/scipy and the HF evaluation stack. In vLLM that binding is exactly what the adapter replaces, so a stub
    module providing only ``_install_dense`` (returning the adapter's per-request binding) is registered instead."""
    if _RUNNER in sys.modules and not getattr(sys.modules[_RUNNER], '_v27_vllm_stub', False):
        raise RuntimeError('the real HF runner is loaded; the vLLM adapter needs its stub binding')
    if _RUNNER not in sys.modules:
        mod = ModuleType(_RUNNER)
        mod._v27_vllm_stub = True

        def _install_dense(adapter):
            if _BINDING is None:
                raise RuntimeError('no vLLM adapter binding is open')
            return _BINDING
        mod._install_dense = _install_dense
        sys.modules[_RUNNER] = mod


class DiffusionGemmaEncoderModel(nn.Module):
    """Stub whose class name matches the HF encoder: the core registers its invalidate pre-hook on it."""

    def forward(self, *args, **kwargs):
        return None


class _StubAttention(nn.Module):
    def __init__(self, layer_idx, is_sliding):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.is_sliding = bool(is_sliding)
        self.layer_type = 'sliding_attention' if is_sliding else 'full_attention'

    def forward(self, *args, **kwargs):
        return None


class _StubModel(nn.Module):
    def __init__(self, layer_types):
        super().__init__()
        self.encoder = DiffusionGemmaEncoderModel()
        self.layers = nn.ModuleList(_StubAttention(i, t == 'sliding_attention') for i, t in enumerate(layer_types))
        self.eval()


class _StubAdapter:
    """The few adapter attributes the core reads, backed by the stub model."""

    def __init__(self, layer_types):
        self.model = _StubModel(layer_types)

    @staticmethod
    def is_blasst_attention_module(name, module):
        return isinstance(module, _StubAttention)


class _CacheLayer:
    def __init__(self, keys, values):
        self.keys, self.values = keys, values


class _PrefixCache:
    """What the core's identify hook reads from the HF DynamicCache: the layer's stored prefix and the absolute length."""
    is_compileable = False

    def __init__(self, n_layers):
        self.layers = [None] * n_layers
        self.length = 0

    def get_seq_length(self):
        return self.length


class VllmMethodAdapter:
    def __init__(self, layer_types, config=None, condition=None, arm='method'):
        if arm not in ('method', 'allkept'):
            raise ValueError(arm)
        if arm == 'method' and (config is None or condition is None):
            raise ValueError('the method arm needs a frozen v21 effective config and its condition')
        self.layer_types = list(layer_types)
        self.global_layers = [i for i, t in enumerate(self.layer_types) if t != 'sliding_attention']
        self.config, self.condition, self.arm = config, condition, arm
        self.runtime = self.stub = self._stack = None
        self.bound = False               # True only while a request is in flight (dummy/warm-up runs pass through)
        self.pending_sample = False      # a denoising forward was prepared and its sample has not been seen yet
        self.step_ctx = None             # dict(phase, step, seq_len, slot) of the forward being prepared
        self.buffers = {}                # layer -> dict(k, v, prefix, nk, pk, pv)
        self.canvas = None
        self.cache = _PrefixCache(len(self.layer_types))
        self.calls = dict(global_calls=0, prefix_copies=0, canvas_refreshes=0, invalidates=0, begins=0, observes=0,
                          passthrough=0, order_errors=0)

    # ------------------------------------------------------------------ request lifecycle
    def begin_request(self):
        if self._stack is not None:
            raise RuntimeError('previous request still bound')
        self.buffers.clear()
        self.cache = _PrefixCache(len(self.layer_types))
        self.step_ctx = None
        for k in self.calls:
            self.calls[k] = 0
        self.pending_sample = False
        self.bound = True
        if self.arm == 'allkept':
            return
        global _BINDING
        _stub_runner()
        from experiments.numerical_qk_reuse import v21
        stub = _StubAdapter(self.layer_types)
        binding = SimpleNamespace(runtime=SimpleNamespace(attention_override=None),
                                  modules=[m for m in stub.model.modules() if isinstance(m, _StubAttention)
                                           and not m.is_sliding],
                                  close=lambda: None)
        self._stack = ExitStack()
        _BINDING = binding
        try:
            self.runtime = self._stack.enter_context(v21.install(stub, self.config, self.condition))
        finally:
            _BINDING = None
        self.stub, self.binding = stub, binding

    def end_request(self):
        counters = None
        if self.runtime is not None:
            counters = self.runtime['counters']()
        if self._stack is not None:
            self._stack.close()
        self._stack = self.runtime = self.stub = None
        self.buffers.clear()
        self.bound, self.step_ctx = False, None
        return dict(adapter=dict(self.calls), method=counters)

    # ------------------------------------------------------------------ clock (called from patched vLLM code)
    def on_prepare(self, phase_encoder, step, seq_len, num_tokens):
        """Before a forward: phase_encoder True for prefill/commit, else denoising step `step` of the canvas."""
        self.step_ctx = dict(encoder=bool(phase_encoder), step=int(step), seq_len=int(seq_len), n=int(num_tokens))
        if self.pending_sample:
            self.calls['order_errors'] += 1                       # the previous denoising sample was never observed
        if phase_encoder:
            self.calls['invalidates'] += 1
            if self.stub is not None:
                self.stub.model.encoder()                         # fires the core's invalidate pre-hooks
            return
        if self.canvas is None or self.canvas.shape[1] != num_tokens:
            self.canvas = torch.zeros(1, int(num_tokens), dtype=torch.long, device='cuda')
        self.calls['begins'] += 1
        self.pending_sample = True
        if self.runtime is not None:
            self.runtime['state'].begin(MAX_DENOISING_STEPS - int(step), self.canvas)

    def on_sample(self, scaled_logits):
        """After a denoising sample: the temperature-scaled logits [1, CL, V] (fast T takes their argmax)."""
        self.calls['observes'] += 1
        if not self.pending_sample:
            self.calls['order_errors'] += 1
        self.pending_sample = False
        if self.runtime is not None:
            self.runtime['state'].observe_logits(scaled_logits, None, None)

    # ------------------------------------------------------------------ GLOBAL decoder attention
    def active_for(self, layer_name):
        ctx = self.step_ctx
        if ctx is None or ctx['encoder']:
            return None
        m = _LAYER_RE.search(layer_name)
        if m is None:
            return None
        layer = int(m.group(1))
        return layer if layer in self.global_layers else None

    def _buffers(self, layer, key_cache, value_cache, block_table, prefix, n):
        """Contiguous [1, hk, prefix + n, d] K/V; prefix copied once per canvas, canvas region refreshed per call."""
        page = key_cache.shape[1]
        b = self.buffers.get(layer)
        nk = prefix + n
        if b is None or b['prefix'] != prefix or b['nk'] != nk:
            hk, d = key_cache.shape[2], key_cache.shape[3]
            kbuf = torch.empty((1, hk, nk, d), dtype=key_cache.dtype, device=key_cache.device)
            vbuf = torch.empty_like(kbuf)
            pages = block_table[: (nk + page - 1) // page].long()
            kbuf[0].copy_(key_cache[pages].reshape(-1, hk, d)[:nk].transpose(0, 1))
            vbuf[0].copy_(value_cache[pages].reshape(-1, hk, d)[:nk].transpose(0, 1))
            b = dict(k=kbuf, v=vbuf, prefix=prefix, nk=nk, pk=kbuf[:, :, :prefix], pv=vbuf[:, :, :prefix])
            self.buffers[layer] = b
            self.calls['prefix_copies'] += 1
        else:
            first, last = prefix // page, (nk - 1) // page
            pages = block_table[first: last + 1].long()
            off = prefix - first * page
            hk, d = key_cache.shape[2], key_cache.shape[3]
            b['k'][0, :, prefix:].copy_(key_cache[pages].reshape(-1, hk, d)[off: off + n].transpose(0, 1))
            b['v'][0, :, prefix:].copy_(value_cache[pages].reshape(-1, hk, d)[off: off + n].transpose(0, 1))
            self.calls['canvas_refreshes'] += 1
        return b

    def forward(self, impl, layer_idx, query, kv_cache, attn_metadata, output):
        ctx = self.step_ctx
        n = int(attn_metadata.num_actual_tokens)
        if n != ctx['n']:
            raise ValueError('forward token count differs from the prepared step')
        prefix = ctx['seq_len'] - n
        key_cache, value_cache = kv_cache.transpose(1, 2).split(impl.head_size, dim=-1)
        b = self._buffers(layer_idx, key_cache, value_cache, attn_metadata.block_table[0], prefix, n)
        q = query[:n].transpose(0, 1).unsqueeze(0)                  # [1, H, n, D] view, as HF's decoder passes it
        self.calls['global_calls'] += 1
        if self.arm == 'allkept':
            from experiments.numerical_qk_reuse import v27_fa4
            out = v27_fa4.dense(q, b['k'], b['v'], float(impl.scale))
        else:
            self.cache.layers[layer_idx] = _CacheLayer(b['pk'], b['pv'])
            self.cache.length = prefix
            module = self.stub.model.layers[layer_idx]
            module(None, None, None, past_key_values=self.cache)    # the core's identify / sketch hooks
            out, _ = self.binding.runtime.attention_override(module, q, b['k'], b['v'], None,
                                                             scaling=float(impl.scale), is_causal=False,
                                                             sliding_window=None)
        output[:n].view(n, -1).copy_(out.reshape(n, -1))
        return output


# ---------------------------------------------------------------------- vLLM patches
_ACTIVE = None


def install_vllm_patches(adapter: VllmMethodAdapter):
    """Patch vLLM classes in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) before the engine is built."""
    global _ACTIVE
    _ACTIVE = adapter
    import vllm.model_executor.models.diffusion_gemma as dg
    from vllm.v1.attention.backends import flash_attn as fa

    if getattr(dg, '_v27_patched', False):
        return
    prepare_attn = dg.DiffusionGemmaModelState.prepare_attn
    sample_step = dg._compiled_sample_step
    forward = fa.FlashAttentionImpl.forward

    def prepare_attn_patched(self, input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups,
                             kv_cache_config, for_capture=False, ubatch_idx=0):
        a = _ACTIVE
        if a is not None and a.bound:
            a.step_ctx = None
            if not for_capture and input_batch.num_reqs == 1:
                slot = int(input_batch.idx_mapping_np[0])
                st = self.diffusion_states
                phase, step, seq_len = torch.stack([st.is_encoder_phase[slot].long(), st.step[slot].long(),
                                                    input_batch.seq_lens[0].long()]).tolist()
                draft = input_batch.num_draft_tokens > 0
                a.on_prepare(bool(phase) or not draft, step, seq_len, int(input_batch.num_tokens))
        return prepare_attn(self, input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups,
                            kv_cache_config, for_capture=for_capture, ubatch_idx=ubatch_idx)

    def sample_step_patched(*args, **kwargs):
        scaled = sample_step(*args, **kwargs)
        a = _ACTIVE
        if a is not None and a.bound and a.step_ctx is not None and not a.step_ctx['encoder']:
            a.on_sample(scaled)
        return scaled

    def forward_patched(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None,
                        output_block_scale=None):
        a = _ACTIVE
        if a is not None and a.bound and attn_metadata is not None and output_scale is None:
            layer_idx = a.active_for(getattr(layer, 'layer_name', ''))
            if layer_idx is not None:
                return a.forward(self, layer_idx, query, kv_cache, attn_metadata, output)
            if a.step_ctx is not None:
                a.calls['passthrough'] += 1
        return forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale,
                       output_block_scale)

    dg.DiffusionGemmaModelState.prepare_attn = prepare_attn_patched
    dg._compiled_sample_step = sample_step_patched
    fa.FlashAttentionImpl.forward = forward_patched
    dg._v27_patched = True
