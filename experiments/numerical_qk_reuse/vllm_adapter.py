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
- FA4 execution (the consumer and every dense call the core makes through ``v27_fa4.sparse_lists``) runs on vLLM's own
  paged cache with a 2-way split: two batch entries alias the same pages through the page table, each takes one
  contiguous share of every kept-tile list, and the two partial outputs are merged exactly by their LSE. Same kernel,
  same lists; the split only fills the GPU (one 256-query canvas gives FA4 64 CTAs on 132 SMs). It is needed because
  on SM90 the kernel's own num_splits re-does the whole list per split with block sparsity, while vLLM's dense call
  gets an effective 2-way split from FA4's dynamic-causal path (bench: v27_fa4_sparse_split_alias_bench.py).

LOCAL layers, prefill, commits, the sampler and everything else stay vLLM's own code. The method arm must run with
``cudagraph_mode=PIECEWISE`` so the attention op executes eagerly every step (a FULL decode graph would freeze the
routing logic at capture time); the dense reference is measured in vLLM's default mode and in PIECEWISE.

Arms:
  'native'  : patches installed and GLOBAL calls intercepted, but vLLM's own attention runs (cost of the hooks alone).
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
    def __init__(self, layer_types, config=None, condition=None, arm='method', profile=False, lifecycle='legacy',
                 canvas_buffers='legacy', kv_copy_backend='torch', merge_backend='torch'):
        if arm not in ('method', 'allkept', 'native'):
            raise ValueError(arm)
        if lifecycle not in ('legacy', 'request_clear'):
            raise ValueError('lifecycle must be legacy or request_clear')
        if canvas_buffers not in ('legacy', 'release_after_invalidate'):
            raise ValueError('canvas_buffers must be legacy or release_after_invalidate')
        if kv_copy_backend not in ('torch', 'triton'):
            raise ValueError('kv_copy_backend must be torch or triton')
        if merge_backend not in ('torch', 'triton'):
            raise ValueError('merge_backend must be torch or triton')
        if arm == 'method' and (config is None or condition is None):
            raise ValueError('the method arm needs a frozen v21 effective config and its condition')
        if arm == 'method':
            # The vLLM sample hook supplies logits only. These HF variants need
            # the accepted-token mask, which this adapter does not yet expose.
            # Reject before request binding, model construction or GPU work.
            if config.get('sensitivity') == 'cgate':
                raise ValueError('vLLM adapter does not support C gate (sensitivity=cgate): '
                                 'the sample hook does not provide the accepted-token mask')
            if config.get('density_gate') is not None:
                raise ValueError('vLLM adapter does not support density_gate: '
                                 'the sample hook does not provide the accepted-token mask')
        self.layer_types = list(layer_types)
        self.global_layers = [i for i, t in enumerate(self.layer_types) if t != 'sliding_attention']
        self.config, self.condition, self.arm = config, condition, arm
        self.lifecycle = lifecycle      # explicit V28 execution setting, readable by runner receipts
        self.canvas_buffers = canvas_buffers
        self.kv_copy_backend = kv_copy_backend
        self.merge_backend = merge_backend
        self._merge_cache = {}
        self.runtime = self.stub = self._stack = None
        self.bound = False               # True only while a request is in flight (dummy/warm-up runs pass through)
        self.profile, self._events = bool(profile), []   # CUDA events around every intercepted GLOBAL call
        self.pending_sample = False      # a denoising forward was prepared and its sample has not been seen yet
        self.step_ctx = None             # dict(phase, step, seq_len, slot) of the forward being prepared
        self.buffers = {}                # layer -> dict(k, v, prefix, nk, pk, pv)
        self._buffer_streams = {}        # opt-in only: layer -> (device, CUDA stream handle)
        self._canvas_invalidate_epoch = 0
        self._canvas_release_epoch = None
        self.canvas = None
        self.cache = _PrefixCache(len(self.layer_types))
        self.paged = None                # vLLM paged K/V of the GLOBAL call in flight (for the split FA4 consumer)
        self.splits = 2
        self._split_cache = []           # (lists object, split lists) for held maps
        self.calls = dict(global_calls=0, prefix_copies=0, canvas_refreshes=0, invalidates=0, begins=0, observes=0,
                          passthrough=0, order_errors=0, split_fa4_calls=0, split_list_builds=0)
        if kv_copy_backend == 'triton':
            self.calls.update(triton_kv_copy_calls=0, triton_kv_copy_elements=0)
        if merge_backend == 'triton':
            self.calls.update(triton_lse_merge_calls=0, triton_lse_identity_builds=0)
        if self.canvas_buffers == 'release_after_invalidate':
            self.calls.update(canvas_release_calls=0, canvas_release_layers=0, canvas_release_bytes=0,
                              canvas_invalidate_epoch=0, canvas_release_epoch=0)

    # ------------------------------------------------------------------ request lifecycle
    def begin_request(self):
        if self._stack is not None or (self.lifecycle == 'request_clear' and self.bound):
            raise RuntimeError('previous request still bound')
        self.buffers.clear()
        self._clear_canvas_metadata()
        self.cache = _PrefixCache(len(self.layer_types))
        self.step_ctx = None
        for k in self.calls:
            self.calls[k] = 0
        self.pending_sample = False
        self.bound = True
        self._events = []
        if self.arm in ('allkept', 'native'):
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
        try:
            return self._end_request_impl()
        finally:
            self._clear_canvas_metadata()

    def _end_request_impl(self):
        if self.lifecycle == 'request_clear':
            return self._end_request_clear()
        timing = None
        if self._events:
            torch.cuda.synchronize()
            ms = [a.elapsed_time(b) for a, b in self._events]
            timing = dict(global_calls_timed=len(ms), global_call_ms_mean=round(sum(ms) / len(ms), 4),
                          global_ms_total=round(sum(ms), 2))
            self._events = []
        counters = None
        if self.runtime is not None:
            counters = self.runtime['counters']()
        if self._stack is not None:
            self._stack.close()
        self._stack = self.runtime = self.stub = None
        self.buffers.clear()
        self.bound, self.step_ctx = False, None
        return dict(adapter=dict(self.calls), method=counters, timing=timing)

    def _end_request_clear(self):
        """Close the core before dropping request references, even on a failed close.

        The caller retains the existing request-boundary synchronization before
        end_request. No allocator flush, new synchronization or canvas-boundary
        reset is introduced here; carry and async routing stay unchanged.
        """
        try:
            timing = None
            if self._events:
                torch.cuda.synchronize()
                ms = [a.elapsed_time(b) for a, b in self._events]
                timing = dict(global_calls_timed=len(ms), global_call_ms_mean=round(sum(ms) / len(ms), 4),
                              global_ms_total=round(sum(ms), 2))
            counters = self.runtime['counters']() if self.runtime is not None else None
            return dict(adapter=dict(self.calls), method=counters, timing=timing)
        finally:
            try:
                if self._stack is not None:
                    self._stack.close()
            finally:
                # Prefix views keep the entire contiguous K/V storage alive.
                if self.cache is not None:
                    self.cache.layers.clear()
                # The stub binding's close is a no-op; its override otherwise
                # retains the closed router and its DP/map caches.
                binding = getattr(self, 'binding', None)
                if binding is not None:
                    binding.runtime.attention_override = None
                self._stack = self.runtime = self.stub = self.binding = None
                self.cache = self.paged = self.canvas = None
                self.buffers.clear()
                self._split_cache.clear()
                self._events.clear()
                self.bound, self.step_ctx, self.pending_sample = False, None, False

    def _clear_canvas_metadata(self):
        self._merge_cache.clear()
        self._buffer_streams.clear()
        self._canvas_invalidate_epoch = 0
        self._canvas_release_epoch = None

    @staticmethod
    def _stream_identity(device=None):
        stream = torch.cuda.current_stream(device)
        return str(stream.device), int(stream.cuda_stream)

    def _release_canvas_buffers(self, epoch):
        """Drop old contiguous KV only after a successful encoder invalidation.

        Observation/projection reads KV on its creation stream; the async
        selector reads separately allocated tail/sketch/summary arrays, whose
        existing record_stream calls protect them. Same-stream allocator reuse
        orders replacement allocations after outstanding KV readers, without
        a host or device synchronization. Carry/split maps remain untouched.
        """
        if epoch != self._canvas_invalidate_epoch or epoch <= 0 or epoch == self._canvas_release_epoch:
            raise RuntimeError('canvas release requires a fresh successful encoder invalidation')
        layers = list(self.buffers)
        # Validate every layer before modifying either owner. Pointer/stream
        # object identity alone is insufficient: use the CUDA handle + device.
        for layer in layers:
            created = self._buffer_streams.get(layer)
            if created is None or self._stream_identity(self.buffers[layer]['k'].device) != created:
                raise RuntimeError('canvas buffer CUDA stream changed; refusing release')
        if self.cache is None or any(source is not None and layer not in self.buffers
                                     for layer, source in enumerate(self.cache.layers)):
            raise RuntimeError('canvas cache owner has no guarded buffer; refusing release')
        nbytes = sum(b[key].numel() * b[key].element_size()
                     for b in self.buffers.values() for key in ('k', 'v'))
        self.buffers.clear()
        self.cache.layers[:] = [None] * len(self.cache.layers)
        self._buffer_streams.clear()
        self._canvas_release_epoch = epoch
        self.calls['canvas_release_epoch'] = epoch
        if layers:
            self.calls['canvas_release_calls'] += 1
            self.calls['canvas_release_layers'] += len(layers)
            self.calls['canvas_release_bytes'] += nbytes

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
            if self.canvas_buffers == 'release_after_invalidate':
                self._canvas_invalidate_epoch += 1
                self.calls['canvas_invalidate_epoch'] = self._canvas_invalidate_epoch
                self._release_canvas_buffers(self._canvas_invalidate_epoch)
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
        stream = None
        if self.canvas_buffers == 'release_after_invalidate':
            stream = self._stream_identity(key_cache.device)
            if b is not None and self._buffer_streams.get(layer) != stream:
                raise RuntimeError('canvas buffer CUDA stream changed; refusing use')
        nk = prefix + n
        if b is None or b['prefix'] != prefix or b['nk'] != nk:
            hk, d = key_cache.shape[2], key_cache.shape[3]
            kbuf = torch.empty((1, hk, nk, d), dtype=key_cache.dtype, device=key_cache.device)
            vbuf = torch.empty_like(kbuf)
            if self.kv_copy_backend == 'triton':
                self._copy_paged(key_cache, value_cache, block_table, kbuf, vbuf, 0, nk)
            else:
                pages = block_table[: (nk + page - 1) // page].long()
                kbuf[0].copy_(key_cache[pages].reshape(-1, hk, d)[:nk].transpose(0, 1))
                vbuf[0].copy_(value_cache[pages].reshape(-1, hk, d)[:nk].transpose(0, 1))
            b = dict(k=kbuf, v=vbuf, prefix=prefix, nk=nk, pk=kbuf[:, :, :prefix], pv=vbuf[:, :, :prefix])
            self.buffers[layer] = b
            if stream is not None:
                self._buffer_streams[layer] = stream
            self.calls['prefix_copies'] += 1
        else:
            first, last = prefix // page, (nk - 1) // page
            if self.kv_copy_backend == 'triton':
                self._copy_paged(key_cache, value_cache, block_table, b['k'], b['v'], prefix, n)
            else:
                pages = block_table[first: last + 1].long()
                off = prefix - first * page
                hk, d = key_cache.shape[2], key_cache.shape[3]
                b['k'][0, :, prefix:].copy_(key_cache[pages].reshape(-1, hk, d)[off: off + n].transpose(0, 1))
                b['v'][0, :, prefix:].copy_(value_cache[pages].reshape(-1, hk, d)[off: off + n].transpose(0, 1))
            self.calls['canvas_refreshes'] += 1
        return b

    def _copy_paged(self, key, value, table, out_key, out_value, start, count):
        from experiments.numerical_qk_reuse.v29_paged_copy import copy_paged_kv
        copy_paged_kv(key, value, table, out_key, out_value, start, count)
        self.calls['triton_kv_copy_calls'] += 1
        self.calls['triton_kv_copy_elements'] += 2 * key.shape[2] * count * key.shape[3]

    def forward(self, impl, layer_idx, query, kv_cache, attn_metadata, output):
        ctx = self.step_ctx
        n = int(attn_metadata.num_actual_tokens)
        if n != ctx['n']:
            raise ValueError('forward token count differs from the prepared step')
        prefix = ctx['seq_len'] - n
        key_cache, value_cache = kv_cache.transpose(1, 2).split(impl.head_size, dim=-1)
        b = self._buffers(layer_idx, key_cache, value_cache, attn_metadata.block_table[0], prefix, n)
        page = key_cache.shape[1]
        self.paged = dict(k=key_cache, v=value_cache, nk=prefix + n,
                          table=attn_metadata.block_table[0, : (prefix + n + page - 1) // page])
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
        self.paged = None
        output[:n].view(n, -1).copy_(out.reshape(n, -1))
        return output

    # ------------------------------------------------------------------ split FA4 over the paged cache
    def _split(self, lists):
        for ref, split in self._split_cache:
            if ref is lists:
                return split
        from experiments.numerical_qk_reuse import v27_fa4
        S = self.splits
        order, cnt = lists.full_block_idx, lists.full_block_cnt          # [1, H, QB, KT], [1, H, QB]
        kt = order.shape[-1]
        ar = torch.arange(kt, device=order.device)
        lo = [(cnt * i) // S for i in range(S + 1)]
        idx = torch.cat([torch.gather(order, -1, (lo[i][..., None] + ar).clamp_max(kt - 1)) for i in range(S)])
        counts = torch.cat([lo[i + 1] - lo[i] for i in range(S)]).to(torch.int32)
        zeros = torch.zeros_like(counts)
        split = v27_fa4._BST(mask_block_cnt=zeros, mask_block_idx=torch.zeros(zeros.shape + (1,), device=zeros.device,
                                                                               dtype=torch.int32),
                             full_block_cnt=counts.contiguous(), full_block_idx=idx.to(torch.int32).contiguous(),
                             block_size=lists.block_size)
        self._split_cache = self._split_cache[-63:] + [(lists, split)]
        self.calls['split_list_builds'] += 1
        return split

    def sparse_lists(self, original, q, k, v, lists, scale):
        ctx = self.paged
        if ctx is None or k.shape[-2] != ctx['nk'] or q.shape[0] != 1:
            return original(q, k, v, lists, scale)
        from experiments.numerical_qk_reuse import v27_fa4
        fwd = v27_fa4.load()
        S = self.splits
        split = self._split(lists)
        qs = q.transpose(1, 2).expand(S, -1, -1, -1)                      # [S, Q, H, D], stride-0 batch
        table = ctx['table'][None].expand(S, -1).contiguous()
        used = torch.full((S,), ctx['nk'], device=q.device, dtype=torch.int32)
        o, lse = fwd(qs, ctx['k'], ctx['v'], softmax_scale=scale, causal=False, page_table=table, seqused_k=used,
                     block_sparse_tensors=split, num_splits=1, return_lse=True)[:2]
        self.calls['split_fa4_calls'] += 1
        if self.merge_backend == 'triton':
            return self._merge_alias2(o, lse)
        w = torch.softmax(lse, dim=0).permute(0, 2, 1)[..., None]         # [S, Q, H, 1], exact LSE merge
        return (o.float() * w).sum(0, keepdim=True).to(o.dtype)           # [1, Q, H, D]

    def _merge_alias2(self, partials, lse):
        from experiments.numerical_qk_reuse.v29_lse_merge import Alias2MappedMerge
        if self.splits != 2:
            raise ValueError('fused identity merge requires exactly two alias splits')
        key = (partials.shape[2], partials.shape[1], str(partials.device))
        merger = self._merge_cache.get(key)
        if merger is None:
            merger = Alias2MappedMerge.identity(key[0], key[1], partials.device)
            self._merge_cache[key] = merger
            self.calls['triton_lse_identity_builds'] += 1
        result = merger(partials, lse)
        self.calls['triton_lse_merge_calls'] += 1
        return result


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
                ev = None
                if a.profile:
                    ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    ev[0].record()
                if a.arm == 'native':
                    a.calls['global_calls'] += 1
                    r = forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale,
                                output_block_scale)
                else:
                    r = a.forward(self, layer_idx, query, kv_cache, attn_metadata, output)
                if ev is not None:
                    ev[1].record()
                    a._events.append(ev)
                return r
            if a.step_ctx is not None:
                a.calls['passthrough'] += 1
        return forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale,
                       output_block_scale)

    from experiments.numerical_qk_reuse import v27_fa4
    sparse_lists = v27_fa4.sparse_lists

    def sparse_lists_patched(q, k, v, lists, scale):
        a = _ACTIVE
        if a is not None and a.bound and a.paged is not None:
            return a.sparse_lists(sparse_lists, q, k, v, lists, scale)
        return sparse_lists(q, k, v, lists, scale)

    v27_fa4.sparse_lists = sparse_lists_patched
    dg.DiffusionGemmaModelState.prepare_attn = prepare_attn_patched
    dg._compiled_sample_step = sample_step_patched
    fa.FlashAttentionImpl.forward = forward_patched
    dg._v27_patched = True
