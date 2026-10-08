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
  'mage'    : port of MAGE (arXiv 2602.14209, prior art) on the same execution path: at the first denoising call of
              each canvas, exact dense output plus a per-KV-head top-k of 64-key tiles by softmax mass averaged over
              the canvas queries and the GQA group (MAGE eq. 5); every later call of the canvas reuses it. Fixed
              budget mage_k tokens; canvas tiles always kept (port decision: the current block is always attended).
  'allkept' : the same K/V buffers and FA4 all-kept call (v27_fa4.dense) on every GLOBAL decoder call -- the
              adapter's own cost with no skipping (the analogue of D_fa4_allkept).
"""
from __future__ import annotations

import math
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


MAGE_GRANULARITIES = ('kvhead', 'qhead', 'qblock', 'qblock_max', 'kvhead_max', 'kvblock_max')
ROW_WEIGHTS = (None, 'cgate', 'conf', 'margin', 'temporal', 'mt', 'ct', 'jcgate')
TRIGGER_SIGNALS = ('accept', 'settle')


class VllmMethodAdapter:
    def __init__(self, layer_types, config=None, condition=None, arm='method', profile=False, lifecycle='legacy',
                 canvas_buffers='legacy', kv_copy_backend='torch', merge_backend='torch', mage_k=1024,
                 logit_stats='legacy', dp_build='legacy', observe_backend='triton', mage_select='torch',
                 trace_canvas=False, dense_when=None, regroup_diag=False, mage_critical=None,
                 mage_coverage=None, mage_select_step=0, mage_granularity='kvhead', mage_keep_frac=None,
                 residual=None, drop_guard=None, drift_diag=False, dense_below=None, mage_carry_first=False,
                 risk_group=None, mage_reselect=None, mage_row_weight=None, mage_beta=3.0, mage_reselect_k=None,
                 mage_reselect_trigger=None, mage_trigger_signal='accept', mage_reselect_kmin=None,
                 mage_cg_tau=2.5, mage_cg_gamma_q=0.65, mage_clock_trace=False, mage_sink=0, mage_recent=0,
                 mage_trigger_relative=False, mage_pool=None, mage_kcover=None, mage_kq=0.75, mage_kmax=16384,
                 mage_sticky=None,
                 cg_stop=None, stall_rescue=None, stall_eps=0.01, local_kv_budget=None,
                 local_kernel='compact_triton', value_selector=None, value_threshold=None, value_audit=False,
                 value_max_tokens=None, value_clean_timing=False):
        if arm not in ('method', 'allkept', 'native', 'mage'):
            raise ValueError(arm)
        if value_selector is not None:
            from .v31_value_selectors import SELECTORS
            if value_selector not in SELECTORS:
                raise ValueError('unknown value selector')
            if (arm != 'mage' or mage_select != 'fa4' or mage_granularity != 'qblock_max'
                    or local_kv_budget is not None or mage_keep_frac is not None
                    or mage_pool is not None or mage_kcover is not None or mage_row_weight is not None
                    or mage_critical is not None or mage_coverage is not None
                    or residual is not None or drop_guard is not None):
                raise ValueError('value selectors require fixed-budget GLOBAL qblock_max lifecycle, native LOCAL')
            if value_selector in SELECTORS[:2] and (value_threshold is None or float(value_threshold) < 0
                                                    or not math.isfinite(float(value_threshold))):
                raise ValueError('online selectors require a finite uniform nonnegative threshold')
        self.value_selector, self.value_threshold = value_selector, value_threshold
        self.value_clean_timing = bool(value_clean_timing)
        self._value_projection = None
        self._value_workspace = None
        if value_selector is not None and value_max_tokens is not None:
            from .v31_value_workspace import ValueWorkspace
            self._value_workspace = ValueWorkspace(int(value_max_tokens))
        self._value_pending = []
        self.value_audit, self._value_audit_events = bool(value_audit), []
        if lifecycle not in ('legacy', 'request_clear'):
            raise ValueError('lifecycle must be legacy or request_clear')
        if canvas_buffers not in ('legacy', 'release_after_invalidate'):
            raise ValueError('canvas_buffers must be legacy or release_after_invalidate')
        if kv_copy_backend not in ('torch', 'triton'):
            raise ValueError('kv_copy_backend must be torch or triton')
        if merge_backend not in ('torch', 'triton'):
            raise ValueError('merge_backend must be torch or triton')
        if logit_stats not in ('legacy', 'fused'):
            raise ValueError('logit_stats must be legacy or fused')
        if dp_build not in ('legacy', 'chunked'):
            raise ValueError('dp_build must be legacy or chunked')
        if observe_backend not in ('triton', 'fa4'):
            raise ValueError('observe_backend must be triton or fa4')
        if mage_select not in ('torch', 'fa4'):
            raise ValueError('mage_select must be torch or fa4')
        if (mage_critical is not None or mage_coverage is not None) and mage_select != 'fa4':
            raise ValueError('mage_critical / mage_coverage are implemented on the FA4 selection only (mage_select=fa4)')
        if mage_granularity not in MAGE_GRANULARITIES:
            raise ValueError(f'mage_granularity must be one of {MAGE_GRANULARITIES}')
        if mage_granularity != 'kvhead' and (mage_select != 'fa4' or mage_critical is not None
                                             or mage_coverage is not None):
            raise ValueError('mage_granularity other than kvhead needs mage_select=fa4, without critical / coverage')
        if mage_keep_frac is not None and (not 0.0 < float(mage_keep_frac) <= 1.0 or mage_select != 'fa4'):
            raise ValueError('mage_keep_frac must be in (0, 1] and needs mage_select=fa4')
        if mage_carry_first and int(mage_select_step) < 1:
            raise ValueError('mage_carry_first replaces the first exact call of a canvas: needs mage_select_step >= 1')
        if mage_reselect is not None:
            mage_reselect = tuple(sorted({int(x) for x in mage_reselect}))
            if (not mage_reselect or mage_reselect[0] <= int(mage_select_step) or mage_select != 'fa4'
                    or mage_critical is not None or mage_coverage is not None):
                raise ValueError('mage_reselect: canvas call indices after mage_select_step, FA4 selection, no '
                                 'critical / coverage')
        if mage_reselect_trigger is not None:
            th = mage_reselect_trigger if isinstance(mage_reselect_trigger, (list, tuple)) else [mage_reselect_trigger]
            th = tuple(sorted({float(x) for x in th}))
            if not th or not all(0.0 < f <= 1.0 for f in th) or mage_reselect is not None or mage_select != 'fa4':
                raise ValueError('mage_reselect_trigger: progress thresholds in (0, 1], FA4 selection, instead of '
                                 'mage_reselect')
            mage_reselect_trigger = th
        if mage_trigger_signal not in TRIGGER_SIGNALS or (mage_trigger_signal != 'accept' and mage_reselect_trigger is None):
            raise ValueError(f'mage_trigger_signal must be one of {TRIGGER_SIGNALS}, with mage_reselect_trigger')
        if mage_reselect_kmin is not None and (mage_reselect_trigger is None or int(mage_reselect_kmin) < 64):
            raise ValueError('mage_reselect_kmin: a token floor (>= 64) for progress-following re-selection budgets, '
                             'with mage_reselect_trigger')
        if mage_row_weight == 'jcgate' and mage_reselect_trigger is None:
            raise ValueError("mage_row_weight 'jcgate' needs the progress clock (mage_reselect_trigger)")
        protect = int(mage_sink) + int(mage_recent) // 64
        if int(mage_sink) < 0 or int(mage_recent) < 0 or int(mage_recent) % 64 or (
                protect and (mage_select != 'fa4' or mage_keep_frac is not None
                             or mage_granularity in ('kvhead', 'kvblock_max', 'kvhead_max')
                             or protect >= int(mage_k) // 64)):
            raise ValueError('mage_sink (tiles) / mage_recent (tokens, a multiple of 64): FA4 query-head selection with '
                             'a token budget mage_k larger than the protected tiles')
        if mage_pool is not None and (int(mage_pool) < 2 or mage_select != 'fa4' or mage_granularity != 'qblock_max'
                                      or mage_keep_frac is not None
                                      or (mage_reselect is None and mage_reselect_trigger is None)):
            raise ValueError('mage_pool: a pool factor >= 2 for re-selections (mage_reselect / trigger) of the FA4 '
                             'qblock_max selection with a token budget')
        if mage_kcover is not None and (not 0.0 < float(mage_kcover) < 1.0 or not 0.0 <= float(mage_kq) <= 1.0
                                        or int(mage_kmax) < int(mage_k) or mage_select != 'fa4'
                                        or mage_granularity != 'qblock_max' or mage_keep_frac is not None):
            raise ValueError('mage_kcover: a mass coverage in (0, 1) with a quantile in [0, 1] and mage_kmax >= mage_k, '
                             'on the FA4 qblock_max selection with a token budget')
        if cg_stop is not None and int(cg_stop) < 1:
            raise ValueError('cg_stop: a stable-run length >= 1 (steps)')
        if stall_rescue is not None and (int(stall_rescue) < 1 or arm not in ('method', 'mage') or float(stall_eps) < 0):
            raise ValueError('stall_rescue: a step count >= 1 on a sparse arm (method / mage)')
        if mage_sticky is not None and (float(mage_sticky) < 0 or (mage_reselect is None and mage_reselect_trigger is None)
                                        or mage_granularity != 'qblock_max' or not mage_carry_first):
            raise ValueError('mage_sticky: a log-share bonus >= 0 for held tiles in re-selections (mage_reselect / '
                             'trigger) of the qblock_max selection with mage_carry_first')
        if mage_trigger_relative and mage_reselect_trigger is None:
            raise ValueError('mage_trigger_relative rescales the progress signal: needs mage_reselect_trigger')
        if mage_clock_trace and mage_reselect_trigger is None:
            raise ValueError('mage_clock_trace traces the progress clock: needs mage_reselect_trigger')
        if mage_cg_tau <= 0 or not 0.0 <= mage_cg_gamma_q < 1.0:
            raise ValueError('invalid C-gate parameters')
        if mage_reselect_trigger is not None and int(mage_select_step) < 1:
            raise ValueError('mage_reselect_trigger re-selects after the step-1 selection: needs mage_select_step >= 1')
        if mage_row_weight not in ROW_WEIGHTS:
            raise ValueError(f'mage_row_weight must be one of {ROW_WEIGHTS}')
        if mage_reselect_k is not None and ((mage_reselect is None and mage_reselect_trigger is None)
                                            or int(mage_reselect_k) < 64
                                            or mage_keep_frac is not None):
            raise ValueError('mage_reselect_k: a token budget (>= 64) for re-selections, with mage_reselect and mage_k')
        if mage_row_weight is not None and ((mage_reselect is None and mage_reselect_trigger is None)
                                            or mage_granularity != 'qblock_max'):
            raise ValueError('mage_row_weight weights the rows of a re-selection: needs mage_reselect and qblock_max')
        if risk_group not in (None, 'kv'):
            raise ValueError("risk_group must be None or 'kv'")
        if risk_group is not None and (arm != 'method' or config is None or config.get('risk_topk') is None
                                       or config.get('risk_budget') is not None or config.get('q_block') == 64):
            raise ValueError('risk_group needs the method arm with a fixed-fraction top-k config (risk_topk, '
                             'no risk_budget) on 128-row blocks')
        from experiments.numerical_qk_reuse.v31_residual import RESIDUAL_MODES
        if residual is not None and residual not in RESIDUAL_MODES:
            raise ValueError(f'residual must be one of {RESIDUAL_MODES}')
        if drop_guard is not None and not 0.0 < float(drop_guard) < 1.0:
            raise ValueError('drop_guard must be in (0, 1)')
        if arm == 'method' and (config is None or condition is None):
            raise ValueError('the method arm needs a frozen v21 effective config and its condition')
        if arm == 'method':
            # C gate (v31): the accepted-token mask is recomputed from the sampler's own temperature-scaled logits
            # with its entropy-bound acceptance rule (see accepted_mask). The density gate still needs the
            # sampler's stop statistics and stays rejected.
            if config.get('density_gate') is not None:
                raise ValueError('vLLM adapter does not support density_gate: '
                                 'the sample hook does not provide the accepted-token mask')
        self.layer_types = list(layer_types)
        self.global_layers = [i for i, t in enumerate(self.layer_types) if t != 'sliding_attention']
        self.local_kv_budget = None if local_kv_budget is None else int(local_kv_budget)
        if local_kernel not in ('compact_triton', 'fa4'):
            raise ValueError("local_kernel must be 'compact_triton' or 'fa4'")
        self.local_kernel = local_kernel
        self.local_router = None
        if self.local_kv_budget is not None:
            if arm not in ('mage', 'native'):
                raise ValueError('LOCAL routing currently supports mage and native controls')
            from experiments.numerical_qk_reuse.v31_local_sparse import LocalSparse
            self.local_router = LocalSparse(self.local_kv_budget, kernel=self.local_kernel)
        self.config, self.condition, self.arm = config, condition, arm
        self.lifecycle = lifecycle      # explicit V28 execution setting, readable by runner receipts
        self.canvas_buffers = canvas_buffers
        self.kv_copy_backend = kv_copy_backend
        self.merge_backend = merge_backend
        # v31 execution variant: 'fused' computes the sampler hook's per-position statistics (argmax, top-1
        # probability, entropy -> acceptance mask) in one pass over the logits (v31_logit_stats); 'legacy' = torch ops
        self.logit_stats = logit_stats
        # v31 execution variants of the observation path (same selector formulas):
        #   dp_build='chunked'    dense-prefix state by a parallel chunked scan (v31_dp_chunked) instead of the
        #                         sequential per-(block, head) loop
        #   observe_backend='fa4' observation calls without a per-row projected V (compact pooled mu) run the
        #                         official FA4 dense kernel with the in-kernel prefix log-mass (v31_fa4_observe)
        self.dp_build, self.observe_backend = dp_build, observe_backend
        # MAGE port: 'torch' = chunked FP32 QK for the selection (reference); 'fa4' = the same eq. 5 statistics from
        # the FA4 in-kernel tile log-mass (the selection call's output is that FA4 dense output)
        self.mage_select = mage_select
        # v31 two-level selection (named variant on the MAGE port, FA4 selection only): MAGE's shared fixed budget
        # (mean mass over the canvas queries and the query heads of a KV head, top k) UNION the per-query-head
        # critical prefix tiles whose mass share reaches mage_critical for ANY row of a 128-row block
        self.mage_critical = None if mage_critical is None else float(mage_critical)
        # v31 coverage-adaptive budget (named variant on the MAGE port, FA4 selection only): per KV head keep the
        # fewest prefix tiles (at least mage_k / 64) whose mean mass, together with the always-kept canvas tiles,
        # covers mage_coverage of the head's mean attention mass: concentrated heads stay at the floor, diffuse heads
        # (aggregation over the whole context) keep more
        self.mage_coverage = None if mage_coverage is None else float(mage_coverage)
        # v31 selection-timing control (named variant on the MAGE port): run the first mage_select_step GLOBAL calls
        # of each canvas dense and select from the queries of the next call (MAGE selects at step 0, all-mask
        # canvas; the method observes at step 1 after two exact steps). Separates when to select from how.
        if int(mage_select_step) < 0:
            raise ValueError('mage_select_step must be >= 0')
        self.mage_select_step = int(mage_select_step)
        # v31 selection-granularity ladder (named variants on the MAGE port, ablation from MAGE toward the method's
        # mass ranking): the unit that owns a top-k set and how its rows are aggregated --
        #   kvhead      MAGE: mean row-normalized mass over the canvas queries and the query heads of a KV head;
        #   qhead       mean over the canvas queries of each query head;
        #   qblock      mean over the rows of each (query head, 128-row block);
        #   qblock_max  max over the rows of each (query head, 128-row block) of the row's share of its PREFIX mass
        #               (the method's risk_value='mass' statistic, from the exact FA4 observation);
        #   kvblock_max the same max share, taken over the rows of a 128-row block AND the query heads of its KV head:
        #               one set per (KV head, block), shared by those query heads (same per-head budget, so the same
        #               work; the shared lists let the heads of a GQA group reuse each K/V tile read, like MAGE's);
        #   kvhead_max  the same max share over all canvas rows and the group's heads: one set per KV head (MAGE's
        #               exact list structure and cost, with the max-share statistic instead of the mean mass).
        # mage_keep_frac: keep PT - floor((1 - f) PT) prefix tiles per unit (topk_skip's rule, so f = 0.12 matches the
        # method's k12 at every length) instead of mage_k / 64.
        self.mage_granularity = mage_granularity
        self.mage_keep_frac = None if mage_keep_frac is None else float(mage_keep_frac)
        # v31 first-call carry on the MAGE port (named variant, the method's carry_first): with mage_select_step >= 1,
        # canvas call 0 runs on the layer's selection of the PREVIOUS canvas instead of exact attention -- tiles wholly
        # in that canvas's prefix keep their decision, every newer tile (the previous canvas, now prefix, and the
        # current canvas) is kept. Valid only as the direct continuation (prefix grown by exactly that canvas, same
        # block layout); the request's first canvas and any invalid carry run the exact call as before.
        self.mage_carry_first = bool(mage_carry_first)
        # v31 group-shared method selection (named variant, method arm, fixed-fraction top-k configs only): the
        # core's per-(query head, block) top-k (v27_dense_prefix.topk_skip) becomes one decision per (KV head, block)
        # on the max of the heads' worst-row values (v31_group_select): same tiles per head, identical lists in a group
        self.risk_group = risk_group
        # v31 progress-aware re-selection (named variant on the MAGE port): at the canvas calls listed in mage_reselect
        # (0-based denoising step of the canvas) the layer observes again (exact output) and re-selects, replacing the
        # held lists (the carry then uses the latest selection). mage_row_weight weights the rows of that re-selection
        # from the sampler of the step before it -- 'cgate': rows the sampler ACCEPTED get weight 0 (Junyu Liao's C gate
        # idea: spend the budget on positions still being decided); 'conf': 1 + beta * sqrt(1 - p_top) clamped to
        # [1, 1 + beta] (his confidence prior / query sensitivity T, beta 3 as in query_adaptive.weight). In the
        # worst-row max-share statistic a weight enters as + log(w) per row; a block whose rows all weigh 0 falls back
        # to the unweighted statistic. Weights are computed only on the step before a re-selection.
        self.mage_reselect = mage_reselect
        self.mage_row_weight, self.mage_beta = mage_row_weight, float(mage_beta)
        self._mage_count, self._mage_row_w, self._mage_units_w = {}, None, None
        # round 2: Junyu Liao's query-sensitivity family (query_adaptive.weight, beta 3, m_ref 1) as row weights --
        # 'margin' M = 1 + beta / (top1 - top2 logit margin + 1), 'conf' C = 1 + beta sqrt(1 - p_top), 'temporal'
        # T = 1 + beta [argmax changed over the last step], 'mt' / 'ct' = sqrt(M T) / sqrt(C T), each clamped to
        # [1, 1 + beta]; 'cgate' = rows not accepted. mage_reselect_k: the re-selection's own token budget (a budget
        # schedule over denoising progress; per-unit counts stay equal, so CTAs stay balanced).
        self.mage_reselect_k = None if mage_reselect_k is None else int(mage_reselect_k)
        self._mage_prev_argmax, self._k_override = None, None
        # round 3: mage_reselect_trigger f -- the re-selection is triggered by denoising PROGRESS instead of a fixed step:
        # after each denoising step the sampler's acceptance mask (official entropy-bound rule, fused row statistics)
        # gives the accepted fraction of the canvas rows; the first time it reaches f, every layer re-selects at the
        # next call (once per canvas; row weights as above, from that same step). One small host read per step until
        # the canvas has triggered (the adapter's prepare hook already reads the step metadata once per step).
        self.mage_reselect_trigger = mage_reselect_trigger          # tuple of thresholds (round 4) or None
        self._trig_canvas, self._trig_at = None, None
        # round 4: the progress clock carries Junyu Liao's full C gate (query_adaptive.State.enable_cgate; COLLABORATION
        # CANDIDATE). Per canvas row, from the previous completed steps: q = EMA (gamma_q 0.65) of 'renoised' (= not
        # accepted, q starts at 1), the stable run r (reset on an argmax flip, else (r + 1) * accepted) through
        # g = 1 - exp(-r / tau) (tau 2.5), and u = sqrt(1 - p_top). Settledness = g (1 - q) (1 - u) in [0, 1].
        #  - mage_trigger_signal 'settle': the progress signal is the canvas mean settledness instead of the accepted
        #    fraction; mage_reselect_trigger may list several thresholds (one re-selection per crossing);
        #  - mage_row_weight 'jcgate': the re-selection weighs its rows by Junyu's sensitivity
        #    s = clip(1 + beta (1 - settledness), 1, 1 + beta) (unsettled rows steer the choice);
        #  - mage_reselect_kmin: the budget follows progress -- a re-selection at progress phi keeps
        #    max(kmin, K (1 - phi)) tokens per unit (rounded down to 64), K = mage_reselect_k or mage_k. Every unit keeps
        #    the same count at every step, so CTAs stay balanced: Junyu's allocation over queries becomes an allocation
        #    over denoising time.
        # The histories are device tensors updated once per step (no host read); the progress signal is one scalar read
        # per step until the canvas has crossed its last threshold.
        self.mage_trigger_signal = mage_trigger_signal
        self.mage_reselect_kmin = None if mage_reselect_kmin is None else int(mage_reselect_kmin)
        self.mage_cg_tau, self.mage_cg_gamma_q = float(mage_cg_tau), float(mage_cg_gamma_q)
        self._trig_count, self._trig_k, self._jc = 0, None, None
        self.mage_clock_trace, self._clock = bool(mage_clock_trace), []
        # round 4d: relative progress -- phi_rel = (phi - phi_1) / (1 - phi_1) with phi_1 the signal after the canvas's
        # first step (that step never triggers): rows the sampler accepts at once (trailing end-of-text positions in
        # short answers) no longer count as progress.
        self.mage_trigger_relative, self._trig_phi0 = bool(mage_trigger_relative), None
        # round 5: two-level selection -- the step-1 dense observation also marks a pool of mage_pool x k tiles per
        # unit; re-selections observe inside the block-sparse kernel over that pool (_mage_select_pool).
        self.mage_pool, self._last_pool = (None if mage_pool is None else int(mage_pool)), None
        # round 7: C gate as step control (_cg_track / _cg_stop / _stall_check)
        self.cg_stop = None if cg_stop is None else int(cg_stop)
        self.stall_rescue = None if stall_rescue is None else int(stall_rescue)
        self.stall_eps = float(stall_eps)
        self._cg = self._cg_canvas = None
        self._stall_best, self._stall_since = -1.0, 0
        # round 8: sticky re-selection -- held tiles get + mage_sticky on their log-share score in a re-selection
        self.mage_sticky, self._mage_held = (None if mage_sticky is None else float(mage_sticky)), None
        # round 6: coverage-calibrated balanced budget -- see _coverage_tiles
        self.mage_kcover = None if mage_kcover is None else float(mage_kcover)
        self.mage_kq, self.mage_kmax = float(mage_kq), int(mage_kmax)
        # round 4c: in-budget protection -- the first mage_sink prefix tiles (attention sink) and the last mage_recent
        # prefix tokens (the most recently committed canvases) win every unit's top-k before the scored tiles, so each
        # unit still keeps exactly k tiles. Re-selections use the same protection.
        self.mage_sink, self.mage_recent_tiles = int(mage_sink), int(mage_recent) // 64
        # v31 pooled residual (named variant, opt-in, any sparse arm): dropped wholly-prefix tiles are added back as
        # their centroid key / mean value (v31_residual); centroids cached per (layer, canvas)
        self.residual = residual
        self._tile_means, self._cur_layer, self._residual_tiles = {}, None, 0
        # v31 dropped-mass guard (named variant, opt-in, any sparse arm): at the first call of each held keep map, a
        # (query head, 128-row block) whose dropped prefix tiles are estimated to hold more than drop_guard of its
        # attention mass keeps its whole prefix (v31_residual.dropped_share); the guarded map is held with the original.
        # It runs once per new map: once per canvas for MAGE, once per re-decision for the method.
        self.drop_guard = None if drop_guard is None else float(drop_guard)
        self._guard_cache = []
        # diagnostic receipts (opt-in): step-to-step drift of the GLOBAL queries and outputs within a canvas -- the
        # prefix K/V are fixed inside a canvas, so a row whose query did not move could reuse its prefix partial
        self.drift_diag = bool(drift_diag)
        # v31 length gate (opt-in, any sparse arm): a GLOBAL call whose key length (prefix + canvas) is below
        # dense_below keys runs vLLM's own dense attention -- the same code path as the dense PIECEWISE baseline --
        # because below the measured crossover the sparse path has no saving (fixed per-call costs); the arm's
        # selector state simply starts at the first call above the gate
        if dense_below is not None and int(dense_below) <= 0:
            raise ValueError('dense_below must be a positive key count')
        self.dense_below = None if dense_below is None else int(dense_below)
        self._drift_prev, self._drift_acc = {}, None
        # diagnostic receipts (opt-in): denoising steps per canvas and the canvas mean token entropy per step, i.e.
        # the quantity the official sampler compares with its confidence threshold to stop a canvas
        self.trace_canvas = bool(trace_canvas)
        # diagnostic (opt-in): at every dense-prefix threshold decision, the kept fraction of prefix tiles the same need
        # matrix would give at 128-row blocks (executed), natural 64-row halves, regrouped 64-row halves (rows sorted by
        # need count, chw/value_aware idea; or by a random projection of the need vector) and per row (ideal)
        self.regroup_diag = bool(regroup_diag)
        # v31 step-level dense fallback (named variant; method and MAGE arms): 'conv:THETA' runs every GLOBAL call of the
        # next denoising step with vLLM's own dense FA4 once the canvas mean token entropy of the previous step is below
        # THETA x the sampler's confidence threshold (the canvas is about to converge); 'step:S' does so from the S-th
        # denoising step of a canvas on. The selector is bypassed on those steps; its sampler-side state still updates.
        self.dense_when = None
        if dense_when:
            kind, val = str(dense_when).split(':')
            if kind not in ('conv', 'step'):
                raise ValueError('dense_when must be conv:THETA or step:S')
            self.dense_when = (kind, float(val))
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
        self.mage_k, self.mage_state, self.canvas_id = int(mage_k), {}, 0
        self._mage_warm = {}
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
        if self.local_router is not None:
            self.local_router.clear()
        self._clear_canvas_metadata()
        self.cache = _PrefixCache(len(self.layer_types))
        self.step_ctx = None
        for k in self.calls:
            self.calls[k] = 0
        self.pending_sample = False
        self.bound = True
        self._events = []
        self.mage_state, self.canvas_id = {}, 0
        self._mage_warm = {}
        self._mage_count, self._mage_row_w, self._mage_units_w = {}, None, None
        self._mage_prev_argmax, self._k_override = None, None
        self._trig_canvas, self._trig_at = None, None
        self._trig_count, self._trig_k, self._jc = 0, None, None
        self._clock, self._trig_phi0 = [], None
        self._cg = self._cg_canvas = None
        self._stall_best, self._stall_since = -1.0, 0
        self._tile_means, self._residual_tiles, self._guard_cache = {}, 0, []
        self._drift_prev, self._drift_acc = {}, None
        self._kept_prefix, self._prefix_total = None, 0     # realized sparsity of the sparse GLOBAL calls
        self._sparse_kept, self._sparse_total, self._global_tiles = None, 0, 0
        self._global_canvas_tiles = 0
        self._canvas_steps, self._cur_steps, self._ent_trace, self._conf_threshold = [], 0, [], None
        self._dense_next = self._dense_now = False
        self._canvas_step = 0
        self._regroup_acc, self._regroup_n = None, 0
        if self.arm == 'mage':
            self.calls.update(mage_selections=0, mage_reused_calls=0, mage_kept_prefix_tiles=0, mage_prefix_tiles=0)
        self._value_pending = []
        self._value_audit_events = []
        self._value_phases = {}
        self._value_global_step_phases = {}
        self._value_local_steps = {}
        self._value_decode_modes = {}
        self._value_native_local_tiles = self._value_native_local_calls = 0
        self._value_nonfinite_outputs = []
        self._value_local_events = []
        if self.arm in ('allkept', 'native', 'mage'):
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
        return dict(adapter=dict(self.calls, **self._kept_receipt(), **self._local_receipt(), **self._regroup_receipt(),
                                 **self._drift_receipt(), **self._clock_receipt(), **self._value_receipt(), **self._value_phase_receipt()), method=counters,
                    timing=timing,
                    trace=self._trace_receipt() or None)

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
            return dict(adapter=dict(self.calls, **self._kept_receipt(), **self._local_receipt(), **self._regroup_receipt(),
                                     **self._drift_receipt(), **self._clock_receipt(), **self._value_receipt(), **self._value_phase_receipt()), method=counters,
                    timing=timing,
                    trace=self._trace_receipt() or None)
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

    @torch.no_grad()
    def _regroup_account(self, state, reference, sensitivity, log_threshold, nq):
        b, h, qb, pt, _ = state.lognorm.shape
        if not pt or nq != qb * 128:
            return
        hk = reference.shape[1]
        kh = torch.arange(h, device=reference.device) // (h // hk)
        risk = state.lognorm - torch.log(reference.float().clamp_min(1e-12))[:, kh][:, :, None, None, None]
        if sensitivity is not None:
            risk = risk + torch.log(sensitivity.float()).view(b, 1, qb, 1, 128)
        need = ((risk >= float(log_threshold)) & (state.eligible[..., None] != 0)).permute(0, 1, 2, 4, 3)  # [b,h,qb,128,pt]
        k128 = need.any(3).float().mean()
        k64 = need.view(b, h, qb, 2, 64, pt).any(4).float().mean()
        def grouped(key):
            order = torch.argsort(key, dim=-1, stable=True)
            g = torch.gather(need, 3, order[..., None].expand(-1, -1, -1, -1, pt))
            return g.view(b, h, qb, 2, 64, pt).any(4).float().mean()
        gen = torch.Generator(device=need.device).manual_seed(5)
        proj = torch.randn(pt, device=need.device, generator=gen)
        vals = torch.stack([k128, k64, grouped(need.sum(-1)), grouped(need.float() @ proj), need.float().mean()])
        # decisions run on the main stream and (after an observation) asynchronously on the core's route stream: keep
        # one tensor per decision and reduce only at the end of the request, after the per-step device sync
        if self._regroup_acc is None:
            self._regroup_acc = []
        self._regroup_acc.append(vals)
        self._regroup_n += 1

    def _regroup_receipt(self):
        if not getattr(self, 'regroup_diag', False) or self._regroup_acc is None:
            return {}
        torch.cuda.synchronize()
        m = torch.stack(self._regroup_acc).mean(0).tolist()
        return dict(regroup_decisions=self._regroup_n, kept128=round(m[0], 5), kept64=round(m[1], 5),
                    kept64_regroup_count=round(m[2], 5), kept64_regroup_proj=round(m[3], 5), kept_per_row=round(m[4], 5))

    def _trace_receipt(self):
        if not getattr(self, 'trace_canvas', False):
            return {}
        steps = list(self._canvas_steps) + ([self._cur_steps] if self._cur_steps else [])
        ent = [round(float(x), 4) for x in torch.stack(self._ent_trace).tolist()] if self._ent_trace else []
        out, i = [], 0
        for n in steps:                                   # split the per-step entropies by canvas
            out.append(ent[i:i + n])
            i += n
        return dict(canvas_steps=steps, canvas_entropy=out, confidence_threshold=self._conf_threshold)

    def _kept_receipt(self):
        """Realized sparsity, in wholly-prefix 64-key tiles x query heads x 128-row query blocks:
        - kept_prefix_fraction (legacy, records before 2026-10-04): every call through the split FA4 lists, which
          INCLUDES the method's bootstrap dense calls (v27_fa4.dense routes an all-kept list there);
        - sparse_kept_prefix_fraction: the genuinely sparse calls only (all-kept lists excluded);
        - global_prefix_work_fraction: all GLOBAL calls of the request (dense bootstrap / observation / MAGE selection /
          dense fallback count every prefix tile, sparse calls their kept tiles) over the dense equivalent -- the
          compute-matched comparison between arms with different numbers of dense steps."""
        kept, total = getattr(self, '_kept_prefix', None), getattr(self, '_prefix_total', 0)
        if kept is None or not total:
            return {}
        out = dict(kept_prefix_tiles=int(kept.item()), sparse_prefix_tiles=int(total),
                   kept_prefix_fraction=round(float(kept.item()) / total, 5))
        skept, stotal, gtiles = (getattr(self, '_sparse_kept', None), getattr(self, '_sparse_total', 0),
                                 getattr(self, '_global_tiles', 0))
        sk = float(skept.item()) if skept is not None else 0.0
        if stotal:
            out.update(sparse_only_kept_tiles=int(round(sk)), sparse_only_prefix_tiles=int(stotal),
                       sparse_kept_prefix_fraction=round(sk / stotal, 5))
        rtiles = getattr(self, '_residual_tiles', 0)
        if rtiles:
            out.update(residual_prefix_tiles=round(rtiles, 1))
        if gtiles:
            out.update(global_prefix_tiles=int(gtiles),
                       global_prefix_work_fraction=round((gtiles - stotal + sk + rtiles) / gtiles, 5))
        canvas_tiles = getattr(self, '_global_canvas_tiles', 0)
        out.update(global_eligible_tiles=int(gtiles + canvas_tiles),
                   global_kept_tiles=int(round(gtiles - stotal + sk + canvas_tiles)))
        return out

    def _local_receipt(self):
        return {} if self.local_router is None else self.local_router.receipt()

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
            self._dense_next = self._dense_now = False
            self._canvas_step = 0
        else:
            self._canvas_step += 1
            self._dense_now = bool(self._dense_next) or (self.dense_when is not None and self.dense_when[0] == 'step'
                                                           and self._canvas_step >= self.dense_when[1])
        if self.trace_canvas:
            if phase_encoder and self._cur_steps:
                self._canvas_steps.append(self._cur_steps)
                self._cur_steps = 0
            elif not phase_encoder:
                self._cur_steps += 1
        if phase_encoder:
            self.calls['invalidates'] += 1
            self.canvas_id += 1
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

    def needs_accepted(self):
        state = None if self.runtime is None else self.runtime.get('state')
        return state is not None and getattr(state, 'cgate', None) is not None

    def fused_stats_eligible(self):
        """The fused statistics replace State.observe_logits on its fast-T path when the C gate is on (T alone only
        takes an argmax, which one torch op already does at full bandwidth)."""
        state = None if self.runtime is None else self.runtime.get('state')
        return (self.logit_stats == 'fused' and state is not None and getattr(state, 'fast_t', False)
                and getattr(state, 'cgate', None) is not None
                and self.config.get('sensitivity') not in ('unit_v30', 'confidence_v30'))

    def on_sample(self, scaled_logits, accepted=None, stats=None):
        """After a denoising sample: the temperature-scaled logits [1, CL, V] (fast T takes their argmax) and, for the
        C gate, the sampler's acceptance mask [1, CL]."""
        self.calls['observes'] += 1
        if not self.pending_sample:
            self.calls['order_errors'] += 1
        self.pending_sample = False
        if self.runtime is not None:
            cur_step = None
            if self.config.get('sensitivity') in ('unit_v30','confidence_v30'):
                cur_step = MAX_DENOISING_STEPS - self.step_ctx['step']
                # The official sampler pads a terminal partial canvas to CL.
                # Only real query rows enter the selector; do not include padding.
                n = self.step_ctx['n']
                if scaled_logits.ndim != 3 or scaled_logits.shape[0] != 1 or not 0 < n <= scaled_logits.shape[1]:
                    raise ValueError('V30 sampler/query geometry mismatch')
                scaled_logits = scaled_logits[:, :n, :]
            if accepted is not None:
                n = self.step_ctx['n']
                if scaled_logits.ndim != 3 or scaled_logits.shape[0] != 1 or not 0 < n <= scaled_logits.shape[1]:
                    raise ValueError('C-gate sampler/query geometry mismatch')
                scaled_logits, accepted = scaled_logits[:, :n, :], accepted[:, :n]
                self.calls['cgate_observes'] = self.calls.get('cgate_observes', 0) + 1
            if stats is not None:
                from experiments.numerical_qk_reuse.v31_logit_stats import RowStats, observe_logits_from_stats
                n = self.step_ctx['n']
                if stats.argmax.ndim != 2 or stats.argmax.shape[0] != 1 or not 0 < n <= stats.argmax.shape[1]:
                    raise ValueError('fused-stats sampler/query geometry mismatch')
                stats = RowStats(*(t[:, :n] for t in (stats.argmax, stats.max, stats.lse, stats.entropy)))
                observe_logits_from_stats(self.runtime['state'], stats, accepted, cur_step)
                self.calls['fused_stat_observes'] = self.calls.get('fused_stat_observes', 0) + 1
                return
            self.runtime['state'].observe_logits(scaled_logits, accepted, cur_step)

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

    def local_for(self, layer_name):
        """Return a LOCAL layer index when the separately enabled local router owns it."""
        ctx = self.step_ctx
        if self.local_router is None or ctx is None or ctx['encoder']:
            return None
        m = _LAYER_RE.search(layer_name)
        if m is None:
            return None
        layer = int(m.group(1))
        return layer if layer not in self.global_layers else None

    def forward_local(self, impl, layer_idx, query, kv_cache, attn_metadata, output):
        """Run the local selector on the native paged cache, or fall back to vLLM dense."""
        n = int(attn_metadata.num_actual_tokens)
        ctx = self.step_ctx
        prefix = ctx['seq_len'] - n
        key_cache, value_cache = kv_cache.transpose(1, 2).split(impl.head_size, dim=-1)
        q = query[:n].transpose(0, 1).unsqueeze(0)
        table = attn_metadata.block_table[0]
        out = self.local_router.forward(layer_idx, q, key_cache, value_cache, table, prefix, n,
                                        float(impl.scale), self)
        if out is None:
            return None
        output[:n].view(n, -1).copy_(out.reshape(n, -1))
        return output

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
        # vLLM's hybrid cache can give GLOBAL layers KV128 pages when LOCAL
        # layers use KV64. FA4 sparse TMA loads require KV64 pages. Split the
        # native page through a zero-copy view and expand its logical table.
        from experiments.numerical_qk_reuse.v31_local_sparse import alias64
        sparse_k, sparse_v, sparse_table = alias64(key_cache, value_cache,
                                                  attn_metadata.block_table[0], prefix + n)
        self.paged = dict(k=sparse_k, v=sparse_v, nk=prefix + n, prefix=prefix,
                          table=sparse_table[0])
        q = query[:n].transpose(0, 1).unsqueeze(0)                  # [1, H, n, D] view, as HF's decoder passes it
        self.calls['global_calls'] += 1
        self._cur_layer = layer_idx
        self._global_tiles += q.shape[1] * -(-n // 128) * (prefix // 64)
        self._global_canvas_tiles += q.shape[1] * -(-n // 128) * (-(-(prefix + n) // 64) - prefix // 64)
        audit_before = None
        if self.value_audit:
            audit_before = (dict(self.calls), self._sparse_total, self._sparse_kept)
        if self.arm == 'allkept':
            from experiments.numerical_qk_reuse import v27_fa4
            out = v27_fa4.dense(q, b['k'], b['v'], float(impl.scale))
        elif self.arm == 'mage':
            out = self._mage(layer_idx, q, b, float(impl.scale), prefix, n)
        else:
            self.cache.layers[layer_idx] = _CacheLayer(b['pk'], b['pv'])
            self.cache.length = prefix
            module = self.stub.model.layers[layer_idx]
            module(None, None, None, past_key_values=self.cache)    # the core's identify / sketch hooks
            out, _ = self.binding.runtime.attention_override(module, q, b['k'], b['v'], None,
                                                             scaling=float(impl.scale), is_causal=False,
                                                             sliding_window=None)
        self.paged = None
        if audit_before is not None:
            counts, previous_total, previous_kept = audit_before
            phase = next((name for name, key in (
                ('initial','mage_selections'),('refresh','mage_reselections'),
                ('held','mage_reused_calls'),('carried','mage_carried_calls'))
                if self.calls.get(key,0)>counts.get(key,0)), 'dense')
            units = q.shape[1]*-(-n//128)
            eligible = units*-(-(prefix+n)//64)
            if self._sparse_total > previous_total:
                kept = self._sparse_kept-(previous_kept if previous_kept is not None else 0)
                kept = kept+units*(-(-(prefix+n)//64)-prefix//64)
            else:
                kept = torch.tensor(eligible,device=q.device,dtype=torch.float64)
            self._value_global_step_phases.setdefault((self.canvas_id,self._canvas_step),set()).add(phase)
            row = self._value_phases.setdefault(phase,[0,0,[]])
            row[0] += 1
            row[1] += eligible
            row[2].append(kept)
            self._value_nonfinite_outputs.append(~torch.isfinite(out).all())
        diagnostic = getattr(self, 'value_diagnostic', None)
        if diagnostic is not None and self.arm == 'mage':
            diagnostic.record(self, layer_idx, q, b, float(impl.scale), prefix, n)
        if self.drift_diag:
            self._drift_account(layer_idx, q, out, n)
        output[:n].view(n, -1).copy_(out.reshape(n, -1))
        return output

    DRIFT_THRESHOLDS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)

    @torch.no_grad()
    def _drift_account(self, layer_idx, q, out, n):
        """Relative change of each (head, row) query and output vs the previous GLOBAL call of this layer in the
        same canvas; counts at DRIFT_THRESHOLDS for rows, for 128-row blocks (max over the block) and for outputs."""
        h = q.shape[1]
        cur_q = q[0].float()                                                    # [H, n, D]
        cur_o = out.reshape(n, h, -1).transpose(0, 1).float()                  # [H, n, D]
        prev = self._drift_prev.get(layer_idx)
        self._drift_prev[layer_idx] = (self.canvas_id, n, cur_q.to(torch.bfloat16), cur_o.to(torch.bfloat16))
        if prev is None or prev[0] != self.canvas_id or prev[1] != n:
            return
        pq, po = prev[2].float(), prev[3].float()
        dq = (cur_q - pq).norm(dim=-1) / pq.norm(dim=-1).clamp_min(1e-12)     # [H, n]
        do = (cur_o - po).norm(dim=-1) / po.norm(dim=-1).clamp_min(1e-12)
        qb = -(-n // 128)
        blocks = torch.nn.functional.pad(dq, (0, qb * 128 - n)).view(h, qb, 128).amax(-1)
        thr = getattr(self, '_drift_thr', None)
        if thr is None or thr.device != q.device:
            thr = self._drift_thr = torch.tensor(self.DRIFT_THRESHOLDS, device=q.device)
        acc = torch.stack([(dq[..., None] <= thr).sum((0, 1)), (blocks[..., None] <= thr).sum((0, 1)),
                           (do[..., None] <= thr).sum((0, 1))]).double()
        totals = acc.new_tensor([dq.numel(), blocks.numel(), do.numel()])  # diagnostic runs are not timing runs
        if self._drift_acc is None:
            self._drift_acc = [acc, totals]
        else:
            self._drift_acc[0] += acc
            self._drift_acc[1] += totals

    def _drift_receipt(self):
        acc = getattr(self, '_drift_acc', None)
        if acc is None:
            return {}
        frac = (acc[0] / acc[1][:, None]).tolist()
        return dict(drift_thresholds=list(self.DRIFT_THRESHOLDS), drift_q_rows=[round(x, 5) for x in frac[0]],
                    drift_q_blocks=[round(x, 5) for x in frac[1]], drift_out_rows=[round(x, 5) for x in frac[2]],
                    drift_pairs=int(acc[1][0].item()))

    # ------------------------------------------------------------------ MAGE port (prior-art baseline)
    def _mage(self, layer_idx, q, b, scale, prefix, n):
        from experiments.numerical_qk_reuse import v27_fa4
        nk = prefix + n
        st = self.mage_state.get(layer_idx)
        if self.mage_reselect is not None or self.mage_reselect_trigger is not None:
            cnt = self._mage_count.get(layer_idx)
            if cnt is None or cnt[:2] != [self.canvas_id, nk]:
                cnt = self._mage_count[layer_idx] = [self.canvas_id, nk, -1]
            cnt[2] += 1                                                  # 0-based call index of this canvas
            due = (cnt[2] in self.mage_reselect if self.mage_reselect is not None
                   else self._trig_at == (self.canvas_id, cnt[2]))
            if st is not None and st['canvas'] == self.canvas_id and st['nk'] == nk and due:
                self._mage_units_w = self._mage_row_w if self.mage_row_weight is not None else None
                self._mage_held = st.get('kept') if self.mage_sticky is not None else None
                self._k_override = self._trig_k if self._trig_k is not None else self.mage_reselect_k
                try:
                    pool = st.get('pool') if self.mage_pool is not None else None
                    if pool is not None and pool.shape[-1] == -(-nk // 64):
                        out, kept = self._mage_select_pool(q, b['k'], b['v'], scale, prefix, n, pool)
                    else:
                        out, kept = self._select_initial_refresh(layer_idx, q, b, scale, prefix, n)
                finally:
                    self._mage_units_w, self._k_override, self._mage_held = None, None, None
                st['lists'] = v27_fa4.block_sparse_tensors(kept)
                if self.mage_carry_first:
                    st['kept'] = kept
                self.calls['mage_reselections'] = self.calls.get('mage_reselections', 0) + 1
                return out
        if st is None or st['canvas'] != self.canvas_id or st['nk'] != nk:
            if self.mage_select_step:
                w = self._mage_warm.get(layer_idx)
                if w is None or w[:2] != [self.canvas_id, nk]:
                    w = self._mage_warm[layer_idx] = [self.canvas_id, nk, 0]
                if w[2] < self.mage_select_step:                   # exact steps before the selection
                    w[2] += 1
                    carried = self._mage_carried(st, prefix, n) if self.mage_carry_first and w[2] == 1 else None
                    if carried is not None:
                        self.calls['mage_carried_calls'] = self.calls.get('mage_carried_calls', 0) + 1
                        return v27_fa4.sparse_lists(q, b['k'], b['v'], carried, scale)
                    self.calls['mage_warm_dense_calls'] = self.calls.get('mage_warm_dense_calls', 0) + 1
                    return v27_fa4.dense(q, b['k'], b['v'], scale)
            if self.mage_select == 'fa4':
                self._last_pool = None
                out, kept = self._select_initial_refresh(layer_idx, q, b, scale, prefix, n)
            else:
                out = v27_fa4.dense(q, b['k'], b['v'], scale)             # exact first-step attention output
                kept = self._mage_select(q, b['k'], scale, prefix, n)
            self.mage_state[layer_idx] = dict(canvas=self.canvas_id, nk=nk, prefix=prefix, lists=v27_fa4.block_sparse_tensors(kept),
                                              kept=kept if self.mage_carry_first else None, pool=self._last_pool)
            self._last_pool = None
            self.calls['mage_selections'] += 1
            return out
        self.calls['mage_reused_calls'] += 1
        return v27_fa4.sparse_lists(q, b['k'], b['v'], st['lists'], scale)

    def _select_initial_refresh(self, layer_idx, q, b, scale, prefix, n):
        if self.value_selector is None:
            return self._mage_select_fa4(q, b['k'], b['v'], scale, prefix, n)
        from experiments.diffusion_gemma_jl_output_aware.projections import Projections
        from .v31_value_projection import refresh
        from .v31_value_kernels import statistics_cuda
        from .v31_value_selectors import mandatory_map, select
        from .v31_fa4_observe import observe_dense
        events = None
        if self.value_audit:
            events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            events[0].record()
        k, v = b['k'], b['v']
        h, hk, nk, d = q.shape[1], k.shape[1], k.shape[2], v.shape[3]
        qb, pt = -(-n//128), prefix//64
        # Keep the initial/refresh model output on the inherited exact FA4 path.
        observation = (torch.empty((h,qb,pt,128),dtype=torch.float32,device=q.device)
            if self._value_workspace is None else self._value_workspace.take(
                'observation',(h*qb,pt,128),torch.float32,q.device).view(h,qb,pt,128))
        out = observe_dense(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale, observation)
        if self._value_projection is None:
            self._value_projection = Projections()
        matrix = self._value_projection.get(layer_idx, hk, d, 'gaussian', 32, 1729, q.device)
        sketch = torch.empty((1, hk, nk, 32), dtype=torch.float32, device=q.device)
        norms = torch.empty((1, hk, nk), dtype=torch.float32, device=q.device)
        valid = torch.ones((1, hk, nk), dtype=torch.bool, device=q.device)
        # Every refresh projects CURRENT V, including changed canvas/boundary V.
        nu = refresh(v, matrix, sketch, norms, valid, 0)[0]
        stats = statistics_cuda(q,k,sketch,nu,scale,pt,workspace=self._value_workspace)
        # Retain FA4's own observed prefix masses; the PZ pass supplies only mu.
        stats.log_mass[:, :pt].view(h, qb, pt, 128).copy_(observation)
        mandatory = mandatory_map(stats, self.mage_sink, self.mage_recent_tiles)
        held = self._mage_held
        held = None if held is None else held.reshape(h*qb, -1)
        budget = self._k_override if self._k_override is not None else self.mage_k
        if events is not None:
            events[1].record()
        kept, counters = select(stats, self.value_selector, budget=max(1, budget//64),
            threshold=self.value_threshold, mandatory=mandatory, held=held,
            sticky=0. if held is None else self.mage_sticky)
        # Device counters are read once at request end, outside generation timing.
        if events is not None:
            events[2].record()
            self._value_audit_events.append(events)
            self._value_pending.append(torch.stack((kept[..., :pt].sum(), stats.invalid.sum())))
        self.calls['value_selection_calls'] = self.calls.get('value_selection_calls', 0)+1
        self.calls['value_statistics_qk_rectangles'] = self.calls.get('value_statistics_qk_rectangles',0)+h*qb*stats.log_mass.shape[1]
        self.calls['value_candidate_evaluations'] = self.calls.get('value_candidate_evaluations', 0)+counters['candidate_evaluations']
        self.calls['value_peak_summary_bytes'] = max(self.calls.get('value_peak_summary_bytes', 0), stats.nbytes)
        self.calls['mage_prefix_tiles'] += pt*h*qb
        if held is not None:
            self.calls['mage_sticky_units'] = self.calls.get('mage_sticky_units', 0)+1
        return out, kept

    def _value_receipt(self):
        if self.value_selector is None:
            return {}
        counts = torch.stack(self._value_pending).sum(0).tolist() if self._value_pending else [0, 0]
        timings = None
        if self._value_audit_events:
            torch.cuda.synchronize()
            timings = dict(discovery_ms=sum(a.elapsed_time(b) for a, b, _ in self._value_audit_events),
                           selection_ms=sum(b.elapsed_time(c) for _, b, c in self._value_audit_events))
        return dict(value_selector=self.value_selector, value_threshold=self.value_threshold,
            value_selected_prefix_tiles=int(counts[0]) if self.value_audit else None,
            value_invalid_rows=int(counts[1]) if self.value_audit else None,
            value_audit=self.value_audit, value_audit_timings=timings,
            value_projection_seed=1729, value_projection_rank=32,
            value_projection_kernel='inherited_gaussian32_refresh_plus_bounded_rms4096',
            value_discovery_work='additional full-support QK and rank32 PV; separately counted from native model-output consumer tiles',
            value_workspace_capacity_tiles=None if self._value_workspace is None else self._value_workspace.max_tiles,
            value_workspace_allocated_bytes=0 if self._value_workspace is None else self._value_workspace.allocated_bytes,
            value_statistics_kernel='triton_fused_tile_softmax_pz',
            value_prefix_mass_source='native_fa4_observation',
            value_selection_kernel='triton_log_mass_online' if self.value_selector.startswith(('value_v1_','value_v2_')) else 'triton_parallel_cached_deletion',
            value_projection_manifest={} if self._value_projection is None else self._value_projection.manifest,
            value_scope='global_only', value_local_router_installed=self.local_router is not None,
            value_state_role='selector_only', value_output_path='unchanged_fa4',
            value_sticky_rule='log_risk_bonus_online_or_log_removal_penalty_fixed')

    def _value_phase_receipt(self):
        if not self.value_audit:
            return {}
        phases = {}
        for phase,(calls,eligible,kept) in self._value_phases.items():
            count = int(torch.stack(kept).sum().item())
            phases[phase] = dict(calls=calls,eligible_tiles=eligible,kept_tiles=count,
                                 skipped_tiles=eligible-count,sparsity=1-count/eligible)
        local_phases = {}
        for key,(calls,tiles) in self._value_local_steps.items():
            labels = self._value_global_step_phases.get(key,set())
            phase = next(iter(labels)) if len(labels)==1 else 'mixed_or_unassigned'
            row = local_phases.setdefault(phase,dict(calls=0,eligible_rectangles=0,skipped_tiles=0,sparsity=0.))
            row['calls'] += calls
            row['eligible_rectangles'] += tiles
        local = self._value_native_local_tiles
        global_eligible = sum(row['eligible_tiles'] for row in phases.values())
        global_skipped = sum(row['skipped_tiles'] for row in phases.values())
        return dict(value_phase_tiles=phases,native_local_phase_rectangles=local_phases,
            decode_cudagraph_modes=dict(self._value_decode_modes),
            value_nonfinite_attention_calls=int(torch.stack(self._value_nonfinite_outputs).sum().item()) if self._value_nonfinite_outputs else 0,
            native_local_calls_audited=self._value_native_local_calls,
            native_local_attention_ms=sum(a.elapsed_time(b) for a,b in self._value_local_events),
            native_local_eligible_rectangles=local,native_local_skipped_tiles=0,
            overall_eligible_rectangles=global_eligible+local,
            overall_skipped_rectangles=global_skipped,
            overall_rectangle_sparsity=global_skipped/(global_eligible+local) if global_eligible+local else None,
            overall_denominator='actual decode calls, GLOBAL full support and LOCAL window-intersecting H x Q128 x KV64 rectangles; geometry-derived, not CUDA CTA count')

    def _value_native_local_account(self,layer_name,heads,n):
        ctx = self.step_ctx
        match = _LAYER_RE.search(layer_name)
        if not self.value_audit or ctx is None or ctx['encoder'] or match is None:
            return
        if int(match.group(1)) in self.global_layers:
            return
        nk = ctx['seq_len']
        prefix = nk-n
        tiles = 0
        for begin in range(0,n,128):
            lo = max(0,prefix+begin-1023)
            hi = min(nk-1,prefix+min(begin+127,n-1)+1023)
            tiles += hi//64-lo//64+1
        step = self._value_local_steps.setdefault((self.canvas_id,self._canvas_step),[0,0])
        step[0] += 1
        step[1] += heads*tiles
        self._value_native_local_tiles += heads*tiles
        self._value_native_local_calls += 1
        return True

    def _mage_carried(self, st, prefix, n):
        """Lists for canvas call 0 from the previous canvas's selection (mage_carry_first), or None when the carry is
        not the direct continuation: previous canvas, prefix grown by exactly that canvas, same row-block layout."""
        from experiments.numerical_qk_reuse import v27_fa4
        if st is None or st.get('kept') is None or st['canvas'] != self.canvas_id - 1 or st['nk'] != prefix:
            return None
        old, qb, kt = st['prefix'] // 64, -(-n // 128), -(-(prefix + n) // 64)
        if st['kept'].shape[2] != qb or old > st['kept'].shape[-1] or kt < st['kept'].shape[-1]:
            return None
        kept = torch.ones((*st['kept'].shape[:3], kt), dtype=torch.bool, device=st['kept'].device)
        kept[..., :old] = st['kept'][..., :old]
        return v27_fa4.block_sparse_tensors(kept)

    @torch.no_grad()
    def _mage_select(self, q, k, scale, prefix, n, chunk=8192):
        """MAGE eq. 5 at 64-key tile granularity: softmax mass per tile from exact first-step attention, averaged
        over the canvas queries and the query heads of each KV head; top (mage_k / 64) prefix tiles per KV head."""
        _, H, _, D = q.shape
        HK, nk = k.shape[1], k.shape[2]
        G, kt = H // HK, -(-nk // 64)
        qg = q[0].reshape(HK, G, n, D).float()
        tile_lse = torch.empty((HK, G, n, kt), device=q.device, dtype=torch.float32)
        for c0 in range(0, nk, chunk):
            c1 = min(nk, c0 + chunk)
            s = torch.matmul(qg, k[0, :, c0:c1].float().transpose(-1, -2).unsqueeze(1)) * scale   # [HK,G,n,len]
            pad = -(c1 - c0) % 64
            if pad:
                s = torch.nn.functional.pad(s, (0, pad), value=float('-inf'))
            tile_lse[..., c0 // 64: c0 // 64 + s.shape[-1] // 64] = s.reshape(HK, G, n, -1, 64).logsumexp(-1)
        mass = (tile_lse - tile_lse.logsumexp(-1, keepdim=True)).exp()
        score = mass.mean(dim=(1, 2))                                          # [HK, kt]
        first_canvas_tile = prefix // 64
        k_tiles = max(1, min(self.mage_k // 64, first_canvas_tile))
        kept_kv = torch.zeros((HK, kt), device=q.device, dtype=torch.bool)
        if first_canvas_tile:
            kept_kv.scatter_(1, score[:, :first_canvas_tile].topk(k_tiles, dim=-1).indices, True)
        kept_kv[:, first_canvas_tile:] = True                                  # the canvas (current block) itself
        self.calls['mage_kept_prefix_tiles'] += int(k_tiles) * HK
        self.calls['mage_prefix_tiles'] += int(first_canvas_tile) * HK
        qb = -(-n // 128)
        return kept_kv.repeat_interleave(G, 0)[None, :, None, :].expand(1, H, qb, kt).contiguous()

    @torch.no_grad()
    def _mage_select_fa4(self, q, k, v, scale, prefix, n):
        """MAGE eq. 5 from the FA4 observation: per-row prefix-tile log-mass from the dense FA4 pass, the
        remaining (canvas / boundary) tiles from an FP32 tail product, row-normalized, averaged over the canvas queries
        and the query heads of each KV head; top (mage_k / 64) prefix tiles per KV head. Returns (output, kept)."""
        from experiments.numerical_qk_reuse.v31_fa4_observe import observe_dense
        _, H, _, D = q.shape
        HK, nk = k.shape[1], k.shape[2]
        G, kt = H // HK, -(-nk // 64)
        first_canvas_tile = prefix // 64
        qb = -(-n // 128)
        z = torch.empty((H, qb, first_canvas_tile, 128), device=q.device, dtype=torch.float32)
        out = observe_dense(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale, z)
        head_lse = z.permute(0, 1, 3, 2).reshape(H, qb * 128, first_canvas_tile)[:, :n]          # [H, n, PT]
        koff = first_canvas_tile * 64
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            s = torch.matmul(q[0].float().reshape(HK, G, n, D), k[0, :, koff:].float().transpose(-1, -2).unsqueeze(1))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        s = s.reshape(H, n, nk - koff) * scale
        pad = -(nk - koff) % 64
        if pad:
            s = torch.nn.functional.pad(s, (0, pad), value=float('-inf'))
        tail_lse = s.reshape(H, n, -1, 64).logsumexp(-1)                                         # [H, n, KT-PT]
        tile_lse = torch.cat([head_lse, tail_lse], -1)                                            # [H, n, KT]
        mass = (tile_lse - tile_lse.logsumexp(-1, keepdim=True)).exp()
        if self.mage_keep_frac is not None:
            import math
            k_tiles = max(1, first_canvas_tile - int(math.floor((1.0 - self.mage_keep_frac) * first_canvas_tile + 1e-9)))
        else:
            budget = self._k_override if self._k_override is not None else self.mage_k
            k_tiles = max(1, min(budget // 64, first_canvas_tile))
            if self.mage_kcover is not None and self._k_override is None and first_canvas_tile:
                k_tiles = self._coverage_tiles(head_lse, n, qb, first_canvas_tile, k_tiles)
        if self.mage_granularity != 'kvhead':
            kept = self._mage_units(mass, head_lse, H, n, qb, kt, first_canvas_tile, k_tiles, G)
            if self.mage_pool is not None:                                       # round 5: the candidate pool
                self._last_pool = self._mage_units(mass, head_lse, H, n, qb, kt, first_canvas_tile,
                                                   min(first_canvas_tile, self.mage_pool * k_tiles), G, account=False)
            return out, kept
        score = mass.reshape(HK, G, n, kt).mean(dim=(1, 2))                                        # [HK, KT]
        kept_kv = torch.zeros((HK, kt), device=q.device, dtype=torch.bool)
        if first_canvas_tile and self.mage_coverage is not None:
            pre = score[:, :first_canvas_tile]                                                  # [HK, PT]
            covered = score[:, first_canvas_tile:].sum(-1, keepdim=True)                       # canvas mass
            srt, order = pre.sort(-1, descending=True)
            need = ((covered + srt.cumsum(-1)) < self.mage_coverage).sum(-1) + 1                 # tiles to reach p
            need = need.clamp(min=k_tiles, max=first_canvas_tile)
            sel = torch.arange(first_canvas_tile, device=q.device)[None] < need[:, None]
            kept_kv[:, :first_canvas_tile].scatter_(1, order, sel)
            self.calls['mage_kept_prefix_tiles'] += int(need.sum())
        elif first_canvas_tile:
            kept_kv.scatter_(1, score[:, :first_canvas_tile].topk(k_tiles, dim=-1).indices, True)
            self.calls['mage_kept_prefix_tiles'] += int(k_tiles) * HK
        kept_kv[:, first_canvas_tile:] = True
        self.calls['mage_prefix_tiles'] += int(first_canvas_tile) * HK
        kept = kept_kv.repeat_interleave(G, 0)[None, :, None, :].expand(1, H, qb, kt).contiguous()
        if self.mage_critical is not None and first_canvas_tile:
            m = mass[..., :first_canvas_tile]                                                   # [H, n, PT]
            pad = qb * 128 - n
            if pad:
                m = torch.nn.functional.pad(m, (0, 0, 0, pad))
            critical = m.view(H, qb, 128, first_canvas_tile).amax(2) >= self.mage_critical       # [H, QB, PT]
            added = critical & ~kept[0, :, :, :first_canvas_tile]
            kept[0, :, :, :first_canvas_tile] |= critical
            self.calls['mage_critical_added_tiles'] = self.calls.get('mage_critical_added_tiles', 0) + int(added.sum())
        return out, kept

    def _progress_observe(self, argmax, p_top, acc, scaled, n_rows, entropy_bound):
        """The progress clock, after every denoising step while the canvas has thresholds left: argmax / p_top / acc are
        the step's per-row statistics [n_rows] (acc = the official acceptance mask). Updates the C-gate histories,
        reads the progress signal and, when it crosses the next threshold(s), schedules a re-selection at the next
        call with its row weights and budget."""
        if self._trig_canvas != self.canvas_id:                                 # a new canvas: reset the clock
            self._trig_canvas, self._trig_count, self._jc, self._trig_k = self.canvas_id, 0, None, None
            self._trig_phi0 = None
            if self.mage_clock_trace:
                self._clock.append([])
        done = self._trig_count >= len(self.mage_reselect_trigger)
        if done and not self.mage_clock_trace:
            return
        settled = None
        if self.mage_trigger_signal == 'settle' or self.mage_row_weight == 'jcgate' or self.mage_clock_trace:
            accf = acc.float()
            jc = self._jc
            q = torch.ones_like(p_top) if jc is None else jc['q']
            r = torch.zeros_like(p_top) if jc is None else jc['r']
            q = self.mage_cg_gamma_q * q + (1 - self.mage_cg_gamma_q) * (1 - accf)          # renoised = not accepted
            stay = (r + 1) * accf
            r = stay if jc is None else torch.where(argmax != jc['prev'], torch.zeros_like(stay), stay)
            u = (1 - p_top.float()).clamp_min(0).sqrt()
            self._jc = dict(q=q, r=r, prev=argmax)
            settled = (1 - torch.exp(-r / self.mage_cg_tau)) * (1 - q) * (1 - u)
        signal = settled if self.mage_trigger_signal == 'settle' else acc.float()
        if self.mage_clock_trace:
            pa, ps = torch.stack([acc.float().mean(), settled.mean()]).tolist()
            self._clock[-1].append([round(pa, 3), round(ps, 3)])
            if done:
                return
            phi = ps if self.mage_trigger_signal == 'settle' else pa
        else:
            phi = float(signal.mean().item())                                  # the one host read of the step
        if self.mage_trigger_relative:
            if self._trig_phi0 is None:                                        # the canvas's first step: reference
                self._trig_phi0 = phi
                return
            phi = (phi - self._trig_phi0) / (1.0 - self._trig_phi0) if self._trig_phi0 < 1.0 else 0.0
        self.calls['trigger_checks'] = self.calls.get('trigger_checks', 0) + 1
        crossed = sum(1 for f in self.mage_reselect_trigger if phi >= f)
        if crossed <= self._trig_count:
            return
        self._trig_count = crossed
        self._trig_at = (self.canvas_id, self._canvas_step)
        self.calls['triggers'] = self.calls.get('triggers', 0) + 1
        self.calls['trigger_step_sum'] = self.calls.get('trigger_step_sum', 0) + self._canvas_step
        if self.mage_reselect_kmin is not None:
            base = self.mage_reselect_k if self.mage_reselect_k is not None else self.mage_k
            self._trig_k = max(self.mage_reselect_kmin, int(base * (1.0 - phi)) // 64 * 64)
            self.calls['trigger_k_sum'] = self.calls.get('trigger_k_sum', 0) + self._trig_k
        mode = self.mage_row_weight
        if mode == 'cgate':
            self._mage_row_w = (~acc).float()
        elif mode == 'jcgate':
            self._mage_row_w = (1 + self.mage_beta * (1 - settled)).clamp(1, 1 + self.mage_beta)
        elif mode is not None:
            self._mage_row_w = self._row_weight_from_logits(scaled, n_rows, entropy_bound)

    def _cg_track(self, argmax, p_top, acc):
        """Junyu's C-gate histories for the canvas in flight, updated once per denoising step (device ops only):
        returns (settledness g (1 - q) (1 - u), stable run r), both [n_rows]."""
        if self._cg_canvas != self.canvas_id:
            self._cg_canvas, self._cg, self._stall_best, self._stall_since = self.canvas_id, None, -1.0, 0
        accf = acc.float()
        cg = self._cg
        q = torch.ones_like(p_top, dtype=torch.float32) if cg is None else cg['q']
        r = torch.zeros_like(p_top, dtype=torch.float32) if cg is None else cg['r']
        q = self.mage_cg_gamma_q * q + (1 - self.mage_cg_gamma_q) * (1 - accf)
        stay = (r + 1) * accf
        r = stay if cg is None else torch.where(argmax != cg['prev'], torch.zeros_like(stay), stay)
        u = (1 - p_top.float()).clamp_min(0).sqrt()
        self._cg = dict(q=q, r=r, prev=argmax)
        return (1 - torch.exp(-r / self.mage_cg_tau)) * (1 - q) * (1 - u), r

    def _cg_stop(self, args, r):
        """cg_stop: after the official sample step (args = its bound arguments), converge the canvas when every row's
        stable run reached cg_stop and the official rule has not converged it. One host read of two flags."""
        slot = args['decode_slots'][:1]
        phase = args['is_encoder_phase']
        ready, done = torch.stack([(r >= self.cg_stop).all(), phase[slot][0]]).tolist()
        self.calls['cg_stop_checks'] = self.calls.get('cg_stop_checks', 0) + 1
        if not ready or done:
            return False
        CL = int(args['CL'])
        phase[slot] = True                                                   # commit next, as a converged canvas
        args['canvas'][slot] = args['argmax_canvas'][slot]
        args['draft_tokens'][slot, :CL] = args['canvas'][slot]
        args['sc_embeds'][slot] = 0
        self.calls['cg_stops'] = self.calls.get('cg_stops', 0) + 1
        self.calls['cg_stop_step_sum'] = self.calls.get('cg_stop_step_sum', 0) + getattr(self, '_canvas_step', 0)
        return True

    def _stall_check(self, settled):
        """stall_rescue: dense GLOBAL attention for the rest of the canvas once its mean settledness has not risen by
        more than stall_eps for stall_rescue steps. One host read per step until the rescue starts."""
        m = float(settled.mean().item())
        if m > self._stall_best + self.stall_eps:
            self._stall_best, self._stall_since = m, 0
            return
        self._stall_since += 1
        if self._stall_since >= self.stall_rescue:
            self._dense_next = True                                          # held until the canvas commits
            self.calls['stall_rescues'] = self.calls.get('stall_rescues', 0) + 1
            self.calls['stall_rescue_step_sum'] = self.calls.get('stall_rescue_step_sum', 0) + getattr(self, '_canvas_step', 0)

    def _clock_receipt(self):
        return dict(clock_trace=self._clock) if self.mage_clock_trace else {}

    def _row_weight_from_logits(self, scaled, n, entropy_bound):
        """Row weights [n] for the next re-selection from the sampler's temperature-scaled logits [1, CL, V]
        (and, for the temporal kinds, the argmax stored one step earlier)."""
        mode, beta = self.mage_row_weight, self.mage_beta
        if mode == 'cgate':
            return (~accepted_mask(scaled, entropy_bound)[0, :n]).float()
        x = scaled[0, :n].float()
        parts = {}
        if mode in ('margin', 'mt'):
            top2 = x.topk(2, dim=-1).values
            parts['M'] = 1 + beta / ((top2[:, 0] - top2[:, 1]).clamp_min(0) + 1.0)
        if mode in ('conf', 'ct'):
            p_top = (x.amax(-1) - torch.logsumexp(x, dim=-1)).exp().clamp(0, 1)
            parts['C'] = 1 + beta * (1 - p_top).clamp_min(0).sqrt()
        if mode in ('temporal', 'mt', 'ct'):
            prev = self._mage_prev_argmax
            changed = (x.argmax(-1) != prev[:n]).float() if prev is not None and prev.shape[0] >= n else torch.ones(n, device=x.device)
            parts['T'] = 1 + beta * changed
        w = {'margin': lambda: parts['M'], 'conf': lambda: parts['C'], 'temporal': lambda: parts['T'],
             'mt': lambda: (parts['M'] * parts['T']).sqrt(), 'ct': lambda: (parts['C'] * parts['T']).sqrt()}[mode]()
        return w.clamp(1, 1 + beta)

    def _mage_units(self, mass, head_lse, H, n, qb, kt, pt, k_tiles, G=1, account=True):
        """Selection-granularity ladder: top k_tiles prefix tiles per query head (qhead), per (query head, 128-row
        block) (qblock, qblock_max), or per KV head of G query heads, per block (kvblock_max) or for the whole canvas
        (kvhead_max), shared by those heads; canvas / boundary tiles always kept. Returns kept [1, H, QB, KT]."""
        dev = head_lse.device if head_lse is not None else mass.device         # the pool path passes mass=None
        kept = torch.zeros((1, H, qb, kt), device=dev, dtype=torch.bool)
        kept[..., pt:] = True
        if not pt:
            return kept
        pad = qb * 128 - n
        if self.mage_granularity in ('kvblock_max', 'kvhead_max'):
            share = head_lse - head_lse.logsumexp(-1, keepdim=True)                               # log share of prefix mass
            share = torch.nn.functional.pad(share, (0, 0, 0, pad), value=float('-inf')).view(H // G, G, qb, 128, pt)
            score = share.amax(dim=(1, 3))                                                        # [HK, QB, PT]
            if self.mage_granularity == 'kvhead_max':
                score = score.amax(1, keepdim=True)                                               # [HK, 1, PT]
            unit = torch.zeros(score.shape, device=dev, dtype=torch.bool)
            unit.scatter_(-1, score.topk(k_tiles, dim=-1).indices, True)
            kept[0, :, :, :pt] = unit.expand(-1, qb, -1).repeat_interleave(G, 0)                  # same set per group
            self.calls['mage_kept_prefix_tiles'] += int(k_tiles) * H * qb
            self.calls['mage_prefix_tiles'] += int(pt) * H * qb
            return kept
        if self.mage_granularity == 'qhead':
            score = mass[..., :pt].mean(1)[:, None].expand(H, qb, pt)                              # [H, QB, PT]
        elif self.mage_granularity == 'qblock':
            m = torch.nn.functional.pad(mass[..., :pt], (0, 0, 0, pad)).view(H, qb, 128, pt)
            rows = torch.nn.functional.pad(torch.ones(n, device=mass.device), (0, pad)).view(qb, 128).sum(-1)
            score = m.sum(2) / rows.clamp_min(1)[None, :, None]
        else:                                                                                   # qblock_max
            share = head_lse - head_lse.logsumexp(-1, keepdim=True)                               # log share of prefix mass
            share = torch.nn.functional.pad(share, (0, 0, 0, pad), value=float('-inf')).view(H, qb, 128, pt)
            score = share.amax(2)
            w = self._mage_units_w
            if w is not None:                                                                   # progress-aware rows
                logw = torch.nn.functional.pad(torch.log(w.float()[:n]), (0, pad), value=float('-inf')).view(qb, 128)
                weighted = (share + logw[None, :, :, None]).amax(2)                             # [H, QB, PT]
                live = torch.isfinite(logw).any(-1)                                             # [QB]: a row to weigh
                score = torch.where(live[None, :, None], weighted, score)
                self.calls['mage_weighted_units'] = self.calls.get('mage_weighted_units', 0) + 1
        held = self._mage_held
        if held is not None and held.shape[-1] >= pt and held.shape[2] == score.shape[1]:          # round 8: hysteresis
            score = score + self.mage_sticky * held[0, :, :, :pt].to(score.dtype)
            self.calls['mage_sticky_units'] = self.calls.get('mage_sticky_units', 0) + 1
        if self.mage_sink or self.mage_recent_tiles:                                           # in-budget protection
            score = score.clone()
            score[..., :min(self.mage_sink, pt)] = float('inf')
            if self.mage_recent_tiles:
                score[..., max(0, pt - self.mage_recent_tiles):] = float('inf')
        kept[0, :, :, :pt].scatter_(-1, score.topk(k_tiles, dim=-1).indices, True)
        if account:
            self.calls['mage_kept_prefix_tiles'] += int(k_tiles) * H * qb
            self.calls['mage_prefix_tiles'] += int(pt) * H * qb
        return kept

    def _coverage_tiles(self, head_lse, n, qb, pt, k_min):
        """One per-unit tile count for the canvas: every (query head, 128-row block) unit's prefix-tile distribution is
        the row mean of the per-row prefix softmax; need_u = the top tiles covering mass mage_kcover; the result is the
        mage_kq quantile of need_u clipped to [k_min, mage_kmax / 64] (and to the prefix). One host read."""
        import math
        H = head_lse.shape[0]
        p = torch.softmax(head_lse.float(), -1)                                                  # [H, n, PT]
        pad = qb * 128 - n
        rows = torch.nn.functional.pad(torch.ones(n, device=p.device), (0, pad)).view(qb, 128).sum(-1)
        unit = torch.nn.functional.pad(p, (0, 0, 0, pad)).view(H, qb, 128, pt).sum(2) / rows[None, :, None]
        srt = unit.sort(-1, descending=True).values.cumsum(-1)                                   # [H, QB, PT]
        need = (srt < self.mage_kcover).sum(-1) + 1                                              # [H, QB]
        flat = need.flatten()
        rank = max(1, min(flat.numel(), math.ceil(self.mage_kq * flat.numel())))        # nearest-rank quantile:
        k = int(flat.kthvalue(rank).values.item())                                     # >= kq of the units covered
        k = max(k_min, min(k, self.mage_kmax // 64, pt))
        self.calls['kcover_tiles_sum'] = self.calls.get('kcover_tiles_sum', 0) + k
        self.calls['kcover_selections'] = self.calls.get('kcover_selections', 0) + 1
        return k

    @staticmethod
    def _split_tensors(kept, S):
        """[1, H, QB, KT] keep map -> (counts [S, H, QB] int32, idx [S, H, QB, KT] int32): each unit's kept tiles
        (ascending) dealt into S contiguous parts whose sizes differ by at most one (balanced CTAs)."""
        order = torch.argsort((~kept).to(torch.int8), dim=-1, stable=True)
        cnt = kept.sum(-1)
        kt = order.shape[-1]
        ar = torch.arange(kt, device=order.device)
        lo = [(cnt * i) // S for i in range(S + 1)]
        idx = torch.cat([torch.gather(order, -1, (lo[i][..., None] + ar).clamp_max(kt - 1)) for i in range(S)])
        counts = torch.cat([lo[i + 1] - lo[i] for i in range(S)])
        return counts.to(torch.int32).contiguous(), idx.to(torch.int32).contiguous()

    def _pool_observe(self, q, scale, pool, n, prefix):
        """One block-sparse FA4 call over the pool tiles of every unit (as MASK blocks, so the observing mask writes the
        prefix-tile log-mass z [H, QB, PT, 128], -inf outside the pool), on the paged cache of the call in flight, S
        alias splits with equal tiles per CTA. Returns (output [1, Q, H, D], z)."""
        from experiments.numerical_qk_reuse import v27_fa4
        from experiments.numerical_qk_reuse import v31_fa4_observe as ob
        ctx = self.paged
        fwd = v27_fa4.load()
        H = q.shape[1]
        pt, qb = prefix // 64, -(-n // 128)
        S = self.splits
        counts, idx = self._split_tensors(pool, S)
        zeros = torch.zeros_like(counts)
        lists = v27_fa4._BST(mask_block_cnt=counts, mask_block_idx=idx, full_block_cnt=zeros,
                             full_block_idx=torch.zeros(zeros.shape + (1,), device=zeros.device, dtype=torch.int32),
                             block_size=(128, 64))
        z = torch.full((H, qb, pt, 128), float('-inf'), device=q.device, dtype=torch.float32)
        qs = q.transpose(1, 2).expand(S, -1, -1, -1)                      # [S, Q, H, D], stride-0 batch
        table = ctx['table'][None].expand(S, -1).contiguous()
        used = torch.full((S,), ctx['nk'], device=q.device, dtype=torch.int32)
        old = ob._sm90.AttentionMask
        ob._sm90.AttentionMask = ob.ObservingMask
        try:
            o, lse = fwd(qs, ctx['k'], ctx['v'], softmax_scale=scale, causal=False, page_table=table, seqused_k=used,
                         block_sparse_tensors=lists, num_splits=1, return_lse=True,
                         aux_tensors=[z, ob._scale_tensor(scale, q.device)])[:2]
        finally:
            ob._sm90.AttentionMask = old
        if self.merge_backend == 'triton':
            return self._merge_alias2(o, lse), z
        w = torch.softmax(lse, dim=0).permute(0, 2, 1)[..., None]
        return (o.float() * w).sum(0, keepdim=True).to(o.dtype), z

    @torch.no_grad()
    def _mage_select_pool(self, q, k, v, scale, prefix, n, pool):
        """Round 5 re-selection over the candidate pool [1, H, QB, KT] observed inside the block-sparse kernel
        (pool tiles as mask blocks -> the observing mask writes z; S alias splits, equal tiles per CTA). Falls back to
        the dense observation when the paged context of the call is not available. Returns (output, kept)."""
        ctx = self.paged
        if ctx is None or k.shape[-2] != ctx['nk'] or q.shape[0] != 1:
            self.calls['mage_pool_fallbacks'] = self.calls.get('mage_pool_fallbacks', 0) + 1
            return self._mage_select_fa4(q, k, v, scale, prefix, n)
        H = q.shape[1]
        G = H // k.shape[1]
        pt, qb, kt = prefix // 64, -(-n // 128), pool.shape[-1]
        out, z = self._pool_observe(q, scale, pool, n, prefix)
        head_lse = z.permute(0, 1, 3, 2).reshape(H, qb * 128, pt)[:, :n]                         # [H, n, PT]
        budget = self._k_override if self._k_override is not None else self.mage_k
        k_tiles = max(1, min(budget // 64, pt))
        kept = self._mage_units(None, head_lse, H, n, qb, kt, pt, k_tiles, G)
        self.calls['mage_pool_reselections'] = self.calls.get('mage_pool_reselections', 0) + 1
        self.calls['mage_pool_tiles'] = self.calls.get('mage_pool_tiles', 0) + int(pool[0, :, :, :pt].sum())
        return out, kept

    # ------------------------------------------------------------------ split FA4 over the paged cache
    def _split(self, lists):
        for ref, split in self._split_cache:
            if ref is lists:
                return split[0] if isinstance(split, tuple) else split
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
        # kept wholly-prefix tiles of this map (list entries are tile indices; the first cnt entries are kept)
        pt = (self.paged['prefix'] // 64) if self.paged is not None else 0
        rows = lists.block_size[0] / 128.0                                 # 64-row (q64) maps count half blocks
        kept_prefix = None if getattr(self, 'value_clean_timing', False) else ((order < pt) & (ar < cnt[..., None])).sum() * rows
        self._split_cache = self._split_cache[-63:] + [(lists, (split, kept_prefix,
                                                                order.shape[1] * order.shape[2] * pt * rows))]
        self.calls['split_list_builds'] += 1
        return split

    def _kept_account(self, lists):
        if getattr(self, 'value_clean_timing', False):
            return
        from experiments.numerical_qk_reuse import v27_fa4
        for ref, entry in self._split_cache:
            if ref is lists and isinstance(entry, tuple):
                _, kept, total = entry
                self._kept_prefix = kept.double() if self._kept_prefix is None else self._kept_prefix + kept
                self._prefix_total += total
                if not any(lists is x for x in v27_fa4._ALLKEPT.values()):   # a dense call routed as all-kept
                    prev = getattr(self, '_sparse_kept', None)
                    self._sparse_kept = kept.double() if prev is None else prev + kept
                    self._sparse_total = getattr(self, '_sparse_total', 0) + total
                return

    def sparse_lists(self, original, q, k, v, lists, scale):
        ctx = self.paged
        if ctx is None or k.shape[-2] != ctx['nk'] or q.shape[0] != 1:
            return original(q, k, v, lists, scale)
        from experiments.numerical_qk_reuse import v27_fa4
        fwd = v27_fa4.load()
        S = self.splits
        if self.drop_guard is not None and not any(lists is x for x in v27_fa4._ALLKEPT.values()):
            lists = self._guarded(q, k, v, lists, scale)
        split = self._split(lists)
        self._kept_account(lists)
        qs = q.transpose(1, 2).expand(S, -1, -1, -1)                      # [S, Q, H, D], stride-0 batch
        table = ctx['table'][None].expand(S, -1).contiguous()
        used = torch.full((S,), ctx['nk'], device=q.device, dtype=torch.int32)
        o, lse = fwd(qs, ctx['k'], ctx['v'], softmax_scale=scale, causal=False, page_table=table, seqused_k=used,
                     block_sparse_tensors=split, num_splits=1, return_lse=True)[:2]
        self.calls['split_fa4_calls'] += 1
        if self.residual is not None and not any(lists is x for x in v27_fa4._ALLKEPT.values()):
            return self._with_residual(q, k, v, lists, scale, o, lse)
        if self.merge_backend == 'triton':
            return self._merge_alias2(o, lse)
        w = torch.softmax(lse, dim=0).permute(0, 2, 1)[..., None]         # [S, Q, H, 1], exact LSE merge
        return (o.float() * w).sum(0, keepdim=True).to(o.dtype)           # [1, Q, H, D]

    def _centroids(self, k, v, pt):
        from experiments.numerical_qk_reuse import v31_residual as rs
        key = (self._cur_layer, self.canvas_id, pt)
        means = self._tile_means.get(key)
        if means is None:
            for old in [x for x in self._tile_means if x[1] != self.canvas_id]:   # earlier canvases: never reused
                del self._tile_means[old]
            means = self._tile_means[key] = rs.tile_means(k, v, pt)
            self.calls['residual_centroid_builds'] = self.calls.get('residual_centroid_builds', 0) + 1
        return means

    def _guarded(self, q, k, v, lists, scale):
        """The keep map with the dropped-mass guard applied, built once per held map object."""
        for ref, guarded in self._guard_cache:
            if ref is lists:
                return guarded
        from experiments.numerical_qk_reuse import v27_fa4, v31_residual as rs
        pt = self.paged['prefix'] // 64
        guarded = lists
        if pt:
            kept = rs.kept_from_lists(lists)
            share = rs.dropped_share(q, k, self._centroids(k, v, pt)[0], kept, pt, scale, q_block=lists.block_size[0])
            flag = share > self.drop_guard                                    # [H, QB]
            n_flag = int(flag.sum())
            self._residual_tiles += flag.numel() * pt / 64.0 * lists.block_size[0] / 128.0   # centroid scoring cost
            self.calls['guard_units'] = self.calls.get('guard_units', 0) + flag.numel()
            self.calls['guard_flagged_units'] = self.calls.get('guard_flagged_units', 0) + n_flag
            if n_flag:
                kept = kept.clone()
                kept[..., :pt] |= flag[..., None]
                guarded = v27_fa4.block_sparse_tensors(kept[None], q_block=lists.block_size[0])
        self._guard_cache = self._guard_cache[-63:] + [(lists, guarded)]
        return guarded

    def _with_residual(self, q, k, v, lists, scale, o, lse):
        """Exact sparse partials [S, Q, H, D] / [S, H, Q] plus the dropped tiles' centroid partial, LSE-merged."""
        from experiments.numerical_qk_reuse import v31_residual as rs
        pt = self.paged['prefix'] // 64
        if not pt:
            return self._merge_alias2(o, lse) if self.merge_backend == 'triton' else rs.merge(o, lse, o.dtype)
        means = self._centroids(k, v, pt)
        kept = rs.kept_from_lists(lists)
        o_d, lse_d = rs.residual_partial(q, means[0], means[1], kept, pt, scale, q_block=lists.block_size[0])
        self.calls['residual_calls'] = self.calls.get('residual_calls', 0) + 1
        h, qb = kept.shape[0], kept.shape[1]
        self._residual_tiles += h * qb * pt / 64.0 * lists.block_size[0] / 128.0   # one centroid key per tile
        if self.merge_backend == 'triton':                                 # the plain path's merge, then 2-way
            sparse = self._merge_alias2(o, lse)                            # [1, Q, H, D]
            lse_s = torch.logsumexp(lse.float(), dim=0, keepdim=True)      # [1, H, Q]
            return rs.merge(torch.cat([sparse.float(), o_d]), torch.cat([lse_s, lse_d]), o.dtype)
        return rs.merge(torch.cat([o.float(), o_d]), torch.cat([lse.float(), lse_d]), o.dtype)

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


def accepted_mask(scaled, entropy_bound):
    """The official sampler's acceptance mask, recomputed from the same temperature-scaled logits it returns:
    positions sorted by token entropy are accepted while (cumulative entropy - running max) <= entropy_bound
    (diffusion_gemma._compiled_sample_step, phase 4). Same ops on the same tensor; only an exact tie at the bound can
    resolve differently under the compiled function's fusion."""
    x = scaled.float()
    logp = x.log_softmax(dim=-1)
    ent = -(logp.exp() * logp).sum(dim=-1)
    s, idx = torch.sort(ent, dim=-1)
    keep = (torch.cumsum(s, dim=-1) - torch.cummax(s, dim=-1).values) <= entropy_bound
    return torch.zeros_like(keep).scatter_(1, idx, keep)


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
                if a.value_audit and not a.step_ctx['encoder']:
                    mode=str(cudagraph_mode)
                    a._value_decode_modes[mode]=a._value_decode_modes.get(mode,0)+1
        return prepare_attn(self, input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups,
                            kv_cache_config, for_capture=for_capture, ubatch_idx=ubatch_idx)

    import inspect
    signature = inspect.signature(getattr(sample_step, '_torchdynamo_orig_callable',
                                          getattr(sample_step, '__wrapped__', sample_step)))

    def sample_step_patched(*args, **kwargs):
        scaled = sample_step(*args, **kwargs)
        a = _ACTIVE
        if a is not None and a.bound and a.step_ctx is not None and not a.step_ctx['encoder']:
            accepted = stats = None
            if a.fused_stats_eligible():
                from experiments.numerical_qk_reuse.v31_logit_stats import accepted_from_entropy, row_stats
                stats = row_stats(scaled)                                  # all CL rows, as the sampler sorts them
                if a.needs_accepted():
                    accepted = accepted_from_entropy(
                        stats.entropy, float(signature.bind(*args, **kwargs).arguments['entropy_bound']))
            elif a.needs_accepted():
                accepted = accepted_mask(scaled, float(signature.bind(*args, **kwargs).arguments['entropy_bound']))
            conv = a.dense_when is not None and a.dense_when[0] == 'conv'
            if a.trace_canvas or conv:
                from experiments.numerical_qk_reuse.v31_logit_stats import row_stats
                ts = stats if stats is not None else row_stats(scaled)
                mean_entropy = ts.entropy.mean()                 # all CL rows, as the sampler's mean_entropy
                if a._conf_threshold is None:
                    a._conf_threshold = float(signature.bind(*args, **kwargs).arguments['confidence_threshold'])
                if a.trace_canvas:
                    a._ent_trace.append(mean_entropy)
                if conv:                                         # one scalar read per step, at the step boundary
                    a._dense_next = bool(mean_entropy.item() < a.dense_when[1] * a._conf_threshold)
            if (a.arm == 'mage' and a.mage_row_weight in ('temporal', 'mt', 'ct') and a.mage_reselect is not None
                    and a._canvas_step + 1 in a.mage_reselect):             # two steps ahead: keep the argmax
                a._mage_prev_argmax = scaled[0, :a.step_ctx['n']].argmax(-1)
            if (a.arm == 'mage' and a.mage_reselect_trigger is not None
                    and (a._trig_canvas != a.canvas_id or a._trig_count < len(a.mage_reselect_trigger)
                         or a.mage_clock_trace)):                       # a traced clock keeps reading every step
                from experiments.numerical_qk_reuse.v31_logit_stats import accepted_from_entropy, row_stats
                n_rows = a.step_ctx['n']
                eb = float(signature.bind(*args, **kwargs).arguments['entropy_bound'])
                rs = stats if stats is not None else row_stats(scaled)                 # reuse the fused statistics
                acc = (accepted if accepted is not None else accepted_from_entropy(rs.entropy, eb))[0, :n_rows]
                a._progress_observe(rs.argmax[0, :n_rows], (rs.max - rs.lse).exp()[0, :n_rows], acc, scaled, n_rows, eb)
            if a.cg_stop is not None or (a.stall_rescue is not None and not a._dense_next):
                from experiments.numerical_qk_reuse.v31_logit_stats import accepted_from_entropy, row_stats
                n_rows = a.step_ctx['n']
                bound = signature.bind(*args, **kwargs).arguments
                rs = stats if stats is not None else row_stats(scaled)
                acc = (accepted if accepted is not None
                       else accepted_from_entropy(rs.entropy, float(bound['entropy_bound'])))[0, :n_rows]
                settled, run = a._cg_track(rs.argmax[0, :n_rows], (rs.max - rs.lse).exp()[0, :n_rows], acc)
                stopped = a._cg_stop(bound, run) if a.cg_stop is not None else False
                if a.stall_rescue is not None and not a._dense_next and not stopped:
                    a._stall_check(settled)
            if (a.arm == 'mage' and a.mage_row_weight is not None and a.mage_reselect is not None
                    and a._canvas_step in a.mage_reselect):                 # the next step re-selects: weigh its rows
                a._mage_row_w = a._row_weight_from_logits(
                    scaled, a.step_ctx['n'], float(signature.bind(*args, **kwargs).arguments['entropy_bound']))
            a.on_sample(scaled, accepted, stats=stats)
        return scaled

    def forward_patched(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None,
                        output_block_scale=None):
        a = _ACTIVE
        local_ev = None
        if a is not None and a.bound and attn_metadata is not None and output_scale is None:
            layer_idx = a.active_for(getattr(layer, 'layer_name', ''))
            if layer_idx is not None:
                ev = None
                if a.profile:
                    ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    ev[0].record()
                gated = (a.arm in ('method', 'mage') and a.dense_below is not None and a.step_ctx is not None
                         and a.step_ctx['seq_len'] < a.dense_below)
                if a.arm == 'native' or (a.arm in ('method', 'mage') and (a._dense_now or gated)):
                    a.calls['global_calls'] += 1
                    n = a.step_ctx['n']
                    a._global_tiles += self.num_heads * -(-n // 128) * ((a.step_ctx['seq_len'] - n) // 64)
                    prefix = a.step_ctx['seq_len'] - n
                    a._global_canvas_tiles += self.num_heads * -(-n // 128) * (-(-a.step_ctx['seq_len'] // 64) - prefix // 64)
                    if a.arm != 'native':
                        counter = 'dense_gate_calls' if gated and not a._dense_now else 'dense_fallback_calls'
                        a.calls[counter] = a.calls.get(counter, 0) + 1
                    r = forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale,
                                output_block_scale)
                else:
                    r = a.forward(self, layer_idx, query, kv_cache, attn_metadata, output)
                if ev is not None:
                    ev[1].record()
                    a._events.append(ev)
                return r
            local_idx = a.local_for(getattr(layer, 'layer_name', ''))
            if local_idx is not None:
                r = a.forward_local(self, local_idx, query, kv_cache, attn_metadata, output)
                return forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                               output_scale, output_block_scale) if r is None else r
            if a.step_ctx is not None:
                local_audited = a.value_audit and a._value_native_local_account(
                    getattr(layer,'layer_name',''),self.num_heads,int(attn_metadata.num_actual_tokens))
                if local_audited:
                    local_ev = (torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
                    local_ev[0].record()
                a.calls['passthrough'] += 1
        result = forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale,
                         output_block_scale)
        if local_ev is not None:
            local_ev[1].record()
            a._value_local_events.append(local_ev)
        return result

    from experiments.numerical_qk_reuse import v27_fa4
    sparse_lists = v27_fa4.sparse_lists

    def sparse_lists_patched(q, k, v, lists, scale):
        a = _ACTIVE
        if a is not None and a.bound and a.paged is not None:
            return a.sparse_lists(sparse_lists, q, k, v, lists, scale)
        return sparse_lists(q, k, v, lists, scale)

    v27_fa4.sparse_lists = sparse_lists_patched
    if adapter.regroup_diag:
        from experiments.numerical_qk_reuse import v27_dense_prefix as dpmod
        inner_route = dpmod.route

        def route_diag(scores, z, reference, state, *, sensitivity=None, log_threshold, **kw):
            r = inner_route(scores, z, reference, state, sensitivity=sensitivity, log_threshold=log_threshold, **kw)
            a = _ACTIVE
            if a is not None and kw.get('risk_topk') is None and kw.get('risk_budget') is None:
                a._regroup_account(state, reference, sensitivity, log_threshold, scores.shape[2])
            return r
        dpmod.route = route_diag
    if adapter.dp_build == 'chunked':
        from experiments.numerical_qk_reuse import v27_dense_prefix
        from experiments.numerical_qk_reuse.v31_dp_chunked import build_chunked
        v27_dense_prefix.build = build_chunked
    if adapter.risk_group == 'kv':
        from experiments.numerical_qk_reuse import v27_dense_prefix as dp_topk
        from experiments.numerical_qk_reuse.v31_group_select import grouped_topk_skip

        def topk_skip_grouped(lognorm, eligible, sensitivity, reference, nq, keep):
            a = _ACTIVE
            if a is not None:
                a.calls['grouped_topk_calls'] = a.calls.get('grouped_topk_calls', 0) + 1
            return grouped_topk_skip(lognorm, eligible, sensitivity, reference, nq, keep)
        dp_topk.topk_skip = topk_skip_grouped
    if adapter.observe_backend == 'fa4':
        from experiments.numerical_qk_reuse import v27_consumer64
        from experiments.numerical_qk_reuse.v31_fa4_observe import fused_observe_fa4
        triton_observe = v27_consumer64.fused_observe

        def fused_observe_routed(q, k, v, sketch, scale, prefix_tiles, summary, splits=2, mu=True,
                                 mu_precision='tf32x3', output=True):
            a = _ACTIVE
            if mu:                                         # exact mu needs the per-row projected V: Triton kernel
                if a is not None:
                    a.calls['triton_observations'] = a.calls.get('triton_observations', 0) + 1
                return triton_observe(q, k, v, sketch, scale, prefix_tiles, summary, splits=splits, mu=mu,
                                      mu_precision=mu_precision, output=output)
            if a is not None:
                a.calls['fa4_observations'] = a.calls.get('fa4_observations', 0) + 1
            return fused_observe_fa4(q, k, v, sketch, scale, prefix_tiles, summary, splits=splits, mu=mu,
                                     mu_precision=mu_precision, output=output)
        v27_consumer64.fused_observe = fused_observe_routed
    dg.DiffusionGemmaModelState.prepare_attn = prepare_attn_patched
    dg._compiled_sample_step = sample_step_patched
    fa.FlashAttentionImpl.forward = forward_patched
    dg._v27_patched = True
