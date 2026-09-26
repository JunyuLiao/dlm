"""v14 CVM-T: cached value-risk margin + live temporal protection (new, explicitly
named simplification; NOT the unchanged M1 and NOT specified by the meeting).

Anchor (true observation, A8 clock and every new canvas / cache epoch): Junyu's
fresh value-aware kernel on CURRENT Q/K/V, via a PRIVATE v5 build that
  * never skips mandatory tiles (j >= protect_tile: current canvas and the partial
    prefix/canvas edge tile), inside decide() so retained state and output agree;
  * optionally exports each row's UNWEIGHTED log risk log_rho[i,J] (before log s_i,
    before the physical max), with the scan's own first-support/inactive sentinels.
Ordinary step (CVM_T): one planner launch
      restore[I,J] = any_{valid i in I} log s_i(t) + log_rho_a[i,J] >= log tau
      keep_t = keep_anchor OR keep_previous OR restore      (add-only inside the epoch)
followed by the preselected-support Hopper consumer on CURRENT QK/softmax/V.
B8_P: identical anchors/export/consumer, bitmap held (no restore). T_P: fresh kernel
with the same mandatory set every step. Row-specific s is never averaged.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import torch

SOURCE = Path(__file__).parent / 'csrc'
FMHA = Path('/home/exouser/dyh/numerical_qk_reuse_native_20260924/build_cp1/sources_db080045f7a5fbce')
CUTLASS = Path('/home/exouser/chw/dlm_test/cutlass/include')
NVCC = Path('/usr/local/cuda/bin/nvcc')
_LOADED = None
MODES = ('TP', 'B8P', 'CVM')


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_v5(build_dir: Path) -> dict:
    """Private v5 kernel (.so, C ABI) + g++ ATen bridge (namespace value_direction_hopper_v5)."""
    from torch.utils.cpp_extension import include_paths, library_paths
    import sysconfig
    files = [SOURCE / 'value_direction_v5.cu', SOURCE / 'value_direction_v5.h', SOURCE / 'torch_bridge_v5.cpp',
             FMHA / 'fmha/hopper/utils_hgmma_bf16.h', FMHA / 'fmha/hopper/utils_warpgroup.h']
    hashes = {p.name: sha256(p) for p in files}
    nvcc_version = subprocess.run([str(NVCC), '--version'], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    flags = ['-std=c++17', '-O3', '-arch=sm_90a', '--shared', '-Xcompiler=-fPIC', '-lineinfo', '--ptxas-options=-v']
    key = hashlib.sha256(json.dumps([hashes, nvcc_version, flags, torch.__version__, 5], sort_keys=True).encode()).hexdigest()[:16]
    out = Path(build_dir) / f'v5_{key}'
    snapshot = out / 'sources'
    snapshot.mkdir(parents=True, exist_ok=True)
    for p in files[:3]:
        (snapshot / p.name).write_bytes(p.read_bytes())
    kernel = out / f'value_direction_v5_{key}.so'
    cmd = [str(NVCC), *flags, '-I' + str(CUTLASS), '-I' + str(FMHA), '-I' + str(snapshot), '-I/usr/local/cuda/include/cccl',
           str(snapshot / 'value_direction_v5.cu'), '-o', str(kernel), '-L/usr/local/cuda/lib64', '-lcudart', '-lcuda',
           '-Xlinker=-rpath,/usr/local/cuda/lib64']
    t0 = time.monotonic()
    if not kernel.exists():
        r = subprocess.run(cmd, capture_output=True, text=True)
        (out / 'nvcc.log').write_text(r.stdout + r.stderr)
        if r.returncode:
            raise RuntimeError(f'nvcc failed: {out / "nvcc.log"}')
    nvcc_s = time.monotonic() - t0
    bridge = out / f'value_direction_torch_v5_{key}.so'
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    bcmd = (['c++', '-shared', '-fPIC', '-std=c++17', '-O3', f'-D_GLIBCXX_USE_CXX11_ABI={abi}', '-DTORCH_API_INCLUDE_EXTENSION_H']
            + [f'-isystem{p}' for p in include_paths()] + ['-isystem' + sysconfig.get_paths()['include'],
            '-I/usr/local/cuda/include', '-I' + str(snapshot), str(snapshot / 'torch_bridge_v5.cpp'), str(kernel),
            '-L/usr/local/cuda/lib64', '-lcudart'] + [f'-L{p}' for p in library_paths()]
            + ['-lc10_cuda', '-lc10', '-ltorch_cpu', '-ltorch', '-ltorch_python', '-Wl,-rpath,' + str(out),
               '-Wl,-rpath,/usr/local/cuda/lib64', '-o', str(bridge)])
    if not bridge.exists():
        r = subprocess.run(bcmd, capture_output=True, text=True)
        (out / 'bridge.log').write_text(r.stdout + r.stderr)
        if r.returncode:
            raise RuntimeError(f'bridge failed: {out / "bridge.log"}')
    identity = dict(schema='v14_v5_build_v1', key=key, abi_version=5, sources=hashes, nvcc_version=nvcc_version, flags=flags,
                    command=cmd, bridge_command=bcmd, torch=torch.__version__, kernel=str(kernel), kernel_sha256=sha256(kernel),
                    bridge=str(bridge), bridge_sha256=sha256(bridge), namespace='value_direction_hopper_v5', nvcc_seconds=nvcc_s,
                    derived_from=dict(value_direction_cu=sha256(SOURCE / 'value_direction.cu'), torch_bridge_cpp=sha256(SOURCE / 'torch_bridge.cpp')))
    (out / 'build_identity.json').write_text(json.dumps(identity, indent=2, sort_keys=True) + '\n')
    return identity


def load_v5(identity_path) -> dict:
    global _LOADED
    identity = json.loads(Path(identity_path).read_text())
    if identity['torch'] != torch.__version__:
        raise RuntimeError('v5 bridge built for another torch')
    for name in ('kernel', 'bridge'):
        if sha256(identity[name]) != identity[f'{name}_sha256']:
            raise RuntimeError(f'v5 {name} differs from its build identity')
    if _LOADED is None:
        torch.ops.load_library(identity['bridge'])
        _LOADED = identity
    elif _LOADED['bridge'] != identity['bridge']:
        raise RuntimeError('another v5 bridge already loaded')
    return identity


class V5Kernel:
    """Drop-in for cuda.Kernel inside Junyu's Attention (torch-op path only)."""

    def __init__(self):
        from .fused import Output
        self.Output = Output
        self.abi_version = 5
        self.export = False
        self.protect_tile = -1
        self.last_export = None
        self.last_masks = None

    def __call__(self, q, k, v, z, reference, *, mask=None, scale=None, log_threshold=-float('inf'), mode='value',
                 trace=False, overlap=True, timings=None, precision='ieee', debug_scores=None, split_pv=False, tma=False,
                 sensitivity=None):
        from .masks import PackedMask
        if timings is not None or debug_scores is not None or split_pv:
            raise ValueError('v5 adapter supports the production torch-op path only')
        packed = isinstance(mask, PackedMask)
        kind = 3 if packed else 0 if mask is None else 1 if mask.dtype == torch.bool else 2
        tensor = mask.tensor if packed else q if mask is None else mask
        out = torch.ops.value_direction_hopper_v5.attention(
            q, k, v, z, reference, tensor, kind, q.shape[-1] ** -.5 if scale is None else scale, log_threshold,
            {'dense': 0, 'value': 1, 'blasst': 2}[mode], {'ieee': 0, 'tf32x3': 1, 'tf32x3_register': 2, 'tf32x3_shared': 3}[precision],
            trace, tma, int(overlap), sensitivity, self.export, int(self.protect_tile))
        self.last_export = out[6] if self.export else None
        self.last_masks = (out[1], out[2])                     # skipped, eligible of THIS call
        return self.Output(*out[:6])


try:
    import triton as tr
    import triton.language as tl

    @tr.jit(do_not_specialize=['Q', 'KT', 'PROTECT'])
    def _plan(LOGRHO, S, SKIP, Q, KT, PROTECT, LOG_TAU, QB: tl.constexpr, H: tl.constexpr, HAS_S: tl.constexpr,
              JB: tl.constexpr):
        """restore[I,J] = any valid row i in I: log s_i + log_rho[i,J] >= log tau  ->  SKIP[I,J] = 0."""
        qb, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        rows = qb * 128 + tl.arange(0, 128)
        valid = rows < Q
        if HAS_S:
            ls = tl.log(tl.load(S + b * Q + rows, valid, other=1.))
        else:
            ls = tl.zeros((128,), tl.float32)
        for j0 in range(0, PROTECT, JB):
            js = j0 + tl.arange(0, JB)
            jv = js < PROTECT
            lr = tl.load(LOGRHO + ((b * H + h) * KT + js[:, None]) * Q + rows[None, :],
                         jv[:, None] & valid[None, :], other=-float('inf'))
            risk = tl.where(jv[:, None] & valid[None, :], lr + ls[None, :], -float('inf'))
            restore = tl.max(risk, 1) >= LOG_TAU
            dest = SKIP + ((b * H + h) * QB + qb) * KT + js
            tl.store(dest, tl.zeros((JB,), tl.int1), jv & restore)
except ImportError:  # CPU-only reference tests
    tr = None


def plan_reference(log_rho, sensitivity, skipped, protect_tile, log_tau):
    """CPU/torch reference of the planner (same rule, row-specific s, Q128 tiles)."""
    b, h, kt, q = log_rho.shape
    ls = torch.zeros(b, q, dtype=log_rho.dtype) if sensitivity is None else torch.log(sensitivity.to(log_rho.dtype))
    risk = log_rho + ls[:, None, None, :]                        # [B,H,KT,Q]
    qb = (q + 127) // 128
    pad = qb * 128 - q
    if pad:
        risk = torch.nn.functional.pad(risk, (0, pad), value=-float('inf'))
    restore = risk.reshape(b, h, kt, qb, 128).amax(-1).permute(0, 1, 3, 2) >= log_tau   # [B,H,QB,KT]
    restore[..., protect_tile:] = False
    out = skipped.clone()
    out[restore] = False
    return out


def plan(log_rho, sensitivity, skipped, protect_tile, log_tau):
    """In-place GPU planner: skipped[..., :protect_tile] &= ~restore."""
    b, h, qb, kt = skipped.shape
    q = log_rho.shape[-1]
    if protect_tile <= 0:
        return skipped
    _plan[(qb, h, b)](log_rho, sensitivity if sensitivity is not None else log_rho, skipped, q, kt, int(protect_tile),
                      float(log_tau), QB=qb, H=h, HAS_S=sensitivity is not None, JB=16, num_warps=4)
    return skipped


class CVMRouter:
    """GLOBAL-layer router for T_P / B8_P / CVM_T (installed through global_scope)."""

    def __init__(self, adapter, library, torch_library, v5_build, thresholds, *, mode, support_build, period=8,
                 guard_mode='fused'):
        from experiments.diffusion_gemma_jl_output_aware.projections import Projections
        from . import integration as junyu
        from . import support as support_consumer
        if mode not in MODES:
            raise ValueError(mode)
        self.mode, self.period, self.thresholds = mode, int(period), thresholds
        self.v5_identity = load_v5(v5_build)
        self.fresh = junyu.Attention(adapter, library, thresholds, mode='value', projections=Projections(),
                                     torch_library=torch_library, collect=False)
        self.kernel = V5Kernel()
        self.fresh.kernel = self.kernel                    # Junyu's call path, private v5 kernel
        self.support_identity = support_consumer.load(support_build) if mode != 'TP' else None
        self._support = support_consumer
        self.guard_mode = guard_mode
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        self._native = sdpa_attention_forward
        self.canvas, self.step, self.epoch = -1, -1, 0
        self.layers = {}
        self.counts = dict(calls=0, anchors=0, ordinary=0, fallback_native=0, restored_tiles=0, planner_calls=0,
                           kept_prefix_tiles=0, prefix_tiles=0)
        self.handles = []
        for name, module in adapter.model.named_modules():
            if type(module).__name__ == 'DiffusionGemmaEncoderModel':
                self.handles.append(module.register_forward_pre_hook(self.invalidate))
        self.audit = False
        self.audit_rows = []

    # State (NativeReuseState) drives these
    @property
    def query_sensitivity(self):
        return self.fresh.query_sensitivity

    @query_sensitivity.setter
    def query_sensitivity(self, value):
        self.fresh.query_sensitivity = value

    @property
    def policy_selector(self):
        return self.fresh.policy_selector

    @policy_selector.setter
    def policy_selector(self, value):
        self.fresh.policy_selector = value

    def invalidate(self, *_):
        self.epoch += 1
        self.layers.clear()

    def begin_step(self, canvas, step):
        if canvas != self.canvas:
            self.layers.clear()
        self.canvas, self.step = canvas, step

    @torch.no_grad()
    def __call__(self, module, q, k, v, mask, *, dropout=0., scaling=None, is_causal=None, sliding_window=None, **kwargs):
        if mask is not None or sliding_window or module.is_sliding or dropout:
            raise ValueError('CVM router is qualified for GLOBAL unmasked decoder calls only')
        b, h, nq, d = q.shape
        nk = k.shape[-2]
        protect = (nk - nq) // 64                            # complete immutable-prefix tiles
        layer = int(module.layer_idx)
        self.counts['calls'] += 1
        kw = dict(dropout=dropout, scaling=scaling, is_causal=is_causal, sliding_window=sliding_window, **kwargs)
        if protect <= 0:                                     # structural no-op: nothing prunable
            self.counts['fallback_native'] += 1
            return self._native(module, q, k, v, mask, **kw)
        self.kernel.protect_tile = protect
        if self.mode == 'TP':
            self.kernel.export = False
            self.counts['anchors'] += 1
            return self.fresh(module, q, k, v, mask, **kw)
        ident = (self.canvas, self.epoch, b, h, nq, nk, d, str(q.dtype))
        st = self.layers.get(layer)
        if st is None or st['ident'] != ident or self.step % self.period == 0:
            self.kernel.export = True
            out = self.fresh(module, q, k, v, mask, **kw)
            self.kernel.export = False
            skipped, eligible = self.kernel.last_masks        # this anchor call's own decision
            self.layers[layer] = dict(ident=ident, anchor_step=self.step, skipped=skipped, eligible=eligible,
                                      log_rho=self.kernel.last_export, protect=protect)
            self.counts['anchors'] += 1
            return out
        self.counts['ordinary'] += 1
        skipped = st['skipped']
        if self.mode == 'CVM':
            before = int(skipped[..., :protect].sum()) if self.audit else None
            tau = float(self.thresholds['global']['log_threshold'])
            plan(st['log_rho'], self.query_sensitivity, skipped, protect, tau)
            self.counts['planner_calls'] += 1
            if self.audit:
                self.audit_rows.append(dict(layer=layer, step=self.step, restored=before - int(skipped[..., :protect].sum()),
                                            kept_prefix=int((~skipped[..., :protect] & st['eligible'][..., :protect]).sum()),
                                            prefix_tiles=int(st['eligible'][..., :protect].sum())))
        out, lse, invalid, _ = self._support.attention(q, k, v, skipped, st['eligible'], scale=scaling if scaling is not None else d ** -.5,
                                                       window=0, layout=1)
        returned = out.transpose(1, 2)                       # [B,Q,H,D] contiguous storage
        if self.guard_mode == 'fused':
            from experiments.numerical_qk_reuse.cached_executor import fused_guard
            fused_guard(returned, invalid)
        else:
            torch._assert_async(~invalid.any(), 'invalid consumer rows')
            torch._assert_async(torch.isfinite(returned).all(), 'non-finite consumer output')
        return returned, None

    def counters(self):
        return dict(self.counts, mode=self.mode, period=self.period, v5_build=self.v5_identity['key'],
                    support_build=None if self.support_identity is None else self.support_identity['key'],
                    guard_mode=self.guard_mode, live_layers=sorted(self.layers),
                    metadata_bytes=sum(st['log_rho'].numel() * 4 + st['skipped'].numel() * 2 for st in self.layers.values()))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.layers.clear()
        self.fresh.cache.close()
