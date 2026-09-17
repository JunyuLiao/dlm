"""Batched diagnostic replay of fixed retained supports, never deployed routing."""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib
from experiments.diffusion_gemma_value_aware_gpu.kernels import _logaddexp
from .config import Config
from .kernels import METHOD


@tr.jit
def _trace(Z, MU, N, WN, REF, SKIP, OUT, Q: tl.constexpr, T: tl.constexpr,
           R: tl.constexpr, RP: tl.constexpr, PRIMARY: tl.constexpr,
           MODE: tl.constexpr, KAPPA: tl.constexpr, MASS: tl.constexpr):
    h = tl.program_id(0)
    q = tl.arange(0, 128)
    rr = tl.arange(0, RP)
    logz = tl.full((128,), -float('inf'), tl.float32)
    previous = tl.full((128, RP), 0., tl.float32)
    ref = tl.maximum(tl.load(REF+h), 1e-12)
    for j in range(T):
        off = (h*Q+q)*T+j
        z = tl.load(Z+off, q < Q, other=-float('inf'))
        n = tl.load(N+off, q < Q, other=0)
        mu = tl.load(MU+off[:, None]*R+rr[None, :], (q[:, None] < Q) & (rr[None, :] < R), other=0.)
        wn = tl.load(WN+off, q < Q, other=0.)
        combined = _logaddexp(logz, z)
        alpha = lib.exp(z-tl.where(combined > -float('inf'), combined, 0.))
        alpha = tl.where(n > 0, alpha, 0.)
        diff = mu-previous
        first = rr[None, :] < PRIMARY
        distance = tl.sqrt(tl.sum(tl.where(first, diff*diff, 0.), 1))/ref
        magnitude = tl.sqrt(tl.sum(tl.where(first, mu*mu, 0.), 1))
        kappa = magnitude/(wn+1e-12)
        centered = alpha*distance
        risk = centered
        if MODE == 1:
            risk = alpha*magnitude/ref
        elif MODE == 2:
            risk = alpha
        elif MODE == 3:
            second = alpha*tl.sqrt(tl.sum(tl.where(~first, diff*diff, 0.), 1))/ref
            risk = tl.where((kappa < KAPPA) & (alpha >= MASS), tl.maximum(centered, second), centered)
        supported = (n > 0) & (logz > -float('inf'))
        tl.store(OUT+off*7+0, alpha, q < Q)
        tl.store(OUT+off*7+1, distance, q < Q)
        tl.store(OUT+off*7+2, kappa, q < Q)
        tl.store(OUT+off*7+3, risk, q < Q)
        tl.store(OUT+off*7+4, centered, q < Q)
        tl.store(OUT+off*7+5, supported.to(tl.float32), q < Q)
        tl.store(OUT+off*7+6, (n > 0).to(tl.float32), q < Q)
        skip = tl.load(SKIP+h*T+j).to(tl.int1)
        addz = tl.where((n > 0) & ~skip, z, -float('inf'))
        newz = _logaddexp(logz, addz)
        safe = tl.where(newz > -float('inf'), newz, 0.)
        a, b = lib.exp(logz-safe), lib.exp(addz-safe)
        previous = a[:, None]*previous+b[:, None]*mu
        logz = newz


def trace(state, ref, skip, config=None):
    c = config or Config()
    b, h, q, t = state['logz'].shape
    if b != 1 or q > 128 or ref.shape != (b, h):
        raise ValueError('Diagnostic replay expects one physical query tile, expanded head references')
    r = state['mu'].shape[-1]
    primary = c.rank if c.method == 'cancellation_guard' else r
    output = torch.empty((b, h, q, t, 7), dtype=torch.float32, device=ref.device)
    _trace[(h,)](state['logz'].contiguous(), state['mu'].contiguous(), state['count'].contiguous(),
        state['mean_norm'].contiguous(), ref.contiguous(), skip.contiguous(), output,
        q, t, r, tr.next_power_of_2(r), primary, METHOD[c.method], c.cancellation_cutoff,
        c.substantial_mass, num_warps=8 if r >= 128 else 4, enable_fp_fusion=False)
    return {key:output[..., i] for i, key in enumerate(('alpha', 'distance', 'kappa', 'risk', 'centered', 'supported', 'active'))}
