"""Trusted FP32 implementation. Full-dimensional data are not routing inputs."""
import torch
import torch.nn.functional as F
from .config import Config


def block_statistics(scores, valid, sketches, projected_norm=None):
    """QK plus projected token V -> exact within-block sketch means.

    No full-dimensional block PV is formed, except when explicitly called with
    identity values as the expensive reference control/diagnostic.
    """
    valid = torch.broadcast_to(valid, scores.shape)
    pad = (-scores.shape[-1]) % 64
    s = F.pad(scores.float().masked_fill(~valid, -torch.inf), (0, pad), value=-torch.inf).unflatten(-1, (-1, 64))
    v = F.pad(valid, (0, pad)).unflatten(-1, (-1, 64))
    count = v.sum(-1)
    maximum = s.amax(-1)
    weights = torch.exp(s-torch.where(count > 0, maximum, 0.)[..., None]).masked_fill(~v, 0.)
    ell = weights.sum(-1)
    weights = weights/ell.clamp_min(1e-30)[..., None]
    z = F.pad(sketches.float(), (0, 0, 0, pad)).unflatten(-2, (-1, 64))
    mu = torch.einsum('bhqtk,bhtkr->bhqtr', weights, z)
    norms = sketches.norm(dim=-1) if projected_norm is None else projected_norm
    zn = F.pad(norms, (0, pad)).unflatten(-1, (-1, 64))
    mean_norm = (weights*zn[..., None, :, :]).sum(-1)
    logz = torch.where(count > 0, maximum+ell.clamp_min(1e-30).log(), -torch.inf)
    return dict(logz=logz, mu=mu, count=count, b=maximum, mean_norm=mean_norm,
                active=count > 0, eligible=(count > 0).any(-2))


def route(state, ref, config=None, return_trace=False, forced_skip=None):
    c = config or Config()
    z = state['logz']
    previous = torch.full_like(z[..., 0], -torch.inf)
    output = torch.zeros_like(state['mu'][..., 0, :])
    masks, traces = [], []
    for j in range(z.shape[-1]):
        active = state['active'][..., j]
        support = torch.isfinite(previous)
        combined = torch.logaddexp(previous, z[..., j])
        alpha = torch.exp(z[..., j]-torch.where(torch.isfinite(combined), combined, 0.)).masked_fill(~active, 0.)
        mu = state['mu'][..., j, :]
        delta = alpha[..., None]*(mu-output)
        if c.method == 'mass_exact':
            risk = alpha
        elif c.method == 'contribution':
            risk = alpha*mu.norm(dim=-1)/ref[..., None]
        elif c.method == 'cancellation_guard':
            risk = delta[..., :c.rank].norm(dim=-1)/ref[..., None]
            kappa = mu[..., :c.rank].norm(dim=-1)/(state['mean_norm'][..., j]+1e-12)
            second = delta[..., c.rank:].norm(dim=-1)/ref[..., None]
            risk = torch.where((kappa < c.cancellation_cutoff) & (alpha >= c.substantial_mass),
                               torch.maximum(risk, second), risk)
        else:
            risk = delta.norm(dim=-1)/ref[..., None]
        log_risk = risk.log()
        log_risk = torch.where(active, torch.where(support, log_risk, torch.inf), -torch.inf)
        worst = log_risk.amax(-1)
        skip = state['eligible'][..., j] & (worst < c.log_threshold)
        if forced_skip is not None:
            skip = forced_skip[..., j]
        if return_trace:
            traces.append(dict(alpha=alpha, previous=output.clone(), mu=mu,
                delta=delta, log_risk=log_risk, support=support, active=active, worst=worst))
        masks.append(skip)
        include = active & ~skip[..., None]
        newz = torch.logaddexp(previous, z[..., j].masked_fill(~include, -torch.inf))
        safe = torch.where(torch.isfinite(newz), newz, 0.)
        a = torch.exp(previous-safe)
        b = torch.exp(z[..., j].masked_fill(~include, -torch.inf)-safe)
        output = a[..., None]*output+b[..., None]*mu
        previous = newz
    mask = torch.stack(masks, -1)
    return (mask, traces) if return_trace else mask
