"""Auditable streaming decisions and explicit attention reference operators.

The new criteria use retained online state BEFORE the current block. The
legacy BLASST reference deliberately uses the maximum of ALL preceding blocks,
including skipped ones, matching the repository's aggressive extension.
"""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from experiments.diffusion_gemma_oracle.routing import select

Q_TILE, KV_TILE = 128, 64
BETAS = {.25: -.6744897501960817, .5: 0., .75: .6744897501960817, .9: 1.2815515655446004}
POOLINGS = ('max', 'mean', 'rms', 'p95', 'vector_mean')


@dataclass(frozen=True)
class Config:
    method: str = 'dense'
    pooling: str = 'rms'
    log_threshold: float = -math.inf
    mode: str = 'topk'
    amount: float = .5
    value_proxy: bool = False
    # Separate evidence-driven ablation, disabled for the initial screen.
    aggregation: str = 'all_rows'
    safeguard_factor: float = 4.


def value_summaries(v, key_valid, width=KV_TILE):
    """B,KVH,K,D -> block-specific metadata; padding never affects pooling.

    mean is mean(norm(V)); vector_mean is norm(mean(V)), kept distinct.
    Metadata reference is RMS token norm across all currently valid KV tokens,
    shared by pooling choices. It requires a metadata reduction, not dense O.
    """
    v = v.float(); n = v.shape[-2]; pad = (-n) % width
    key_valid = torch.broadcast_to(key_valid, v.shape[:-1])
    x = F.pad(v, (0, 0, 0, pad)).unflatten(-2, (-1, width))
    valid = F.pad(key_valid, (0, pad)).unflatten(-1, (-1, width))
    count = valid.sum(-1)
    norms = x.norm(dim=-1)
    mean = (x * valid[..., None]).sum(-2) / count.clamp_min(1)[..., None]
    sorted_norms = norms.masked_fill(~valid, torch.inf).sort(-1).values
    # Linear-interpolated quantile, matching torch.quantile on valid tokens.
    rank = (count - 1).clamp_min(0) * .95
    low = sorted_norms.gather(-1, rank.floor().long()[..., None]).squeeze(-1)
    high = sorted_norms.gather(-1, rank.ceil().long()[..., None]).squeeze(-1)
    p95 = torch.where(count > 0, low + (high - low) * (rank - rank.floor()), 0.)
    sums_sq = (norms.square() * valid).sum(-1)
    ref = (sums_sq.sum(-1) / count.sum(-1).clamp_min(1)).sqrt().clamp_min(1e-12)
    return dict(max=norms.masked_fill(~valid, 0.).amax(-1),
        mean=(norms * valid).sum(-1) / count.clamp_min(1),
        rms=(sums_sq / count.clamp_min(1)).sqrt(), p95=p95,
        vector_mean=mean.norm(dim=-1), vectors=mean,
        radius=(x - mean[..., None, :]).norm(dim=-1).masked_fill(~valid, 0.).amax(-1),
        counts=count, sums_sq=sums_sq, ref=ref,
        token_norms=v.norm(dim=-1).masked_fill(~key_valid, 0.))


class ValueCache:
    """Per-layer/native-KV-head cache, checking content before prefix reuse.

    Cache lifetime is one generation. Only complete unchanged prefix blocks
    reuse metadata; mixed boundary/canvas blocks are refreshed. The explicit
    equality check is an emulation audit cost; a deployment would use cache
    write/version notifications. No stale-prefix assumption is made.
    """
    def __init__(self):
        self.entries = {}; self.reused_blocks = 0; self.refreshed_blocks = 0

    def get(self, layer, v, valid, prefix):
        end = (max(0, prefix) // KV_TILE) * KV_TILE
        old = self.entries.get(layer)
        reuse = bool(end and old is not None and old[0].shape == v[..., :end, :].shape
            and torch.equal(old[0], v[..., :end, :]) and torch.equal(old[1], valid[..., :end]))
        if reuse:
            a = old[2]; self.reused_blocks += end // KV_TILE
        elif end:
            a = value_summaries(v[..., :end, :], valid[..., :end])
            self.entries[layer] = (v[..., :end, :].clone(), valid[..., :end].clone(), a)
            self.refreshed_blocks += end // KV_TILE
        if end < v.shape[-2]:
            b = value_summaries(v[..., end:, :], valid[..., end:])
            self.refreshed_blocks += (v.shape[-2] - end + KV_TILE - 1) // KV_TILE
        if not end: return b
        if end == v.shape[-2]: return a
        result = {}
        for name in a:
            if name == 'ref': continue
            result[name] = torch.cat((a[name], b[name]), dim=-2 if name == 'vectors' else -1)
        result['ref'] = (result['sums_sq'].sum(-1) / result['counts'].sum(-1).clamp_min(1)).sqrt().clamp_min(1e-12)
        return result


def expand_summaries(metadata, repeats):
    return {k: v.repeat_interleave(repeats, 1) for k, v in metadata.items()}


def block_state(scores, valid, values, width=KV_TILE):
    """Sufficient statistics shared by screening and execution for ONE Q tile.

    block_mean is exact within-block softmax-weighted V, unlike vector pooling.
    QK stays in native model precision/order; diagnostics and state are FP32.
    """
    valid = torch.broadcast_to(valid, scores.shape)
    scores = scores.float().masked_fill(~valid, -torch.inf)
    pad = (-scores.shape[-1]) % width
    sb = F.pad(scores, (0, pad), value=-torch.inf).unflatten(-1, (-1, width))
    vb = F.pad(valid, (0, pad)).unflatten(-1, (-1, width))
    vv = F.pad(values.float(), (0, 0, 0, pad)).unflatten(-2, (-1, width))
    count = vb.sum(-1); active = count > 0
    b = sb.amax(-1)
    exp = torch.exp(sb - torch.where(active, b, 0.)[..., None]).masked_fill(~vb, 0.)
    ell = exp.sum(-1)
    logz = torch.where(active, b + ell.clamp_min(1e-30).log(), -torch.inf)
    mean = torch.einsum('bhqtk,bhtkd->bhqtd', exp / ell.clamp_min(1e-30)[..., None], vv)
    total = torch.logsumexp(logz, -1, keepdim=True)
    mass = torch.exp(logz - torch.where(torch.isfinite(total), total, 0.)).masked_fill(~active, 0.)
    return dict(b=b, logz=logz, count=count, active=active, eligible=active.any(-2),
        block_mean=mean, mass=mass, contrib=mass[..., None] * mean, sb=sb, vb=vb)


def dense_previous(state):
    """Stable online state before J, all prior blocks retained (screen only)."""
    if '_previous' in state: return state['_previous']
    z = state['logz']
    previous_z = F.pad(z.logcumsumexp(-1)[..., :-1], (1, 0), value=-torch.inf)
    previous_m = F.pad(state['b'].cummax(-1).values[..., :-1], (1, 0), value=-torch.inf)
    # Do not divide cumulative FINAL-normalized contributions by cumulative
    # final mass: a large future logit can underflow both, corrupting past O.
    running = torch.zeros_like(state['block_mean'][..., 0, :]); outputs = [running]
    for j in range(z.shape[-1]-1):
        newz = torch.logaddexp(previous_z[..., j], z[..., j])
        safez = torch.where(torch.isfinite(newz), newz, 0.)
        a = torch.exp(previous_z[..., j]-safez); b = torch.exp(z[..., j]-safez)
        running = a[..., None]*running + b[..., None]*state['block_mean'][..., j, :]
        outputs.append(running)
    state['_previous'] = previous_m, previous_z, torch.stack(outputs,-2)
    return state['_previous']


def log_risk(config, b, n, m, logz, output, meta, j=None, aligned=None):
    """Log-domain criteria. m/logz/output always refer to BEFORE current tile."""
    pool = meta[config.pooling] if j is None else meta[config.pooling][..., j]
    ref = meta['ref'][..., None] if j is None else meta['ref']
    ratio = (pool / ref).clamp_min(1e-30).log().unsqueeze(-2 if j is None else -1)
    gap = b - m
    bound_z = b + n.clamp_min(1).float().log()
    alpha = bound_z - torch.logaddexp(logz, bound_z)
    method = config.method
    if method == 'blasst': risk = gap
    elif method == 'value': risk = gap + ratio
    elif method == 'mass': risk = alpha
    elif method == 'mass_value': risk = alpha + ratio
    elif method == 'risk': risk = alpha + F.softplus(ratio)
    elif method == 'aligned': risk = aligned - m
    elif method == 'centered':
        mean = meta['vectors'] if j is None else meta['vectors'][..., j, :]
        radius = meta['radius'] if j is None else meta['radius'][..., j]
        deviation = (mean.unsqueeze(-3 if j is None else -2) - output).norm(dim=-1)
        deviation = deviation + radius.unsqueeze(-2 if j is None else -1)
        # Dimensionless threshold: divide the proposed deletion-risk norm by ref.
        risk = alpha + (deviation / ref.unsqueeze(-2 if j is None else -1)).clamp_min(1e-30).log()
    elif method == 'compensate':
        radius = meta['radius'] if j is None else meta['radius'][..., j]
        risk = alpha + (radius / ref).clamp_min(1e-30).log().unsqueeze(-2 if j is None else -1)
    elif method == 'zero_pv': risk = alpha + ratio
    else: raise ValueError(method)
    # Before any retained support, keep the block. Important even when V=0.
    return torch.where(n > 0, torch.where(torch.isfinite(logz), risk, torch.inf), -torch.inf)


def aligned_logits(state, meta):
    norms = meta['token_norms']; pad = (-norms.shape[-1]) % KV_TILE
    lognorm = F.pad((norms / meta['ref'][..., None]).clamp_min(1e-30).log(), (0, pad), value=-torch.inf)
    return (state['sb'] + lognorm.unflatten(-1, (-1, KV_TILE))[..., None, :, :]).amax(-1)


def aggregate_risk(risk, active, config):
    worst = risk.masked_fill(~active, -torch.inf).amax(-2)
    if config.aggregation == 'all_rows': return worst
    if config.aggregation != 'mean_safeguard': raise ValueError(config.aggregation)
    # Mean of nonnegative row risks, with worst row <= safeguard * threshold.
    mean = torch.logsumexp(risk.masked_fill(~active, -torch.inf), -2) - active.sum(-2).clamp_min(1).log()
    return torch.maximum(mean, worst - math.log(config.safeguard_factor))


def screen_risks(state, meta, config):
    m, z, o = dense_previous(state)
    aligned = aligned_logits(state, meta) if config.method == 'aligned' else None
    risk = log_risk(config, state['b'], state['count'], m, z, o, meta, aligned=aligned)
    return aggregate_risk(risk, state['active'], config), risk


def streaming_mask(state, meta, config):
    """Sequential, exact-state emulation; no current dense output in routing.

    Returns replaced physical tiles (deletion OR PV replacement by method).
    The actual model operator is applied separately to avoid precision drift
    on retained tiles. Running output is needed only for centered routing.
    """
    b, z, n = state['b'], state['logz'], state['count']
    m = torch.full_like(b[..., 0], -torch.inf); logz = m.clone(); seen_m = m.clone()
    out = torch.zeros_like(state['block_mean'][..., 0, :])
    masks = []; aligned = aligned_logits(state, meta) if config.method == 'aligned' else None
    for j in range(b.shape[-1]):
        previous = seen_m if config.method == 'blasst' else m
        r = log_risk(config, b[..., j], n[..., j], previous, logz, out, meta, j,
            None if aligned is None else aligned[..., j])
        score = aggregate_risk(r[..., None], state['active'][..., j, None], config).squeeze(-1)
        skip = state['eligible'][..., j] & (score < config.log_threshold)
        masks.append(skip)
        seen_m = torch.maximum(seen_m, b[..., j])
        include = state['active'][..., j] & (~skip[..., None] if config.method not in ('compensate', 'zero_pv') else True)
        addz = z[..., j].masked_fill(~include, -torch.inf)
        newz = torch.logaddexp(logz, addz)
        safez = torch.where(torch.isfinite(newz), newz, 0.)
        a, c = torch.exp(logz - safez), torch.exp(addz - safez)
        block_output = state['block_mean'][..., j, :]
        if config.method == 'compensate':
            block_output = torch.where(skip[..., None, None], meta['vectors'][..., j, None, :], block_output)
        elif config.method == 'zero_pv':
            block_output = block_output.masked_fill(skip[..., None, None], 0.)
        if config.method == 'centered': out = a[..., None] * out + c[..., None] * block_output
        logz = newz; m = torch.maximum(m, b[..., j].masked_fill(~include, -torch.inf))
    return torch.stack(masks, -1)


def rescue(keep, state):
    empty = state['active'].any(-1) & ~(state['active'] & keep[..., None, :]).any(-1)
    restore = F.one_hot(state['mass'].argmax(-1), keep.shape[-1]).bool() & empty[..., None]
    return keep | restore.any(-2), empty


def reference_output(scores, valid, values, replaced, config, means=None, position_mask=None):
    """Explicit reference for deletion, denominator-preserving zero/mean PV.

    Softmax is FP32 then cast to model dtype, same as matched dense eager.
    Compensation adds exact block probability mass times its vector mean;
    it never changes the softmax denominator or scales raw QK by value norms.
    """
    positions = (replaced.repeat_interleave(KV_TILE, -1)[..., :scores.shape[-1]][..., None, :]
        if position_mask is None else position_mask)
    compensation = config.method in ('compensate', 'zero_pv')
    allowed = valid if compensation else valid & ~positions
    has = allowed.any(-1, keepdim=True)
    probs = torch.softmax(torch.where(has, scores.float().masked_fill(~allowed, -torch.inf), 0.), -1)
    probs = probs.masked_fill(~allowed, 0.)
    kept_probs = probs.masked_fill(positions, 0.) if compensation else probs
    output = kept_probs.to(values.dtype) @ values
    if config.method == 'compensate':
        pad = (-scores.shape[-1]) % KV_TILE
        mass = F.pad(probs, (0, pad)).unflatten(-1, (-1, KV_TILE)).sum(-1)
        changed = replaced[..., None, :] if position_mask is None else position_mask[..., ::KV_TILE]
        add = torch.einsum('bhqt,bhtd->bhqd', mass * changed, means)
        output = (output.float() + add).to(values.dtype)
    return output


def diagnostics(state, replaced, config, meta):
    keep = state['eligible'] & ~replaced
    mass = (state['mass'] * keep[..., None, :]).sum(-1)
    dense = state['contrib'].sum(-2)
    numerator = (state['contrib'] * keep[..., None, :, None]).sum(-2)
    if config.method == 'compensate':
        numerator += torch.einsum('bhqt,bhtd->bhqd', state['mass'] * replaced[..., None, :], meta['vectors'])
    out = numerator if config.method in ('compensate', 'zero_pv') else numerator / mass.clamp_min(1e-30)[..., None]
    has = state['active'].any(-1)
    return dict(mass_sum=(mass * has).sum(-1), rows=has.sum(-1),
        error_sq=((out-dense)*has[..., None]).square().sum((-1,-2)), dense_sq=dense.square().sum((-1,-2)))


def sol_proxy(query, keys, valid, metadata, scale, value_aware=False):
    """Mean-Q times mean-K per physical block, one prefix+canvas population."""
    qvalid = valid.any(-1)
    kvvalid = valid.any(-2)
    qm = (query.float() * qvalid[..., None]).sum(-2) / qvalid.sum(-1).clamp_min(1)[..., None]
    pad = (-keys.shape[-2]) % KV_TILE
    k = F.pad(keys.float(), (0, 0, 0, pad)).unflatten(-2, (-1, KV_TILE))
    v = F.pad(kvvalid, (0, pad)).unflatten(-1, (-1, KV_TILE))
    km = (k * v[..., None]).sum(-2) / v.sum(-1).clamp_min(1)[..., None]
    proxy = torch.einsum('bhd,bhtd->bht', qm, km) * scale
    if value_aware: proxy = proxy + metadata['rms'].clamp_min(1e-12).log()
    return proxy


def proxy_keep(proxy, eligible, config):
    n = eligible.sum(-1, keepdim=True)
    mean = proxy.masked_fill(~eligible, 0.).sum(-1, keepdim=True) / n.clamp_min(1)
    sd = ((proxy-mean).square().masked_fill(~eligible, 0.).sum(-1, keepdim=True) / n.clamp_min(1)).sqrt()
    z = (proxy-mean) / sd.clamp_min(1e-6)
    if config.mode == 'gaussian':
        keep = eligible & ((z >= BETAS[config.amount]) | (n < 2) | (sd < 1e-6))
    elif config.mode == 'topk': keep = select(proxy, eligible, 'topk', config.amount)
    elif config.mode == 'topp':
        # Explicit nonnegative weights: exp(proxy - eligible row maximum).
        weights = torch.exp(proxy - proxy.masked_fill(~eligible, -torch.inf).amax(-1, keepdim=True))
        keep = select(weights, eligible, 'topp', config.amount)
    else: raise ValueError(config.mode)
    return keep, z, ((n < 2) | (sd < 1e-6)).squeeze(-1)
