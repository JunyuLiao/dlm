"""Independent fixed-budget value/reuse operators, initially diagnostic-only.

One unit is one query head and one query tile. Prefix blocks are eligible;
boundary/canvas tokens are mandatory. All moments use a common, explicit
log-normalizer. No prior selector is imported. No sampler or native path is
modified by this module.
"""
from dataclasses import dataclass

import torch


@dataclass
class Moments:
    mass: torch.Tensor                 # [prefix blocks, rows]
    numerator: torch.Tensor            # [prefix blocks, rows, full V dim]
    tail_mass: torch.Tensor            # [rows]
    tail_numerator: torch.Tensor       # [rows, full V dim]
    log_scale: torch.Tensor            # [rows], full-support log Z
    dm: torch.Tensor | None = None     # unnormalized moment derivative / Z
    dc: torch.Tensor | None = None
    tail_dm: torch.Tensor | None = None
    tail_dc: torch.Tensor | None = None
    value_rms: torch.Tensor | None = None

    def selected(self, mask):
        mass = self.tail_mass + self.mass[mask].sum(0)
        num = self.tail_numerator + self.numerator[mask].sum(0)
        return num, mass

    def output(self, mask):
        num, mass = self.selected(mask)
        return num / mass.clamp_min(torch.finfo(num.dtype).tiny)[..., None]

    def tangent(self, mask):
        if self.dm is None:
            raise ValueError('no causal query direction was supplied')
        num, mass = self.selected(mask)
        dm = self.tail_dm + self.dm[mask].sum(0)
        dc = self.tail_dc + self.dc[mask].sum(0)
        return (dc - (num / mass[..., None]) * dm[..., None]) / mass[..., None]

    def dense(self):
        return self.output(torch.ones(len(self.mass), dtype=torch.bool, device=self.mass.device))


@torch.inference_mode()
def moments(q, k, v, prefix, scale, direction=None, block=64, chunk=128):
    """Exact FP32/64 moments using batched block GEMMs; bounded working memory.

    q=[Q,D], k/v=[KV,D]. Prefix floor(prefix/block) blocks are selectable.
    The partial prefix boundary and the canvas are included in the mandatory
    tail. Directional moments differentiate unnormalized exp logits while
    holding the anchor normalization constant, so quotient-rule tangents are
    exact. Autograd is not used and no future query is accepted implicitly.
    """
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2 or k.shape != v.shape:
        raise ValueError('expected one head with matching K/V [tokens, dim]')
    if q.shape[1] != k.shape[1] or not 0 <= prefix <= len(k):
        raise ValueError('invalid attention geometry')
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    scores = q @ k.T * scale
    logz = scores.logsumexp(-1)
    p = (scores - logz[:, None]).exp()
    dl = direction.to(dtype) @ k.T * scale if direction is not None else None
    pt = prefix // block
    # Block moments are materialized only for this head/query tile, not all
    # 16 heads together. Chunked GEMMs amortize dispatch across candidate tiles.
    m = p[:, :pt * block].reshape(len(q), pt, block).sum(-1).T.contiguous()
    c = torch.empty((pt, len(q), v.shape[1]), dtype=dtype, device=q.device)
    dm, dc = (torch.empty_like(m), torch.empty_like(c)) if dl is not None else (None, None)
    for lo in range(0, pt, chunk):
        hi = min(pt, lo + chunk)
        weights = p[:, lo * block:hi * block].reshape(len(q), hi - lo, block).permute(1, 0, 2)
        values = v[lo * block:hi * block].reshape(hi - lo, block, -1)
        c[lo:hi] = weights @ values
        if dl is not None:
            dw = weights * dl[:, lo * block:hi * block].reshape(len(q), hi - lo, block).permute(1, 0, 2)
            dm[lo:hi] = dw.sum(-1)
            dc[lo:hi] = dw @ values
    tp = p[:, pt * block:]
    tm, tc = tp.sum(-1), tp @ v[pt * block:]
    tdm = tdc = None
    if dl is not None:
        tdp = tp * dl[:, pt * block:]
        tdm, tdc = tdp.sum(-1), tdp @ v[pt * block:]
    return Moments(m, c, tm, tc, logz, dm, dc, tdm, tdc, v.square().sum(-1).mean().sqrt())


def mass_score(m):
    # V31's qblock_max uses prefix-normalized row mass, excluding the tail.
    return (m.mass / m.mass.sum(0).clamp_min(1e-30)[None]).amax(-1)


def fixed_topk(score, k):
    if k < 0:
        raise ValueError('negative budget')
    result = torch.zeros_like(score, dtype=torch.bool)
    result[torch.argsort(score, descending=True, stable=True)[:min(k, len(score))]] = True
    return result


def mass_neighborhood(m, anchor, extra=32):
    return anchor | fixed_topk(mass_score(m), int(anchor.sum()) + extra)


def c_gate(previous_accepted, previous_changed, previous_reversed):
    """Causal row weights; must be supplied from the preceding sampler call."""
    return (1 + (~previous_accepted).float() + previous_changed.float()
            + previous_reversed.float()).clamp(1, 4)


def squared_rows(error, gram=None):
    # gram = W_O(head)^T W_O(head); exact single-head post-O squared norm.
    if gram is None:
        return error.square().sum(-1)
    return ((error @ gram) * error).sum(-1).clamp_min(0)


def objective(output, target, weight, gram=None):
    return (squared_rows(output - target, gram) * weight).mean(-1)


@torch.inference_mode()
def online_risks(m, threshold=.021):
    """Full-dimensional V2 preserve-mass diagnostic, in physical key order.

    Return proposal risks only, not a variable-budget consumer mask. The full
    denominator advances on every block; the running output only advances on
    accepted blocks. Normalization uses the exact full-support RMS V norm.
    This is a full-V proposal variant; inherited V2 uses a rank-32 sketch.
    """
    if m.value_rms is None:
        raise ValueError('V RMS is required for the full-V online risk')
    running = torch.zeros_like(m.tail_numerator)
    denominator = torch.zeros_like(m.tail_mass)
    risks = []
    for j in range(len(m.mass)):
        denominator = denominator + m.mass[j]
        eta = m.mass[j] / denominator.clamp_min(1e-30)
        mu = m.numerator[j] / m.mass[j].clamp_min(1e-30)[:, None]
        risk = (eta * (mu - running).norm(dim=-1) / m.value_rms.clamp_min(1e-30)).max()
        take = (risk >= threshold) | (j == 0)
        running = torch.where(take, running + eta[:, None] * (mu - running), running)
        risks.append(risk)
    return torch.stack(risks) if risks else m.mass.new_empty(0)


def online_prefix_risks(m):
    """Causal running full-prefix output, *not* V2's thresholded accumulator.

    Both input moments and returned scores are diagnostic. No final-output
    information enters a block's running reference. True final-output error
    remains an optional admission guard at a dense observation.
    """
    running_mass = m.mass.cumsum(0)
    running_num = m.numerator.cumsum(0)
    previous_mass = running_mass - m.mass
    previous_num = running_num - m.numerator
    previous = previous_num / previous_mass.clamp_min(1e-30)[..., None]
    mu = m.numerator / m.mass.clamp_min(1e-30)[..., None]
    eta = m.mass / running_mass.clamp_min(1e-30)
    return (eta * (mu - previous).norm(dim=-1)).amax(-1)


@torch.inference_mode()
def bounded_proposal(m, anchor, *, swaps=2, extra=32, shortlist=16,
                     gram=None, weight=None, lifetime=False, online=False,
                     mass_guard=.025, max_error_ratio=1.02):
    """Rank bounded independent swaps, then accept a combined proposal.

    All pair scores are computed at the unchanged anchor; no exact greedy
    reoptimization occurs. Disjoint ranked pairs form one proposal. Back off
    its batch size until the combined signed-error/mass guards pass. K stays
    identical, including no-op and undersubscribed-neighborhood cases.
    """
    k = int(anchor.sum())
    if k == 0 or k == len(anchor) or swaps == 0:
        return anchor.clone(), dict(swaps=0, pairs=0, reason='no_candidates')
    weight = torch.ones_like(m.tail_mass) if weight is None else weight
    if weight.shape != m.tail_mass.shape or bool((weight < 0).any()) or not bool(torch.isfinite(weight).all()):
        raise ValueError('invalid per-row weights')
    score = mass_score(m)
    pool = mass_neighborhood(m, anchor, extra)
    outside = torch.where(pool & ~anchor)[0]
    remove = torch.where(anchor)[0]
    remove = remove[torch.argsort(score[remove], stable=True)[:shortlist]]
    priority = online_risks(m) if online else score
    outside = outside[torch.argsort(priority[outside], descending=True, stable=True)[:shortlist]]
    if len(outside) == 0:
        return anchor.clone(), dict(swaps=0, pairs=0, reason='no_candidates')
    num, den = m.selected(anchor)
    base = num / den[..., None]
    target = m.dense()
    reference_loss = objective(base, target, weight, gram)
    reference_max = squared_rows(base - target, gram).max()
    target_next = base_next = None
    if lifetime:
        allmask = torch.ones_like(anchor)
        target_next = target + m.tangent(allmask)
        base_next = base + m.tangent(anchor)
        reference_loss = reference_loss + objective(base_next, target_next, weight, gram)
    pairs = torch.cartesian_prod(outside, remove)
    losses = []
    # Bounded batches prevent a 16x16x128x512 temporary from multiplying with
    # derivative and projection buffers. GEMMs replace per-pair Python work.
    for batch in pairs.split(16):
        add, rem = batch[:, 0], batch[:, 1]
        pd = den + m.mass[add] - m.mass[rem]
        pn = num + m.numerator[add] - m.numerator[rem]
        po = pn / pd[..., None].clamp_min(1e-30)
        loss = objective(po, target, weight, gram)
        if lifetime:
            dm = m.tail_dm + m.dm[anchor].sum(0) + m.dm[add] - m.dm[rem]
            dc = m.tail_dc + m.dc[anchor].sum(0) + m.dc[add] - m.dc[rem]
            tangent = (dc - po * dm[..., None]) / pd[..., None]
            loss = loss + objective(po + tangent, target_next, weight, gram)
        losses.append(loss)
    losses = torch.cat(losses)
    # Only bounded proposal indices/scores cross to the host in this reference
    # implementation. A native promotion requires tensorized selection.
    order = losses.argsort(stable=True).tolist()
    pairs_cpu, loss_cpu = pairs.tolist(), losses.tolist()
    chosen, used_add, used_rem = [], set(), set()
    for i in order:
        a, r = pairs_cpu[i]
        if loss_cpu[i] >= float(reference_loss):
            break
        if a not in used_add and r not in used_rem:
            chosen.append((a, r)); used_add.add(a); used_rem.add(r)
        if len(chosen) >= swaps:
            break
    for count in range(len(chosen), 0, -1):
        candidate = anchor.clone()
        for a, r in chosen[:count]:
            candidate[a], candidate[r] = True, False
        out = m.output(candidate)
        loss = objective(out, target, weight, gram)
        if lifetime:
            loss = loss + objective(out + m.tangent(candidate), target_next, weight, gram)
        _, newden = m.selected(candidate)
        good = (loss < reference_loss - 1e-8
                and objective(out, target, weight, gram) <= objective(base, target, weight, gram) + 1e-6
                and squared_rows(out - target, gram).max() <= reference_max * max_error_ratio**2 + 1e-6
                and (den - newden).max() <= mass_guard + 1e-6)
        if good:
            assert int(candidate.sum()) == k
            return candidate, dict(swaps=count, pairs=len(pairs), reason='accepted')
    return anchor.clone(), dict(swaps=0, pairs=len(pairs), reason='joint_guard_or_no_gain')


@dataclass
class OmittedCache:
    mask: torch.Tensor
    log_scale: torch.Tensor
    mass: torch.Tensor
    numerator: torch.Tensor
    residual: torch.Tensor
    dm: torch.Tensor | None
    dc: torch.Tensor | None

    @classmethod
    def build(cls, anchor, mask):
        # Mandatory tail is always in the selected side, never cached omitted.
        return cls(mask.clone(), anchor.log_scale.clone(), anchor.mass[~mask].sum(0),
                   anchor.numerator[~mask].sum(0), anchor.dense() - anchor.output(mask),
                   anchor.dm[~mask].sum(0) if anchor.dm is not None else None,
                   anchor.dc[~mask].sum(0) if anchor.dc is not None else None)

    def reconstruct(self, current, mask, kind='moments', steps=0):
        if not torch.equal(mask, self.mask):
            raise ValueError('changed support requires cache reset')
        selected = current.output(mask)
        if kind == 'residual':
            return selected + self.residual
        if kind not in ('moments', 'tangent'):
            raise ValueError('unknown correction')
        old_mass, old_num = self.mass, self.numerator
        if kind == 'tangent':
            if self.dm is None or steps < 0:
                raise ValueError('missing causal derivative or invalid age')
            old_mass = old_mass + steps * self.dm
            old_num = old_num + steps * self.dc
            # First-order unnormalized mass can become negative. Fall back to
            # cached positive moments per row; never silently clamp a bad mean.
            valid = (old_mass > 0) & torch.isfinite(old_num).all(-1)
            old_mass = torch.where(valid, old_mass, self.mass)
            old_num = torch.where(valid[:, None], old_num, self.numerator)
        num, mass = current.selected(mask)
        common = torch.maximum(current.log_scale, self.log_scale)
        a, b = (current.log_scale - common).exp(), (self.log_scale - common).exp()
        return (a[:, None] * num + b[:, None] * old_num) / (a * mass + b * old_mass)[:, None]
