"""Value-direction-aware block selectors for the v31 cross-step reuse pipeline (this study's variants).

PROVENANCE. The first cross-step-reuse implementation, and the online projected-V routing state in
``integration.py`` / ``reference.py``, are Yuhan's inherited work. Gaussian32 / value-direction-aware
routing and the query-sensitivity (C_gate) protection are Junyu's earlier method families. Every
selector in this file is a NEW selector authored by this study and evaluated INSIDE the unchanged
inherited reuse pipeline. Nothing here is attributed to either earlier author.

SCOPE. Block selection only. At an initial/refresh observation call the caller already owns

  * ``z[H, PT, N]``  -- per-(query row, 64-key physical KV tile) tile log-mass
    ``z = logsumexp_u s_iu``, ``s_iu`` the native scaled attention logit after the structural mask,
    written by the FA4 in-kernel observation (``v31_fa4_observe.ObservingMask``);
  * ``mu[H, PT, N, R]`` -- attention-weighted within-tile projected mean
    ``mu_ij = (sum_{u in j} exp(s_iu) z_u) / (sum_{u in j} exp(s_iu))`` with ``z_u = v_u R``,
    ``R in R^{d_v x 32}``, ``R_ab ~ N(0, 1/32)``, the repository's frozen Gaussian32 bank
    (seed 1729, per native KV head, recorded matrix hash);
  * ``nu[H, N]`` -- the current valid-KV reference scale (RMS of the valid V norms), broadcast from
    the native KV head exactly as the inherited online router's ``ref``;
  * the canvas/boundary tiles ``j >= PT``, which are always kept and are part of the V3 full support.

The return value is only a keep map in the same ``[1, H, QB, KT]`` bool form the unchanged FA4
sparse consumer already consumes. The observation call's attention output is FA4's own native
current-step BF16 output, unchanged: this module never substitutes a projected, cached or
reshaped output for it.

LAYOUT. ``N = QB * 128`` is the padded query-row axis of the map's query blocks; ``rows[H, N]`` marks
the rows that exist (a partial last block). All tile-major so that one selection unit -- the active
control's unit, one (query head, 128-row query block) -- is a contiguous ``[N/128, PT]`` window.

MATHEMATICS. ``Agg_i`` is the maximum over the valid rows of the unit (the aggregation of the active
``qblock_max`` control, preserved across every arm of this study and never tuned per method);
``eps = 1e-12``; ``nu_i = max(nu, eps)``.

  stable tile mass and within-tile projected mean
      Z_ij = sum_{u in j} exp(s_iu)          mu_ij = (sum_{u in j} exp(s_iu) z_u) / Z_ij

  V1 / V2 (faithful online selectors; identical ordering, aggregation, quota and budget).
  Scan optional tiles in ascending physical tile order, keeping an internal normalized projected
  output ``o_hat_i`` and a running mass ``Z_i``:
      eta_ij = Z_ij / (Z_i + Z_ij)
      rho_j  = Agg_i( eta_ij ||mu_ij - o_hat_i||_2 / nu_i )            (higher favours retention)
      V1 retained:  o_hat_i <- (1 - eta_ij) o_hat_i + eta_ij mu_ij,  Z_i <- Z_i + Z_ij
      V1 skipped:   both unchanged  (retained-support model only)
      V2 retained:  same update;    V2 skipped: Z_i <- Z_i + Z_ij, o_hat_i UNCHANGED
  V2's denominator is advanced consistently. The implementation stores the UNNORMALIZED numerator
  ``P_i`` and advances ``P_i <- P_i + Z_ij mu_ij`` with ``Z_i <- Z_i + Z_ij`` on a skip, so
  ``o_hat_i = P_i / Z_i`` is unchanged. Advancing only the denominator of a fixed numerator would be
  wrong and is not done. V2 is a routing surrogate only: it does not make the later masked attention
  operator mass-preserving.

  V3 (full-support selectors matching masked attention in sketch space).
  Over the complete eligible support (prefix tiles AND the always-kept canvas/boundary tiles):
      alpha_ij = Z_ij / sum_l Z_il        c_ij = alpha_ij mu_ij        O_i = sum_j c_ij
      O_i(S)   = (sum_{j in S} c_ij) / (sum_{j in S} alpha_ij)
      F(S)     = Agg_i( ||O_i(S) - O_i||_2 / nu_i )
  V3a (singleton deletion ranking, scores computed ONCE on the full support):
      d_j = Agg_i( alpha_ij ||mu_ij - O_i||_2 / ((1 - alpha_ij) nu_i) )
  V3b (cumulative backward greedy pruning from the full support):
      g_ij = c_ij - alpha_ij O_i,  A_i(S) = sum_{j in S} alpha_ij,  r_i(S) = sum_{j in S} g_ij
      G_j(S) = Agg_i( ||r_i(S) - g_ij||_2 / ([A_i(S) - alpha_ij] nu_i) )
  V3b is a greedy heuristic, not a global subset solver, and it need not improve monotonically.

QUOTA (identical for every arm). Mandatory tiles: the canvas/boundary tiles ``j >= PT`` plus the
control's in-budget protection (sink / recent), all recorded in the receipt. Optional prefix tiles
are scanned in ascending tile order under the prescribed causal quota: reserve ``k`` optional slots,
apply the frozen threshold while optional slots remain, force skip when no optional slot remains,
force keep when all remaining blocks are needed to fill the quota, and force keep the first valid
support so a previous output never approximates a block that has no support yet. Ties: ``>=`` against
the frozen threshold keeps a boundary tie (never a silent strict inequality) and the greedy argmin
takes the LOWEST tile index.

APPROXIMATIONS ARE NAMED, NEVER ALIASED. ``v3b`` is the exact backward greedy over every optional
candidate. ``v3b_drop`` is a bounded batch drop-and-refine (every pass drops the worst
``ceil(fraction * remaining)``), ``v3b_shortlist`` runs the exact greedy on a V3a-ranked shortlist;
both are approximations, always reported under their own names with their candidate counts and pass
counts, and their quality loss is measured against exact ``v3b`` wherever both are runnable.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import torch

RANK = 32
PROJECTION_SEED = 1729
EPS = 1e-12
VALUE_SELECTORS = ('v1', 'v2', 'v3a', 'v3b', 'v3b_drop', 'v3b_shortlist')
APPROXIMATE_SELECTORS = ('v3b_drop', 'v3b_shortlist')
# Above this optional-tile count the exact backward greedy's candidate-evaluation count
# (O(PT^2 * N * R)) stops being a bounded cost; the caller must then use a named approximation
# or record the arm as blocked. Never silently substituted.
V3B_EXACT_MAX_TILES = 256


# --------------------------------------------------------------------------- projection identity


def gaussian32_bank(layer, kv_heads, width, device, seed=PROJECTION_SEED):
    """The repository's frozen deterministic Gaussian32 bank: per native KV head
    ``R[head] = randn(width, 32)/sqrt(32)`` from a CPU generator seeded by
    ``sha256('jl_output_v1/gaussian/32/<seed>/<layer>/<head>/<width>')``. Returns
    ``(matrix[HK, width, 32] fp32, records)`` with the per-head matrix sha256."""
    from experiments.diffusion_gemma_jl_output_aware.projections import Projections, digest
    get = Projections().get
    matrices = torch.stack([get(layer, kv_heads, width, 'gaussian', RANK, seed, 'cpu')[h]
                            for h in range(kv_heads)])
    records = []
    for head in range(kv_heads):
        material = f'jl_output_v1/gaussian/{RANK}/{seed}/{layer}/{head}/{width}'
        derived = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], 'little') % (2 ** 63 - 1)
        records.append(dict(layer=int(layer), native_kv_head=int(head), value_width=int(width),
                            family='gaussian', rank=RANK, seed=int(seed), derived_seed=derived,
                            sha256=digest(matrices[head])))
    return matrices.to(device), records


def bank_digest(matrix):
    """sha256 of one projection matrix's FP32 bytes (the recorded matrix hash)."""
    return hashlib.sha256(matrix.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def project_values(values, bank, out_dtype=None):
    """``values`` [1, HK, nk, d] -> projected sketch ``[1, HK, nk, 32]`` (FP32 multiply)."""
    if values.ndim != 4 or values.shape[1] != bank.shape[0] or values.shape[-1] != bank.shape[1]:
        raise ValueError('project_values expects [1, HK, nk, d] matching the bank')
    x = values[0].float() if values.dtype in (torch.bfloat16, torch.float16) else values[0]
    z = torch.matmul(x, bank.to(x.dtype))
    return (z if out_dtype is None else z.to(out_dtype)).unsqueeze(0)


def valid_kv_reference(values, valid):
    """The inherited valid-KV reference scale: per native KV head the RMS of the valid V norms,
    ``sqrt(sum_u ||v_u||^2 / count_valid)``, clamped to ``>= 1e-12``. Identical definition to the
    inherited online router's ``ref``."""
    x = values[0].float() if values.dtype in (torch.bfloat16, torch.float16) else values[0]
    norm_sq = x.norm(dim=-1).square()
    mask = valid[0].to(norm_sq.dtype)
    count = mask.sum(-1).clamp_min(1.0)
    return ((norm_sq * mask).sum(-1) / count).sqrt().clamp_min(EPS)


# --------------------------------------------------------------------------- statistics container


@dataclass
class ValueStats:
    """Tile-major compact statistics of one observation call.

    ``z``    [H, PT, N]      FP32 tile log-mass (``-inf`` where the row/tile has no legal key)
    ``mu``   [H, PT, N, R]   FP32 attention-weighted within-tile projected mean (zeros if absent)
    ``nu``   [H, N]          FP32 per-row reference scale (already broadcast from its KV head)
    ``rows`` [H, N]          bool: the query rows that exist
    ``n``                    the real row count (``N = QB * 128 >= n``)
    ``kv_heads``, ``group``  GQA geometry
    ``prefix_tiles`` ``total_tiles``  optional / all physical KV64 tiles
    """
    z: torch.Tensor
    mu: torch.Tensor | None
    nu: torch.Tensor
    rows: torch.Tensor
    n: int
    kv_heads: int
    group: int
    prefix_tiles: int
    total_tiles: int

    def __post_init__(self):
        h, pt, n_pad = self.z.shape
        if n_pad % 128:
            raise ValueError('the padded row axis must be a multiple of 128')
        if self.n > n_pad:
            raise ValueError('real row count exceeds the padded axis')
        if self.rows.shape != (h, n_pad) or self.nu.shape != (h, n_pad):
            raise ValueError('rows/nu must be [H, N]')
        if self.mu is not None and self.mu.shape != (h, pt, n_pad, RANK):
            raise ValueError('mu must be [H, PT, N, 32]')
        self.heads, self.blocks = h, n_pad // 128
        keep = self.rows[:, None, :]
        self.z = torch.where(self.rows[:, None, :], self.z, torch.full_like(self.z, -math.inf))
        self.nu = self.nu.clamp_min(EPS)
        if self.mu is not None:
            # Mask in place. torch.where here would allocate a zeros_like AND a result of the
            # full [H, PT, N, 32] size, which at a 120k-token context is tens of GiB and OOMs the
            # engine. Multiplying by the 0/1 mask is the same value and allocates nothing.
            # Skipped entirely when every row is real, the usual case.
            if not bool(self.rows.all()):
                self.mu.mul_(self.rows[:, None, :, None])

    def finite_mass(self):
        """``Z_ij`` scaled by each row's largest tile log-mass. A padded row (all ``-inf``) and a
        wholly masked tile (all ``-inf``) aggregate to zero mass, never NaN."""
        z = self.z                                                     # [H, KT, N]
        mass = torch.exp(z - z.amax(1, keepdim=True))                   # scaled per row over TILES
        return torch.where(torch.isfinite(z), mass, torch.zeros_like(mass))

    def summary_bytes(self):
        total = self.z.numel() * self.z.element_size() + self.nu.numel() * self.nu.element_size()
        if self.mu is not None:
            total += self.mu.numel() * self.mu.element_size()
        return int(total)


def block_max(per_row, rows, blocks=None):
    """``Agg_i`` = maximum over the valid rows of each 128-row query block.

    ``per_row`` [H, X, N] -> [H, X, blocks]. Non-existent rows hold ``-inf`` and can never win."""
    h, x, n_pad = per_row.shape
    blk = blocks or n_pad // 128
    out = per_row.reshape(h, x, blk, 128)
    mask = rows.reshape(h, 1, blk, 128).expand(h, x, blk, 128)
    return torch.where(mask, out, torch.full_like(out, -math.inf)).amax(-1)


def block_reduce(per_row, rows, fn, blocks=None):
    """``Agg_i`` with an arbitrary max/sum over the valid rows of each 128-row block."""
    return block_max(per_row, rows, blocks) if fn == 'max' else _block_sum(per_row, rows, blocks)


def _block_sum(per_row, rows, blocks=None):
    h, x, n_pad = per_row.shape
    blk = blocks or n_pad // 128
    out = per_row.reshape(h, x, blk, 128)
    mask = rows.reshape(h, 1, blk, 128).expand(h, x, blk, 128)
    return torch.where(mask, out, torch.zeros_like(out)).sum(-1)


# --------------------------------------------------------------------------- counters


@dataclass
class Counters:
    candidates: int = 0
    evaluations: int = 0
    passes: int = 0
    forced_keep: int = 0
    forced_skip: int = 0
    threshold_keeps: int = 0
    exact: bool = True
    extra: dict = field(default_factory=dict)

    def as_dict(self):
        return dict(candidates=int(self.candidates), evaluations=int(self.evaluations),
                    passes=int(self.passes), forced_keep=int(self.forced_keep),
                    forced_skip=int(self.forced_skip), threshold_keeps=int(self.threshold_keeps),
                    exact=bool(self.exact), **self.extra)


def protect_map(prefix_tiles, total_tiles, sink, recent, heads, blocks, device):
    """The control's in-budget protection as a ``[H, blocks, total_tiles]`` bool: the sink tiles at
    the front of the prefix and the most recent prefix tiles. ``sink``/``recent`` are in TOKENS."""
    prot = torch.zeros((heads, blocks, total_tiles), dtype=torch.bool, device=device)
    if prefix_tiles <= 0:
        return prot
    if sink:
        prot[..., :min(int(sink) // 64, prefix_tiles)] = True
    if recent:
        prot[..., max(0, prefix_tiles - int(recent) // 64):prefix_tiles] = True
    return prot


# --------------------------------------------------------------------------- V1 / V2 (online scan)


def scan_online(stats, budget, threshold, keep_skipped_mass, counters=None, protect=None,
                return_state=False):
    """V1 (``keep_skipped_mass=False``) / V2 (``True``): the faithful online sequential selectors.

    The loop length is the optional tile count; every head, unit and row is decided inside one
    batched tensor step, and no step reads a value back to the host. This is the trusted CPU/GPU
    reference; the Triton kernel in ``v31_value_scan.py`` must reproduce it exactly."""
    z, mu, rows = stats.z, stats.mu, stats.rows
    h, pt, n_pad = z.shape
    blocks = stats.blocks
    dev = z.device
    ptiles = min(int(budget), pt)
    if protect is not None:
        # the control gives protected tiles an +inf score and then tops up to k, so they are
        # mandatory AND charged against the optional budget
        ptiles = max(0, ptiles - int(protect[:, :, :pt].sum(-1).sum()))
    keep = torch.zeros((h, blocks, stats.total_tiles), dtype=torch.bool, device=dev)
    keep[..., pt:] = True
    if protect is not None:
        keep |= protect
    mass = stats.finite_mass()                                          # [H, PT, N]
    acc = mass.dtype if mu is None else torch.promote_types(mass.dtype, mu.dtype)
    running = torch.zeros((h, 1, n_pad), device=dev, dtype=acc)
    numerator = (torch.zeros((h, 1, n_pad, RANK), device=dev, dtype=acc) if mu is not None else None)
    # A unit has support as soon as it has retained any tile: every row of a unit shares its decision.
    has_support = torch.zeros((h, blocks), device=dev, dtype=torch.bool)
    optional = torch.full((h, blocks), ptiles, device=dev, dtype=torch.int64)
    counters = counters or Counters()
    counters.candidates += int(pt)
    nu = stats.nu[:, None, :]
    prot = protect
    for j in range(pt):
        remaining = pt - j
        m = mass[:, j:j + 1, :]                                         # [H, 1, N]
        eta = torch.where(running + m > 0, m / (running + m).clamp_min(EPS), torch.zeros_like(m))
        if mu is not None:
            est = numerator / running.clamp_min(EPS)[..., None]
            term = eta * (mu[:, j:j + 1, :, :] - est).norm(dim=-1) / nu
        else:
            term = eta
        rho = block_max(term, rows)                                     # [H, 1, blocks]
        exhausted = (optional <= 0)[:, None, :]
        must_fill = (optional >= remaining)[:, None, :]
        first = (~has_support)[:, None, :]
        take = torch.where(exhausted, torch.zeros_like(rho, dtype=torch.bool),
                           torch.where(must_fill, torch.ones_like(rho, dtype=torch.bool),
                                       (rho >= threshold) | first))
        forced = take & (exhausted | must_fill | first)
        if prot is not None:
            p = prot[:, :, j:j + 1].transpose(1, 2)
            take, forced = take | p, forced | p
        keep[:, :, j] = take[:, 0]
        keep_rows = take[..., None].expand(h, 1, blocks, 128).reshape(h, 1, n_pad)
        if mu is not None:
            # A RETAINED tile applies the same convex update in both variants. In unnormalized form
            #   P' = Z * o_hat + Z_ij mu_ij   (because (Z + Z_ij)(1 - eta) / Z = 1 with
            #   eta = Z_ij / (Z + Z_ij) and (Z + Z_ij) eta = Z_ij), which is exactly o_hat' below.
            numerator = numerator + (keep_rows.to(acc) * m)[..., None] * mu[:, j:j + 1, :, :]
            if keep_skipped_mass:
                # V2's SKIP advances the denominator AND the numerator by o_hat * Z_ij, so the
                # normalized routing output is EXACTLY unchanged. Advancing only the denominator
                # would be wrong; advancing the numerator by Z_ij mu_ij would make V2's routing
                # state the full-support mean, which is V3's semantics, not V2's.
                skip_rows = (~keep_rows).to(acc)
                numerator = numerator + skip_rows[..., None] * (
                    numerator / running.clamp_min(EPS)[..., None]) * m[..., None]
        running = running + (m if keep_skipped_mass else torch.where(keep_rows, m, torch.zeros_like(m)))
        has_support = has_support | take[:, 0]
        optional = optional - take[:, 0].to(torch.int64)
        counters.forced_keep += int(forced.sum())
        counters.threshold_keeps += int((take & ~forced).sum())
        counters.forced_skip += int((~take).sum())
    counters.passes += 1
    out = keep.reshape(1, h, blocks, stats.total_tiles)
    if return_state:
        # the internal routing state, for the tests that must distinguish it from masked attention
        o_hat = None if numerator is None else (numerator / running.clamp_min(EPS)[..., None])[:, 0]
        return out, counters, dict(running=running[:, 0], numerator=(None if numerator is None
                                                                     else numerator[:, 0]),
                                   o_hat=o_hat)
    return out, counters


# --------------------------------------------------------------------------- V3 full support


def full_support(stats):
    """``alpha``, ``c`` and ``O`` over the complete eligible support (prefix + always-kept tiles).

    The caller concatenates the canvas/boundary tiles onto ``stats.z`` / ``stats.mu``, so the sums
    below are over every tile the observation call attended, whether or not the tile is charged
    against the optional prefix budget."""
    z, mu = stats.z, stats.mu
    h, kt, n_pad = z.shape
    live = torch.isfinite(z)
    shift = z.amax(1, keepdim=True)
    mass = torch.where(live, torch.exp(z - shift), torch.zeros_like(z))
    total = mass.sum(1, keepdim=True).clamp_min(EPS)                     # over TILES
    alpha = mass / total * rows_to_float(stats.rows)
    c = alpha[..., None] * mu
    ref = c.sum(1)                                                      # over TILES
    return alpha, c, ref, live


def rows_to_float(rows):
    return rows[:, None, :].to(torch.float32)


def v3a_select(stats, budget, counters=None, protect=None):
    """Singleton deletion ranking on the full support; the scores are computed ONCE."""
    alpha, c, ref, live = full_support(stats)
    h, kt, n_pad = stats.z.shape
    pt, blocks = stats.prefix_tiles, stats.blocks
    dev = stats.z.device
    keep = torch.zeros((h, blocks, kt), dtype=torch.bool, device=dev)
    keep[..., pt:] = True
    if protect is not None:
        keep |= protect
    counters = counters or Counters()
    counters.candidates += int(pt)
    gap = (stats.mu - ref[:, None, :, :]).norm(dim=-1)                  # [H, KT, N]
    denom = (1.0 - alpha).clamp_min(EPS) * stats.nu[:, None, :]
    d = alpha * gap / denom
    score = block_max(d, stats.rows)                                     # [H, KT, blocks]
    removable = live.any(-1)                                             # [H, KT] some row has support
    score = torch.where(removable[:, :, None], score, torch.full_like(score, -math.inf))
    if protect is not None:
        score = torch.where(protect.transpose(-1, -2), torch.full_like(score, math.inf), score)
    score[..., pt:] = torch.full_like(score[..., pt:], math.inf)         # never removable
    # only the OPTIONAL prefix tiles are ranked; the canvas/boundary tiles are mandatory and are not
    # candidates at all, so they can never enter the top-k
    score = score[:, :pt]
    ptiles = min(int(budget), pt)
    idx = score.transpose(-1, -2).topk(ptiles, dim=-1).indices            # [H, blocks, ptiles]
    chosen = torch.zeros_like(keep[:, :, :pt]).scatter_(-1, idx, True)
    keep[:, :, :pt] |= chosen
    counters.evaluations += int(h * blocks * pt)
    if protect is not None:
        counters.forced_keep += int((keep[:, :, :pt] & protect[:, :, :pt] & ~chosen).sum())
    counters.passes += 1
    return keep.reshape(1, h, blocks, kt), counters, dict(alpha=alpha, c=c, ref=ref)


def v3b_greedy(stats, budget, counters=None, protect=None, candidates=None, exact=True,
               drop_fraction=0.0, shortlist=0):
    """Cumulative backward greedy pruning.

    With the default ``drop_fraction=0`` and no restricted candidate set this IS the exact backward
    greedy over every optional prefix tile, and ``exact=True`` says so. Any non-zero
    ``drop_fraction`` (bounded batch drop-and-refine) or a restricted candidate set (``candidates`` /
    ``shortlist``) is an APPROXIMATION and the caller must pass ``exact=False`` and report the
    candidate count; the counters mark it so an approximation can never be reported as exact."""
    if exact and (drop_fraction or candidates is not None or shortlist):
        raise ValueError('an approximated v3b must be declared exact=False and report its candidates')
    alpha, c, ref, live = full_support(stats)
    h, kt, n_pad = stats.z.shape
    pt, blocks = stats.prefix_tiles, stats.blocks
    dev = stats.z.device
    keep = torch.zeros((h, blocks, kt), dtype=torch.bool, device=dev)
    keep[..., pt:] = True
    if protect is not None:
        keep |= protect
    ptiles = min(int(budget), pt)
    live_set = torch.zeros((h, blocks, pt), dtype=torch.bool, device=dev)
    live_set |= live[:, :pt].any(-1)[:, None, :].expand(h, blocks, pt)
    protected = torch.zeros((h, blocks, pt), dtype=torch.bool, device=dev)
    if protect is not None:
        protected = protect[:, :, :pt].clone()
    keep_prefix = protected.clone()
    counters = counters or Counters(exact=exact)
    if candidates is not None:
        if candidates.shape != live_set.shape:
            raise ValueError('candidate mask must be [heads, blocks, prefix_tiles]')
        keep_prefix |= protected & ~candidates.to(dev)
        live_set &= candidates.to(dev)
    elif shortlist and shortlist > 0:
        # the V3a singleton-deletion ranking over the OPTIONAL prefix tiles only
        score = _deletion_scores(stats, alpha, ref, live)[:, :pt]
        if protect is not None:
            score = torch.where(protect[:, :, :pt].transpose(-1, -2),
                                torch.full_like(score, math.inf), score)
        m = min(pt, max(int(shortlist), ptiles))     # never smaller than the budget itself
        picked = score.transpose(-1, -2).topk(m, dim=-1).indices          # [H, blocks, m]
        pool = torch.zeros((h, blocks, pt), dtype=torch.bool, device=dev).scatter_(-1, picked, True)
        live_set &= pool
        counters.extra['shortlist'] = int(m)
    counters.candidates += int(live_set.sum())
    live_set &= ~keep_prefix
    keep_prefix |= protected
    # protected tiles are mandatory AND are charged against the optional budget, as in the control
    ptiles = max(0, ptiles - int(protected.sum(-1).sum()))
    # a (per row, per tile) and g (per row, per tile, per rank); BF16 for the candidate score pass
    a = alpha[:, :pt].permute(0, 2, 1).contiguous()                       # [H, N, PT]
    g = (c[:, :pt] - alpha[:, :pt, :, None] * ref[:, None, :, :]).permute(0, 2, 1, 3).contiguous()
    lm = live_set[:, :, None, :].expand(h, blocks, 128, pt).reshape(h, n_pad, pt)
    A = (a * lm).sum(-1)                                                  # [H, N]
    r = (g * lm[..., None]).sum(-2)                                      # [H, N, R]
    nu = stats.nu                                                          # [H, N]
    while True:
        count = live_set.sum(-1)                                          # [H, blocks]
        # ||r - g_j||^2 = ||r||^2 - 2 r.g_j + ||g_j||^2, from the cached summaries only
        rr = (r * r).sum(-1, keepdim=True)                                # [H, N, 1]
        cross = torch.einsum('hnr,hntr->hnt', r, g)                      # [H, N, PT]
        gg = (g * g).sum(-1)                                            # [H, N, PT]
        nrm2 = (rr - 2.0 * cross + gg).clamp_min(0.0)
        d2 = (A[:, :, None] - a).clamp_min(EPS)                            # [H, N, PT]
        per_row = (nrm2.sqrt() / d2 / nu[..., None]).permute(0, 2, 1)        # [H, PT, N]
        gscore = block_max(per_row, stats.rows)                              # [H, PT, blocks]
        alive = live_set.transpose(1, 2) & torch.isfinite(gscore)
        # the exact greedy takes the argmin, so a dead candidate must score +inf; the worst-first
        # ordering for drop-and-refine must push dead candidates LAST, so it uses the -inf variant
        unit = torch.where(alive, gscore, torch.full_like(gscore, math.inf)).transpose(1, 2)
        if drop_fraction and 0.0 < drop_fraction < 1.0:
            quota = ((count - ptiles).to(torch.float32) * float(drop_fraction)).ceil_().to(torch.int64)
            worst = torch.where(alive, gscore, torch.full_like(gscore, -math.inf)).transpose(1, 2)
            order = worst.argsort(-1, descending=True)                       # worst first
            rank = torch.empty_like(order)
            rank.scatter_(-1, order, torch.arange(pt, device=dev).expand_as(order))
            drop = live_set & (rank < quota.clamp_min(0)[..., None])
        else:
            # exact greedy: remove the single best ALIVE candidate (argmin, lowest tile index on ties)
            best = unit.argmin(-1)
            drop = torch.zeros_like(live_set).scatter_(-1, best[..., None], True)
            drop &= (count - ptiles > 0)[..., None]
        if not bool(drop.any()):
            break
        dm = drop[:, :, None, :].expand(h, blocks, 128, pt).reshape(h, blocks * 128, pt)
        A = A - (a * dm).sum(-1)
        r = r - (g * dm[..., None]).sum(-2)
        live_set &= ~drop
        counters.evaluations += int(h * n_pad * pt)
        counters.passes += 1
    keep[:, :, :pt] = keep_prefix | live_set
    counters.forced_keep += int(keep_prefix.sum())
    counters.forced_skip += int((~live_set).logical_and(~keep_prefix).sum())
    return keep.reshape(1, h, blocks, kt), counters, dict(alpha=alpha, c=c, ref=ref)


def _deletion_scores(stats, alpha, ref, live):
    gap = (stats.mu - ref[:, None, :, :]).norm(dim=-1)
    denom = (1.0 - alpha).clamp_min(EPS) * stats.nu[:, None, :]
    score = block_max(alpha * gap / denom, stats.rows)
    return torch.where(live.any(-1)[:, :, None], score, torch.full_like(score, -math.inf))


# --------------------------------------------------------------------------- statistics construction


def tile_statistics(scores, sketch, rows, chunk_tiles=32):
    """Reference tile statistics from an explicit masked score matrix (the trusted oracle).

    ``scores`` [H, N, nk] FP32: the native scaled attention logits AFTER the structural mask, with
    ``-inf`` where the (row, key) pair is illegal. ``sketch`` [HK, nk, 32]: the frozen Gaussian32
    projection of V (the head's own KV head is used). Returns ``(z[H, KT, N], mu[H, KT, N, 32])``
    with FlashAttention-style max/LSE reductions (``z = tilemax + log(sum exp(s - tilemax))``) and
    never a materialised unbounded exponential or a token attention matrix that outlives a chunk:
    the key axis is reduced tile by tile in chunks of ``chunk_tiles`` tiles."""
    h, n_pad, nk = scores.shape
    if rows.shape != (h, n_pad):
        raise ValueError('scores/rows must agree on (heads, padded rows)')
    if sketch.dim() != 3 or sketch.shape[0] * (h // sketch.shape[0]) != h or sketch.shape[1] != nk:
        raise ValueError('sketch must be [HK, nk, 32] with HK dividing the query heads')
    kt = -(-nk // 64)
    acc = torch.float64 if scores.dtype == torch.float64 else torch.float32
    z = torch.empty((h, kt, n_pad), device=scores.device, dtype=acc)
    mu = torch.empty((h, kt, n_pad, RANK), device=scores.device, dtype=acc)
    kha = torch.arange(h, device=scores.device) // (h // sketch.shape[0])
    per_head = sketch[kha].reshape(h, 1, nk, RANK)
    for t0 in range(0, kt, int(chunk_tiles)):
        t1 = min(kt, t0 + int(chunk_tiles))
        block = scores[:, :, t0 * 64:t1 * 64].reshape(h, n_pad, t1 - t0, 64)
        mx = block.amax(-1)                                               # [H, N, T]
        live = torch.isfinite(mx)
        safe = torch.where(live, mx, torch.zeros_like(mx))
        p = torch.where(torch.isfinite(block), torch.exp(block - safe[..., None]),
                        torch.zeros_like(block))
        ell = p.sum(-1).clamp_min(EPS)
        z[:, t0:t1] = torch.where(live, safe + torch.log(ell),
                                  torch.full_like(safe, -math.inf)).transpose(1, 2)
        sk = per_head[:, :, t0 * 64:t1 * 64, :].reshape(h, 1, t1 - t0, 64, RANK)
        num = torch.einsum('hntc,hntcr->hntr', p, sk.expand(h, n_pad, t1 - t0, 64, RANK))
        mu[:, t0:t1] = (num / ell[..., None]).transpose(1, 2)
    return z, mu


def append_tail_tiles(prefix_z, prefix_mu, tail_scores, sketch, total_tiles):
    """Append the always-kept canvas/boundary tiles to the prefix statistics so that V3's reference
    ``O_i`` covers the complete eligible support. ``prefix_z`` [H, PT, N] carries the padded-row
    mask, which is recovered from its ``-inf`` entries. ``tail_scores`` [H, N, n_tail] are the same
    FP32 scaled masked logits the active control already forms for the canvas region; ``sketch`` is
    [HK, nk, 32] over the FULL key extent and is sliced to the tail's keys."""
    rows = torch.isfinite(prefix_z).any(1)
    if sketch.shape[1] != tail_scores.shape[2]:
        sketch = sketch[:, sketch.shape[1] - tail_scores.shape[2]:, :]
    tz, tmu = tile_statistics(tail_scores, sketch, rows)
    pt = prefix_z.shape[1]
    if pt + tz.shape[1] != total_tiles:
        raise ValueError(f'tail tiles ({tz.shape[1]}) + prefix tiles ({pt}) != total tiles {total_tiles}')
    return torch.cat([prefix_z, tz], 1), torch.cat([prefix_mu, tmu], 1)


# --------------------------------------------------------------------------- masked-attention oracles


def masked_attention_sketch(alpha, c, keep):
    """``O_i(S) = (sum_{j in S} c_ij) / (sum_{j in S} alpha_ij)`` for a ``[1,H,blocks,kt]`` map.

    The trusted formula the V3 selectors minimize; used by the tests to recompute the achieved
    objective directly from the statistics."""
    h, kt, n_pad = alpha.shape
    blocks = keep.shape[2]
    if blocks * 128 != n_pad:
        raise ValueError('the keep map must cover the padded row axis')
    w = keep[0][..., None].expand(h, blocks, kt, 128).permute(0, 2, 1, 3).reshape(h, kt, n_pad)
    w = w.to(alpha.dtype)
    num = (c * w[..., None]).sum(1)
    den = (alpha * w).sum(1).clamp_min(EPS)
    return num / den[..., None], den


def objective(stats, keep, alpha=None, c=None, ref=None):
    """``F(S) = Agg_i(||O_i(S) - O_i|| / nu_i)`` on the full support (the V3 fixed-budget goal).

    Returns ``(F [H, blocks], O(S) [H, N, 32])`` so a caller can report the achieved objective and
    the retained output together."""
    if alpha is None:
        alpha, c, ref, _ = full_support(stats)
    out, _ = masked_attention_sketch(alpha, c, keep)
    err = (out - ref).norm(dim=-1) / stats.nu                                # [H, N]
    return block_max(err[:, None, :], stats.rows)[:, 0], out