"""Predeclared candidates; no final-score-based family/dimension selection."""
from dataclasses import dataclass

PRIMARY_SEED = 1729
SENSITIVITY_SEEDS = (2718, 31415)
FAMILIES = ('gaussian', 'sign')
DIMENSIONS = (8, 16, 32)
TARGETS = (.5, .75)
Q_TILE, KV_TILE = 128, 64


@dataclass(frozen=True)
class Config:
    method: str = 'centered'
    family: str = 'gaussian'
    rank: int = 16
    projection_seed: int = PRIMARY_SEED
    guard_seed: int = SENSITIVITY_SEEDS[0]
    log_threshold: float = -float('inf')
    cancellation_cutoff: float = .25
    substantial_mass: float = .1

    def __post_init__(self):
        if self.method not in ('centered', 'contribution', 'mass_exact', 'cancellation_guard'):
            raise ValueError('Unknown projected routing method')
        if self.family not in (*FAMILIES, 'identity'):
            raise ValueError('Unknown projection family')
        if self.family != 'identity' and self.rank not in DIMENSIONS:
            raise ValueError('Unrequested projection dimension')
        if self.method == 'cancellation_guard' and self.projection_seed == self.guard_seed:
            raise ValueError('The cancellation safeguard requires an independent sketch')


PROJECTED = {f'jl_{f}_r{r}': dict(family=f, rank=r) for f in FAMILIES for r in DIMENSIONS}
# Both families at the middle dimension are prespecified controls, not winners.
PROJECTED.update({f'contribution_{f}_r16': dict(method='contribution', family=f, rank=16) for f in FAMILIES})
PROJECTED.update({f'cancel_guard_{f}_r16': dict(method='cancellation_guard', family=f, rank=16) for f in FAMILIES})
PROJECTED.update(full_centered=dict(family='identity'), mass_exact=dict(method='mass_exact'))
BASELINES = dict(blasst_original=dict(method='blasst'), blasst_aggressive=dict(method='blasst'),
                 value=dict(method='value', pooling='vector_mean'),
                 mass=dict(method='mass'), risk=dict(method='risk', pooling='mean'),
                 unweighted_centered=dict(method='centered'))
METHODS = tuple(BASELINES) + tuple(PROJECTED)
