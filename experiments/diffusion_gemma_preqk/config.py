from dataclasses import dataclass, asdict, replace
from pathlib import Path

ROOT = Path('results/diffusion_gemma_preqk')
ORACLE = Path('results/diffusion_gemma_oracle')
BASE = Path('results/diffusion_gemma_solattn_blasst_multibench_controlled')
TILE = 64


@dataclass(frozen=True)
class RouterConfig:
    estimator: str = 'last_mass'
    protection: str = 'none'
    random_protection: bool = False
    kv_neighborhood: str = 'none'
    query_neighborhood: str = 'none'
    value_weight: str = 'none'
    rho: float = .8
    refresh_interval: int = 4
    exploration_tiles: int = 1
    renormalization: str = 'estimated_coverage'
    allocation: str = 'uniform'
    history_source: str = 'sparse'
    execution: str = 'reference'
    sparsity: float = .5

    def __post_init__(self):
        assert self.estimator in ('last_mass','last_mask','frequency','ema_mass','peak','proxy')
        assert self.protection in ('none','beginning','prefix_end','canvas_end','diagonal')
        assert self.kv_neighborhood in ('none','mean','max')
        assert self.query_neighborhood in ('none','mean','max')
        assert self.value_weight in ('none','rms','max')
        assert self.history_source in ('dense_reference','sparse')
        assert self.renormalization in ('conditional','estimated_coverage')
        assert self.execution in ('reference','gather')
        assert 0 <= self.sparsity < 1 and 0 <= self.rho < 1
        assert self.refresh_interval >= 0 and self.exploration_tiles >= 0

    def to_dict(self):
        return asdict(self)


def screening_configs():
    base=RouterConfig(history_source='dense_reference',refresh_interval=0,exploration_tiles=0)
    rows={name:replace(base,estimator=name) for name in ('last_mass','last_mask','frequency','ema_mass','peak','proxy')}
    for position in ('beginning','prefix_end','canvas_end','diagonal'):
        rows['protect_'+position]=replace(base,protection=position)
        rows['random_'+position]=replace(base,protection=position,random_protection=True)
    for dimension in ('kv','query'):
        for reduction in ('mean','max'):
            rows[dimension+'_'+reduction]=replace(base,**{dimension+'_neighborhood':reduction})
    for reduction in ('rms','max'):
        rows['value_'+reduction]=replace(base,value_weight=reduction)
    return rows
