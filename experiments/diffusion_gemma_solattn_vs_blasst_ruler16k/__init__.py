"""DiffusionGemma Sol-Attn versus BLASST RULER16K reference study."""

from .config import (
    ANALYTIC_BETAS,
    DEFAULT_TASKS,
    DEFAULT_MODEL_PATH,
    DEFAULT_MODEL_REVISION,
    TARGET_SPARSITIES,
    ExperimentConfig,
    analytic_beta,
    canonical_conditions,
)
from .routing import (
    BlasstRoutingResult,
    SolRoutingResult,
    blasst_route,
    mean_pooled_tile_proxies,
    route_attention_call,
    sol_gaussian_route,
)

__all__ = [
    "ANALYTIC_BETAS",
    "DEFAULT_TASKS",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_MODEL_REVISION",
    "BlasstRoutingResult",
    "SolRoutingResult",
    "TARGET_SPARSITIES",
    "ExperimentConfig",
    "analytic_beta",
    "blasst_route",
    "canonical_conditions",
    "mean_pooled_tile_proxies",
    "route_attention_call",
    "sol_gaussian_route",
]
