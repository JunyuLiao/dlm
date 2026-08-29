"""Configuration and protocol constants for the DiffusionGemma RULER16K study.

The protocol deliberately uses *target skipped-tile sparsity* as its public
quantity.  Internally a router works with retained density, which is always
``1 - target_sparsity``.  Keeping the two names separate avoids the common
mistake of passing a sparsity target to a density threshold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


DEFAULT_TASKS = (
    "niah_multikey_1",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "fwe",
)

# The experiment protocol is tied to the released DiffusionGemma checkpoint
# used by the repository's existing RULER studies.  Keeping these defaults in
# the isolated package makes an omitted ``--revision`` deterministic while
# still allowing callers to pass an explicit local model path.
DEFAULT_MODEL_PATH = "google/diffusiongemma-26B-A4B-it"
DEFAULT_MODEL_REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"

TARGET_SPARSITIES = (0.25, 0.50, 0.75, 0.90)
TILE_SIZE = 64

# Universal Sol-Attn Gaussian cut points.  Values are explicitly recorded so
# reports remain stable even if a scipy implementation changes its last bit.
ANALYTIC_BETAS: dict[float, float] = {
    0.25: -0.674490,
    0.50: 0.0,
    0.75: 0.674490,
    0.90: 1.281552,
}


def analytic_beta(target_sparsity: float) -> float:
    """Return the universal Gaussian ``Phi^-1(target_sparsity)`` cut point."""

    value = float(target_sparsity)
    for key, beta in ANALYTIC_BETAS.items():
        if abs(value - key) < 1.0e-12:
            return beta
    raise ValueError(
        "the canonical Sol-Attn protocol defines beta only for target "
        f"sparsities {tuple(ANALYTIC_BETAS)}"
    )


def condition_name(method: str, target_sparsity: float) -> str:
    """Return the stable condition name used in manifests and reports."""

    if method == "dense":
        return "dense"
    value = float(target_sparsity)
    suffix = int(round(value * 100))
    if method == "sol_gaussian":
        return f"sol_gaussian_s{suffix}"
    if method == "blasst_calibrated":
        return f"blasst_calibrated_s{suffix}"
    raise ValueError(f"unknown method: {method}")


@dataclass(frozen=True)
class ExperimentConfig:
    """Core run configuration shared by calibration, execution, and reports."""

    method: str
    target_sparsity: float = 0.0
    q_tile_size: int = TILE_SIZE
    kv_tile_size: int = TILE_SIZE
    beta: float | None = None
    lambda_local: float | None = None
    lambda_global: float | None = None
    manifest_path: str | None = None
    model_path: str | None = None
    model_revision: str | None = None
    seed: int = 42
    output_dir: str = "results/diffusion_gemma_solattn_vs_blasst_ruler16k"

    def __post_init__(self) -> None:
        if self.method not in {"dense", "sol_gaussian", "blasst_calibrated"}:
            raise ValueError("method must be dense, sol_gaussian, or blasst_calibrated")
        if self.method == "dense":
            if self.target_sparsity not in (0.0,):
                raise ValueError("dense condition must have target_sparsity=0")
        elif not 0.0 < float(self.target_sparsity) < 1.0:
            raise ValueError("target_sparsity must lie strictly between zero and one")
        if self.q_tile_size != TILE_SIZE or self.kv_tile_size != TILE_SIZE:
            raise ValueError("the canonical comparison uses 64x64 logical tiles")
        if self.method == "sol_gaussian":
            expected = analytic_beta(self.target_sparsity)
            if self.beta is not None and abs(float(self.beta) - expected) > 2.0e-6:
                raise ValueError("Sol-Attn beta must be the analytic universal cut point")
        if self.method == "blasst_calibrated":
            for name, value in (("lambda_local", self.lambda_local), ("lambda_global", self.lambda_global)):
                if value is None or not 0.0 < float(value) < 1.0:
                    raise ValueError(f"{name} must lie strictly between zero and one")

    @property
    def retained_density(self) -> float:
        return 1.0 if self.method == "dense" else 1.0 - float(self.target_sparsity)

    @property
    def beta_used(self) -> float | None:
        return None if self.method != "sol_gaussian" else analytic_beta(self.target_sparsity)

    @property
    def name(self) -> str:
        return condition_name(self.method, self.target_sparsity)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            condition=self.name,
            retained_density=self.retained_density,
            beta_used=self.beta_used,
        )
        return result


def canonical_conditions(
    *,
    local_lambdas: Mapping[float, float] | None = None,
    global_lambdas: Mapping[float, float] | None = None,
) -> list[ExperimentConfig]:
    """Build dense plus the eight requested sparse conditions."""

    result = [ExperimentConfig(method="dense")]
    for target in TARGET_SPARSITIES:
        result.append(
            ExperimentConfig(
                method="sol_gaussian",
                target_sparsity=target,
                beta=analytic_beta(target),
            )
        )
    for target in TARGET_SPARSITIES:
        if local_lambdas is None or global_lambdas is None:
            raise ValueError("BLASST conditions require calibrated local/global lambdas")
        result.append(
            ExperimentConfig(
                method="blasst_calibrated",
                target_sparsity=target,
                lambda_local=float(local_lambdas[target]),
                lambda_global=float(global_lambdas[target]),
            )
        )
    return result


__all__ = [
    "ANALYTIC_BETAS",
    "DEFAULT_TASKS",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_MODEL_REVISION",
    "TARGET_SPARSITIES",
    "TILE_SIZE",
    "ExperimentConfig",
    "analytic_beta",
    "canonical_conditions",
    "condition_name",
]
