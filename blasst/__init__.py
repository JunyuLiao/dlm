"""BLASST sparse FlashAttention integration for LLaDA."""

from .flash_attention import BlasstStats, blasst_flash_attn_func, collect_blasst_stats, install_blasst
from .calibration import (
    PhysicalCalibrationRegression,
    PhysicalTileScoreCollector,
    calibration_matches_runtime_defaults,
    lambda_for_physical_sparsity,
    load_physical_calibration,
    select_largest_eligible_lambda,
)
from .triton_bidirectional import (
    DiffusionKernelController,
    DiffusionLambdaSchedule,
    KernelStats,
    blasst_bidirectional_flash_attn_func,
    get_kernel_stats,
    install_bidirectional_blasst_kernel,
    install_diffusion_blasst_kernel,
    reset_kernel_stats,
)

__all__ = [
    "BlasstStats",
    "DiffusionKernelController",
    "DiffusionLambdaSchedule",
    "KernelStats",
    "PhysicalTileScoreCollector",
    "PhysicalCalibrationRegression",
    "calibration_matches_runtime_defaults",
    "blasst_flash_attn_func",
    "blasst_bidirectional_flash_attn_func",
    "collect_blasst_stats",
    "get_kernel_stats",
    "install_blasst",
    "install_bidirectional_blasst_kernel",
    "install_diffusion_blasst_kernel",
    "lambda_for_physical_sparsity",
    "load_physical_calibration",
    "select_largest_eligible_lambda",
    "reset_kernel_stats",
]
