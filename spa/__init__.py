"""Training-free Shared-Page Attention (SPA) research references."""

from .reference import (
    SPAConfig,
    SPAMetrics,
    SPASupport,
    build_oracle_support,
    build_oracle_support_from_probabilities,
    dense_attention_reference,
    exact_shared_support_attention,
    evaluate_support,
    mean_support_jaccard,
)

__all__ = [
    "SPAConfig",
    "SPAMetrics",
    "SPASupport",
    "build_oracle_support",
    "build_oracle_support_from_probabilities",
    "dense_attention_reference",
    "exact_shared_support_attention",
    "evaluate_support",
    "mean_support_jaccard",
]
