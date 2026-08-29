"""Observation-first analysis of DiffusionGemma denoising attention."""

from .hooks import install_attention_observer, install_attention_replay
from .proxy import AttentionObservationConfig, AttentionObserver
from .replay import ReplayAttention, ReplayConfig

__all__ = [
    "AttentionObservationConfig",
    "AttentionObserver",
    "ReplayAttention",
    "ReplayConfig",
    "install_attention_observer",
    "install_attention_replay",
]
