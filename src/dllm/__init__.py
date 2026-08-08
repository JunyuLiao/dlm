"""Model-independent diffusion-LLM inference framework."""

from .models import GenerationRequest, GenerationResult, ModelAdapter, create_adapter

__all__ = [
    "GenerationRequest",
    "GenerationResult",
    "ModelAdapter",
    "create_adapter",
]

