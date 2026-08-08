"""Shared attention execution interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class AttentionContext:
    active_query_mask: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

