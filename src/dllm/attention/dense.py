from __future__ import annotations

import torch
import torch.nn.functional as F


def dense_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    kwargs = {
        "attn_mask": attention_mask,
        "dropout_p": dropout_p,
        "is_causal": is_causal,
    }
    if scale is not None:
        kwargs["scale"] = scale
    return F.scaled_dot_product_attention(query, key, value, **kwargs)

