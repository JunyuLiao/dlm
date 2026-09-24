"""Phase A1: native-SDPA support vs legacy Junyu query-relative window.

Uses the actual production function ``dllm.attention.blasst.core._attention_validity``
(the same call our numerical adapter and fresh Junyu T both use) rather than
reimplementing the window formula by hand. The compared quantity -- which key
positions are legal for a given query -- is a pure function of shape/position,
so this audit needs no live model forward; the geometry constants (sliding
window, canvas/block length) are read from the actual executed model config
and runner request shape, not guessed.

Native support (ground truth): the installed Transformers 5.11
``sdpa_attention_forward`` ignores the ``sliding_window`` keyword entirely
(see integrations/sdpa_attention.py), and DiffusionGemma's decoder attention
is bidirectional (``is_causal=False``) over whatever the DynamicCache already
holds. ``DynamicSlidingWindowLayer.update`` already trims each sliding-layer's
stored encoder prefix to at most ``sliding_window - 1`` tokens at commit time.
So native local-layer support = every key in the supplied (already-capped)
prefix plus the full canvas, for every query, with no further narrowing.

Legacy Junyu support additionally applies ``_attention_validity``'s
query-relative lower bound ``k_position >= q_position - sliding_window + 1``,
computed from each query's absolute position. Because canvas queries have
much larger absolute positions than the (already small) capped prefix, this
formula excludes early prefix keys for later canvas queries even though
native dense attends them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dllm.attention.blasst.core import _attention_validity


def audit_one(prefix: int, canvas: int, window: int, heads: int = 1) -> dict:
    kv_len = prefix + canvas
    q = torch.zeros(1, heads, canvas, 1)
    k = torch.zeros(1, heads, kv_len, 1)
    native = _attention_validity(None, q, k, is_causal=False, sliding_window=None)
    legacy = _attention_validity(None, q, k, is_causal=False, sliding_window=window)
    native_legal = int(native[0, 0].sum().item())
    legacy_legal = int(legacy[0, 0].sum().item())
    excluded = native & ~legacy
    excluded_prefix = int(excluded[0, 0, :, :prefix].sum().item())
    excluded_canvas = int(excluded[0, 0, :, prefix:].sum().item())
    extra_in_legacy = int((legacy & ~native).sum().item())
    last_query_excluded_prefix = int(excluded[0, 0, -1, :prefix].sum().item()) if canvas else 0
    return dict(prefix=prefix, canvas=canvas, sliding_window=window,
               native_legal_pairs=native_legal, legacy_legal_pairs=legacy_legal,
               excluded_prefix_pairs=excluded_prefix, excluded_canvas_pairs=excluded_canvas,
               legacy_adds_pairs_beyond_native=extra_in_legacy,
               last_query_excluded_prefix_keys=last_query_excluded_prefix,
               last_query_prefix_available=prefix)


def run(sliding_window: int, canvas: int, prefixes: list[int]) -> dict:
    if any(p < 0 for p in prefixes) or canvas <= 0 or sliding_window <= 0:
        raise ValueError("prefixes must be nonnegative and canvas/window positive")
    rows = [audit_one(p, canvas, sliding_window) for p in prefixes]
    return dict(
        schema="native_mask_audit_v1",
        function="dllm.attention.blasst.core._attention_validity",
        native_definition="sliding_window=None (matches sdpa_attention_forward, which ignores "
                          "the sliding_window kwarg, and DynamicSlidingWindowLayer's own "
                          "already-capped stored prefix)",
        legacy_definition="sliding_window=<config value> (Junyu's query-relative lower bound, "
                          "currently inherited by fresh T, M1 and M3)",
        model_sliding_window=sliding_window,
        canvas_length=canvas,
        rows=rows,
        note="Positive excluded_prefix_pairs means legacy silently drops native-legal prefix "
            "keys for later canvas queries; this is a real support mismatch but is shared "
            "identically by T, M1 and M3, so it does not by itself explain M1 failing "
            "relative to T.",
    )


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sliding-window", type=int, default=1024,
                        help="text_config.sliding_window from the actual executed model config.json")
    parser.add_argument("--canvas", type=int, default=256, help="native runner block_size")
    parser.add_argument("--prefixes", type=int, nargs="+",
                        default=[0, 1, 64, 256, 512, 768, 1000, 1023],
                        help="Encoder-committed prefix lengths to audit, up to sliding_window-1")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse(argv)
    payload = run(args.sliding_window, args.canvas, args.prefixes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
