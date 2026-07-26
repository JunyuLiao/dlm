#!/usr/bin/env python3
"""Offline-compile all BLASST microgroup variants and report SM90 PTX structure."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton-cache")

def compile_variant(group_rows: int, collect_stats: bool) -> dict[str, int]:
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource

    from blasst.triton_bidirectional import _blasst_bidirectional_fwd

    pointers = {"Q", "K", "V", "O", "STATS", "LOG_THRESHOLDS", "QUERY_PERM", "KV_ORDER"}
    signature = {
        name: (
            "*bf16"
            if name in {"Q", "K", "V", "O"}
            else "*i64"
            if name == "STATS"
            else "*fp32"
            if name == "LOG_THRESHOLDS"
            else "*i32"
        )
        for name in _blasst_bidirectional_fwd.arg_names
        if name in pointers
    }
    signature["softmax_scale"] = "fp32"
    for name in _blasst_bidirectional_fwd.arg_names:
        if name.startswith("stride_") or name in {"seqlen", "nheads"}:
            signature[name] = "i32"
    groups = 128 // group_rows
    constants = {
        "BLOCK_M": 128,
        "BLOCK_N": 64,
        "HEAD_DIM": 128,
        "PIPELINE_STAGES": 2,
        "COLLECT_STATS": collect_stats,
        "USE_QUERY_PERM": True,
        "USE_KV_ORDER": True,
        "SKIP_GROUP_ROWS": group_rows,
        "DENSE_FALLBACK_THRESHOLD": groups,
    }
    compiled = triton.compile(
        ASTSource(_blasst_bidirectional_fwd, signature, constexprs=constants),
        target=GPUTarget("cuda", 90, 32),
        options={"num_warps": 4, "num_stages": 2},
    )
    ptx = compiled.asm["ptx"]
    return {
        "skip_group_rows": group_rows,
        "groups": groups,
        "shared_bytes": compiled.metadata.shared,
        "wgmma_instructions_static": ptx.count("wgmma.mma_async"),
        "mma_sync_instructions_static": ptx.count("mma.sync"),
        "branch_instructions_static": ptx.count("bra"),
        "global_load_instructions_static": ptx.count("ld.global"),
    }


def compile_pre_qk(collect_stats: bool) -> dict[str, int | str]:
    """Compile the independent metadata-aware parent-tile specialization."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource

    from blasst.triton_bidirectional import _blasst_bidirectional_pre_qk_fwd

    signature: dict[str, str] = {}
    for name in _blasst_bidirectional_pre_qk_fwd.arg_names:
        if name in {"Q", "K", "V", "O"}:
            signature[name] = "*bf16"
        elif name == "STATS":
            signature[name] = "*i64"
        elif name in {"PREVIOUS_LOG_SCORES", "CURRENT_LOG_SCORES"}:
            signature[name] = "*fp16"
        elif name in {"LOG_THRESHOLDS", "PROXY_LOG_THRESHOLDS"}:
            signature[name] = "*fp32"
        elif name == "softmax_scale":
            signature[name] = "fp32"
        elif name.startswith("stride_") or name in {"seqlen", "nheads"}:
            signature[name] = "i32"
    constants = {
        "BLOCK_M": 128, "BLOCK_N": 64, "HEAD_DIM": 128,
        "PIPELINE_STAGES": 2, "COLLECT_STATS": collect_stats,
        "WARMUP_TILES": 2, "PERIODIC_REFRESH": 8,
        "ANCHOR_LOCAL": True, "ANCHOR_SINK": True, "LOCAL_RADIUS": 1,
    }
    compiled = triton.compile(
        ASTSource(_blasst_bidirectional_pre_qk_fwd, signature, constexprs=constants),
        target=GPUTarget("cuda", 90, 32),
        options={"num_warps": 4, "num_stages": 2},
    )
    ptx = compiled.asm["ptx"]
    return {
        "variant": "pre_qk_128x64",
        "shared_bytes": compiled.metadata.shared,
        "wgmma_instructions_static": ptx.count("wgmma.mma_async"),
        "mma_sync_instructions_static": ptx.count("mma.sync"),
        "branch_instructions_static": ptx.count("bra"),
        "global_load_instructions_static": ptx.count("ld.global"),
        "global_store_instructions_static": ptx.count("st.global"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--collect-stats", action="store_true")
    parser.add_argument("--group", type=int, choices=(16, 32, 64, 128), help=argparse.SUPPRESS)
    parser.add_argument("--pre-qk", action="store_true")
    args = parser.parse_args()
    if args.group is not None:
        print(json.dumps(compile_variant(args.group, args.collect_stats)))
        return
    if args.pre_qk:
        print(json.dumps(compile_pre_qk(args.collect_stats), indent=2))
        return
    variants = []
    for group in (128, 64, 32, 16):
        command = [sys.executable, str(pathlib.Path(__file__).resolve()), "--group", str(group)]
        if args.collect_stats:
            command.append("--collect-stats")
        variants.append(json.loads(subprocess.check_output(command, text=True)))
    report = {
        "target": "sm90",
        "note": (
            "Static PTX counts prove specialized matrix instructions and branch structure, "
            "not runtime instruction retirement or latency."
        ),
        "variants": variants,
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
