"""Resumable diagnostic experiments, preserving the original RULER protocol."""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

from dllm.evaluation.ruler.runner import RulerRunConfig, run_evaluation
from dllm.attention.blasst import BLASST_MASK_SEMANTICS, validate_blasst_output_directory
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.runner import _runner_manifest, _study_rows

BASE = Path("results/diffusion_gemma_solattn_vs_blasst_ruler16k").resolve()
ROOT = Path("results/diffusion_gemma_blasst_diagnosis").resolve()
MODEL = "/home/exouser/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b"
REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"
RULER = "/home/exouser/dyh/RULER_c3f5e3b_clean"


def endpoint(root: Path, smoke: bool = False, reverse: bool = False, eager_dense: bool = False):
    study, rows = _study_rows(BASE / "manifest.json", "final")
    if smoke:
        rows = rows[:2]
    out = root / (("reverse_smoke" if reverse else "endpoint_smoke") if smoke else ("lambda1_reverse" if reverse else "lambda1_forward"))
    if eager_dense:
        out = root / 'dense_eager'
    else:
        validate_blasst_output_directory(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = _runner_manifest(study, rows, out / "runner_manifest.json", adapter="diffusion_gemma")
    config = RulerRunConfig(
        model_adapter="diffusion_gemma", model_path=MODEL, revision=REVISION,
        manifest_path=str(manifest), ruler_root=RULER, output_dir=str(out),
        num_samples=len(rows), context_length=16384, attention_backend="eager-dense" if eager_dense else "blasst-reference",
        blasst_lambda=1.0, blasst_policy={"local_blasst_lambda": 1.0, "global_blasst_lambda": 1.0},
        q_tile_size=64, kv_tile_size=64, collect_attention_stats=not eager_dense, stats_level="head",
        temperature=0.0, precision="bfloat16", progress_every=1,
    )
    from dllm.attention.blasst.core import blasst_2d_attention_forward
    (out / "diagnostic_policy.json").write_text(json.dumps({"kv_tile_order": "reverse" if reverse else "forward", "lambda_local": 1., "lambda_global": 1., "blasst_mask_semantics": "dense" if eager_dense else BLASST_MASK_SEMANTICS}))
    return run_evaluation(config, attention_override=partial(blasst_2d_attention_forward, blasst_tile_order="reverse") if reverse else None)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("endpoint-smoke", "endpoint", "reverse-smoke", "reverse", "diagnostics", "dense-eager"))
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == "diagnostics":
        from .diagnostics import collect
        result = collect(args.output_dir)
    else:
        result = endpoint(args.output_dir, smoke=args.command.endswith("smoke"), reverse=args.command.startswith("reverse"), eager_dense=args.command=='dense-eager')
    print(json.dumps({"command": args.command, "summary_keys": list(result)}, sort_keys=True))
