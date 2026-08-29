"""Audit and report for the Math500-only prefix proxy study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .plots import generate_plots


def build_report(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    # Always regenerate from the final shards so report metadata cannot point
    # at stale plots or an earlier score transform.
    summary = generate_plots(root)
    metadata = []
    for path in sorted((root / "collection").glob("*/metadata.json")):
        metadata.append(json.loads(path.read_text()))
    audits = []
    for item in metadata:
        adapter = item["adapter"]
        shards = sorted((root / "shards" / adapter).glob("*.npz"))
        attention_types = set()
        rows = 0
        prefix_tiles = 0
        for shard in shards:
            import numpy as np

            with np.load(shard) as payload:
                attention_types.update(str(value) for value in payload["attention_type"])
                rows += len(payload["attention_type"])
                prefix_tiles += int(payload["prefix_tiles"].sum())
        audits.append({
            "adapter": adapter,
            "problems": item["num_problems"],
            "shards": len(shards),
            "parity_passed": item["parity"]["passed"],
            "attention_types": sorted(attention_types),
            "rows": rows,
            "prefix_tiles": prefix_tiles,
            "problem_ids": item["problem_ids"],
        })
    if len(audits) == 2:
        assert audits[0]["problem_ids"] == audits[1]["problem_ids"]
    corpus = metadata[0].get("corpus", "unknown") if metadata else "unknown"
    audit = {
        "study": f"{corpus}_prefix_proxy_distribution",
        "corpus": corpus,
        "num_problems": 10,
        "q_block_size": 64,
        "kv_block_size": 64,
        "population": "prefix KV tiles only",
        "score": "row-standardized raw pre-softmax Mean(Q_i) Mean(K_j)^T * native scale",
        "row_aggregation": "equal prompt, denoising call, layer, head, and query-block mass; each row contributes total mass one",
        "row_standardization": True,
        "raw_score_retained_for_diagnostics": True,
        "threshold_fitting": False,
        "identical_problem_ids_across_models": len(audits) == 2 and audits[0]["problem_ids"] == audits[1]["problem_ids"],
        "collections": audits,
        "plots": summary.get("plots", []),
    }
    (root / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    lines = [
        f"# {corpus} prefix-proxy distribution study",
        "",
        f"This is a {corpus}-only distribution study over 10 fixed examples. It models prefix KV tiles because DiffusionGemma has only a small number of canvas tiles per row. No Gaussian-tail threshold fitting, routing evaluation, or speedup claim is included.",
        "",
        "## Experimental setting",
        "",
        "- Logical blocks: 64 query × 64 KV tokens.",
        "- Score: raw pre-softmax `Mean(Q_i) Mean(K_j)^T × native scale` after normalization/RoPE and GQA expansion; plots use row-standardized `z=(s−μ_row)/σ_row`.",
        "- Population: prefix KV tiles only; canvas tiles are excluded from the plotted population.",
        "- Aggregation: equal prompt, denoising-call, layer, head, and query-block weight; each row contributes total mass one across its prefix tiles.",
        "- Stored values: deterministic raw per-row reservoirs and row summaries; plots standardize each row using its stored mean and standard deviation.",
        "- Native dense attention output is unchanged; parity is checked on the first two problems per model.",
        "- Comparison scope: Sol-Attn reports image/video diffusion-transformer behavior and uses an online threshold plus approximate correction; this bundle reproduces only its mean-pooled proxy/routing-threshold idea, with fresh exact logical masking for discrete block-diffusion language models.",
        "- Interpretation: row standardization removes each row's mean and scale, but it cannot remove skew, multimodality, sparse lexical alignment, finite-row effects, or denoising-state changes.",
        "",
        "## Availability",
        "",
    ]
    for item in audits:
        types = ", ".join(item["attention_types"]) if item["attention_types"] else "none"
        lines.append(f"- `{item['adapter']}`: {item['problems']} problems, {item['rows']:,} rows, {item['prefix_tiles']:,} prefix tiles, attention types observed: {types}.")
    lines.extend([
        "",
        "Fast-dLLM-v2 has no observed local-attention layer in this configuration, so its local panel is intentionally empty. DiffusionGemma has both global and local panels.",
        "",
        "## Plots",
        "",
        "- `plots/01_standardized_prefix_proxy_distributions.png`: row-standardized prefix tile proxy distributions by model and attention type.",
        "- `plots/02_row_mean_prefix_proxy_distributions.png`: distributions of row-wise prefix means by model and attention type.",
        "",
        "## Prefix-tile counts per row",
        "",
    ])
    for key, group in summary.get("groups", {}).items():
        if group.get("rows"):
            lines.append(f"- `{key}`: mean {group['prefix_tiles'] / group['rows']:.2f} eligible prefix tiles per row.")
    lines.extend([
        "",
        "## Gaussian comparison",
        "",
        "Row standardization forces every contributing row to mean zero and variance one; it does not imply Gaussian shape. The dashed N(0,1) overlay and measured tail densities expose skew and excess-tail deviations that mean/std normalization cannot remove.",
    ])
    for key, group in summary.get("groups", {}).items():
        if group.get("gaussian_tail_densities"):
            tails = group["gaussian_tail_densities"]
            lines.append(f"- `{key}` realized tails at Gaussian 75/50/25/10% cutoffs: {tails['beta_-0.674_rho75']:.3f}, {tails['beta_0_rho50']:.3f}, {tails['beta_0.674_rho25']:.3f}, {tails['beta_1.282_rho10']:.3f}.")
            asymmetric = group.get("asymmetric_tail_probabilities", {})
            lines.append(
                f"  Shape diagnostics: skewness {group.get('weighted_skewness', float('nan')):+.3f}, "
                f"excess kurtosis {group.get('weighted_excess_kurtosis', float('nan')):+.3f}; "
                f"P(z>2)={asymmetric.get('gt_2', float('nan')):.4f}, "
                f"P(z<-2)={asymmetric.get('lt_minus_2', float('nan')):.4f}, "
                f"P(z>3)={asymmetric.get('gt_3', float('nan')):.4f}, "
                f"P(z<-3)={asymmetric.get('lt_minus_3', float('nan')):.4f}."
            )
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return {"audit": audit, "report": str(root / "report.md"), "plots": summary.get("plots", [])}
