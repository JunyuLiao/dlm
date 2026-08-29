"""Assemble the distribution, profile, and routing artifacts into one audit bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DECISION_DENSITIES = (75, 50, 25)
DECISION_MODES = ("gaussian", "profiled", "oracle")
EXPECTED_CONDITIONS = {
    "dense",
    *(f"{mode}_rho{density:02d}" for mode in DECISION_MODES for density in DECISION_DENSITIES),
    *(f"{mode}_rho{density:02d}_prefix_only" for mode in DECISION_MODES for density in DECISION_DENSITIES),
    # Ten percent is a failure-point probe for all-KV routing only.
    "gaussian_rho10",
    "oracle_rho10",
}

INTENTIONALLY_SKIPPED_CONDITIONS = {
    *(f"random_rho{density:02d}" for density in (75, 50, 25, 10)),
    *(f"random_rho{density:02d}_prefix_only" for density in (75, 50, 25, 10)),
    *(f"{mode}_rho10_prefix_only" for mode in DECISION_MODES),
    # Profiled 10% is explicitly optional in the staged protocol.
    "profiled_rho10",
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _condition_lines(path: Path) -> int:
    prediction_path = path / "predictions.jsonl"
    if not prediction_path.exists():
        return 0
    with prediction_path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _routing_status(root: Path) -> dict[str, Any]:
    models = {}
    # Revised runs use a new directory when an older partial shard has an
    # incompatible fingerprint.  Merge those directories without deleting or
    # rewriting the historical artifacts.
    aliases = {
        "diffusion_gemma": ("diffusion_gemma", "diffusion_gemma_decision_revised"),
        "fast_dllm_v2": ("fast_dllm_v2", "fast_dllm_v2_decision_revised"),
    }
    for model, names in aliases.items():
        source_roots = [root / name for name in names if (root / name).exists()]
        completed: set[str] = set()
        partial: dict[str, int] = {}
        superseded_partial: dict[str, int] = {}
        reports = []
        for model_root in source_roots:
            report = _read(model_root / "ruler_report.json")
            if report is not None:
                reports.append(str((model_root / "ruler_report.json").relative_to(root)))
            for path in model_root.iterdir():
                if not path.is_dir():
                    continue
                lines = _condition_lines(path)
                if lines >= 50 and (path / "summary.json").exists():
                    completed.add(path.name)
                elif 0 < lines < 50:
                    partial[path.name] = max(partial.get(path.name, 0), lines)
        for condition, lines in list(partial.items()):
            if condition in completed:
                superseded_partial[condition] = lines
                del partial[condition]
        completed_required = completed & EXPECTED_CONDITIONS
        completed_extra = completed - EXPECTED_CONDITIONS
        models[model] = {
            "source_directories": [str(path.relative_to(root)) for path in source_roots],
            "completed_required_conditions": sorted(completed_required),
            "completed_extra_conditions": sorted(completed_extra),
            "completed_conditions": sorted(completed),
            "missing_conditions": sorted(EXPECTED_CONDITIONS - completed_required),
            "partial_conditions": partial,
            "superseded_partial_conditions": superseded_partial,
            "intentionally_skipped_conditions": sorted(INTENTIONALLY_SKIPPED_CONDITIONS),
            "report_files": reports,
        }
    return models


def _diagnostic_status(root: Path, prefixes: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for prefix in prefixes:
        matches = sorted(path for path in root.glob(f"{prefix}*") if path.is_dir())
        if not matches:
            result[prefix] = {"status": "not_run", "directories": []}
            continue
        conditions: dict[str, str] = {}
        for match in matches:
            for child in sorted(match.iterdir()):
                if not child.is_dir():
                    continue
                lines = _condition_lines(child)
                conditions[f"{match.name}/{child.name}"] = "complete" if lines >= 2 and (child / "summary.json").exists() else ("partial" if lines else "empty")
        result[prefix] = {
            "status": "complete" if conditions and all(value == "complete" for value in conditions.values()) else "partial",
            "directories": [str(path.relative_to(root)) for path in matches],
            "conditions": conditions,
        }
    return result


def build_canonical_report(bundle_root: str | Path) -> dict[str, Any]:
    root = Path(bundle_root)
    distribution_root = root.parent.parent.parent / "proxy_diagnostics" / "ruler16k_prefix_distribution"
    distribution = _read(distribution_root / "summary.json")
    distribution_audit = _read(distribution_root / "audit.json")
    profile = _read(root / "profiled_threshold_model.json")
    gpu_stage_status = _read(root / "gpu_stage_status.json")
    routing = _routing_status(root)
    routing_complete = all(not item["missing_conditions"] for item in routing.values())
    diagnostics = _diagnostic_status(root, ("physical_", "blocksize_", "smoke_staged_"))
    result = {
        "schema_version": 1,
        "study": "distribution_first_sparse_routing_ruler16k",
        "distribution": distribution,
        "distribution_audit": distribution_audit,
        "profile": profile,
        "routing": routing,
        "routing_completeness": {
            "complete": routing_complete,
            "expected_conditions_per_model": len(EXPECTED_CONDITIONS),
            "note": "The expected set follows the staged protocol: 75/50/25 decision conditions, all-KV 10% failure probes, and no 50-example random controls.",
        },
        "diagnostics": diagnostics,
        "gpu_stage_status": gpu_stage_status,
        "physical_sparse_reference": {
            "implemented": True,
            "execution": "two-pass retained-tile QK/AV reference",
            "measured_speedup": False,
            "timing_fields": ["proxy_mask_seconds", "sparse_attention_seconds"],
        },
    }
    (root / "canonical_report.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# Distribution-first sparse routing: RULER16K canonical report",
        "",
        "This bundle evaluates fresh 64x64 logical tile routing on fixed tokenizer-specific RULER16K manifests. Prefix-only proxy distributions use 10 examples per model with 256 samples per row; routing sweeps use 50 balanced examples. Logical/physical densities are mask measurements, not speedups.",
        "",
        "## Distribution",
        "",
        f"- Summary present: `{distribution is not None}`; audit present: `{distribution_audit is not None}`.",
        f"- Population: `{(distribution or {}).get('population', 'missing')}`.",
        "- Row standardization is used for plotting and threshold calibration; it does not establish Gaussianity.",
    ]
    for key, group in (distribution or {}).get("groups", {}).items():
        if group.get("weighted_skewness") is not None:
            lines.append(
                f"- `{key}`: skewness `{group['weighted_skewness']:+.3f}`, "
                f"excess kurtosis `{group['weighted_excess_kurtosis']:+.3f}`."
            )
    lines.extend([
        "",
        "## Profiled threshold",
        "",
        f"- Profile present: `{profile is not None}`.",
        f"- Held-out simple-profile gate: `{(profile or {}).get('simple_profiling_passed', False)}`.",
        "- The profile is empirical-CDF, model/attention-type scoped, and uses a disjoint deterministic shard split.",
        "",
        "## Routing coverage",
        "",
    ])
    for model, status in routing.items():
        lines.append(f"- `{model}` completed {len(status['completed_required_conditions'])}/{len(EXPECTED_CONDITIONS)} required conditions.")
        lines.append(f"  Missing required: {', '.join(status['missing_conditions']) if status['missing_conditions'] else 'none'}.")
        lines.append(f"  Extra completed diagnostics: {', '.join(status['completed_extra_conditions']) if status['completed_extra_conditions'] else 'none'}.")
        lines.append(f"  Partial: {', '.join(f'{k} ({v}/50)' for k, v in sorted(status['partial_conditions'].items())) if status['partial_conditions'] else 'none'}.")
        lines.append(f"  Intentionally skipped: random full sweeps, prefix-only 10% probes, and optional profiled 10%.")
    if not routing_complete:
        lines.extend([
            "",
            "The required routing set is incomplete. Missing conditions are not interpreted as passing or failing; legacy partial shards are retained separately and are not silently promoted to completed results.",
        ])
    lines.extend(["", "## Diagnostics", ""])
    for name, item in diagnostics.items():
        lines.append(f"- `{name}`: **{item['status']}**; directories: {', '.join(item['directories']) if item['directories'] else 'none'}.")
    if gpu_stage_status is not None:
        gpu_complete = gpu_stage_status.get("status") == "complete"
        lines.extend([
            "",
            f"## {'GPU diagnostics' if gpu_complete else 'Deferred GPU diagnostics'}",
            "",
            f"- Status: **{gpu_stage_status.get('status', 'unknown')}**; recorded at `{gpu_stage_status.get('recorded_at_utc', 'unknown')}`.",
            *([f"- Reason: {gpu_stage_status.get('reason', 'not recorded')}"] if not gpu_complete else []),
            ("- Physical timing and block-size smoke artifacts are complete; logical density remains a mask measurement, not a speedup."
             if gpu_complete else
             "- No physical speedup or block-size result is claimed; logical density remains a mask measurement, not a speedup."),
        ])
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Sol-Attn's reported setting is continuous image/video diffusion with an online proxy threshold and approximation correction. This study isolates the mean-pooled proxy and exact keep/drop behavior in discrete block-diffusion language models; it does not claim that the full Sol-Attn algorithm or its speedups transfer unchanged.",
        "",
        "Discrete masked/multinomial denoising changes token identities between calls, language retrieval can concentrate on a few lexical positions inside a block, local layers have short finite candidate rows, and global/local heads have different geometry. These mechanisms can produce skew, heavy tails, or multimodality after row standardization. Therefore standardized prefix proxies may be approximately centered and unit-scale without being standard Gaussian.",
    ])
    (root / "canonical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_canonical_report(args.bundle_root), indent=2, sort_keys=True, default=str))
