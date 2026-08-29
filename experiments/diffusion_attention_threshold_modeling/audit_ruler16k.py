"""Final compact audit for the staged RULER16K routing bundle."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .canonical_report import EXPECTED_CONDITIONS

MODEL_SOURCES = {
    "diffusion_gemma": ("diffusion_gemma", "diffusion_gemma_decision_revised"),
    "fast_dllm_v2": ("fast_dllm_v2", "fast_dllm_v2_decision_revised"),
}
SMOKE_EXPECTED = {
    "dense", "gaussian_rho75", "gaussian_rho50", "gaussian_rho25",
    "gaussian_rho10", "oracle_rho50", "random_rho50",
    "gaussian_rho50_prefix_only",
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _complete(root: Path, condition: str, expected: int) -> bool:
    return _jsonl_count(root / condition / "predictions.jsonl") >= expected and (
        root / condition / "summary.json"
    ).exists()


def _manifest_audit(root: Path) -> dict[str, Any]:
    result = {}
    for model in MODEL_SOURCES:
        path = root / "manifests" / model / "manifest.json"
        manifest = _read(path)
        sample_path = None
        hash_ok = False
        if manifest:
            sample_path = Path(manifest.get("samples", {}).get("path", ""))
            if not sample_path.is_absolute():
                sample_path = path.parent / sample_path
            hash_ok = _sha256(sample_path) == manifest.get("samples", {}).get("sha256")
        result[model] = {
            "manifest_present": manifest is not None,
            "requested_num_samples": (manifest or {}).get("requested_num_samples"),
            "actual_num_samples": (manifest or {}).get("actual_num_samples"),
            "context_length": (manifest or {}).get("context_length"),
            "samples_hash_ok": hash_ok,
            "passed": bool(
                manifest
                and manifest.get("requested_num_samples") == 50
                and manifest.get("actual_num_samples") == 50
                and manifest.get("context_length") == 16384
                and hash_ok
            ),
        }
    return result


def _shard_audit(root: Path) -> dict[str, Any]:
    result = {}
    for model, sources in MODEL_SOURCES.items():
        complete = set()
        partial = {}
        for source_name in sources:
            source = root / source_name
            if not source.exists():
                continue
            for child in source.iterdir():
                if not child.is_dir():
                    continue
                count = _jsonl_count(child / "predictions.jsonl")
                if count >= 50 and (child / "summary.json").exists():
                    complete.add(child.name)
                elif count:
                    partial[child.name] = max(partial.get(child.name, 0), count)
        for name in list(partial):
            if name in complete:
                del partial[name]
        required = complete & EXPECTED_CONDITIONS
        result[model] = {
            "required_complete": sorted(required),
            "missing_required": sorted(EXPECTED_CONDITIONS - required),
            "partial_conditions": partial,
            "passed": not (EXPECTED_CONDITIONS - required),
        }
    return result


def _smoke_audit(root: Path) -> dict[str, Any]:
    result = {}
    for model_dir in ("smoke_staged_diffusion_gemma", "smoke_staged_fast_dllm_v2"):
        base = root / model_dir
        conditions = {}
        for condition in sorted(SMOKE_EXPECTED):
            summary = _read(base / condition / "summary.json")
            count = _jsonl_count(base / condition / "predictions.jsonl")
            routing = (summary or {}).get("routing_aggregate") or {}
            dense = condition == "dense"
            conditions[condition] = {
                "predictions": count,
                "summary_present": summary is not None,
                "full_cuda": (summary or {}).get("full_checkpoint_on_cuda"),
                "layers": len(routing.get("by_layer", {})),
                "heads": len(routing.get("by_head", {})),
                "steps": len(routing.get("by_step", {})),
                "regions": sorted(routing.get("region_counts", {})),
                "no_nonfinite_values": _finite(summary),
                "passed": bool(
                    count == 2
                    and summary is not None
                    and (summary or {}).get("full_checkpoint_on_cuda") is True
                    and _finite(summary)
                    and (dense or (
                        routing.get("calls", 0) > 0
                        and routing.get("by_layer")
                        and routing.get("by_head")
                        and routing.get("by_step")
                    ))
                ),
            }
        result[model_dir] = {
            "conditions": conditions,
            "passed": all(item["passed"] for item in conditions.values()),
        }
    return result


def _profile_audit(root: Path) -> dict[str, Any]:
    profile = _read(root / "profiled_threshold_model.json") or {}
    overlap = False
    for split in (profile.get("splits") or {}).values():
        overlap = overlap or bool(
            set(split.get("calibration", [])) & set(split.get("validation", []))
        )
    return {
        "present": bool(profile),
        "calibration_only_audit": profile.get("calibration_only_audit"),
        "split_overlap": overlap,
        "simple_profiling_passed": profile.get("simple_profiling_passed"),
        "selected_family": (profile.get("selected") or {}).get("family"),
        "selected_scope": (profile.get("selected") or {}).get("scope"),
        "passed": bool(profile and profile.get("calibration_only_audit") and not overlap),
    }


def _report_audit(root: Path) -> dict[str, Any]:
    ruler = _read(root / "ruler_report.json")
    canonical = _read(root / "canonical_report.json")
    required = (
        root / "ruler_report.json",
        root / "ruler_report.md",
        root / "canonical_report.json",
        root / "canonical_report.md",
    )
    plots = sorted((root / "plots").glob("*.png")) if (root / "plots").exists() else []
    paired = (ruler or {}).get("audit", {}).get(
        "all_completed_conditions_share_prompts_and_seeds"
    )
    no_speedup_claim = not (ruler or {}).get("audit", {}).get(
        "theoretical_density_described_as_speedup"
    )
    return {
        "required_files": {
            str(path.relative_to(root)): path.exists() for path in required
        },
        "plot_count": len(plots),
        "paired_seed_audit_passed": paired,
        "no_logical_density_claimed_as_speedup": no_speedup_claim,
        "canonical_complete": (canonical or {}).get("routing_completeness", {}).get(
            "complete"
        ),
        "passed": bool(
            ruler
            and canonical
            and all(path.exists() for path in required)
            and plots
            and paired
            and no_speedup_claim
        ),
    }


def _diagnostic_audit(root: Path) -> dict[str, Any]:
    physical = {}
    for model in ("diffusion_gemma", "fast_dllm_v2"):
        base = root / ("physical_" + model)
        physical[model] = {
            condition: _complete(base, condition, 2)
            for condition in ("dense", "profiled_rho50", "profiled_rho25")
        }
    blocksize = {}
    for model in ("diffusion_gemma", "fast_dllm_v2"):
        for q in (32, 64, 128):
            base = root / ("blocksize_" + model + "_q" + str(q))
            blocksize[model + "_q" + str(q)] = {
                condition: _complete(base, condition, 2)
                for condition in ("dense", "gaussian_rho50")
            }
    return {
        "physical": physical,
        "blocksize": blocksize,
        "passed": (
            all(all(values.values()) for values in physical.values())
            and all(all(values.values()) for values in blocksize.values())
        ),
    }


def build_final_audit(bundle_root: str | Path) -> dict[str, Any]:
    root = Path(bundle_root)
    manifests = _manifest_audit(root)
    smoke = _smoke_audit(root)
    shards = _shard_audit(root)
    profile = _profile_audit(root)
    diagnostics = _diagnostic_audit(root)
    reports = _report_audit(root)
    result = {
        "schema_version": 1,
        "manifest_seed_prompt_identity": manifests,
        "smoke_coverage": smoke,
        "shard_completeness": shards,
        "calibration_only_fitting": profile,
        "diagnostics": diagnostics,
        "report_generation": reports,
    }
    result["passed"] = (
        all(item["passed"] for item in manifests.values())
        and all(item["passed"] for item in smoke.values())
        and all(item["passed"] for item in shards.values())
        and profile["passed"]
        and diagnostics["passed"]
        and reports["passed"]
    )
    (root / "final_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Final RULER16K staged audit",
        "",
        "Overall passed: " + str(result["passed"]),
    ]
    for name in (
        "manifest_seed_prompt_identity",
        "smoke_coverage",
        "shard_completeness",
        "calibration_only_fitting",
        "diagnostics",
        "report_generation",
    ):
        section = result[name]
        if "passed" in section:
            passed = section["passed"]
        else:
            passed = all(item.get("passed", False) for item in section.values())
        lines.append(name + " passed: " + str(passed))
    (root / "final_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    args = parser.parse_args()
    audit = build_final_audit(args.bundle_root)
    print(json.dumps({"bundle_root": str(args.bundle_root), "passed": audit["passed"]}))
