"""Summarize one separate native stop-diagnostic session without exposing output text.

Inputs are the session directory written by native_reuse_stop_diagnostic.py.
Optional primary roots provide read-only parity checks against original smoke
attempt-zero receipts. These diagnostics are never formal quality/timing results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any


ARMS = ("native_dense", "fresh_junyu_T", "M1", "M3")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token_hash(tokens: list[int]) -> str:
    encoded = json.dumps(tokens, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _source(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    configs_path = root / "configs.json"
    configs = _read(configs_path) or {}
    hashes = {"configs.json": _sha(configs_path)} if configs_path.is_file() else {}
    for name in ("gate.json", "failure.json", "plain_native_dense.json",
                 "observed_native_dense.json", "fresh_junyu_T.json", "M1.json", "M3.json"):
        path = root / name
        if path.is_file():
            hashes[name] = _sha(path)
    return configs, hashes


def _quadrants(steps: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {f"stable_{str(s).lower()}_confident_{str(c).lower()}": 0
              for s in (False, True) for c in (False, True)}
    for step in steps:
        stable, confident = step["stable"], step["confident"]
        if not isinstance(stable, bool) or not isinstance(confident, bool):
            raise ValueError("Stop predicates must be booleans")
        counts[f"stable_{str(stable).lower()}_confident_{str(confident).lower()}"] += 1
    return {"denominator_decoder_calls": len(steps), "counts": counts}


def _acceptance(steps: list[dict[str, Any]]) -> dict[str, Any]:
    total_positions = sum(step["canvas_positions"] for step in steps)
    return {
        "denominator_decoder_calls": len(steps),
        "denominator_canvas_positions": total_positions,
        "mean_accepted_positions_per_call": (mean(step["accepted"] for step in steps) if steps else None),
        "mean_renoised_positions_per_call": (mean(step["renoised"] for step in steps) if steps else None),
        "accepted_positions_total": sum(step["accepted"] for step in steps),
        "renoised_positions_total": sum(step["renoised"] for step in steps),
    }


def _summarize_arm(path: Path, *, split_iterations: bool) -> dict[str, Any]:
    item = _read(path)
    if item is None:
        return {"status": "NOT_RUN"}
    if item.get("schema") != "numerical_qk_stop_diagnostic_v1":
        raise ValueError(f"Unexpected diagnostic schema: {path}")
    if item.get("quality_eligible") is not False or item.get("timing_eligible") is not False:
        raise ValueError(f"Diagnostic eligibility labels missing or changed: {path}")
    steps, canvases = item["per_step"], item["per_canvas"]
    if not steps or len(steps) != item["actual_decoder_calls"]:
        raise ValueError(f"Decoder-call count mismatch: {path}")
    if sum(canvas["actual_decoder_calls"] for canvas in canvases) != len(steps):
        raise ValueError(f"Per-canvas call count mismatch: {path}")
    for step in steps:
        if step["accepted"] + step["renoised"] != step["canvas_positions"]:
            raise ValueError(f"Acceptance partition mismatch: {path}")
        if (step["stable"] and step["confident"]) != step["native_criterion_stop"]:
            raise ValueError(f"Stop conjunction mismatch: {path}")
    data: dict[str, Any] = {
        "status": "DIAGNOSTIC_ONLY", "id": item["id"], "seed": item["seed"],
        "condition": item["condition"], "fingerprint": item["fingerprint"],
        "prompt_sha256": item["prompt_hash"], "raw_file_sha256": _sha(path),
        "raw_completion_sha256": hashlib.sha256(item["raw_completion"].encode("utf-8")).hexdigest(),
        "completion_token_sha256": _token_hash(item["completion_tokens"]),
        "completion_token_count": len(item["completion_tokens"]),
        "termination_reason": item["termination_reason"],
        "actual_decoder_calls": len(steps), "stop_predicates": _quadrants(steps),
        "acceptance": _acceptance(steps),
        "canvases": [{
            "canvas_index": canvas["canvas_index"],
            "actual_decoder_calls": canvas["actual_decoder_calls"],
            "iteration_cap_final_call": canvas["iteration_cap_final_call"],
            "joint_stop_final_call": canvas["native_stop_final_call"],
            "stable_final_call": canvas["native_stable_final_call"],
            "confident_final_call": canvas["native_confident_final_call"],
            "request_eos_boundary": canvas["request_eos_boundary"],
            "request_output_cap_boundary": canvas["request_output_cap_boundary"],
        } for canvas in canvases],
    }
    if split_iterations:
        early = [step for step in steps if step["decoder_call"] <= 8]
        late = [step for step in steps if step["decoder_call"] >= 9]
        data["iteration_windows"] = {
            "early_zero_based_0_to_7": {"stop_predicates": _quadrants(early),
                                         "acceptance": _acceptance(early)},
            "late_zero_based_8_plus": {"stop_predicates": _quadrants(late),
                                        "acceptance": _acceptance(late)},
        }
    return data


def _primary_candidates(root: Path, condition: str, id_: str, seed: int) -> list[Path]:
    digest = hashlib.sha256(id_.encode()).hexdigest()[:20]
    name = f"{digest}.attempt0.json"
    direct = root / "smoke" / condition / f"seed_{seed}" / name
    if direct.is_file():
        return [direct]
    # A primary root may be the smoke directory itself or one level above it.
    return sorted(path for path in root.rglob(name)
                  if path.parent.name == f"seed_{seed}" and path.parent.parent.name == condition)


def _primary_parity(arm: dict[str, Any], diagnostic_path: Path,
                    primary_roots: list[Path]) -> list[dict[str, Any]]:
    if arm["status"] == "NOT_RUN":
        return []
    diagnostic = _read(diagnostic_path)
    if diagnostic is None:
        raise FileNotFoundError(diagnostic_path)
    result = []
    for root in primary_roots:
        candidates = _primary_candidates(root, arm["condition"], arm["id"], arm["seed"])
        if not candidates:
            result.append({"primary_root": str(root), "status": "NOT_RUN"})
            continue
        for path in candidates:
            original = _read(path)
            if original is None:
                continue
            if (original.get("id"), original.get("seed"), original.get("condition")) != (
                    arm["id"], arm["seed"], arm["condition"]):
                raise ValueError(f"Primary receipt identity mismatch: {path}")
            token_hash = _token_hash(original["completion_tokens"])
            original_calls = original["total_decoder_calls"]
            result.append({
                "primary_root": str(root), "primary_receipt": str(path),
                "primary_raw_file_sha256": _sha(path),
                "primary_completion_token_sha256": token_hash,
                "primary_total_decoder_calls": original_calls,
                "same_id_seed_condition": True,
                "same_prompt_sha256": original.get("prompt_hash") == arm["prompt_sha256"],
                "exact_tokens_match": original["completion_tokens"] == diagnostic["completion_tokens"],
                "decoder_calls_match": original_calls == arm["actual_decoder_calls"],
                "status": "DIAGNOSTIC_PARITY_ONLY",
            })
    return result


def summarize(root: Path, primary_roots: list[Path], *, split_iterations: bool) -> dict[str, Any]:
    root = root.resolve()
    configs, raw_hashes = _source(root)
    gate = _read(root / "gate.json")
    failure = _read(root / "failure.json")
    source_hashes = {name: config.get("source_hashes") for name, config in configs.items()
                     if isinstance(config, dict)}
    result: dict[str, Any] = {
        "schema": "numerical_qk_stop_summary_v1", "root": str(root),
        "scope": "separate diagnostic only; no quality or formal timing replacement",
        "gate": (dict(status="NOT_RUN") if gate is None else {
            "status": "PASS" if gate.get("passed") else "FAIL",
            "id": gate.get("id"), "tokens_match": gate.get("tokens_match"),
            "decoder_counts_match": gate.get("decoder_counts_match"),
            "same_model_instance": gate.get("same_model_instance"),
        }),
        "failure": (None if failure is None else {
            "condition": failure.get("condition"), "error_type": failure.get("error_type"),
            "note": "Session failed; absent arms are NOT_RUN",
        }),
        "raw_input_file_sha256": raw_hashes,
        "source_hashes_by_condition": source_hashes,
        "arms": {},
    }
    seen_identity: tuple[str, int] | None = None
    for condition in ARMS:
        filename = "observed_native_dense.json" if condition == "native_dense" else f"{condition}.json"
        arm = _summarize_arm(root / filename, split_iterations=split_iterations)
        if arm["status"] != "NOT_RUN" and arm["condition"] != condition:
            raise ValueError(f"Diagnostic condition mismatch in {filename}")
        if arm["status"] != "NOT_RUN":
            identity = (arm["id"], arm["seed"])
            if seen_identity is not None and identity != seen_identity:
                raise ValueError(f"Diagnostic ID/seed differs across arms: {filename}")
            seen_identity = identity
            if gate is not None and arm["id"] != gate.get("id"):
                raise ValueError(f"Diagnostic ID differs from gate: {filename}")
        arm["primary_parity"] = _primary_parity(arm, root / filename, primary_roots)
        result["arms"][condition] = arm
    return result


def _markdown(result: dict[str, Any]) -> str:
    lines = ["# Native stop diagnostic summary", "",
             "Separate diagnostic only; these runs do not replace formal quality or timing receipts.", "",
             f"Observer gate: **{result['gate']['status']}**.", ""]
    if result["failure"]:
        lines.append(f"Session failure: {result['failure']['condition']} "
                     f"({result['failure']['error_type']}); absent arms are NOT_RUN.")
        lines.append("")
    lines.extend(["| Arm | Status | Calls | Stable/Confident FF, FT, TF, TT | Accepted mean | Renoised mean | Termination |",
                  "| --- | --- | ---: | --- | ---: | ---: | --- |"])
    for name, arm in result["arms"].items():
        if arm["status"] == "NOT_RUN":
            lines.append(f"| {name} | NOT_RUN | — | — | — | — | — |")
            continue
        counts = arm["stop_predicates"]["counts"]
        quad = ", ".join(str(counts[f"stable_{s}_confident_{c}"])
                         for s, c in (("false", "false"), ("false", "true"),
                                      ("true", "false"), ("true", "true")))
        acceptance = arm["acceptance"]
        lines.append(f"| {name} | diagnostic | {arm['actual_decoder_calls']} | {quad} "
                     f"(n={arm['stop_predicates']['denominator_decoder_calls']}) | "
                     f"{acceptance['mean_accepted_positions_per_call']:.2f} | "
                     f"{acceptance['mean_renoised_positions_per_call']:.2f} | "
                     f"{arm['termination_reason']} |")
    lines.append("")
    for name, arm in result["arms"].items():
        if arm["status"] == "NOT_RUN":
            continue
        canvases = arm["canvases"]
        calls = ", ".join(str(canvas["actual_decoder_calls"]) for canvas in canvases)
        caps = sum(bool(canvas["iteration_cap_final_call"]) for canvas in canvases)
        joint = sum(bool(canvas["joint_stop_final_call"]) for canvas in canvases)
        lines.append(f"{name}: {len(canvases)} canvases; calls per canvas [{calls}]; "
                     f"iteration-cap finals {caps}; joint-stop finals {joint}.")
        for parity in arm["primary_parity"]:
            if parity["status"] == "NOT_RUN":
                lines.append(f"  Primary receipt NOT_RUN under {parity['primary_root']}.")
            else:
                lines.append(f"  Primary parity: tokens={parity['exact_tokens_match']}, "
                             f"calls={parity['decoder_calls_match']}, "
                             f"prompt={parity['same_prompt_sha256']}.")
    lines.append("")
    lines.append("JSON contains input-file, source, and output hashes; no prompt, answer, or token content.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="One stop_diagnostic/<session> directory")
    parser.add_argument("--primary-root", type=Path, nargs="+", action="append", default=[],
                        help="Optional original smoke receipt roots; repeatable")
    parser.add_argument("--split-iterations", action="store_true",
                        help="Also count zero-based early 0-7 vs late 8+ decoder calls per canvas")
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        parser.error("--root must be an existing diagnostic session directory")
    primary_roots = [path.resolve() for group in args.primary_root for path in group]
    for path in primary_roots:
        if not path.is_dir():
            parser.error(f"Primary root does not exist: {path}")
    result = summarize(args.root, primary_roots, split_iterations=args.split_iterations)
    json_path = args.root / "stop_summary.json"
    markdown_path = args.root / "stop_summary.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path),
                      "gate": result["gate"]["status"],
                      "arms": {name: arm["status"] for name, arm in result["arms"].items()}}))


if __name__ == "__main__":
    main()
