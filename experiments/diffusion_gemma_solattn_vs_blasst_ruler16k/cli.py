"""CLI for the isolated DiffusionGemma Sol-Attn versus BLASST study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_MODEL_REVISION,
    DEFAULT_TASKS,
    TARGET_SPARSITIES,
    ExperimentConfig,
    analytic_beta,
)
from dllm.evaluation.ruler.io import sha256_json
from .dataset import build_disjoint_manifests, prepare_manifests_from_ruler
from .runner import calibrate_from_traces, run_calibration_pass, run_one_condition, run_sweep


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="create disjoint 2-per-task calibration and 10-per-task final manifests")
    prepare.add_argument("--raw-root", type=Path, required=True)
    prepare.add_argument("--ruler-root", type=Path, required=True)
    prepare.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    prepare.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    prepare.add_argument("--output-dir", type=Path, default=Path("results/diffusion_gemma_solattn_vs_blasst_ruler16k"))
    prepare.add_argument("--split-seed", type=int, default=42)
    prepare.add_argument("--model-adapter", default="diffusion_gemma")
    prepare.add_argument("--calibration-per-task", type=int, default=2)
    prepare.add_argument("--final-per-task", type=int, default=10)

    calibrate = sub.add_parser("calibrate-blasst", help="collect dense margins or fit a policy from saved traces")
    calibrate.add_argument("--trace-path", type=Path)
    calibrate.add_argument("--study-manifest", type=Path)
    calibrate.add_argument("--adapter", default="diffusion_gemma")
    calibrate.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    calibrate.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    calibrate.add_argument("--ruler-root", type=Path)
    calibrate.add_argument("--output-dir", type=Path, required=True)
    calibrate.add_argument("--device", default="cuda")
    calibrate.add_argument("--precision", default="bfloat16")

    run = sub.add_parser("run", help="run one or all nine dense/Sol-Attn/BLASST conditions")
    run.add_argument("--study-manifest", type=Path, required=True)
    run.add_argument("--adapter", default="diffusion_gemma")
    run.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    run.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    run.add_argument("--ruler-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, default=Path("results/diffusion_gemma_solattn_vs_blasst_ruler16k"))
    run.add_argument("--threshold-policy", type=Path)
    run.add_argument("--conditions", nargs="*")
    run.add_argument("--device", default="cuda")
    run.add_argument("--precision", default="bfloat16")

    report = sub.add_parser("report", help="audit completed shards and regenerate canonical tables/plots")
    report.add_argument("--output-dir", type=Path, default=Path("results/diffusion_gemma_solattn_vs_blasst_ruler16k"))
    report.add_argument("--ruler-root", type=Path)
    return root


def _load_policy(path: Path | None) -> dict:
    if path is None:
        default = Path("results/diffusion_gemma_solattn_vs_blasst_ruler16k/calibration/blasst_policy.json")
        path = default if default.exists() else None
    if path is None:
        raise ValueError("--threshold-policy is required for BLASST conditions")
    policy = json.loads(path.read_text(encoding="utf-8"))
    embedded = policy.get("policy_sha256")
    if embedded:
        content = dict(policy)
        content.pop("policy_sha256", None)
        if str(embedded) != sha256_json(content):
            raise ValueError("BLASST threshold policy content digest mismatch")
    policy["policy_path"] = str(path.resolve())
    return policy


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        from dllm.models import create_adapter
        adapter = create_adapter(args.model_adapter, args.model_path, device="cpu", precision="float32", revision=args.revision).load_tokenizer()
        result = prepare_manifests_from_ruler(
            raw_root=args.raw_root,
            ruler_root=args.ruler_root,
            tokenizer=lambda prompt: adapter.encode_prompt(prompt, {}),
            output_dir=args.output_dir,
            split_seed=args.split_seed,
            calibration_per_task=args.calibration_per_task,
            final_per_task=args.final_per_task,
            tokenizer_provenance={
                "adapter": args.model_adapter,
                "model_path": str(Path(args.model_path).resolve()),
                "revision": args.revision,
            },
        )
    elif args.command == "calibrate-blasst":
        if args.trace_path:
            result = calibrate_from_traces(args.trace_path, args.output_dir / "blasst_policy.json")
        else:
            missing = [name for name in ("study_manifest", "model_path", "ruler_root") if getattr(args, name) is None]
            if missing:
                raise ValueError("calibrate-blasst requires --trace-path or " + ", ".join("--" + name.replace("_", "-") for name in missing))
            result = run_calibration_pass(
                study_manifest_path=args.study_manifest,
                adapter=args.adapter,
                model_path=args.model_path,
                model_revision=args.revision,
                output_dir=args.output_dir,
                ruler_root=args.ruler_root,
                device=args.device,
                precision=args.precision,
            )
    elif args.command == "run":
        policy = None
        if args.threshold_policy:
            policy = _load_policy(args.threshold_policy)
        elif not args.conditions:
            policy = _load_policy(None)
        from .config import condition_name
        from .config import canonical_conditions
        if policy is not None:
            local = {float(key): value for key, value in policy["lambda_local"].items()}
            global_ = {float(key): value for key, value in policy["lambda_global"].items()}
            all_conditions = {item.name: item for item in canonical_conditions(local_lambdas=local, global_lambdas=global_)}
        else:
            all_conditions = {
                "dense": ExperimentConfig(method="dense"),
                **{ExperimentConfig(method="sol_gaussian", target_sparsity=target, beta=analytic_beta(target)).name: ExperimentConfig(method="sol_gaussian", target_sparsity=target, beta=analytic_beta(target)) for target in TARGET_SPARSITIES},
            }
        if args.conditions and any(name.startswith("blasst_calibrated") for name in args.conditions) and policy is None:
            raise ValueError("BLASST conditions require --threshold-policy")
        selected = list(all_conditions.values()) if not args.conditions else [all_conditions[name] for name in args.conditions]
        results = []
        for condition in selected:
            results.append(run_one_condition(
                study_manifest_path=args.study_manifest,
                adapter=args.adapter,
                model_path=args.model_path,
                model_revision=args.revision,
                output_dir=args.output_dir / condition.name,
                condition=condition,
                ruler_root=args.ruler_root,
                threshold_policy=policy,
                device=args.device,
                precision=args.precision,
            ))
        result = {"study": "diffusion_gemma_solattn_vs_blasst_ruler16k", "conditions": results}
        (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    else:
        from .report import build_report
        result = build_report(args.output_dir, args.ruler_root)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
