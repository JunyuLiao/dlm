"""Pinned NVIDIA RULER generation, validation, and official scoring."""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml
from dllm.models import create_adapter

from .io import read_jsonl, sha256_bytes, sha256_file, write_json, write_jsonl


RULER_REPOSITORY = "https://github.com/NVIDIA/RULER.git"
RULER_COMMIT = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
DEFAULT_TASKS = (
    "niah_multikey_1",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "fwe",
)
PAPER_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)
SHORT_CONTEXT_TASKS = ("niah_multikey_2", "fwe")
SOURCE_FILES = (
    "scripts/synthetic.yaml",
    "scripts/data/prepare.py",
    "scripts/data/tokenizer.py",
    "scripts/data/synthetic/constants.py",
    "scripts/data/synthetic/niah.py",
    "scripts/data/synthetic/variable_tracking.py",
    "scripts/data/synthetic/freq_words_extraction.py",
    "scripts/eval/synthetic/constants.py",
    "scripts/eval/evaluate.py",
)


def balanced_counts(tasks: Sequence[str], total: int) -> dict[str, int]:
    if total <= 0:
        raise ValueError("num_samples must be positive")
    if not tasks:
        raise ValueError("at least one RULER task is required")
    quotient, remainder = divmod(total, len(tasks))
    return {
        task: quotient + int(index < remainder)
        for index, task in enumerate(tasks)
    }


def verify_checkout(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    if not (root / ".git").exists():
        raise FileNotFoundError(f"{root} is not an NVIDIA RULER checkout")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if commit != RULER_COMMIT:
        raise ValueError(f"RULER commit {commit} does not match pinned {RULER_COMMIT}")
    hashes = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"missing RULER source: {relative}")
        upstream = subprocess.check_output(
            ["git", "show", f"{RULER_COMMIT}:{relative}"], cwd=root
        )
        if path.read_bytes() != upstream:
            raise ValueError(f"modified RULER source: {relative}")
        hashes[relative] = sha256_file(path)
    return {
        "repository": RULER_REPOSITORY,
        "commit": commit,
        "source_sha256": hashes,
    }


def load_configuration(root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(root)
    customized = yaml.safe_load(
        (root / "scripts/synthetic.yaml").read_text(encoding="utf-8")
    )
    path = root / "scripts/data/synthetic/constants.py"
    spec = importlib.util.spec_from_file_location("dllm_ruler_data_constants", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return customized, module.TASKS


def load_scorers(root: str | Path) -> dict[str, Callable]:
    path = Path(root) / "scripts/eval/synthetic/constants.py"
    spec = importlib.util.spec_from_file_location("dllm_ruler_eval_constants", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {name: config["metric_fn"] for name, config in module.TASKS.items()}


def _subprocess_environment(
    dependency_path: str | None, nltk_data: str | None
) -> dict[str, str]:
    environment = dict(os.environ)
    # RULER's prepare.py historically invokes task generators through the
    # literal `python` command. Keep that subprocess on the same interpreter
    # and ABI as the caller (important for local CUDA/NumPy environments).
    interpreter_dir = str(Path(sys.executable).resolve().parent)
    environment["PATH"] = os.pathsep.join(
        value for value in (interpreter_dir, environment.get("PATH", "")) if value
    )
    if dependency_path:
        current = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (dependency_path, current) if value
        )
    if nltk_data:
        environment["NLTK_DATA"] = nltk_data
    environment["PYTHONHASHSEED"] = "0"
    return environment


def _oversample(task: str, desired: int) -> int:
    if task in ("niah_multivalue", "niah_multiquery"):
        return max(desired * 20, 20)
    if task.startswith("niah_"):
        return max(desired * 6, 6)
    if task == "fwe":
        # FWE lengths vary substantially across tokenizers. DiffusionGemma in
        # particular needs a wider deterministic candidate pool to obtain 20
        # prompts within the declared long-context tolerance.
        return max(desired * 8, 8)
    return desired + max(8, math.ceil(desired * 0.5))


def prepare_manifest(
    *,
    ruler_root: str | Path,
    tokenizer_path: str,
    model_adapter: str,
    context_length: int,
    num_samples: int,
    output_dir: str | Path,
    seed: int = 42,
    tasks: Sequence[str] | None = None,
    revision: str | None = None,
    dependency_path: str | None = None,
    nltk_data: str | None = None,
    tolerance_fraction: float = 0.06,
    tolerance_min_tokens: int = 32,
    generation_extra: Mapping[str, Any] | None = None,
    length_mode: str = "prompt",
) -> Path:
    """Generate a deterministic manifest containing exactly ``num_samples``."""
    ruler_root = Path(ruler_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = verify_checkout(ruler_root)
    if length_mode not in ("prompt", "total"):
        raise ValueError("length_mode must be 'prompt' or 'total'")
    selected_tasks = tuple(
        tasks or (SHORT_CONTEXT_TASKS if context_length == 512 else DEFAULT_TASKS)
    )
    counts = balanced_counts(selected_tasks, num_samples)
    adapter = create_adapter(
        model_adapter,
        tokenizer_path,
        device="cpu",
        precision="float32",
        revision=revision,
    ).load_tokenizer()
    generation_extra = dict(generation_extra or {})
    prompt_configuration = adapter.prompt_configuration(generation_extra)
    customized, base = load_configuration(ruler_root)
    tolerance = max(tolerance_min_tokens, math.ceil(context_length * tolerance_fraction))
    selected: dict[str, list[dict[str, Any]]] = {}
    commands: list[list[str]] = []

    for task_index, task in enumerate(selected_tasks):
        desired = counts[task]
        if desired == 0:
            selected[task] = []
            continue
        config = customized[task]
        task_base = str(config["task"])
        tokens_to_generate = int(base[task_base]["tokens_to_generate"])
        # Paper-style RULER defines max_seq_length as prompt + completion. Its
        # generators already target that limit, so only modest oversampling is
        # needed to absorb tokenizer/chat-template differences. The historical
        # prompt-length mode needs a larger pool because it filters after adding
        # the model's prompt template.
        candidates = desired if length_mode == "total" else _oversample(task, desired)
        generator_seed = seed * 100_000 + context_length + task_index * 1_000
        shard_root = output_dir / "official_raw" / str(context_length)
        shard = shard_root / task / "validation.jsonl"
        command = [
            sys.executable,
            str(ruler_root / "scripts/data/prepare.py"),
            "--save_dir", str(shard_root),
            "--benchmark", "synthetic",
            "--task", task,
            "--tokenizer_path", tokenizer_path,
            "--tokenizer_type", "hf",
            "--max_seq_length",
            str(
                context_length
                if length_mode == "total"
                else context_length + tokens_to_generate
            ),
            "--model_template_type", "base",
            "--num_samples", str(candidates),
            "--random_seed", str(generator_seed),
            "--subset", "validation",
        ]
        commands.append(command)
        if not shard.exists() or len(read_jsonl(shard)) != candidates:
            completed = subprocess.run(
                command,
                cwd=ruler_root,
                env=_subprocess_environment(dependency_path, nltk_data),
                text=True,
                capture_output=True,
                timeout=900,
                check=False,
            )
            if completed.returncode or not shard.exists() or len(read_jsonl(shard)) != candidates:
                raise RuntimeError(
                    "official RULER generation failed\n"
                    + " ".join(command)
                    + f"\nstdout:\n{completed.stdout[-4000:]}"
                    + f"\nstderr:\n{completed.stderr[-4000:]}"
                )
        valid = []
        for row_index, raw in enumerate(read_jsonl(shard)):
            prompt = str(raw["input"]) + str(raw.get("answer_prefix", ""))
            actual = len(adapter.encode_prompt(prompt, generation_extra))
            measured_length = (
                actual + tokens_to_generate if length_mode == "total" else actual
            )
            # Paper-style RULER keeps the generator's complete sample set; it
            # treats max_seq_length as an upper budget rather than filtering
            # examples into a narrow post-template length band. Retain that
            # distribution in total mode. Historical prompt mode still needs
            # the post-tokenization filter to construct fixed-prompt sweeps.
            if (
                length_mode == "prompt"
                and abs(measured_length - context_length) > tolerance
            ):
                continue
            valid.append(
                {
                    "sample_id": f"ruler_{context_length}_{task}_{row_index:04d}",
                    "task": task,
                    "task_base": task_base,
                    "target_length": context_length,
                    "actual_prompt_length": actual,
                    "actual_total_length": actual + tokens_to_generate,
                    "generator_seed": generator_seed,
                    "inference_seed": seed * 1_000_000 + context_length * 100 + task_index * 10_000 + row_index,
                    "prompt": prompt,
                    "prompt_sha256": sha256_bytes(prompt.encode()),
                    "outputs": [str(value) for value in raw["outputs"]],
                    "tokens_to_generate": tokens_to_generate,
                    "official_index": raw.get("index"),
                }
            )
        if len(valid) < desired:
            raise RuntimeError(
                f"RULER {task} yielded {len(valid)} valid prompts; {desired} required"
            )
        selected[task] = valid[:desired]

    records = []
    for index in range(max(counts.values())):
        for task in selected_tasks:
            if index < len(selected[task]):
                records.append(selected[task][index])
    if len(records) != num_samples:
        raise AssertionError(f"prepared {len(records)} records, expected {num_samples}")
    records_path = output_dir / "samples.jsonl"
    write_jsonl(records_path, records)
    manifest = {
        "schema_version": 3,
        "ruler": provenance,
        "tokenizer_path": tokenizer_path,
        "model_adapter": model_adapter,
        "prompt_configuration": prompt_configuration,
        "tokenizer_revision": revision,
        "context_length": context_length,
        "length_mode": length_mode,
        "requested_num_samples": num_samples,
        "actual_num_samples": len(records),
        "seed": seed,
        "tasks": list(selected_tasks),
        "task_counts": counts,
        "length_tolerance": {
            "fraction": tolerance_fraction,
            "minimum_tokens": tolerance_min_tokens,
        },
        "generator_commands": commands,
        "samples": {"path": str(records_path), "sha256": sha256_file(records_path)},
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def postprocess_prediction(prediction: str) -> str:
    import re

    return re.sub(r"[\x00-\x1f]", "\n", prediction.strip()).strip()


def score_predictions(
    rows: list[Mapping[str, Any]], ruler_root: str | Path
) -> tuple[dict[str, float], float]:
    scorers = load_scorers(ruler_root)
    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task"]), []).append(row)
    scores = {}
    weighted = 0.0
    for task, subset in by_task.items():
        bases = {str(row["task_base"]) for row in subset}
        if len(bases) != 1:
            raise ValueError(f"task {task} mixes RULER metric functions")
        scorer = scorers[next(iter(bases))]
        value = float(
            scorer(
                [postprocess_prediction(str(row["prediction"])) for row in subset],
                [[str(answer) for answer in row["outputs"]] for row in subset],
            )
        ) / 100.0
        scores[task] = value
        weighted += value * len(subset)
    return scores, weighted / len(rows)
