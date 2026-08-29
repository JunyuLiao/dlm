"""Deterministic, disjoint RULER16K calibration/final manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from dllm.evaluation.ruler.io import sha256_file, write_json, write_jsonl
from dllm.evaluation.ruler.official import DEFAULT_TASKS, RULER_COMMIT, verify_checkout, load_configuration


def _stable_key(seed: int, *parts: Any) -> str:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _source_id(row: Mapping[str, Any], fallback: int) -> str:
    for key in ("source_id", "official_index", "index", "id", "sample_id", "request_id"):
        if row.get(key) is not None:
            return str(row[key])
    return str(fallback)


def _prompt(row: Mapping[str, Any]) -> str:
    if row.get("prompt") is not None:
        return str(row["prompt"])
    return str(row.get("input", "")) + str(row.get("answer_prefix", ""))


def _generation_budget(row: Mapping[str, Any], budgets: Mapping[str, int] | None, task: str) -> int:
    value = row.get("tokens_to_generate", row.get("generation_budget"))
    if value is None and budgets is not None:
        value = budgets.get(task)
    if value is None:
        raise ValueError(f"no generation budget available for task {task}")
    value = int(value)
    if value <= 0:
        raise ValueError("generation budgets must be positive")
    return value


def _token_count(row: Mapping[str, Any], prompt: str, tokenizer: Callable[[str], Iterable[int]] | None) -> int:
    if tokenizer is not None:
        value = tokenizer(prompt)
        try:
            return len(value)
        except TypeError:
            return int(value)
    for key in ("token_count", "actual_prompt_length", "prompt_tokens"):
        if row.get(key) is not None:
            value = row[key]
            return len(value) if isinstance(value, (list, tuple)) else int(value)
    raise ValueError("a tokenizer or token_count is required to build a manifest")


def build_disjoint_manifests(
    rows_by_task: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    tokenizer: Callable[[str], Iterable[int]] | None = None,
    generation_budgets: Mapping[str, int] | None = None,
    calibration_per_task: int = 2,
    final_per_task: int = 10,
    context_length: int = 16_384,
    split_seed: int = 42,
    inference_seed_base: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Shuffle within each task and take disjoint calibration then final rows.

    The function accepts ordinary dictionaries so tests can construct tiny
    synthetic suites.  Input order never affects the result: sorting uses a
    SHA-256 key over the seed, task, source ID, and prompt hash.
    """

    if calibration_per_task < 0 or final_per_task < 0:
        raise ValueError("split sizes must be non-negative")
    if not rows_by_task:
        raise ValueError("at least one task is required")
    calibration: list[dict[str, Any]] = []
    final: list[dict[str, Any]] = []
    seen_source: set[tuple[str, str]] = set()
    seen_prompt: set[str] = set()
    ordered_tasks = [task for task in DEFAULT_TASKS if task in rows_by_task]
    ordered_tasks.extend(sorted(task for task in rows_by_task if task not in DEFAULT_TASKS))
    for task in ordered_tasks:
        raw_rows = [dict(row) for row in rows_by_task[task]]
        decorated = []
        for index, raw in enumerate(raw_rows):
            prompt = _prompt(raw)
            if not prompt:
                raise ValueError(f"task {task} contains an empty prompt")
            source_raw = _source_id(raw, index)
            # RULER's per-task ``index`` values are not globally unique (for
            # example VT and FWE both legitimately start at zero).  Expose a
            # task-qualified source ID so the disjointness audit can enforce
            # the literal "no source ID overlap" contract across the whole
            # calibration/final pair while retaining the local ID for
            # provenance.
            source = f"{task}:{source_raw}"
            digest = prompt_hash(prompt)
            decorated.append((_stable_key(split_seed, task, source, digest), source, digest, raw, prompt))
        decorated.sort(key=lambda item: item[0])
        need = calibration_per_task + final_per_task
        if len(decorated) < need:
            raise ValueError(f"task {task} has {len(decorated)} rows; {need} required")
        for ordinal, (_key, source, digest, raw, prompt) in enumerate(decorated[:need]):
            source_identity = (str(task), source)
            if source_identity in seen_source or digest in seen_prompt:
                raise ValueError(f"duplicate source ID or prompt hash in task {task}: {source}")
            seen_source.add(source_identity); seen_prompt.add(digest)
            split = "calibration" if ordinal < calibration_per_task else "final"
            row = {
                "task": task,
                "task_base": str(raw.get("task_base", task)),
                "source_id": source,
                "source_id_local": source_raw,
                "prompt": prompt,
                "prompt_hash": digest,
                "token_count": _token_count(raw, prompt, tokenizer),
                "context_length": int(context_length),
                "generation_budget": _generation_budget(raw, generation_budgets, task),
                "seed": int(inference_seed_base * 1_000_000 + len(calibration) + len(final)),
                "official_index": raw.get("official_index", raw.get("index")),
                "answer_prefix": raw.get("answer_prefix", ""),
                "outputs": list(raw.get("outputs", [])),
            }
            row["sample_id"] = f"ruler16k_{split}_{task}_{ordinal:04d}"
            (calibration if split == "calibration" else final).append(row)

    if not calibration or not final:
        raise ValueError("both calibration and final sets must be non-empty")
    audit = {
        "split_seed": int(split_seed),
        "context_length": int(context_length),
        "tasks": ordered_tasks,
        "calibration_count": len(calibration),
        "final_count": len(final),
        "calibration_task_counts": {task: sum(row["task"] == task for row in calibration) for task in ordered_tasks},
        "final_task_counts": {task: sum(row["task"] == task for row in final) for task in ordered_tasks},
        "source_id_overlap": [
            {"task": task, "source_id": source}
            for task, source in sorted(
                {(row["task"], row["source_id"]) for row in calibration}
                & {(row["task"], row["source_id"]) for row in final}
            )
        ],
        "prompt_hash_overlap": sorted({row["prompt_hash"] for row in calibration} & {row["prompt_hash"] for row in final}),
    }
    audit["disjoint"] = not audit["source_id_overlap"] and not audit["prompt_hash_overlap"]
    if not audit["disjoint"]:
        raise AssertionError("calibration and final manifests overlap")
    return calibration, final, audit


def _read_task_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_manifests_from_ruler(
    *,
    raw_root: str | Path,
    ruler_root: str | Path,
    tokenizer: Callable[[str], Iterable[int]],
    output_dir: str | Path,
    generation_budgets: Mapping[str, int] | None = None,
    split_seed: int = 42,
    context_length: int = 16_384,
    calibration_per_task: int = 2,
    final_per_task: int = 10,
    tokenizer_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare and write the canonical pair from pinned RULER raw shards."""

    raw_root, ruler_root, output_dir = Path(raw_root).resolve(), Path(ruler_root).resolve(), Path(output_dir).resolve()
    provenance = verify_checkout(ruler_root)
    if provenance["commit"] != RULER_COMMIT:
        raise ValueError("RULER checkout is not pinned")
    customized, base = load_configuration(ruler_root)
    rows_by_task = {}
    resolved_budgets = dict(generation_budgets or {})
    source_hashes = {}
    for task in DEFAULT_TASKS:
        source = raw_root / task / "validation.jsonl"
        if not source.exists():
            raise FileNotFoundError(source)
        rows_by_task[task] = _read_task_rows(source)
        source_hashes[task] = sha256_file(source)
        task_base = str(customized[task]["task"])
        resolved_budgets.setdefault(task, int(base[task_base]["tokens_to_generate"]))
        for row in rows_by_task[task]:
            row.setdefault("task_base", task_base)
    calibration, final, audit = build_disjoint_manifests(
        rows_by_task,
        tokenizer=tokenizer,
        generation_budgets=resolved_budgets,
        calibration_per_task=calibration_per_task,
        final_per_task=final_per_task,
        context_length=context_length,
        split_seed=split_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cal_path, final_path = output_dir / "calibration.jsonl", output_dir / "final.jsonl"
    write_jsonl(cal_path, calibration); write_jsonl(final_path, final)
    manifest = {
        "schema_version": 1,
        "study": "diffusion_gemma_solattn_vs_blasst_ruler16k",
        "ruler": provenance,
        "context_length": context_length,
        "split_seed": split_seed,
        "tasks": list(DEFAULT_TASKS),
        "tokenizer": dict(tokenizer_provenance or {}),
        "calibration": {"path": str(cal_path), "sha256": sha256_file(cal_path), "count": len(calibration)},
        "final": {"path": str(final_path), "sha256": sha256_file(final_path), "count": len(final)},
        "source_raw_root": str(raw_root),
        "source_sha256": source_hashes,
        "audit": audit,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


__all__ = [
    "build_disjoint_manifests",
    "prepare_manifests_from_ruler",
    "prompt_hash",
]
