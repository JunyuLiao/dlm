"""Deterministic small manifests and benchmark-specific scorers."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

from experiments.diffusion_attention_threshold_modeling.math500 import build_math_verifier, symbolic_verify
from dllm.evaluation.ruler.official import load_scorers


RULER_FINAL = Path("results/diffusion_gemma_solattn_vs_blasst_ruler16k/final.jsonl")
RULER_ROOT = Path("/tmp/nemo-gym-ruler")
if not RULER_ROOT.exists():
    RULER_ROOT = Path("/home/exouser/dyh/RULER_c3f5e3b_clean")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _choice(text: str) -> str | None:
    matches = re.findall(r"(?:answer|choice)\s*:\s*(?:\\boxed\s*\{)?\s*([A-D])", text, flags=re.I)
    if not matches:
        matches = re.findall(r"\\boxed\s*\{\s*([A-D])\s*\}", text, flags=re.I)
    return matches[-1].upper() if matches else None


def _code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n?(.*?)```", text, flags=re.S | re.I)
    return (blocks[-1] if blocks else text).strip()


def _longbench_prompt(item: dict[str, Any]) -> str:
    return (
        "Please read the following text and answer the question.\n\n<text>\n" + item["context"] +
        "\n</text>\n\nWhat is the correct answer to this question: " + item["question"] +
        "\nChoices:\n(A) " + item["choice_A"] + "\n(B) " + item["choice_B"] +
        "\n(C) " + item["choice_C"] + "\n(D) " + item["choice_D"] +
        "\n\nThink step by step. The last line must be Answer: \\boxed{A/B/C/D}."
    )


def _stable_take(rows: list[dict[str, Any]], count: int, *, seed: int, key: str) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda row: _hash(f"{seed}|{key}|{row.get('source_id', row.get('id', row.get('index', '')))}"))
    if len(rows) < count:
        raise RuntimeError(f"{key}: expected at least {count} rows, found {len(rows)}")
    return rows[:count]


def prepare(output_dir: Path, *, model_path: str, seed: int = 42) -> dict[str, Any]:
    """Build exactly 80 paired prompts: 50 RULER, 10 each other benchmark."""
    from dllm.models import create_adapter

    tokenizer = create_adapter("diffusion_gemma", model_path, device="cpu", precision="float32").load_tokenizer()
    rows: list[dict[str, Any]] = []
    # Reuse the existing pinned, audited 10-per-task final RULER manifest.
    ruler = [json.loads(line) for line in RULER_FINAL.read_text().splitlines() if line.strip()]
    if len(ruler) != 50:
        raise RuntimeError("expected the prior 50-item RULER final manifest")
    for item in ruler:
        rows.append({
            "id": f"ruler16k/{item['sample_id']}", "benchmark": "ruler16k", "task": item["task"],
            "prompt": item["prompt"], "expected": item["outputs"], "generation_budget": int(item["generation_budget"]),
            "seed": int(item["seed"]), "source_id": item["source_id"], "task_base": item["task_base"],
        })
    # AIME 2024 ships locally through NeMo Skills.  Take a deterministic ten
    # from the 30 official problems, rather than mixing yearly distributions.
    aime_path = Path("/home/exouser/ljy/Skills/nemo_skills/dataset/aime24/test.txt")
    aime = [json.loads(line) for line in aime_path.read_text().splitlines() if line.strip()]
    for item in _stable_take(aime, 10, seed=seed, key="aime24"):
        prompt = (
            "Solve the following AIME problem carefully. Give the final integer from 000 to 999 "
            "on the last line exactly as Answer: \\boxed{NNN}.\n\n" + str(item["problem"])
        )
        rows.append({"id": f"aime24/{item['id']}", "benchmark": "aime24", "task": "aime24", "prompt": prompt,
                     "expected": str(item["expected_answer"]), "generation_budget": 2048,
                     "seed": seed + len(rows), "source_id": item["id"]})
    # LongBench-v2 is the maintained public LongBench interface in the local
    # evaluation stack. Use only its short context band and spread 10 examples
    # over the available domains so the compact run remains feasible on 16K.
    longbench = load_dataset("THUDM/LongBench-v2", split="train")
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in longbench:
        raw = dict(item)
        # Avoid materialising/tokenising the benchmark's multi-million-token
        # examples merely to reject them.  The character guard is deliberately
        # conservative; final tokenizer accounting remains authoritative.
        if raw["length"] == "short" and len(raw["context"]) <= 60_000 and len(tokenizer.encode_prompt(_longbench_prompt(raw), {})) <= 15_000:
            by_domain[str(raw["domain"])].append(raw)
    candidates: list[dict[str, Any]] = []
    for domain in sorted(by_domain):
        candidates += _stable_take(by_domain[domain], 1, seed=seed, key=f"longbench/{domain}")
    used = {str(item["_id"]) for item in candidates}
    pool = [item for items in by_domain.values() for item in items if str(item["_id"]) not in used]
    candidates += _stable_take(pool, 10 - len(candidates), seed=seed + 1, key="longbench/remainder")
    candidates = candidates[:10]
    if len(candidates) != 10:
        raise RuntimeError("could not construct 10 distinct LongBench-v2 examples")
    for item in candidates:
        prompt = _longbench_prompt(item)
        rows.append({"id": f"longbench_v2/{item['_id']}", "benchmark": "longbench_v2", "task": str(item["domain"]),
                     "prompt": prompt, "expected": str(item["answer"]), "generation_budget": 1024,
                     "seed": seed + len(rows), "source_id": str(item["_id"]), "length": item["length"],
                     "difficulty": item["difficulty"], "domain": item["domain"]})
    # Retain testcase columns so the final code score is official pass@1.
    lcb = load_dataset("livecodebench/code_generation_lite", "release_v6", split="test", revision="refs/pr/7")
    parsed = []
    for item in lcb:
        raw = dict(item)
        date = str(raw.get("contest_date", ""))
        if "2024-08" <= date[:7] <= "2025-05":
            parsed.append(raw)
    for item in _stable_take(parsed, 10, seed=seed, key="livecodebench_v6"):
        starter = str(item.get("starter_code") or "")
        suffix = ("\n\nStart with this function header:\n```\n" + starter + "\n```" if starter else "")
        prompt = str(item["question_content"]) + suffix + "\n\nWrite Python code only in a ```python``` block."
        rows.append({"id": f"livecodebench_v6/{item['question_id']}", "benchmark": "livecodebench_v6",
                     "task": str(item.get("difficulty", "unknown")), "prompt": prompt, "expected": None,
                     "generation_budget": 2048, "seed": seed + len(rows), "source_id": str(item["question_id"]),
                     "lcb": {key: item.get(key) for key in ("question_id", "contest_id", "contest_date", "starter_code", "difficulty", "public_test_cases", "private_test_cases", "metadata")}})
    for row in rows:
        row["prompt_hash"] = _hash(row["prompt"])
        row["prompt_tokens"] = len(tokenizer.encode_prompt(row["prompt"], {}))
    if len(rows) != 80 or len({row["id"] for row in rows}) != 80:
        raise AssertionError("manifest must contain 80 unique examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.jsonl"
    payload = "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows)
    if path.exists() and path.read_text() != payload:
        raise RuntimeError("existing manifest differs from deterministic protocol")
    path.write_text(payload)
    audit = {"schema_version": 1, "count": len(rows), "counts": {name: sum(row["benchmark"] == name for row in rows) for name in sorted({row["benchmark"] for row in rows})}, "unique_ids": len({row["id"] for row in rows}) == len(rows), "seed": seed, "longbench_variant": "LongBench-v2 short"}
    (output_dir / "manifest_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def score(row: dict[str, Any], prediction: str, *, ruler_scorers: dict[str, Any] | None = None, verifier: Any | None = None) -> bool | None:
    benchmark = row["benchmark"]
    if benchmark == "ruler16k":
        scorer = (ruler_scorers or load_scorers(RULER_ROOT))[row["task_base"]]
        return bool(float(scorer([prediction.strip()], [[str(value) for value in row["expected"]]])) / 100.0 > 0.5)
    if benchmark == "aime24":
        verifier = verifier or build_math_verifier()
        return bool(symbolic_verify(verifier, str(row["expected"]), prediction)[0] > 0.5)
    if benchmark == "longbench_v2":
        return _choice(prediction) == str(row["expected"]).upper()
    return None


def code_from_prediction(text: str) -> str:
    return _code(text)
