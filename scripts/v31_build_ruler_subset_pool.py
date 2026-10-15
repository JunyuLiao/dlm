"""Build a sanitized v31-style RULER pool from the pinned generator samples.

The source samples stay in a private directory.  The resulting manifest contains
token ids needed by the runner; do not publish that directory or its gold file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v31_official as op  # noqa: E402


DATASET = "ruler128k_v33ofc_subset"
TASKS = tuple(op.RULER_TASKS)
OPENER = op.RESPONSE_OPENER["diffusion_gemma"]
MODEL_REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"


def sha(value: bytes | str) -> str:
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--revision", default=MODEL_REVISION)
    p.add_argument("--max-model-len", type=int, default=136192)
    args = p.parse_args()

    samples = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    if len(samples) != 104:
        raise ValueError(f"expected 104 samples, got {len(samples)}")
    counts = {task: sum(str(r["task"]) == task for r in samples) for task in TASKS}
    if counts != {task: 8 for task in TASKS}:
        raise ValueError(f"expected eight samples per task, got {counts}")

    from dllm.models import create_adapter

    adapter = create_adapter("diffusion_gemma", args.tokenizer_path, device="cpu", precision="float32",
                             revision=args.revision).load_tokenizer()
    rows, gold = [], {}
    for i, sample in enumerate(samples):
        task = str(sample["task"])
        sample_id = f"{DATASET}/{task}/p{int(i):04d}"
        ids, user_count = op.ruler_prompt_ids(adapter, sample["input"], sample["answer_prefix"], OPENER)
        budget = int(sample["tokens_to_generate"])
        if len(ids) + budget > args.max_model_len:
            raise ValueError(f"row {i} exceeds max-model-len")
        row = {
            "answer_prefix": str(sample["answer_prefix"]),
            "benchmark": DATASET,
            "generation_budget": budget,
            "id": sample_id,
            "index": i,
            "input_token_count": user_count,
            "official_index": sample.get("official_index"),
            "prefilled": True,
            "prompt": str(sample["input"]),
            "prompt_hash": sha(str(sample["input"])),
            "prompt_tokens": [int(x) for x in ids],
            "prompt_token_count": len(ids),
            "prefill_token_count": len(ids) - user_count,
            "response_prefill": OPENER + str(sample["answer_prefix"]),
            "source_id": str(sample["sample_id"]),
            "task": task,
            "task_base": str(sample["task_base"]),
            "thinking": False,
            "target_length": 131072,
        }
        rows.append(row)
        gold[sample_id] = [str(x) for x in sample["outputs"]]

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / f"{DATASET}_generation_manifest.json"
    manifest.write_text(json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    rowinfo = {
        "schema": op.ROWINFO_SCHEMA,
        "dataset": DATASET,
        "manifest_sha256": manifest_sha,
        "max_model_len": args.max_model_len,
        "rows": [
            {
                "id": r["id"], "index": i, "generation_budget": r["generation_budget"],
                "prompt_sha256": op.token_ids_sha256(r["prompt_tokens"]),
                "prompt_token_count": r["prompt_token_count"], "thinking": False,
                "task": r["task"], "task_base": r["task_base"], "prefilled": True,
            }
            for i, r in enumerate(rows)
        ],
    }
    (args.out / f"{DATASET}_rowinfo.json").write_text(json.dumps(rowinfo, indent=1, sort_keys=True) + "\n")
    (args.out / f"{DATASET}_gold.json").write_text(json.dumps(gold, sort_keys=True) + "\n")
    cells = [{"dataset": DATASET, "id": r["id"], "index": i, "prompt_tokens": r["prompt_token_count"], "seed": 1}
             for i, r in enumerate(rows)]
    (args.out / f"cells_{DATASET}.json").write_text(json.dumps(cells, indent=2) + "\n")
    readme = {
        "schema": "v31_ruler_subset_pool_v1", "dataset": DATASET, "context_length": 131072,
        "samples": len(rows), "tasks": counts, "seed": 6464, "per_task": 8,
        "ruler_commit": op.RULER_COMMIT, "max_model_len": args.max_model_len,
        "manifest_sha256": manifest_sha, "source_samples_sha256": hashlib.sha256(args.samples.read_bytes()).hexdigest(),
        "tokenizer_revision": args.revision, "response_opener": OPENER,
        "generation_budget_by_task": {task: sorted({r["generation_budget"] for r in rows if r["task"] == task}) for task in TASKS},
    }
    (args.out / "readme.json").write_text(json.dumps(readme, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"dataset": DATASET, "rows": len(rows), "tasks": counts, "manifest_sha256": manifest_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
