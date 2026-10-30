"""Prepare private benchmarks and disjoint computation-only calibration data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2) + "\n")


def short_data(root):
    old = Path("results/optimized_refresh_benchmarks_20261022/attempt009/private/pools")
    aime = json.loads((old / "aime26/aime26_generation_manifest.json").read_text())
    answers = {str(r["id"]): str(r["answer"]) for r in map(json.loads, (old / "aime26/aime2026_answers.jsonl").read_text().splitlines())}
    he = json.loads((old / "humaneval/gold.json").read_text())
    assert len(aime) == 30 and len(he) == 164
    a = [{"id": r["id"], "prompt": r["prompt"], "gold": answers[str(r["source_id"])],
          "seeds": [42, 43, 44]} for r in aime]
    h = [{"id": key, "prompt": "Complete the Python function below. Return the complete Python solution in a single fenced python code block, without a test harness.\n\n" + value["prompt"],
          "gold": value, "seeds": [42]} for key, value in sorted(he.items(), key=lambda z: int(z[0].split('/')[-1]))]
    math = json.loads((old / "math500/math500_generation_manifest.json").read_text())
    cal_math = [{"id": f"dev_math/{i:03}", "prompt": r["prompt"], "seeds": [142]}
                for i, r in enumerate(math[:6])]
    coding = [
        "Write a Python function reverse_words(text) that reverses whitespace-separated words.",
        "Write a Python function sum_squares(n) returning the sum of squares from 1 through n.",
        "Write a Python function unique_sorted(values) returning sorted distinct integers.",
        "Write a Python function is_palindrome(text) ignoring spaces and case.",
        "Write a Python function merge_counts(a,b) combining two dictionaries of integer counts.",
        "Write a Python function transpose(matrix) returning the transpose of a rectangular list of lists.",
    ]
    cal_code = [{"id": f"dev_code/{i:03}", "prompt": prompt + " Return the complete function in a Python code block.", "seeds": [142]}
                for i, prompt in enumerate(coding)]
    for key in ("llada22", "ultrallada", "trado"):
        for name, rows in [("aime26", a), ("humaneval", h), ("calibration_math", cal_math), ("calibration_code", cal_code)]:
            write(root / "private/data" / key / f"{name}.json", rows)
    return {"aime_source_sha256": digest(old / "aime26/aime26_generation_manifest.json"),
            "humaneval_gold_sha256": digest(old / "humaneval/gold.json"),
            "math_development_source_sha256": digest(old / "math500/math500_generation_manifest.json"),
            "coding_development": "six_handwritten_nonbenchmark_tasks",
            "aime_questions": 30, "humaneval_questions": 164}


def ruler_data(root):
    import yaml
    from dllm.evaluation.ruler.official import PAPER_TASKS, verify_checkout
    from transformers import AutoTokenizer
    from experiments.numerical_qk_reuse.multi_model_runtime import ModelHarness
    ruler = Path("reference/RULER").resolve()
    identity = verify_checkout(ruler)
    config = yaml.safe_load((ruler / "scripts/synthetic.yaml").read_text())
    for key in ("llada22", "ultrallada"):
        harness = ModelHarness(root, key)
        harness.tokenizer = AutoTokenizer.from_pretrained(root / "private/models" / key, trust_remote_code=True)
        for phase, per_task, seed in [("calibration_ruler", 1, 9142), ("ruler128k", 10, 6464)]:
            rows = []
            for ti, task in enumerate(PAPER_TASKS):
                target = root / "private/ruler_generated" / key / phase
                command = [sys.executable, str(ruler / "scripts/data/prepare.py"),
                           "--save_dir", str(target), "--benchmark", "synthetic", "--task", task,
                           "--tokenizer_path", str(root / "private/models" / key),
                           "--tokenizer_type", "hf", "--max_seq_length", "130816",
                           "--model_template_type", "base", "--num_samples", str(per_task),
                           "--random_seed", str(seed + ti * 1000), "--subset", "validation"]
                log = target / f"{task}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                environment = dict(os.environ)
                environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
                with log.open("w") as handle:
                    subprocess.run(command, cwd=ruler, env=environment, stdout=handle,
                                   stderr=subprocess.STDOUT, check=True)
                samples = [json.loads(line) for line in (target / task / "validation.jsonl").read_text().splitlines() if line.strip()]
                assert len(samples) == per_task
                for i, sample in enumerate(samples):
                    prefix = str(sample.get("answer_prefix", ""))
                    count = len(harness.encode(sample["input"], answer_prefix=prefix))
                    if count + 128 > 131072:
                        raise ValueError(f"{key} {task}/{i} exceeds model context: {count}+128")
                    rows.append({"id": f"{task}/{i:03}", "task": task, "task_base": config[task]["task"],
                                 "prompt": sample["input"], "answer_prefix": prefix,
                                 "gold": sample["outputs"], "prompt_tokens": count,
                                 "seeds": [42 if phase == "ruler128k" else 142]})
                print(json.dumps({"event": "ruler_data_task_complete", "model": key,
                                  "phase": phase, "task": task, "count": len(samples)}), flush=True)
            write(root / "private/data" / key / f"{phase}.json", rows)
    return identity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--ruler", action="store_true")
    a = p.parse_args()
    a.root = a.root.resolve()
    result = ruler_data(a.root) if a.ruler else short_data(a.root)
    write(a.root / ("ruler_data_identity.json" if a.ruler else "short_data_identity.json"), result)
    print(json.dumps({"status": "complete", "ruler": a.ruler}), flush=True)


if __name__ == "__main__":
    main()
