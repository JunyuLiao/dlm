"""Independent coverage, binding, work-accounting and privacy audit."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


ARMS = ("dense_native", "allkept_fa4", "mass_topk_reuse_r4",
        "value_v2_gaussian32_reuse_r4", "value_v3a_singleton_reuse_r4")
MODELS = ("llada22", "ultrallada", "trado")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--attempt", default="attempt001")
    a = p.parse_args()
    root = a.root.resolve()
    out = root / "verification" / a.attempt
    out.mkdir(parents=True, exist_ok=False)
    errors = []
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol["spec_sha256"] != sha(Path("docs/MULTI_MODEL_VALUE_EVALUATION_20261030.md")):
        errors.append("frozen specification digest mismatch")
    table = collections.defaultdict(dict)
    execution_bindings = collections.defaultdict(set)
    receipts = []
    accepted_shards = set()
    repeated_attempts = []
    for path in sorted((root / "runs").glob("*/*/*/shard*/attempt*/final_complete.json")):
        leaf = path.parent
        shard_identity = str(leaf.parent.relative_to(root))
        if shard_identity in accepted_shards:
            repeated_attempts.append(str(leaf.relative_to(root)))
            continue
        accepted_shards.add(shard_identity)
        config = json.loads((leaf / "config.json").read_text())
        receipt = json.loads((leaf / "receipt.json").read_text())
        manifest = root / "private/data" / config["model"] / (config["suite"] + ".json")
        inputs = {r["id"]: r for r in json.loads(manifest.read_text())}
        if config["manifest_sha256"] != sha(manifest):
            errors.append("manifest digest mismatch: " + str(leaf.relative_to(root)))
        if config["protocol_sha256"] != sha(root / "protocol.json"):
            errors.append("protocol digest mismatch: " + str(leaf.relative_to(root)))
        if config["model_revision"] != protocol["models"][config["model"]]:
            errors.append("model revision mismatch: " + str(leaf.relative_to(root)))
        if receipt["records_sha256"] != sha(leaf / "records.jsonl"):
            errors.append("records digest mismatch: " + str(leaf.relative_to(root)))
        panel = (config["model"], config["suite"], config["arm"])
        execution_bindings[panel].add((config["runtime_sha256"], config["attention_source_sha256"],
                                      config["host_gpu_uuid"], config.get("statistics_query_blocks", 1)))
        rows = [json.loads(line) for line in (leaf / "records.jsonl").read_text().splitlines() if line.strip()]
        if len(rows) != config["cells"]:
            errors.append("shard cell count mismatch: " + str(leaf.relative_to(root)))
        for row in rows:
            key = (row["id"], row["seed"])
            if key in table[panel]:
                errors.append(f"duplicate cell: {panel}/{key}")
            table[panel][key] = row
            source_row = inputs.get(row["id"])
            if source_row is None or row["seed"] not in source_row["seeds"]:
                errors.append(f"unexpected manifest cell: {panel}/{key}")
            elif row["prompt_sha256"] != hashlib.sha256(source_row["prompt"].encode()).hexdigest():
                errors.append(f"prompt digest mismatch: {panel}/{key}")
            for field in ("manifest_sha256", "source_sha256", "runtime_sha256", "protocol_sha256", "model_revision"):
                if row[field] != config[field]:
                    errors.append(f"record/config binding mismatch: {panel}/{key}/{field}")
            n = row["counts"]
            if n["total_model_forwards"] != n["denoising_forwards"] + n.get("prefill_model_forwards", 0) + n.get("commit_model_forwards", 0):
                errors.append(f"forward accounting mismatch: {panel}/{key}")
            if not 0 <= n["kept_tiles"] <= n["eligible_tiles"]:
                errors.append(f"invalid tile counts: {panel}/{key}")
            if config["arm"] in ARMS[:2] and n["kept_tiles"] != n["eligible_tiles"]:
                errors.append(f"dense control skipped tiles: {panel}/{key}")
            prohibited = {"text", "tokens", "input_ids", "completion_tokens", "gold", "answer", "prompt"}
            if prohibited.intersection(row):
                errors.append(f"private payload in public record: {panel}/{key}")
        receipts.append({"path": str(leaf.relative_to(root)), "receipt_sha256": sha(leaf / "receipt.json"),
                         "gpu_worker_seconds": receipt["gpu_worker_seconds"]})
    panels = {}
    for model in MODELS:
        for suite in (("aime26", "humaneval") if model == "trado" else ("aime26", "humaneval", "ruler128k")):
            dataset_path = root / "private/data" / model / f"{suite}.json"
            dataset = json.loads(dataset_path.read_text())
            expected = {(r["id"], seed) for r in dataset for seed in r["seeds"]}
            count = 90 if suite == "aime26" else 164 if suite == "humaneval" else 130
            if len(expected) != count:
                errors.append(f"wrong planned coverage: {model}/{suite}")
            if suite == "ruler128k":
                tasks = collections.Counter(r["task"] for r in dataset)
                if len(tasks) != 13 or set(tasks.values()) != {10}:
                    errors.append(f"RULER task coverage mismatch: {model}")
                if any(r["prompt_tokens"] + 128 > 131072 for r in dataset):
                    errors.append(f"overlength RULER input: {model}")
            threshold_path = sorted((root / "calibration" / model / suite).glob("attempt*/selected_threshold.json"))
            if not threshold_path:
                errors.append(f"missing calibration: {model}/{suite}")
                continue
            calibration = json.loads(threshold_path[-1].read_text())
            if not calibration["in_range"]:
                errors.append(f"calibration out of range: {model}/{suite}")
            for arm in ARMS:
                panel = (model, suite, arm)
                actual = table.get(panel, {})
                if set(actual) != expected:
                    errors.append(f"coverage mismatch: {panel}: {len(actual)}/{len(expected)}")
                if actual:
                    eligible = sum(r["counts"]["eligible_tiles"] for r in actual.values())
                    kept = sum(r["counts"]["kept_tiles"] for r in actual.values())
                    panels["/".join(panel)] = {"requests": len(actual), "eligible_tiles": eligible,
                                              "kept_tiles": kept, "consumer_sparsity": 1. - kept / eligible}
                if len(execution_bindings[panel]) > 1:
                    errors.append(f"inconsistent execution binding within panel: {panel}")
                for ref_arm in ARMS[:2]:
                    reference = execution_bindings.get((model, suite, ref_arm))
                    if reference and execution_bindings[panel] and execution_bindings[panel] != reference:
                        errors.append(f"attention/runtime/GPU/statistics binding mismatch: {panel}/{ref_arm}")
                for key, row in actual.items():
                    if arm == "value_v2_gaussian32_reuse_r4" and row["threshold"] != calibration["selected_threshold"]:
                        errors.append(f"V2 threshold drift: {panel}/{key}")
                    for ref_arm in ARMS[:2]:
                        other = table.get((model, suite, ref_arm), {}).get(key)
                        if other:
                            for field in ("prompt_sha256", "prompt_tokens", "generation_cap", "model_revision", "runtime_sha256"):
                                if row[field] != other[field]:
                                    errors.append(f"fairness binding mismatch: {panel}/{key}/{field}")
    result = {"status": "pass" if not errors else "fail", "errors": errors,
              "complete_panels": len(table), "expected_panels": 40,
              "complete_requests": sum(len(v) for v in table.values()), "expected_requests": 5110,
              "panels": panels, "receipt_fingerprints": receipts,
              "redundant_complete_attempts_excluded": repeated_attempts,
              "main_panel_sparsity_drift_is_reported_not_retuned": True}
    (out / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "complete_panels", "complete_requests", "errors")}), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
