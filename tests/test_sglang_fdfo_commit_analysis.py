from __future__ import annotations

import math
import json

from benchmarks.sglang_fdfo_commit import analyze


def _row(composition: str, batch_size: int, commit_count: int, model_ms: float):
    return {
        "graph_mode": "eager",
        "block_size": 32,
        "context_length": 1024,
        "requested_batch_size": batch_size,
        "batch_size": batch_size,
        "composition": composition,
        "commit_count": commit_count,
        "normal_count": batch_size - commit_count,
        "commit_fraction": commit_count / batch_size,
        "wall_ms": model_ms + 2,
        "total_gpu_ms": model_ms + 1,
        "model_gpu_ms": model_ms,
        "attention_gpu_ms": model_ms * 0.2,
        "kv_write_gpu_ms": model_ms * 0.01,
        "step_gpu_ms": 0.2,
        "step_cpu_ms": 0.5,
        "logits_gpu_ms": 0.5,
        "cpu_overhead_ms": 1.0,
        "block_times": [{"layer_id": 19, "gpu_ms": 0.7}],
    }


def test_expected_kv_bytes() -> None:
    assert analyze.expected_kv_bytes(32) == 1_310_720
    assert analyze.expected_kv_bytes(64, 4) == 10_485_760


def test_matched_commit_ratio_and_breakdown() -> None:
    rows = [_row("all_normal", 4, 0, 10), _row("all_commit", 4, 4, 11)]
    ratio, cells = analyze.matched_commit_ratio(rows)
    assert cells == 1
    assert math.isclose(ratio, 1.1)
    breakdown = analyze.breakdown_table(rows)
    assert {row["composition"] for row in breakdown} == {"all_normal", "all_commit"}
    assert all(row["kv_percent_model"] == 1.0 for row in breakdown)


def test_opportunity_includes_batching_and_separation_controls() -> None:
    rows = [
        _row("all_normal", 1, 0, 4),
        _row("all_commit", 1, 1, 4),
        _row("all_normal", 3, 0, 8),
        _row("all_commit", 4, 4, 9),
        _row("mixed", 4, 1, 10),
    ]
    opportunity = analyze.compute_opportunity(rows)
    assert opportunity["delayed_commit_batching"]["batch_1_to_largest_throughput_speedup"] > 1
    separation = opportunity["separate_vs_mixed"]
    assert separation["matched_mixed_samples"] == 1
    assert math.isclose(separation["mixed_over_separate_model_time_ratio_median"], 10 / 12)
    assert opportunity["write_only_fantasy"]["end_to_end_percent"] > 0


def test_load_run_separates_prompt_prefill_from_final_commit(tmp_path) -> None:
    workload_dir = tmp_path / "eager" / "block_32"
    profile_dir = workload_dir / "profile"
    profile_dir.mkdir(parents=True)
    workload = {
        "name": "measure",
        "warmup": False,
        "graph_mode": "eager",
        "block_size": 32,
        "context_length": 128,
        "batch_size": 1,
        "repetition": 1,
        "started_time_ns": 100,
        "finished_time_ns": 1000,
    }
    (workload_dir / "workloads.jsonl").write_text(json.dumps(workload) + "\n")
    events = [
        {
            "type": "fdfo_iteration",
            "started_time_ns": 200,
            "batch_size": 1,
            "accept_lengths": [32],
            "prefix_lens": [96],
        },
        {
            "type": "fdfo_iteration",
            "started_time_ns": 300,
            "batch_size": 1,
            "accept_lengths": [32],
            "prefix_lens": [128],
        },
    ]
    (profile_dir / "events-1.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )
    _, rows, _ = analyze.load_run(tmp_path)
    assert rows[0]["composition"] == "prompt_prefill"
    assert rows[0]["commit_count"] == 0
    assert rows[1]["composition"] == "all_commit"
    assert rows[1]["commit_count"] == 1
