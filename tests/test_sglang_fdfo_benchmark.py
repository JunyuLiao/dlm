from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.sglang_fdfo import run


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<chat>{messages[0]['content']}</chat>"


FEW_SHOTS = [
    {"question": f"few shot {index}", "answer": f"work\n#### {index}"}
    for index in range(4)
]


def test_download_gsm8k_validates_hash_and_selection(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    rows = [
        {"question": f"question {index}", "answer": f"reasoning\n#### {index}"}
        for index in range(3)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    expected = run.sha256_file(path)
    assert run.download_gsm8k(path, expected_sha256=expected) == path
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        run.download_gsm8k(path, expected_sha256="0" * 64)

    manifest = run.build_manifest(
        rows,
        FakeTokenizer(),
        2,
        start_index=1,
        seed_base=100,
        few_shots=FEW_SHOTS,
    )
    assert [row["dataset_index"] for row in manifest] == [1, 2]
    assert [row["request_id"] for row in manifest] == [
        "gsm8k-test-0001",
        "gsm8k-test-0002",
    ]
    assert [row["seed"] for row in manifest] == [101, 102]


def test_prompt_construction_is_deterministic() -> None:
    first = run.build_manifest(
        [{"question": "target", "answer": "steps\n#### 7"}],
        FakeTokenizer(),
        1,
        few_shots=FEW_SHOTS,
    )
    second = run.build_manifest(
        [{"question": "target", "answer": "steps\n#### 7"}],
        FakeTokenizer(),
        1,
        few_shots=FEW_SHOTS,
    )
    assert first == second
    assert first[0]["prompt_sha256"] == run.sha256_bytes(
        first[0]["prompt"].encode()
    )
    assert first[0]["raw_prompt"].count("Question:") == 5


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The answer is 1,234.", "1234"),
        ("work\n#### -7", "-7"),
        ("The result is $4.500", "4.5"),
        ("no numeric answer", None),
        ("intermediate 2; final 9", "9"),
    ],
)
def test_numeric_answer_extraction(text: str, expected: str | None) -> None:
    assert run.extract_numeric_answer(text) == expected


def _spec(identifier: str) -> run.RequestSpec:
    return run.RequestSpec(identifier, f"prompt-{identifier}", seed=1, label="2")


def _result(identifier: str, *, latency: float = 1.0, correct: bool = True):
    return {
        "request_id": identifier,
        "http_status": 200,
        "error": None,
        "text": "answer 2" if correct else "answer 3",
        "extracted_answer": "2" if correct else "3",
        "label": "2",
        "correct": correct,
        "latency_seconds": latency,
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "completion_index": 0 if identifier == "a" else 1,
    }


def test_lifecycle_validation_rejects_missing_duplicate_and_failed_requests() -> None:
    specs = [_spec("a"), _spec("b")]
    run.validate_request_results(specs, [_result("a"), _result("b")])
    with pytest.raises(RuntimeError, match="missing"):
        run.validate_request_results(specs, [_result("a")])
    with pytest.raises(RuntimeError, match="duplicate_actual"):
        run.validate_request_results(specs, [_result("a"), _result("a")])
    failed = _result("b")
    failed["error"] = "boom"
    with pytest.raises(RuntimeError, match="failed"):
        run.validate_request_results(specs, [_result("a"), failed])


def test_server_commands_differ_only_by_mode_and_port() -> None:
    config = run.RunnerConfig(model="model", model_revision="revision")
    sync = run.build_server_command(
        config, port=30000, scheduler_cap=4, fdfo=False
    )
    fdfo = run.build_server_command(
        config, port=30001, scheduler_cap=4, fdfo=True
    )

    def normalize(command: list[str]) -> list[str]:
        normalized = list(command)
        port_index = normalized.index("--port") + 1
        normalized[port_index] = "PORT"
        normalized[-1] = "FDFO_MODE"
        return normalized

    assert normalize(sync) == normalize(fdfo)
    assert sync[-1] == "--no-dllm-fdfo"
    assert fdfo[-1] == "--dllm-fdfo"
    graph_index = sync.index("--cuda-graph-bs") + 1
    assert sync[graph_index : graph_index + 4] == ["1", "2", "3", "4"]


def test_dllm_request_payload_uses_supported_greedy_parameters() -> None:
    payload = run.build_request_payload(_spec("request"), max_new_tokens=128)
    assert payload["rid"] == "request"
    assert payload["sampling_params"] == {
        "temperature": 0.0,
        "max_new_tokens": 128,
        "stop": run.STOP_SEQUENCES,
    }
    assert "seed" not in payload["sampling_params"]


def test_runtime_configuration_verifies_fdfo_and_cuda_graph_sizes() -> None:
    config = run.RunnerConfig(model="model", model_revision="revision")
    info = {
        "version": run.SGLANG_VERSION,
        "model_path": "model",
        "revision": "revision",
        "dllm_algorithm": "JointThreshold",
        "dllm_algorithm_config": str(run.THIS_DIR / "joint_threshold.yaml"),
        "dllm_fdfo": True,
        "dtype": "bfloat16",
        "tp_size": 1,
        "mem_fraction_static": 0.9,
        "page_size": 32,
        "max_running_requests": 4,
        "attention_backend": "flashinfer",
        "cuda_graph_config": {"decode": {"bs": [1, 2, 3, 4]}},
    }
    run.assert_server_info(info, config, scheduler_cap=4, fdfo=True)
    info["dllm_fdfo"] = False
    with pytest.raises(RuntimeError, match="dllm_fdfo"):
        run.assert_server_info(info, config, scheduler_cap=4, fdfo=True)


def test_cuda_graph_memory_fallbacks_apply_equally_to_each_arm() -> None:
    reduced = run.RunnerConfig(cuda_graph_batch_sizes=(1, 2, 4, 8, 16))
    assert run.effective_cuda_graph_batch_sizes(reduced, scheduler_cap=4) == [1, 2, 4]
    eager = run.RunnerConfig(disable_cuda_graphs=True)
    command = run.build_server_command(eager, port=30000, scheduler_cap=4, fdfo=True)
    assert "--cuda-graph-bs" not in command
    assert "--disable-decode-cuda-graph" in command
    assert "--disable-prefill-cuda-graph" in command


def test_summary_statistics_and_pairwise_correctness() -> None:
    rows = [_result("a", latency=1), _result("b", latency=3, correct=False)]
    summary = run.summarize_results(rows, wall_time=4)
    assert summary["num_requests"] == 2
    assert summary["requests_per_second"] == 0.5
    assert summary["output_tokens_per_second"] == 5
    assert summary["latency_seconds"]["p50"] == 2
    assert summary["accuracy"] == 0.5

    fdfo = [_result("a", correct=True), _result("b", correct=True)]
    pair = run.compare_pair(rows, fdfo)
    assert pair["answer_equivalence_rate"] == 0.5
    assert pair["changed_answer_ids"] == ["b"]
    assert pair["mcnemar"]["sync_incorrect_fdfo_correct"] == 1


def test_comparison_and_markdown_report_with_mocked_results() -> None:
    def summary(wall: float, throughput: float, accuracy: float = 1.0):
        return {
            "wall_time_seconds": wall,
            "output_tokens_per_second": throughput,
            "requests_per_second": 10,
            "latency_seconds": {
                "p50": 1,
                "p90": 2,
                "p95": 2,
                "p99": 3,
                "max": 3,
            },
            "accuracy": accuracy,
            "malformed_answer_rate": 0.0,
        }

    measured = {
        4: {
            repetition: {
                "sync": summary(10, 100),
                "fdfo": summary(9.5, 105),
            }
            for repetition in range(1, 4)
        },
        16: {
            repetition: {
                "sync": summary(10, 100),
                "fdfo": summary(8, 125),
            }
            for repetition in range(1, 4)
        },
    }
    rows = {
        (cap, repetition, mode): [_result("a"), _result("b")]
        for cap in (4, 16)
        for repetition in range(1, 4)
        for mode in ("sync", "fdfo")
    }
    comparison = run.build_comparison(measured, rows, {"passed": True})
    assert comparison["integration_success"] is True
    assert comparison["meaningful_speedup"] is True
    assert comparison["outcome"] == "meaningful speedup"
    report = run.render_report(comparison)
    assert "| 16 | 10.000 | 8.000 | 1.250x [1.250, 1.250]" in report
    assert "Server startup and warm-up are excluded" in report
