from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch.nn as nn

from dllm.evaluation.ruler.io import sha256_file, write_json, write_jsonl
from dllm.evaluation.ruler.official import RULER_COMMIT, balanced_counts
from dllm.evaluation.ruler.runner import RulerRunConfig, run_evaluation
from dllm.models import GenerationResult


def test_balanced_counts_preserve_exact_total_for_small_and_uneven_runs() -> None:
    tasks = ("a", "b", "c", "d", "e")
    assert balanced_counts(tasks, 2) == {"a": 1, "b": 1, "c": 0, "d": 0, "e": 0}
    counts = balanced_counts(tasks, 13)
    assert counts == {"a": 3, "b": 3, "c": 3, "d": 2, "e": 2}
    assert sum(counts.values()) == 13


class FakeAdapter:
    name = "fake"
    attention_class_names = ()
    mask_token_id = 99
    pad_token_id = 0

    def __init__(self) -> None:
        self.model = nn.Identity()

    def load(self):
        return self

    def encode_prompt(self, prompt, extra=None):
        return list(range(8))

    def prompt_configuration(self, extra=None):
        return {}

    def runtime_metadata(self):
        return {"transformers_version": "fake"}

    def generate(self, request):
        return GenerationResult(
            prompt=request.prompt,
            prompt_tokens=[1],
            completion_tokens=[2],
            text="answer",
            elapsed_seconds=0.01,
        )


def _manifest(
    tmp_path: Path, count: int = 3, model_adapter: str | None = None
) -> Path:
    rows = [
        {
            "sample_id": f"sample-{index}",
            "task": "fwe",
            "task_base": "fwe",
            "target_length": 8,
            "actual_prompt_length": 8,
            "inference_seed": 100 + index,
            "prompt": f"prompt {index}",
            "outputs": ["answer"],
            "tokens_to_generate": 4,
        }
        for index in range(count)
    ]
    samples = tmp_path / "samples.jsonl"
    write_jsonl(samples, rows)
    manifest = tmp_path / "manifest.json"
    payload = {
        "schema_version": 2,
        "ruler": {"commit": RULER_COMMIT},
        "context_length": 8,
        "requested_num_samples": count,
        "actual_num_samples": count,
        "samples": {"path": str(samples), "sha256": sha256_file(samples)},
    }
    if model_adapter is not None:
        payload["model_adapter"] = model_adapter
    write_json(
        manifest,
        payload,
    )
    return manifest


def test_runner_uses_exact_requested_count_and_stable_resume(tmp_path, monkeypatch) -> None:
    import dllm.evaluation.ruler.runner as runner

    monkeypatch.setattr(runner, "create_adapter", lambda *args, **kwargs: FakeAdapter())
    monkeypatch.setattr(
        runner,
        "score_predictions",
        lambda rows, root: ({"fwe": 1.0}, 1.0),
    )
    config = RulerRunConfig(
        model_adapter="diffusion_gemma",
        model_path="fake",
        manifest_path=str(
            _manifest(tmp_path, count=2, model_adapter="diffusion_gemma")
        ),
        ruler_root=str(tmp_path),
        output_dir=str(tmp_path / "run"),
        num_samples=2,
        context_length=8,
        device="cpu",
        precision="float32",
        generation_extra={"max_denoising_steps": 4},
    )
    first = run_evaluation(config)
    second = run_evaluation(config)
    assert first == second
    assert first["requested_num_samples"] == first["actual_num_samples"] == 2
    predictions = (tmp_path / "run" / "predictions.jsonl").read_text().splitlines()
    assert len(predictions) == 2
    run_config = json.loads((tmp_path / "run" / "run_config.json").read_text())
    assert run_config["generation_extra"] == {"max_denoising_steps": 4}
    assert run_config["transformers_version"] == "fake"


def test_runner_rejects_more_samples_than_manifest(tmp_path) -> None:
    config = RulerRunConfig(
        model_adapter="fast_dllm_v2",
        model_path="fake",
        manifest_path=str(_manifest(tmp_path, count=1)),
        ruler_root=str(tmp_path),
        output_dir=str(tmp_path / "run"),
        num_samples=2,
        context_length=8,
        device="cpu",
    )
    with pytest.raises(ValueError, match="manifest count"):
        run_evaluation(config)


def test_runner_rejects_using_only_a_prefix_of_a_larger_manifest(tmp_path) -> None:
    config = RulerRunConfig(
        model_adapter="fast_dllm_v2",
        model_path="fake",
        manifest_path=str(_manifest(tmp_path, count=3)),
        ruler_root=str(tmp_path),
        output_dir=str(tmp_path / "run"),
        num_samples=2,
        context_length=8,
        device="cpu",
    )
    with pytest.raises(ValueError, match="manifest count"):
        run_evaluation(config)
