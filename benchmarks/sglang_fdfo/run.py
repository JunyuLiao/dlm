#!/usr/bin/env python3
"""Reproducible SGLang FDFO A/B benchmark for LLaDA2.1-mini on GSM8K.

The script intentionally uses SGLang's HTTP server.  The repository's dInfer
``ModelRunner`` integration bypasses the request scheduler and cannot exercise
FDFO.  Install ``requirements.txt`` from this directory in an isolated Python
environment before running this benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import datetime as dt
import decimal
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


SGLANG_VERSION = "0.5.16"
SGLANG_TAG_COMMIT = "fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1"
DEFAULT_MODEL = "inclusionAI/LLaDA2.1-mini"
DEFAULT_MODEL_REVISION = "20e64e2ad21644d0e5248586ed9c942cdd45de0f"
GSM8K_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    f"{GSM8K_REVISION}/grade_school_math/data/test.jsonl"
)
GSM8K_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
STOP_SEQUENCES = ["Question:", "</s>", "<|im_end|>"]
UPSTREAM_PARITY_PROMPTS = [
    "Question: Natalia sold clips to 48 friends in April, and half as many in "
    "May. How many clips did she sell altogether? Answer:",
    "The capital of France is",
    "Q: What is 12 times 13? A:",
]

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
DEFAULT_TASK_YAML = (
    REPO_ROOT
    / "third_party/dinfer/evaluations/tasks/gsm8k/gsm8k-llada-mini.yaml"
)


@dataclasses.dataclass(frozen=True)
class RequestSpec:
    request_id: str
    prompt: str
    seed: int
    label: str | None = None
    dataset_index: int | None = None
    question: str | None = None


@dataclasses.dataclass(frozen=True)
class RunnerConfig:
    model: str = DEFAULT_MODEL
    model_revision: str = DEFAULT_MODEL_REVISION
    host: str = "127.0.0.1"
    mem_fraction_static: float = 0.9
    dtype: str = "bfloat16"
    server_timeout_seconds: float = 1800.0
    request_timeout_seconds: float = 900.0
    telemetry_interval_seconds: float = 0.5
    random_seed: int = 1234
    cuda_graph_batch_sizes: tuple[int, ...] | None = None
    disable_cuda_graphs: bool = False


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def download_gsm8k(destination: Path, expected_sha256: str = GSM8K_SHA256) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with urllib.request.urlopen(GSM8K_URL, timeout=120) as response:
                data = response.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(f"failed to download pinned GSM8K data: {exc}") from exc
        temporary.write_bytes(data)
        temporary.replace(destination)
    actual = sha256_file(destination)
    if actual != expected_sha256:
        raise RuntimeError(
            f"GSM8K checksum mismatch for {destination}: expected "
            f"{expected_sha256}, got {actual}"
        )
    return destination


_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def normalize_number(value: str) -> str | None:
    cleaned = value.replace("$", "").replace(",", "").strip()
    try:
        number = decimal.Decimal(cleaned)
    except decimal.InvalidOperation:
        return None
    if number == number.to_integral_value():
        return str(number.quantize(decimal.Decimal(1)))
    return format(number.normalize(), "f")


def extract_numeric_answer(text: str) -> str | None:
    matches = _NUMBER_RE.findall(text)
    return normalize_number(matches[-1]) if matches else None


def target_answer(answer: str) -> str:
    marker = answer.rsplit("####", 1)[-1]
    normalized = extract_numeric_answer(marker)
    if normalized is None:
        raise ValueError(f"GSM8K target has no numeric answer: {answer!r}")
    return normalized


def load_few_shots(task_yaml: Path = DEFAULT_TASK_YAML) -> list[dict[str, str]]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; install the benchmark requirements") from exc
    config = yaml.safe_load(task_yaml.read_text(encoding="utf-8"))
    samples = config["fewshot_config"]["samples"]
    if len(samples) != 4:
        raise RuntimeError(f"expected four fixed GSM8K few-shots, found {len(samples)}")
    return [{"question": item["question"], "answer": item["answer"]} for item in samples]


def format_question(question: str) -> str:
    return f"Question: {question}\nLet's think step by step\nAnswer:"


def build_raw_prompt(question: str, few_shots: Sequence[dict[str, str]]) -> str:
    parts = [
        f"{format_question(sample['question'])} {sample['answer'].rstrip()}"
        for sample in few_shots
    ]
    parts.append(format_question(question))
    return "\n\n".join(parts)


def apply_chat_template(tokenizer: Any, raw_prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_manifest(
    dataset: Sequence[dict[str, Any]],
    tokenizer: Any,
    count: int,
    *,
    start_index: int = 0,
    seed_base: int = 1234,
    few_shots: Sequence[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    if count <= 0 or start_index < 0 or start_index + count > len(dataset):
        raise ValueError("requested manifest range is outside the GSM8K dataset")
    fixed_shots = list(few_shots) if few_shots is not None else load_few_shots()
    rows: list[dict[str, Any]] = []
    for index in range(start_index, start_index + count):
        item = dataset[index]
        raw_prompt = build_raw_prompt(item["question"], fixed_shots)
        prompt = apply_chat_template(tokenizer, raw_prompt)
        rows.append(
            {
                "request_id": f"gsm8k-test-{index:04d}",
                "dataset_index": index,
                "question": item["question"],
                "label": target_answer(item["answer"]),
                "seed": seed_base + index,
                "raw_prompt": raw_prompt,
                "prompt": prompt,
                "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            }
        )
    return rows


def specs_from_manifest(rows: Sequence[dict[str, Any]]) -> list[RequestSpec]:
    return [
        RequestSpec(
            request_id=row["request_id"],
            prompt=row["prompt"],
            seed=int(row["seed"]),
            label=row.get("label"),
            dataset_index=row.get("dataset_index"),
            question=row.get("question"),
        )
        for row in rows
    ]


def preflight_sglang() -> dict[str, str]:
    try:
        package_version = importlib.metadata.version("sglang")
        import sglang
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise RuntimeError(
            "SGLang is not installed. Create an isolated environment and install "
            "benchmarks/sglang_fdfo/requirements.txt."
        ) from exc
    module_path = Path(sglang.__file__).resolve()
    vendored_root = (REPO_ROOT / "third_party/sglang").resolve()
    if package_version != SGLANG_VERSION:
        raise RuntimeError(
            f"expected sglang=={SGLANG_VERSION}, imported {package_version} from {module_path}"
        )
    if module_path.is_relative_to(vendored_root):
        raise RuntimeError(f"benchmark resolved the unsupported vendored SGLang: {module_path}")
    return {"version": package_version, "module_path": str(module_path)}


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def build_server_command(
    config: RunnerConfig,
    *,
    port: int,
    scheduler_cap: int,
    fdfo: bool,
) -> list[str]:
    if scheduler_cap <= 0:
        raise ValueError("scheduler_cap must be positive")
    graph_batch_sizes = effective_cuda_graph_batch_sizes(config, scheduler_cap)
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        config.model,
        "--revision",
        config.model_revision,
        "--trust-remote-code",
        "--dtype",
        config.dtype,
        "--tp-size",
        "1",
        "--mem-fraction-static",
        str(config.mem_fraction_static),
        "--attention-backend",
        "flashinfer",
        "--dllm-algorithm",
        "JointThreshold",
        "--dllm-algorithm-config",
        str(THIS_DIR / "joint_threshold.yaml"),
        "--page-size",
        "32",
        "--max-running-requests",
        str(scheduler_cap),
    ]
    if config.disable_cuda_graphs:
        command.extend(["--disable-decode-cuda-graph", "--disable-prefill-cuda-graph"])
    else:
        command.extend(
            ["--cuda-graph-bs", *[str(value) for value in graph_batch_sizes]]
        )
    command.extend(
        [
            "--host",
            config.host,
            "--port",
            str(port),
            "--random-seed",
            str(config.random_seed),
            "--dllm-fdfo" if fdfo else "--no-dllm-fdfo",
        ]
    )
    return command


def effective_cuda_graph_batch_sizes(
    config: RunnerConfig, scheduler_cap: int
) -> list[int]:
    requested = config.cuda_graph_batch_sizes or tuple(range(1, scheduler_cap + 1))
    effective = [value for value in requested if value <= scheduler_cap]
    if not config.disable_cuda_graphs and not effective:
        raise ValueError(
            f"no requested CUDA-graph batch size is valid for scheduler cap {scheduler_cap}"
        )
    return effective


def child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    paths = environment.get("PYTHONPATH", "").split(os.pathsep)
    blocked = {
        str((REPO_ROOT / "third_party/sglang/python").resolve()),
        str((REPO_ROOT / "third_party/dinfer/python").resolve()),
    }
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in paths if path and str(Path(path).resolve()) not in blocked
    )
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def http_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server(
    base_url: str, process: subprocess.Popen[Any], timeout: float
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    deadline = started + timeout
    last_error: Exception | None = None
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang server exited early with code {process.returncode}")
        try:
            info = http_json(f"{base_url}/server_info", timeout=5)
            return info, time.perf_counter() - started
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
            time.sleep(1)
    raise TimeoutError(f"SGLang server did not become ready: {last_error}")


def assert_server_info(
    info: dict[str, Any], config: RunnerConfig, scheduler_cap: int, fdfo: bool
) -> None:
    expected = {
        "version": SGLANG_VERSION,
        "model_path": config.model,
        "revision": config.model_revision,
        "dllm_algorithm": "JointThreshold",
        "dllm_algorithm_config": str(THIS_DIR / "joint_threshold.yaml"),
        "dllm_fdfo": fdfo,
        "dtype": config.dtype,
        "tp_size": 1,
        "mem_fraction_static": config.mem_fraction_static,
        "page_size": 32,
        "max_running_requests": scheduler_cap,
        "attention_backend": "flashinfer",
    }
    mismatches = {
        key: {"expected": value, "actual": info.get(key)}
        for key, value in expected.items()
        if info.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"SGLang runtime configuration mismatch: {mismatches}")
    cuda_graph_config = info.get("cuda_graph_config") or {}
    decode_config = cuda_graph_config.get("decode") or {}
    prefill_config = cuda_graph_config.get("prefill") or {}
    if config.disable_cuda_graphs:
        backends = (decode_config.get("backend"), prefill_config.get("backend"))
        if backends != ("disabled", "disabled"):
            raise RuntimeError(
                "SGLang runtime did not disable both CUDA-graph phases: "
                f"actual={backends}"
            )
        return
    captured_sizes = decode_config.get("bs") or info.get("cuda_graph_bs_decode")
    expected_sizes = effective_cuda_graph_batch_sizes(config, scheduler_cap)
    if captured_sizes != expected_sizes:
        raise RuntimeError(
            "SGLang runtime CUDA-graph batch sizes do not match the scheduler cap: "
            f"expected={expected_sizes} actual={captured_sizes}"
        )


async def poll_loads(
    session: Any, base_url: str, output_path: Path, stop: asyncio.Event, interval: float
) -> None:
    rows: list[dict[str, Any]] = []
    while not stop.is_set():
        recorded = {"monotonic_seconds": time.perf_counter(), "ok": False}
        try:
            async with session.get(f"{base_url}/v1/loads") as response:
                recorded.update(
                    {
                        "status": response.status,
                        "payload": await response.json(content_type=None),
                        "ok": response.status == 200,
                    }
                )
        except Exception as exc:  # telemetry must not abort a measured run
            recorded["error"] = repr(exc)
        rows.append(recorded)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
    write_jsonl(output_path, rows)


def build_request_payload(spec: RequestSpec, max_new_tokens: int) -> dict[str, Any]:
    return {
        "text": spec.prompt,
        "rid": spec.request_id,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "stop": STOP_SEQUENCES,
        },
    }


async def send_request(
    session: Any,
    base_url: str,
    spec: RequestSpec,
    max_new_tokens: int,
    request_timeout: float,
    completion_counter: list[int],
) -> dict[str, Any]:
    payload = build_request_payload(spec, max_new_tokens)
    started_at_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    started = time.perf_counter()
    status = 0
    response_payload: dict[str, Any] | None = None
    error: str | None = None
    try:
        async with session.post(
            f"{base_url}/generate", json=payload, timeout=request_timeout
        ) as response:
            status = response.status
            body = await response.text()
            try:
                parsed = json.loads(body)
                response_payload = parsed if isinstance(parsed, dict) else {"value": parsed}
            except json.JSONDecodeError:
                error = f"non-JSON response: {body[:1000]}"
            if status != 200 and error is None:
                error = f"HTTP {status}: {body[:1000]}"
    except Exception as exc:
        error = repr(exc)
    finished = time.perf_counter()
    finished_at_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    completion_index = completion_counter[0]
    completion_counter[0] += 1
    meta = (response_payload or {}).get("meta_info") or {}
    echoed_id = meta.get("id") or meta.get("rid") or (response_payload or {}).get("id")
    if echoed_id is not None and str(echoed_id) != spec.request_id:
        error = error or f"server echoed request id {echoed_id!r}"
    text = (response_payload or {}).get("text")
    if isinstance(text, list):
        text = text[0] if text else ""
    extracted = extract_numeric_answer(text) if isinstance(text, str) else None
    completion_tokens = meta.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = meta.get("output_tokens", 0)
    return {
        "request_id": spec.request_id,
        "dataset_index": spec.dataset_index,
        "question": spec.question,
        "label": spec.label,
        "seed": spec.seed,
        "prompt_sha256": sha256_bytes(spec.prompt.encode("utf-8")),
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "http_status": status,
        "error": error,
        "text": text,
        "extracted_answer": extracted,
        "correct": None if spec.label is None else extracted == spec.label,
        "finish_reason": meta.get("finish_reason"),
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": int(completion_tokens or 0),
        "started_monotonic": started,
        "finished_monotonic": finished,
        "latency_seconds": finished - started,
        "completion_index": completion_index,
        "meta_info": meta,
    }


async def run_requests_async(
    base_url: str,
    specs: Sequence[RequestSpec],
    *,
    max_new_tokens: int,
    request_timeout: float,
    loads_path: Path | None,
    telemetry_interval: float,
) -> tuple[list[dict[str, Any]], float]:
    try:
        import aiohttp
    except ImportError as exc:
        raise RuntimeError("aiohttp is required by the SGLang benchmark environment") from exc
    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=0)
    completion_counter = [0]
    stop = asyncio.Event()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        poller = (
            asyncio.create_task(
                poll_loads(session, base_url, loads_path, stop, telemetry_interval)
            )
            if loads_path is not None
            else None
        )
        started = time.perf_counter()
        tasks = [
            asyncio.create_task(
                send_request(
                    session,
                    base_url,
                    spec,
                    max_new_tokens,
                    request_timeout,
                    completion_counter,
                )
            )
            for spec in specs
        ]
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - started
        stop.set()
        if poller is not None:
            await poller
    return results, wall_time


def run_requests(
    base_url: str,
    specs: Sequence[RequestSpec],
    *,
    max_new_tokens: int,
    config: RunnerConfig,
    loads_path: Path | None = None,
) -> tuple[list[dict[str, Any]], float]:
    return asyncio.run(
        run_requests_async(
            base_url,
            specs,
            max_new_tokens=max_new_tokens,
            request_timeout=config.request_timeout_seconds,
            loads_path=loads_path,
            telemetry_interval=config.telemetry_interval_seconds,
        )
    )


def validate_request_results(
    specs: Sequence[RequestSpec], results: Sequence[dict[str, Any]]
) -> None:
    expected = [spec.request_id for spec in specs]
    actual = [str(row.get("request_id")) for row in results]
    duplicate_expected = [key for key, count in Counter(expected).items() if count > 1]
    duplicate_actual = [key for key, count in Counter(actual).items() if count > 1]
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    completion_indices = [row.get("completion_index") for row in results]
    invalid_completion_order = not all(
        isinstance(value, int) for value in completion_indices
    ) or sorted(completion_indices) != list(range(len(results)))
    failed = [
        row.get("request_id")
        for row in results
        if row.get("http_status") != 200
        or row.get("error")
        or not isinstance(row.get("text"), str)
        or not row.get("text")
    ]
    missing_token_counts = [
        row.get("request_id")
        for row in results
        if not isinstance(row.get("prompt_tokens"), int)
        or not isinstance(row.get("completion_tokens"), int)
        or row.get("completion_tokens", 0) <= 0
    ]
    if len(results) != len(specs) or any(
        (
            duplicate_expected,
            duplicate_actual,
            missing,
            unexpected,
            failed,
            missing_token_counts,
            invalid_completion_order,
        )
    ):
        raise RuntimeError(
            "request lifecycle validation failed: "
            f"expected={len(specs)} actual={len(results)} "
            f"duplicate_expected={duplicate_expected} duplicate_actual={duplicate_actual} "
            f"missing={missing} unexpected={unexpected} failed={failed} "
            f"missing_token_counts={missing_token_counts} "
            f"invalid_completion_order={invalid_completion_order}"
        )


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize_results(
    results: Sequence[dict[str, Any]], wall_time: float
) -> dict[str, Any]:
    latencies = [float(row["latency_seconds"]) for row in results]
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in results)
    scored = [row for row in results if row.get("label") is not None]
    malformed = [row for row in scored if row.get("extracted_answer") is None]
    correct = [row for row in scored if row.get("correct") is True]
    return {
        "num_requests": len(results),
        "wall_time_seconds": wall_time,
        "requests_per_second": len(results) / wall_time if wall_time else None,
        "completion_tokens": completion_tokens,
        "output_tokens_per_second": completion_tokens / wall_time if wall_time else None,
        "latency_seconds": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p90": percentile(latencies, 0.90),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies) if latencies else None,
        },
        "accuracy": len(correct) / len(scored) if scored else None,
        "malformed_answer_rate": len(malformed) / len(scored) if scored else None,
        "failed_requests": sum(
            bool(row.get("error")) or row.get("http_status") != 200 for row in results
        ),
    }


class GpuTelemetry:
    HEADER = [
        "timestamp",
        "index",
        "uuid",
        "utilization_gpu_percent",
        "utilization_memory_percent",
        "memory_used_mib",
        "power_draw_watts",
        "clocks_sm_mhz",
    ]

    def __init__(self, output_path: Path, interval_seconds: float):
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self.process: subprocess.Popen[Any] | None = None
        self.handle: Any = None

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.output_path.open("w", encoding="utf-8")
        self.handle.write(",".join(self.HEADER) + "\n")
        self.handle.flush()
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return
        query = (
            "timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,"
            "power.draw,clocks.sm"
        )
        try:
            self.process = subprocess.Popen(
                [
                    executable,
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                    f"--loop-ms={max(100, int(self.interval_seconds * 1000))}",
                ],
                stdout=self.handle,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:
            self.process = None

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.handle is not None:
            self.handle.close()
        self.process = None
        self.handle = None


def gpu_summary(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return None
    numeric_rows: list[dict[str, float]] = []
    for row in rows:
        try:
            numeric_rows.append(
                {
                    "utilization": float(row["utilization_gpu_percent"]),
                    "memory": float(row["memory_used_mib"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    if not numeric_rows:
        return None
    utilization = [row["utilization"] for row in numeric_rows]
    return {
        "samples": len(numeric_rows),
        "utilization_gpu_percent": {
            "mean": statistics.fmean(utilization),
            "p50": percentile(utilization, 0.50),
            "p95": percentile(utilization, 0.95),
        },
        "peak_memory_used_mib": max(row["memory"] for row in numeric_rows),
    }


def stop_server(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=30)


def run_arm(
    config: RunnerConfig,
    *,
    run_dir: Path,
    scheduler_cap: int,
    fdfo: bool,
    specs: Sequence[RequestSpec],
    warmup_specs: Sequence[RequestSpec],
    max_new_tokens: int,
    warmup_max_new_tokens: int = 128,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    port = free_port(config.host)
    base_url = f"http://{config.host}:{port}"
    command = build_server_command(
        config, port=port, scheduler_cap=scheduler_cap, fdfo=fdfo
    )
    write_json(run_dir / "launch.json", {"command": command, "fdfo": fdfo})
    log_handle = (run_dir / "server.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=child_environment(),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    telemetry = GpuTelemetry(run_dir / "gpu.csv", config.telemetry_interval_seconds)
    try:
        server_info, startup_seconds = wait_for_server(
            base_url, process, config.server_timeout_seconds
        )
        assert_server_info(server_info, config, scheduler_cap, fdfo)
        write_json(run_dir / "server_info.json", server_info)
        if warmup_specs:
            warmup, warmup_wall_time = run_requests(
                base_url,
                warmup_specs,
                max_new_tokens=warmup_max_new_tokens,
                config=config,
            )
            write_jsonl(run_dir / "warmup_requests.jsonl", warmup)
            validate_request_results(warmup_specs, warmup)
            write_json(
                run_dir / "warmup_summary.json",
                summarize_results(warmup, warmup_wall_time),
            )
        telemetry.start()
        results, wall_time = run_requests(
            base_url,
            specs,
            max_new_tokens=max_new_tokens,
            config=config,
            loads_path=run_dir / "loads.jsonl",
        )
        telemetry.stop()
        write_jsonl(run_dir / "requests.jsonl", results)
        validate_request_results(specs, results)
        summary = summarize_results(results, wall_time)
        summary.update(
            {
                "fdfo": fdfo,
                "mode": "fdfo" if fdfo else "sync",
                "scheduler_cap": scheduler_cap,
                "server_startup_seconds": startup_seconds,
                "gpu": gpu_summary(run_dir / "gpu.csv"),
                "runtime_verified": True,
                "requests_completed_under_verified_fdfo": (
                    len(results) if fdfo else 0
                ),
                "launch_command": command,
            }
        )
        write_json(run_dir / "summary.json", summary)
        return summary, results
    finally:
        telemetry.stop()
        stop_server(process)
        log_handle.close()


def describe(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "min": None, "max": None, "stdev": None, "cv": None}
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": stdev,
        "cv": stdev / mean if mean else None,
    }


def mcnemar_exact_p(sync_only: int, fdfo_only: int) -> float:
    discordant = sync_only + fdfo_only
    if discordant == 0:
        return 1.0
    lower = min(sync_only, fdfo_only)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def compare_pair(
    sync_rows: Sequence[dict[str, Any]], fdfo_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    sync_by_id = {row["request_id"]: row for row in sync_rows}
    fdfo_by_id = {row["request_id"]: row for row in fdfo_rows}
    if sync_by_id.keys() != fdfo_by_id.keys():
        raise RuntimeError("paired runs do not contain identical request IDs")
    identifiers = sorted(sync_by_id)
    changed = [
        identifier
        for identifier in identifiers
        if sync_by_id[identifier].get("extracted_answer")
        != fdfo_by_id[identifier].get("extracted_answer")
    ]
    sync_only = sum(
        sync_by_id[key].get("correct") is True and fdfo_by_id[key].get("correct") is not True
        for key in identifiers
    )
    fdfo_only = sum(
        fdfo_by_id[key].get("correct") is True and sync_by_id[key].get("correct") is not True
        for key in identifiers
    )
    return {
        "answer_equivalence_rate": 1 - len(changed) / len(identifiers),
        "changed_answer_ids": changed,
        "mcnemar": {
            "sync_correct_fdfo_incorrect": sync_only,
            "sync_incorrect_fdfo_correct": fdfo_only,
            "exact_p_value": mcnemar_exact_p(sync_only, fdfo_only),
        },
    }


def build_comparison(
    measured: dict[int, dict[int, dict[str, dict[str, Any]]]],
    request_rows: dict[tuple[int, int, str], list[dict[str, Any]]],
    smoke: dict[str, Any],
) -> dict[str, Any]:
    caps: dict[str, Any] = {}
    all_correctness_pass = True
    for cap, repetitions in sorted(measured.items()):
        paired: list[dict[str, Any]] = []
        cell_values: dict[str, dict[str, list[float]]] = {
            mode: {
                "wall": [],
                "throughput": [],
                "requests_per_second": [],
                "latency_p50": [],
                "latency_p90": [],
                "latency_p95": [],
                "latency_p99": [],
                "latency_max": [],
                "accuracy": [],
                "gpu_utilization_mean": [],
                "gpu_utilization_p50": [],
                "gpu_utilization_p95": [],
                "gpu_peak_memory": [],
            }
            for mode in ("sync", "fdfo")
        }
        for repetition, modes in sorted(repetitions.items()):
            sync = modes["sync"]
            fdfo = modes["fdfo"]
            correctness = compare_pair(
                request_rows[(cap, repetition, "sync")],
                request_rows[(cap, repetition, "fdfo")],
            )
            accuracy_loss = (sync["accuracy"] or 0) - (fdfo["accuracy"] or 0)
            significant_regression = (
                correctness["mcnemar"]["sync_correct_fdfo_incorrect"]
                > correctness["mcnemar"]["sync_incorrect_fdfo_correct"]
                and correctness["mcnemar"]["exact_p_value"] < 0.05
            )
            correctness_pass = (
                accuracy_loss <= 0.0200000001
                and correctness["answer_equivalence_rate"] >= 0.9799999999
                and not significant_regression
            )
            all_correctness_pass &= correctness_pass
            paired.append(
                {
                    "repetition": repetition,
                    "latency_speedup": sync["wall_time_seconds"] / fdfo["wall_time_seconds"],
                    "throughput_speedup": fdfo["output_tokens_per_second"]
                    / sync["output_tokens_per_second"],
                    "sync_accuracy": sync["accuracy"],
                    "fdfo_accuracy": fdfo["accuracy"],
                    "sync_malformed_answer_rate": sync["malformed_answer_rate"],
                    "fdfo_malformed_answer_rate": fdfo["malformed_answer_rate"],
                    "accuracy_loss": accuracy_loss,
                    "significant_correctness_regression": significant_regression,
                    "correctness_pass": correctness_pass,
                    **correctness,
                }
            )
            for mode, summary in modes.items():
                cell_values[mode]["wall"].append(summary["wall_time_seconds"])
                cell_values[mode]["throughput"].append(summary["output_tokens_per_second"])
                cell_values[mode]["requests_per_second"].append(
                    summary["requests_per_second"]
                )
                for quantile in ("p50", "p90", "p95", "p99", "max"):
                    cell_values[mode][f"latency_{quantile}"].append(
                        summary["latency_seconds"][quantile]
                    )
                cell_values[mode]["accuracy"].append(summary["accuracy"])
                if summary.get("gpu"):
                    cell_values[mode]["gpu_utilization_mean"].append(
                        summary["gpu"]["utilization_gpu_percent"]["mean"]
                    )
                    cell_values[mode]["gpu_utilization_p50"].append(
                        summary["gpu"]["utilization_gpu_percent"]["p50"]
                    )
                    cell_values[mode]["gpu_utilization_p95"].append(
                        summary["gpu"]["utilization_gpu_percent"]["p95"]
                    )
                    cell_values[mode]["gpu_peak_memory"].append(
                        summary["gpu"]["peak_memory_used_mib"]
                    )
        caps[str(cap)] = {
            "paired_runs": paired,
            "latency_speedup": describe([item["latency_speedup"] for item in paired]),
            "throughput_speedup": describe(
                [item["throughput_speedup"] for item in paired]
            ),
            "cells": {
                mode: {
                    "wall_time_seconds": describe(values["wall"]),
                    "output_tokens_per_second": describe(values["throughput"]),
                    "requests_per_second": describe(values["requests_per_second"]),
                    "latency_p50_seconds": describe(values["latency_p50"]),
                    "latency_p90_seconds": describe(values["latency_p90"]),
                    "latency_p95_seconds": describe(values["latency_p95"]),
                    "latency_p99_seconds": describe(values["latency_p99"]),
                    "latency_max_seconds": describe(values["latency_max"]),
                    "accuracy": describe(values["accuracy"]),
                    "gpu_utilization_mean_percent": describe(
                        values["gpu_utilization_mean"]
                    ),
                    "gpu_utilization_p50_percent": describe(
                        values["gpu_utilization_p50"]
                    ),
                    "gpu_utilization_p95_percent": describe(
                        values["gpu_utilization_p95"]
                    ),
                    "gpu_peak_memory_used_mib": describe(values["gpu_peak_memory"]),
                }
                for mode, values in cell_values.items()
            },
        }
    primary = caps.get("16")
    cap4 = caps.get("4")
    meaningful = bool(
        primary
        and primary["latency_speedup"]["median"] >= 1.10
        and sum(item["latency_speedup"] > 1 for item in primary["paired_runs"]) >= 2
        and (cap4 is None or cap4["latency_speedup"]["median"] >= 0.95)
        and all_correctness_pass
    )
    integration_success = bool(smoke.get("passed") and all_correctness_pass)
    return {
        "schema_version": 1,
        "smoke": smoke,
        "caps": caps,
        "correctness_pass": all_correctness_pass,
        "integration_success": integration_success,
        "meaningful_speedup": meaningful,
        "outcome": (
            "meaningful speedup"
            if integration_success and meaningful
            else "integration successful; no meaningful speedup observed"
            if integration_success
            else "acceptance criteria failed"
        ),
    }


def render_report(comparison: dict[str, Any]) -> str:
    def optional_number(value: float | None, suffix: str = "") -> str:
        return "n/a" if value is None else f"{value:.3f}{suffix}"

    def optional_percent(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3%}"

    lines = [
        "# SGLang FDFO comparison",
        "",
        f"- Outcome: **{comparison['outcome']}**",
        f"- Integration success: `{comparison['integration_success']}`",
        f"- Meaningful speedup: `{comparison['meaningful_speedup']}`",
        f"- Correctness pass: `{comparison['correctness_pass']}`",
        f"- CUDA-graph mode: `{comparison.get('cuda_graph_mode', 'captured')}`",
        "",
        "| Cap | Sync median (s) | FDFO median (s) | "
        "Speedup median [min, max] | SD | CV | Correctness |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for cap, result in sorted(comparison["caps"].items(), key=lambda item: int(item[0])):
        sync = result["cells"]["sync"]["wall_time_seconds"]["median"]
        fdfo = result["cells"]["fdfo"]["wall_time_seconds"]["median"]
        speedup = result["latency_speedup"]
        correct = all(item["correctness_pass"] for item in result["paired_runs"])
        lines.append(
            f"| {cap} | {sync:.3f} | {fdfo:.3f} | "
            f"{speedup['median']:.3f}x [{speedup['min']:.3f}, {speedup['max']:.3f}] | "
            f"{speedup['stdev']:.3f} | {speedup['cv']:.3f} | {correct} |"
        )
    lines.extend(
        [
            "",
            "## Paired repetitions",
            "",
            "| Cap | Rep | Wall-time speedup | Token-throughput speedup | "
            "Sync acc. | FDFO acc. | Malformed S/F | Answer equiv. | McNemar p |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cap, result in sorted(comparison["caps"].items(), key=lambda item: int(item[0])):
        for pair in result["paired_runs"]:
            lines.append(
                f"| {cap} | {pair['repetition']} | {pair['latency_speedup']:.3f}x | "
                f"{pair['throughput_speedup']:.3f}x | {pair['sync_accuracy']:.3%} | "
                f"{pair['fdfo_accuracy']:.3%} | "
                f"{pair['sync_malformed_answer_rate']:.3%}/"
                f"{pair['fdfo_malformed_answer_rate']:.3%} | "
                f"{pair['answer_equivalence_rate']:.3%} | "
                f"{pair['mcnemar']['exact_p_value']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Median arm metrics",
            "",
            "| Cap | Mode | Requests/s | Output tokens/s | "
            "Latency p50/p90/p95/p99/max (s) | Accuracy | "
            "GPU util. mean/p50/p95 | "
            "Peak GPU memory (MiB) |",
            "|---:|:---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cap, result in sorted(comparison["caps"].items(), key=lambda item: int(item[0])):
        for mode in ("sync", "fdfo"):
            cell = result["cells"][mode]
            latency = "/".join(
                optional_number(cell[f"latency_{quantile}_seconds"]["median"])
                for quantile in ("p50", "p90", "p95", "p99", "max")
            )
            gpu_utilization = "/".join(
                optional_number(
                    cell[f"gpu_utilization_{statistic}_percent"]["median"], "%"
                )
                for statistic in ("mean", "p50", "p95")
            )
            lines.append(
                f"| {cap} | {mode} | "
                f"{optional_number(cell['requests_per_second']['median'])} | "
                f"{optional_number(cell['output_tokens_per_second']['median'])} | "
                f"{latency} | {optional_percent(cell['accuracy']['median'])} | "
                f"{gpu_utilization} | "
                f"{optional_number(cell['gpu_peak_memory_used_mib']['median'])} |"
            )
    lines.extend(
        [
            "",
            "Speedup is paired as sync wall time divided by FDFO wall time. "
            "Server startup and warm-up are excluded from the measured window.",
            "",
        ]
    )
    return "\n".join(lines)


def command_output(command: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def package_versions(names: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def find_model_snapshot(model: str, revision: str) -> dict[str, Any]:
    hub_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    candidate = hub_root / f"models--{model.replace('/', '--')}" / "snapshots" / revision
    result: dict[str, Any] = {
        "requested_revision": revision,
        "snapshot_path": str(candidate),
        "snapshot_exists": candidate.exists(),
    }
    index = candidate / "model.safetensors.index.json"
    if index.exists():
        result["model_index_sha256"] = sha256_file(index)
    if candidate.exists():
        manifest = []
        for path in sorted(candidate.rglob("*")):
            if not path.is_file():
                continue
            stat = path.stat()
            manifest.append(
                {
                    "path": str(path.relative_to(candidate)),
                    "size": stat.st_size,
                    "link_target": os.readlink(path) if path.is_symlink() else None,
                }
            )
        result["snapshot_manifest_sha256"] = sha256_bytes(
            json.dumps(manifest, sort_keys=True).encode("utf-8")
        )
        result["snapshot_file_count"] = len(manifest)
    return result


def collect_environment(
    config: RunnerConfig,
    sglang_info: dict[str, str],
    dataset_path: Path,
    launch_commands: Sequence[Sequence[str]],
) -> dict[str, Any]:
    return {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository": {
            "root": str(REPO_ROOT),
            "revision": command_output(["git", "rev-parse", "HEAD"]),
            "status": command_output(["git", "status", "--short"]),
        },
        "python": {"executable": sys.executable, "version": sys.version},
        "packages": package_versions(
            [
                "sglang",
                "torch",
                "transformers",
                "flashinfer-python",
                "sglang-kernel",
                "cuda-python",
                "aiohttp",
                "PyYAML",
            ]
        ),
        "pip_freeze": (command_output([sys.executable, "-m", "pip", "freeze"]) or "").splitlines(),
        "sglang": {**sglang_info, "tag_commit": SGLANG_TAG_COMMIT},
        "model": {
            "id": config.model,
            "revision": config.model_revision,
            **find_model_snapshot(config.model, config.model_revision),
        },
        "dataset": {
            "url": GSM8K_URL,
            "revision": GSM8K_REVISION,
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
        },
        "hardware": {
            "nvidia_smi": command_output(["nvidia-smi"]),
            "gpu_query": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
                    "--format=csv,noheader",
                ]
            ),
        },
        "decoding_config": {
            "path": str(THIS_DIR / "joint_threshold.yaml"),
            "sha256": sha256_file(THIS_DIR / "joint_threshold.yaml"),
            "contents": (THIS_DIR / "joint_threshold.yaml").read_text(
                encoding="utf-8"
            ),
        },
        "launch_commands": [list(command) for command in launch_commands],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--scheduler-caps", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/sglang_fdfo")
    parser.add_argument("--dataset-cache", type=Path, default=REPO_ROOT / ".cache/gsm8k/test.jsonl")
    parser.add_argument("--task-yaml", type=Path, default=DEFAULT_TASK_YAML)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--server-timeout-seconds", type=float, default=1800)
    parser.add_argument("--request-timeout-seconds", type=float, default=900)
    parser.add_argument(
        "--cuda-graph-batch-sizes",
        type=int,
        nargs="+",
        help=(
            "Memory fallback: capture only these sizes (values above each "
            "scheduler cap are omitted)"
        ),
    )
    parser.add_argument(
        "--disable-cuda-graphs",
        action="store_true",
        help="Eager-mode fallback; applied identically to both benchmark arms",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="For debugging only; acceptance will fail",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_examples != 100:
        raise ValueError("the acceptance experiment requires exactly 100 examples")
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    if len(set(args.scheduler_caps)) != len(args.scheduler_caps) or any(
        cap <= 0 for cap in args.scheduler_caps
    ):
        raise ValueError("scheduler caps must be unique positive integers")
    if args.cuda_graph_batch_sizes and (
        args.cuda_graph_batch_sizes != sorted(set(args.cuda_graph_batch_sizes))
        or any(value <= 0 for value in args.cuda_graph_batch_sizes)
        or 1 not in args.cuda_graph_batch_sizes
    ):
        raise ValueError(
            "CUDA-graph batch sizes must be unique increasing positive integers "
            "and include 1 for the parity smoke test"
        )
    if args.disable_cuda_graphs and args.cuda_graph_batch_sizes:
        raise ValueError(
            "--disable-cuda-graphs and --cuda-graph-batch-sizes are mutually exclusive"
        )
    sglang_info = preflight_sglang()
    config = RunnerConfig(
        model=args.model,
        model_revision=args.model_revision,
        server_timeout_seconds=args.server_timeout_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
        cuda_graph_batch_sizes=(
            tuple(args.cuda_graph_batch_sizes)
            if args.cuda_graph_batch_sizes is not None
            else None
        ),
        disable_cuda_graphs=args.disable_cuda_graphs,
    )
    dataset_path = download_gsm8k(args.dataset_cache)
    dataset = read_jsonl(dataset_path)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model, revision=config.model_revision, trust_remote_code=True
    )
    few_shots = load_few_shots(args.task_yaml)
    manifest_rows = build_manifest(
        dataset, tokenizer, args.num_examples, few_shots=few_shots
    )
    warmup_count = max(args.scheduler_caps)
    warmup_rows = build_manifest(
        dataset,
        tokenizer,
        warmup_count,
        start_index=args.num_examples,
        seed_base=91234,
        few_shots=few_shots,
    )
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = args.output_dir / timestamp
    output_root.mkdir(parents=True, exist_ok=False)
    write_json(
        output_root / "manifest.json",
        {
            "schema_version": 1,
            "dataset_revision": GSM8K_REVISION,
            "dataset_sha256": GSM8K_SHA256,
            "model": config.model,
            "model_revision": config.model_revision,
            "task_yaml": str(args.task_yaml),
            "task_yaml_sha256": sha256_file(args.task_yaml),
            "generation": {
                "temperature": 0.0,
                "max_new_tokens": args.max_new_tokens,
                "stop": STOP_SEQUENCES,
                "server_random_seed": config.random_seed,
                "request_seed_metadata_formula": "1234 + dataset_index",
                "per_request_seed_supported": False,
            },
            "experiment": {
                "client_concurrency": args.num_examples,
                "scheduler_caps": args.scheduler_caps,
                "repetitions": args.repetitions,
                "warmup_max_new_tokens": 128,
                "server_per_arm": True,
                "repetition_order": "odd=sync,fdfo; even=fdfo,sync",
                "cuda_graphs_disabled": config.disable_cuda_graphs,
                "requested_cuda_graph_batch_sizes": config.cuda_graph_batch_sizes,
            },
            "examples": manifest_rows,
            "warmup_examples": warmup_rows,
        },
    )
    specs = specs_from_manifest(manifest_rows)
    warmup_specs = specs_from_manifest(warmup_rows)
    launch_commands: list[list[str]] = []
    write_json(
        output_root / "environment.json",
        collect_environment(config, sglang_info, dataset_path, launch_commands),
    )
    smoke_result: dict[str, Any] = {"passed": False, "skipped": args.skip_smoke}
    if not args.skip_smoke:
        parity_rows: dict[str, list[dict[str, Any]]] = {}
        parity_specs = [
            RequestSpec(f"parity-{index}", prompt, 7000 + index)
            for index, prompt in enumerate(UPSTREAM_PARITY_PROMPTS)
        ]
        for fdfo in (False, True):
            mode = "fdfo" if fdfo else "sync"
            run_dir = output_root / "smoke/parity" / mode
            summary, parity_rows[mode] = run_arm(
                config,
                run_dir=run_dir,
                scheduler_cap=1,
                fdfo=fdfo,
                specs=parity_specs,
                warmup_specs=[],
                max_new_tokens=128,
            )
            launch_commands.append(summary["launch_command"])
        sync_text = [row["text"] for row in parity_rows["sync"]]
        fdfo_text = [row["text"] for row in parity_rows["fdfo"]]
        if sync_text != fdfo_text:
            raise RuntimeError("cap-1 FDFO output is not byte-identical to sync output")
        lifecycle_specs = specs[:8]
        for fdfo in (False, True):
            mode = "fdfo" if fdfo else "sync"
            summary, _ = run_arm(
                config,
                run_dir=output_root / "smoke/lifecycle" / mode,
                scheduler_cap=4,
                fdfo=fdfo,
                specs=lifecycle_specs,
                warmup_specs=[],
                max_new_tokens=128,
            )
            launch_commands.append(summary["launch_command"])
        smoke_result = {
            "passed": True,
            "skipped": False,
            "single_request_byte_identical": True,
            "lifecycle_requests_per_mode": 8,
        }
        write_json(output_root / "smoke/summary.json", smoke_result)

    measured: dict[int, dict[int, dict[str, dict[str, Any]]]] = {}
    request_rows: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for cap in args.scheduler_caps:
        measured[cap] = {}
        for repetition in range(1, args.repetitions + 1):
            measured[cap][repetition] = {}
            order = [False, True] if repetition % 2 else [True, False]
            for fdfo in order:
                mode = "fdfo" if fdfo else "sync"
                run_dir = output_root / f"runs/cap_{cap}/rep_{repetition}/{mode}"
                summary, rows = run_arm(
                    config,
                    run_dir=run_dir,
                    scheduler_cap=cap,
                    fdfo=fdfo,
                    specs=specs,
                    warmup_specs=warmup_specs[:cap],
                    max_new_tokens=args.max_new_tokens,
                )
                launch_commands.append(summary["launch_command"])
                measured[cap][repetition][mode] = summary
                request_rows[(cap, repetition, mode)] = rows
    comparison = build_comparison(measured, request_rows, smoke_result)
    comparison["cuda_graph_mode"] = (
        "eager fallback" if config.disable_cuda_graphs else "captured"
    )
    comparison["requested_cuda_graph_batch_sizes"] = config.cuda_graph_batch_sizes
    write_json(output_root / "comparison.json", comparison)
    (output_root / "report.md").write_text(
        render_report(comparison), encoding="utf-8"
    )
    write_json(
        output_root / "environment.json",
        collect_environment(config, sglang_info, dataset_path, launch_commands),
    )
    print(f"Results: {output_root}")
    print(f"Outcome: {comparison['outcome']}")
    return 0 if comparison["integration_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
