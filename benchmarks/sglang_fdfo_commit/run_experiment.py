#!/usr/bin/env python3
"""Execute the SGLang FDFO final-commit profiling experiment on one GPU."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import importlib.metadata
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
INJECT_DIR = THIS_DIR / "profile_inject"
MODEL = "inclusionAI/LLaDA2.1-mini"
MODEL_REVISION = "20e64e2ad21644d0e5248586ed9c942cdd45de0f"
SGLANG_VERSION = "0.5.16"
CONTEXT_LENGTHS = (128, 1024, 4096)
BLOCK_SIZES = (8, 16, 32, 64)
BATCH_SIZES = (1, 4, 16)
BASE_TEXTS = (
    "Explain why the sky appears blue during the day.",
    "Solve carefully: if twelve boxes contain seven items each, how many items are there?",
    "Write a concise description of a quiet library in winter.",
    "A train travels east for eighty miles and north for sixty miles. Summarize the trip.",
    "List three properties of prime numbers and give a small example.",
    "Describe the steps needed to make a cup of tea.",
    "What is the difference between weather and climate?",
    "Compute the area of a rectangle with length thirteen and width nine.",
    "Give a short argument for testing software before release.",
    "Explain photosynthesis to a student using plain language.",
    "A farmer has forty apples and gives away seventeen. How many remain?",
    "Write two sentences about a spacecraft approaching Mars.",
    "Compare a river and a road in one short paragraph.",
    "What makes a measurement reproducible in an experiment?",
    "Summarize how a seed grows into a plant.",
    "Calculate five squared plus six squared and show the result.",
)


@dataclass(frozen=True)
class ServerSpec:
    block_size: int
    graph_mode: str
    output_dir: Path


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def command_output(command: Sequence[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server(base_url: str, process: subprocess.Popen, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode}")
        try:
            return http_json(f"{base_url}/server_info", timeout=5)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
            time.sleep(1)
    raise TimeoutError(f"server startup timed out: {last_error}")


def stop_process(process: subprocess.Popen) -> None:
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


def make_config(path: Path, block_size: int) -> None:
    path.write_text(
        "\n".join(
            (
                f"block_size: {block_size}",
                "threshold: 0.5",
                "edit_threshold: 0.0",
                "max_post_edit_steps: 16",
                "penalty_lambda: 0",
                "",
            )
        ),
        encoding="utf-8",
    )


def server_command(spec: ServerSpec, port: int, config_path: Path) -> list[str]:
    # SGLang's dLLM configuration normalizes the paged-cache granularity to
    # the denoising block size. State it explicitly and verify that contract.
    page_size = spec.block_size
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL,
        "--revision",
        MODEL_REVISION,
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--tp-size",
        "1",
        "--mem-fraction-static",
        "0.9",
        "--attention-backend",
        "flashinfer",
        "--dllm-algorithm",
        "JointThreshold",
        "--dllm-algorithm-config",
        str(config_path),
        "--page-size",
        str(page_size),
        "--max-running-requests",
        "16",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--random-seed",
        "1234",
        "--dllm-fdfo",
    ]
    if spec.graph_mode == "eager":
        command.extend(["--disable-decode-cuda-graph", "--disable-prefill-cuda-graph"])
    else:
        command.extend(["--cuda-graph-bs", *[str(value) for value in range(1, 17)]])
    return command


def child_environment(profile_dir: Path, nvtx: bool) -> dict[str, str]:
    environment = os.environ.copy()
    existing = [item for item in environment.get("PYTHONPATH", "").split(os.pathsep) if item]
    blocked = {
        str((REPO_ROOT / "third_party/sglang/python").resolve()),
        str((REPO_ROOT / "third_party/dinfer/python").resolve()),
    }
    clean = [item for item in existing if str(Path(item).resolve()) not in blocked]
    environment["PYTHONPATH"] = os.pathsep.join([str(INJECT_DIR), *clean])
    environment["SGLANG_FDFO_PROFILE_DIR"] = str(profile_dir)
    environment["SGLANG_FDFO_PROFILE_NVTX"] = "1" if nvtx else "0"
    environment["FLASHINFER_WORKSPACE_BASE"] = "/tmp/fdfo-flashinfer"
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def exact_length_inputs(tokenizer: Any, length: int, batch_size: int) -> list[list[int]]:
    rows: list[list[int]] = []
    for index in range(batch_size):
        prefix = f"Request {index}. "
        text = prefix + BASE_TEXTS[index % len(BASE_TEXTS)]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            raise RuntimeError("tokenizer produced an empty input")
        repeated = (tokens * (length // len(tokens) + 1))[:length]
        rows.append([int(token) for token in repeated])
    return rows


async def send_one(
    session: Any,
    base_url: str,
    rid: str,
    input_ids: list[int],
    max_new_tokens: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload = {
        "input_ids": input_ids,
        "rid": rid,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
    }
    started_ns = time.time_ns()
    error = None
    status = 0
    response_value: dict[str, Any] | None = None
    try:
        async with session.post(
            f"{base_url}/generate", json=payload, timeout=timeout_seconds
        ) as response:
            status = response.status
            body = await response.text()
            try:
                parsed = json.loads(body)
                response_value = parsed if isinstance(parsed, dict) else {"value": parsed}
            except json.JSONDecodeError:
                error = f"non-JSON response: {body[:500]}"
            if status != 200 and error is None:
                error = f"HTTP {status}: {body[:500]}"
    except Exception as exc:
        error = repr(exc)
    finished_ns = time.time_ns()
    meta = (response_value or {}).get("meta_info") or {}
    return {
        "rid": rid,
        "status": status,
        "error": error,
        "started_time_ns": started_ns,
        "finished_time_ns": finished_ns,
        "latency_ms": (finished_ns - started_ns) / 1e6,
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": meta.get("completion_tokens") or meta.get("output_tokens"),
    }


async def run_batch_async(
    base_url: str,
    input_rows: list[list[int]],
    rid_prefix: str,
    max_new_tokens: int,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    import aiohttp

    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            asyncio.create_task(
                send_one(
                    session,
                    base_url,
                    f"{rid_prefix}-{index}",
                    input_ids,
                    max_new_tokens,
                    timeout_seconds,
                )
            )
            for index, input_ids in enumerate(input_rows)
        ]
        return await asyncio.gather(*tasks)


def run_workload(
    base_url: str,
    tokenizer: Any,
    output_dir: Path,
    *,
    block_size: int,
    graph_mode: str,
    context_length: int,
    batch_size: int,
    repetition: int,
    blocks_per_request: int,
    timeout_seconds: float,
    warmup: bool = False,
) -> dict[str, Any]:
    name = (
        f"{'warmup' if warmup else 'measure'}-graph_{graph_mode}-block_{block_size}"
        f"-ctx_{context_length}-bs_{batch_size}-rep_{repetition}"
    )
    input_rows = exact_length_inputs(tokenizer, context_length, batch_size)
    started_ns = time.time_ns()
    rows = asyncio.run(
        run_batch_async(
            base_url,
            input_rows,
            name,
            block_size * blocks_per_request,
            timeout_seconds,
        )
    )
    finished_ns = time.time_ns()
    failures = [row for row in rows if row["status"] != 200 or row["error"]]
    record = {
        "name": name,
        "warmup": warmup,
        "graph_mode": graph_mode,
        "block_size": block_size,
        "context_length": context_length,
        "batch_size": batch_size,
        "repetition": repetition,
        "blocks_per_request": blocks_per_request,
        "max_new_tokens": block_size * blocks_per_request,
        "started_time_ns": started_ns,
        "finished_time_ns": finished_ns,
        "wall_ms": (finished_ns - started_ns) / 1e6,
        "requests": rows,
        "failure_count": len(failures),
    }
    append_jsonl(output_dir / "workloads.jsonl", record)
    if failures:
        raise RuntimeError(f"workload {name} failed: {failures}")
    return record


class Telemetry:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return
        self.handle = self.path.open("w", encoding="utf-8")
        self.handle.write(
            "timestamp,index,utilization_gpu_percent,utilization_memory_percent,memory_used_mib,power_draw_watts,clocks_sm_mhz\n"
        )
        self.handle.flush()
        query = "timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm"
        self.process = subprocess.Popen(
            [
                executable,
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
                "--loop-ms=100",
            ],
            stdout=self.handle,
            stderr=subprocess.DEVNULL,
            text=True,
        )

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


def run_server_sweep(
    spec: ServerSpec,
    tokenizer: Any,
    args: argparse.Namespace,
    contexts: Iterable[int],
    batches: Iterable[int],
) -> list[dict[str, Any]]:
    spec.output_dir.mkdir(parents=True, exist_ok=False)
    profile_dir = spec.output_dir / "profile"
    profile_dir.mkdir()
    config_path = spec.output_dir / "joint_threshold.yaml"
    make_config(config_path, spec.block_size)
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    command = server_command(spec, port, config_path)
    write_json(spec.output_dir / "launch.json", {"command": command})
    log_handle = (spec.output_dir / "server.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=child_environment(profile_dir, args.nvtx),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    telemetry = Telemetry(spec.output_dir / "gpu.csv")
    records: list[dict[str, Any]] = []
    try:
        info = wait_for_server(base_url, process, args.server_timeout_seconds)
        write_json(spec.output_dir / "server_info.json", info)
        expected = {
            "version": SGLANG_VERSION,
            "dllm_fdfo": True,
            "dllm_algorithm": "JointThreshold",
            "max_running_requests": 16,
            "page_size": spec.block_size,
        }
        mismatches = {
            key: {"expected": value, "actual": info.get(key)}
            for key, value in expected.items()
            if info.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"server configuration mismatch: {mismatches}")
        run_workload(
            base_url,
            tokenizer,
            spec.output_dir,
            block_size=spec.block_size,
            graph_mode=spec.graph_mode,
            context_length=128,
            batch_size=4,
            repetition=0,
            blocks_per_request=1,
            timeout_seconds=args.request_timeout_seconds,
            warmup=True,
        )
        telemetry.start()
        for context_length in contexts:
            for batch_size in batches:
                for repetition in range(1, args.repetitions + 1):
                    records.append(
                        run_workload(
                            base_url,
                            tokenizer,
                            spec.output_dir,
                            block_size=spec.block_size,
                            graph_mode=spec.graph_mode,
                            context_length=context_length,
                            batch_size=batch_size,
                            repetition=repetition,
                            blocks_per_request=args.blocks_per_request,
                            timeout_seconds=args.request_timeout_seconds,
                        )
                    )
        return records
    finally:
        telemetry.stop()
        stop_process(process)
        log_handle.close()


def collect_environment() -> dict[str, Any]:
    try:
        sglang_version = importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        sglang_version = None
    return {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "sglang_version": sglang_version,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "nvidia_smi": command_output(["nvidia-smi"]),
        "gpu_query": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ]
        ),
        "git_head": command_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": (command_output(["git", "status", "--short"]) or "").splitlines(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/sglang_fdfo_commit")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=list(BLOCK_SIZES))
    parser.add_argument("--context-lengths", type=int, nargs="+", default=list(CONTEXT_LENGTHS))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(BATCH_SIZES))
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--blocks-per-request", type=int, default=2)
    parser.add_argument("--skip-cuda-graph", action="store_true")
    parser.add_argument("--skip-bandwidth", action="store_true")
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--server-timeout-seconds", type=float, default=1800)
    parser.add_argument("--request-timeout-seconds", type=float, default=900)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if importlib.metadata.version("sglang") != SGLANG_VERSION:
        raise RuntimeError(f"the experiment requires sglang=={SGLANG_VERSION}")
    if args.repetitions < 1 or args.blocks_per_request < 1:
        raise ValueError("repetitions and blocks-per-request must be positive")
    if any(value not in BLOCK_SIZES for value in args.block_sizes):
        raise ValueError(f"supported block sizes are {BLOCK_SIZES}")
    if any(value not in BATCH_SIZES for value in args.batch_sizes):
        raise ValueError(f"supported batch sizes are {BATCH_SIZES}")
    if any(value <= 0 or value % max(args.block_sizes) for value in args.context_lengths):
        raise ValueError("context lengths must be positive multiples of the largest block size")

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = args.output_dir / timestamp
    output_root.mkdir(parents=True, exist_ok=False)
    write_json(output_root / "environment.json", collect_environment())
    write_json(output_root / "arguments.json", vars(args) | {"output_dir": str(args.output_dir)})

    if not args.skip_bandwidth:
        subprocess.run(
            [
                sys.executable,
                str(THIS_DIR / "measure_bandwidth.py"),
                "--output",
                str(output_root / "hbm_bandwidth.json"),
            ],
            cwd=REPO_ROOT,
            check=True,
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=MODEL_REVISION, trust_remote_code=True
    )
    all_records: list[dict[str, Any]] = []
    for block_size in args.block_sizes:
        all_records.extend(
            run_server_sweep(
                ServerSpec(block_size, "eager", output_root / f"eager/block_{block_size}"),
                tokenizer,
                args,
                args.context_lengths,
                args.batch_sizes,
            )
        )
    if not args.skip_cuda_graph:
        all_records.extend(
            run_server_sweep(
                ServerSpec(32, "captured", output_root / "captured/block_32"),
                tokenizer,
                args,
                (1024,),
                args.batch_sizes,
            )
        )
    write_json(output_root / "run_summary.json", {"workloads": len(all_records)})
    subprocess.run(
        [sys.executable, str(THIS_DIR / "analyze.py"), str(output_root)],
        cwd=REPO_ROOT,
        check=True,
    )
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
