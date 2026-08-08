"""Runtime-only timing hooks for SGLang 0.5.16's FDFO implementation.

The module is imported through ``sitecustomize`` only when
``SGLANG_FDFO_PROFILE_DIR`` is set.  Every wrapper delegates to the original
method exactly once and preserves its arguments and return value.  Timing data
is written to per-process JSONL files so SGLang's spawned processes never share
an open file handle.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import torch


SCHEMA_VERSION = 1
_ACTIVE: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "sglang_fdfo_profile_active", default=None
)
_INSTALLED = False


def _write(record: dict[str, Any]) -> None:
    output_dir = Path(os.environ["SGLANG_FDFO_PROFILE_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": SCHEMA_VERSION,
        "pid": os.getpid(),
        "time_ns": time.time_ns(),
        **record,
    }
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path = output_dir / f"events-{os.getpid()}.jsonl"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def _cpu_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            return [int(item) for item in value.detach().cpu().tolist()]
        return [int(item) for item in value]
    except Exception:
        return None


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def _elapsed(pair: tuple[torch.cuda.Event, torch.cuda.Event] | None) -> float | None:
    if pair is None:
        return None
    try:
        return float(pair[0].elapsed_time(pair[1]))
    except Exception:
        return None


def _sum_elapsed(pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]]) -> float | None:
    values = [value for pair in pairs if (value := _elapsed(pair)) is not None]
    return float(sum(values)) if values else None


def _nvtx(label: str):
    if os.environ.get("SGLANG_FDFO_PROFILE_NVTX") != "1":
        return contextlib.nullcontext()
    return torch.cuda.nvtx.range(label)


def _can_detail() -> bool:
    try:
        return torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def _profile_nested(
    original: Callable[..., Any],
    bucket: str,
    label: str,
    *,
    metadata: Callable[[tuple[Any, ...], dict[str, Any]], dict[str, Any]] | None = None,
) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any):
        active = _ACTIVE.get()
        if active is None or not _can_detail():
            return original(*args, **kwargs)
        start = _event()
        end = _event()
        cpu_start = time.perf_counter_ns()
        start.record()
        with _nvtx(label):
            result = original(*args, **kwargs)
        end.record()
        item = {
            "events": (start, end),
            "cpu_ms": (time.perf_counter_ns() - cpu_start) / 1e6,
        }
        if metadata is not None:
            item.update(metadata(args, kwargs))
        active[bucket].append(item)
        return result

    return wrapped


def _iteration_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapped(self: Any, model_runner: Any, forward_batch: Any, algo_states: Any):
        batch_size = int(forward_batch.batch_size)
        carried_state = (
            [state is not None for state in algo_states]
            if algo_states is not None
            else [False] * batch_size
        )
        active: dict[str, Any] = {
            "attention": [],
            "blocks": [],
            "kv_writes": [],
            "logits": [],
            "model_pair": None,
            "model_cpu_ms": None,
            "step_pair": None,
            "step_cpu_ms": None,
            "done": None,
        }
        total_start = _event()
        total_end = _event()
        wall_start_ns = time.perf_counter_ns()
        started_ns = time.time_ns()
        token = _ACTIVE.set(active)
        total_start.record()
        try:
            with _nvtx(f"fdfo_iteration_bs_{batch_size}"):
                result = original(self, model_runner, forward_batch, algo_states)
            return result
        finally:
            total_end.record()
            total_end.synchronize()
            wall_ms = (time.perf_counter_ns() - wall_start_ns) / 1e6
            _ACTIVE.reset(token)

            result_value = locals().get("result")
            accept = result_value[2] if result_value is not None else None
            accept_list = _cpu_list(accept)
            if accept_list is not None:
                accepted = [value > 0 for value in accept_list]
            elif active["done"] is not None:
                accepted = [bool(value) for value in active["done"]]
            else:
                accepted = [False] * batch_size
            commit_count = sum(
                accepted[index] and carried_state[index]
                for index in range(batch_size)
            )
            prompt_prefill_count = sum(
                accepted[index] and not carried_state[index]
                for index in range(batch_size)
            )
            normal_count = batch_size - commit_count - prompt_prefill_count
            if prompt_prefill_count:
                composition = "prompt_prefill"
            elif commit_count == 0:
                composition = "all_normal"
            elif commit_count == batch_size:
                composition = "all_commit"
            else:
                composition = "mixed"

            attention_ms = _sum_elapsed(
                [item["events"] for item in active["attention"]]
            )
            block_items = [
                {
                    "layer_id": item.get("layer_id"),
                    "gpu_ms": _elapsed(item["events"]),
                    "cpu_ms": item["cpu_ms"],
                }
                for item in active["blocks"]
            ]
            kv_items = [
                {
                    "layer_id": item.get("layer_id"),
                    "gpu_ms": _elapsed(item["events"]),
                    "cpu_ms": item["cpu_ms"],
                    "bytes": item.get("bytes", 0),
                    "tokens": item.get("tokens"),
                }
                for item in active["kv_writes"]
            ]
            logits_ms = _sum_elapsed([item["events"] for item in active["logits"]])
            model_gpu_ms = _elapsed(active["model_pair"])
            total_gpu_ms = _elapsed((total_start, total_end))
            step_gpu_ms = _elapsed(active["step_pair"])
            kv_gpu_ms = sum(
                item["gpu_ms"] for item in kv_items if item["gpu_ms"] is not None
            )
            block_gpu_ms = sum(
                item["gpu_ms"] for item in block_items if item["gpu_ms"] is not None
            )
            cpu_overhead_ms = (
                max(0.0, wall_ms - total_gpu_ms)
                if total_gpu_ms is not None
                else None
            )
            _write(
                {
                    "type": "fdfo_iteration",
                    "started_time_ns": started_ns,
                    "batch_size": batch_size,
                    "block_size": int(self.block_size),
                    "num_forward_tokens": int(forward_batch.input_ids.numel()),
                    "seq_lens": _cpu_list(getattr(forward_batch, "seq_lens_cpu", None))
                    or _cpu_list(getattr(forward_batch, "seq_lens", None)),
                    "prefix_lens": _cpu_list(
                        getattr(forward_batch, "extend_prefix_lens_cpu", None)
                    ),
                    "extend_lens": _cpu_list(
                        getattr(forward_batch, "extend_seq_lens_cpu", None)
                    ),
                    "commit_count": commit_count,
                    "prompt_prefill_count": prompt_prefill_count,
                    "normal_count": normal_count,
                    "commit_fraction": commit_count / batch_size,
                    "composition": composition,
                    "accept_lengths": accept_list,
                    "cuda_graph": bool(result_value[4]) if result_value is not None else None,
                    "wall_ms": wall_ms,
                    "total_gpu_ms": total_gpu_ms,
                    "model_gpu_ms": model_gpu_ms,
                    "model_cpu_ms": active["model_cpu_ms"],
                    "attention_gpu_ms": attention_ms,
                    "transformer_blocks_gpu_ms": block_gpu_ms or None,
                    "logits_gpu_ms": logits_ms,
                    "kv_write_gpu_ms": kv_gpu_ms or None,
                    "kv_write_bytes": sum(item["bytes"] for item in kv_items),
                    "step_gpu_ms": step_gpu_ms,
                    "step_cpu_ms": active["step_cpu_ms"],
                    "cpu_overhead_ms": cpu_overhead_ms,
                    "attention_calls": len(active["attention"]),
                    "kv_write_calls": len(kv_items),
                    "block_times": block_items,
                    "kv_writes": kv_items,
                }
            )

    return wrapped


def _model_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any):
        active = _ACTIVE.get()
        if active is None:
            return original(*args, **kwargs)
        start = _event()
        end = _event()
        cpu_start = time.perf_counter_ns()
        start.record()
        with _nvtx("fdfo_model_forward"):
            result = original(*args, **kwargs)
        end.record()
        active["model_pair"] = (start, end)
        active["model_cpu_ms"] = (time.perf_counter_ns() - cpu_start) / 1e6
        return result

    return wrapped


def _step_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any):
        active = _ACTIVE.get()
        if active is None:
            return original(*args, **kwargs)
        start = _event()
        end = _event()
        cpu_start = time.perf_counter_ns()
        start.record()
        with _nvtx("fdfo_token_selection_step"):
            result = original(*args, **kwargs)
        end.record()
        active["step_pair"] = (start, end)
        active["step_cpu_ms"] = (time.perf_counter_ns() - cpu_start) / 1e6
        active["done"] = result
        return result

    return wrapped


def _kv_metadata(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    layer = args[1] if len(args) > 1 else kwargs.get("layer")
    cache_k = args[3] if len(args) > 3 else kwargs.get("cache_k")
    cache_v = args[4] if len(args) > 4 else kwargs.get("cache_v")
    byte_count = 0
    token_count = None
    for tensor in (cache_k, cache_v):
        if isinstance(tensor, torch.Tensor):
            byte_count += tensor.numel() * tensor.element_size()
            token_count = int(tensor.shape[0]) if tensor.ndim else 1
    return {
        "layer_id": getattr(layer, "layer_id", None),
        "bytes": int(byte_count),
        "tokens": token_count,
    }


def _block_metadata(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    block = args[0]
    attention = getattr(block, "attention", None)
    radix_attention = getattr(attention, "attn", None)
    return {"layer_id": getattr(radix_attention, "layer_id", None)}


def _scheduler_wrapper(original: Callable[..., Any], record_type: str):
    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any):
        started_ns = time.time_ns()
        cpu_start = time.perf_counter_ns()
        result = original(*args, **kwargs)
        cpu_ms = (time.perf_counter_ns() - cpu_start) / 1e6
        record: dict[str, Any] = {
            "type": record_type,
            "started_time_ns": started_ns,
            "cpu_ms": cpu_ms,
        }
        if record_type == "scheduler_prepare":
            record["batch_size"] = result.batch_size() if result is not None else 0
            # The scheduler polls this method while idle. Those empty calls are
            # neither batch preparation nor part of a dLLM iteration and would
            # otherwise dominate the event stream.
            if result is None:
                return result
        else:
            batch = args[1] if len(args) > 1 else kwargs.get("batch")
            generation_result = args[2] if len(args) > 2 else kwargs.get("result")
            accept = _cpu_list(
                getattr(generation_result, "accept_length_per_req_cpu", None)
            )
            record.update(
                {
                    "batch_size": batch.batch_size() if batch is not None else None,
                    "commit_count": (
                        sum(value > 0 for value in accept) if accept is not None else None
                    ),
                }
            )
        _write(record)
        return result

    return wrapped


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from sglang.srt.dllm.algorithm.base import DllmAlgorithm
    from sglang.srt.dllm.algorithm.joint_threshold import JointThreshold
    from sglang.srt.dllm.algorithm.low_confidence import LowConfidence
    from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.models.llada2 import LLaDA2MoeBlock

    DllmAlgorithm._run_fdfo = _iteration_wrapper(DllmAlgorithm._run_fdfo)
    from sglang.srt.model_executor.model_runner import ModelRunner

    ModelRunner.forward = _model_wrapper(ModelRunner.forward)
    JointThreshold.step = _step_wrapper(JointThreshold.step)
    LowConfidence.step = _step_wrapper(LowConfidence.step)
    RadixAttention.forward = _profile_nested(
        RadixAttention.forward, "attention", "fdfo_attention_backend"
    )
    LLaDA2MoeBlock.forward = _profile_nested(
        LLaDA2MoeBlock.forward,
        "blocks",
        "fdfo_transformer_block",
        metadata=_block_metadata,
    )
    LogitsProcessor.forward = _profile_nested(
        LogitsProcessor.forward, "logits", "fdfo_logits_processor"
    )
    MHATokenToKVPool.set_kv_buffer = _profile_nested(
        MHATokenToKVPool.set_kv_buffer,
        "kv_writes",
        "fdfo_paged_kv_write",
        metadata=_kv_metadata,
    )
    SchedulerDllmMixin.get_new_batch_dllm = _scheduler_wrapper(
        SchedulerDllmMixin.get_new_batch_dllm, "scheduler_prepare"
    )
    SchedulerDllmMixin.process_batch_result_dllm = _scheduler_wrapper(
        SchedulerDllmMixin.process_batch_result_dllm, "scheduler_result"
    )

    _write(
        {
            "type": "profiler_installed",
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "nvtx": os.environ.get("SGLANG_FDFO_PROFILE_NVTX") == "1",
        }
    )
