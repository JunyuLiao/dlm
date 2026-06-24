#!/usr/bin/env python3
"""Measure GPU / SM utilization during typical LLaDA batched block execution.

Reuses the real forward path from llada_block_step_probe.py and samples GPU
metrics in a background thread while the model runs. Use this to see whether a
normal batch execution keeps the GPU busy or leaves SMs idle.

Example:
  python3 scripts/llada_gpu_util_probe.py \
    --batch-sizes 1 2 4 8 16 \
    --block-size 32 \
    --target-prompt-tokens 300 \
    --num-blocks 1 \
    --warmup-steps 2 \
    --sample-ms 50
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llada_block_step_probe import (
    PromptRecord,
    append_mask_block,
    build_prompt_batch,
    choose_mask_token_id,
    cuda_sync,
    disable_use_cache,
    ensure_all_tied_weights_keys,
    get_num_transfer_tokens,
    patch_llada_transformers_compat,
)
from llada_length_heterogeneity_probe import make_prompt_for_target

try:
    import pynvml
except ImportError:
    pynvml = None  # type: ignore[assignment]

_NVML_INITIALIZED = False
_NVML_HANDLES: dict[int, object] = {}


def get_nvml_handle(device_index: int) -> object:
    global _NVML_INITIALIZED
    if pynvml is None:
        raise RuntimeError("pynvml is not installed")
    if not _NVML_INITIALIZED:
        pynvml.nvmlInit()
        _NVML_INITIALIZED = True
    if device_index not in _NVML_HANDLES:
        _NVML_HANDLES[device_index] = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    return _NVML_HANDLES[device_index]


@dataclass
class GpuSample:
    elapsed_s: float
    phase: str
    gpu_util_pct: float | None
    sm_util_pct: float | None
    mem_util_pct: float | None
    mem_used_mib: float | None


@dataclass
class PhaseStats:
    count: int = 0
    gpu_util_pct_mean: float = 0.0
    gpu_util_pct_max: float = 0.0
    sm_util_pct_mean: float = 0.0
    sm_util_pct_max: float = 0.0
    mem_util_pct_mean: float = 0.0
    mem_util_pct_max: float = 0.0


@dataclass
class RunStats:
    batch_size: int
    block_size: int
    target_prompt_tokens: int
    prompt_tokens_mean: float
    seq_len: int
    forward_steps: int
    forward_ms_mean: float
    forward_ms_max: float
    post_ms_mean: float
    idle_ms_mean: float
    forward: PhaseStats = field(default_factory=PhaseStats)
    post: PhaseStats = field(default_factory=PhaseStats)
    idle: PhaseStats = field(default_factory=PhaseStats)


def build_fixed_length_prompts(
    tokenizer: AutoTokenizer,
    batch_size: int,
    target_prompt_tokens: int,
    trial: int = 0,
) -> list[PromptRecord]:
    prompts: list[PromptRecord] = []
    for request_id in range(batch_size):
        prompt, actual_tokens = make_prompt_for_target(
            tokenizer,
            group_name="gpu_util",
            request_id=request_id,
            trial=trial,
            target_tokens=target_prompt_tokens,
        )
        prompts.append(PromptRecord(request_id, "medium", prompt))
        if request_id == 0:
            print(
                f"[gpu util] target_prompt_tokens={target_prompt_tokens} "
                f"actual_prompt_tokens={actual_tokens}"
            )
    return prompts


class GpuMonitor:
    """Poll GPU metrics in a background thread."""

    def __init__(self, device_index: int = 0, sample_ms: int = 50) -> None:
        self.device_index = device_index
        self.sample_s = sample_ms / 1000.0
        self._phase = "idle"
        self._phase_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[GpuSample] = []
        self._nvml_handle = None
        self._backend = self._choose_backend()

    def _choose_backend(self) -> str:
        if pynvml is not None:
            try:
                self._nvml_handle = get_nvml_handle(self.device_index)
                return "nvml"
            except Exception:
                self._nvml_handle = None
        if shutil.which("nvidia-smi") is None:
            return "none"
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.device_index}",
                    "--query-gpu=utilization.gpu,utilization.memory,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.stdout.strip():
                return "query"
        except (subprocess.SubprocessError, OSError):
            pass
        return "none"

    def _read_metrics_nvml(self) -> tuple[float | None, float | None, float | None, float | None]:
        assert self._nvml_handle is not None
        rates = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
        mem_used_mib = float(mem_info.used) / (1024.0 * 1024.0)
        # NVML gpu util = % of time at least one kernel ran on SMs since last sample.
        sm_active_pct = float(rates.gpu)
        gpu_util_pct = sm_active_pct
        mem_util_pct = float(rates.memory)
        return gpu_util_pct, sm_active_pct, mem_util_pct, mem_used_mib

    def _read_metrics_query(self) -> tuple[float | None, float | None, float | None, float | None]:
        proc = subprocess.run(
            [
                "nvidia-smi",
                f"--id={self.device_index}",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        gpu_util, mem_util, mem_used = [part.strip() for part in proc.stdout.strip().split(",")]
        gpu_val = float(gpu_util)
        # dmon cannot sample faster than 1s, so use the same query field as SM-active proxy.
        return gpu_val, gpu_val, float(mem_util), float(mem_used)

    def _read_metrics(self) -> tuple[float | None, float | None, float | None, float | None]:
        if self._backend == "none":
            return None, None, None, None
        try:
            if self._backend == "nvml":
                return self._read_metrics_nvml()
            return self._read_metrics_query()
        except (subprocess.SubprocessError, OSError, ValueError):
            return None, None, None, None

    def set_phase(self, phase: str) -> None:
        with self._phase_lock:
            self._phase = phase

    def start(self) -> None:
        if self._backend == "none":
            print("[gpu util] WARN: nvidia-smi unavailable; GPU metrics will be empty")
            return
        if self._backend == "nvml":
            print(f"[gpu util] SM sampling via NVML every {int(self.sample_s * 1000)}ms (kernel-active / SM-not-idle)")
        else:
            print(
                f"[gpu util] WARN: pynvml unavailable; falling back to nvidia-smi query every "
                f"{int(self.sample_s * 1000)}ms (install pynvml for accurate 50ms SM sampling)"
            )
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        time.sleep(self.sample_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        start = time.perf_counter()
        while not self._stop.is_set():
            gpu_util, sm_util, mem_util, mem_used = self._read_metrics()
            with self._phase_lock:
                phase = self._phase
            self.samples.append(
                GpuSample(
                    elapsed_s=time.perf_counter() - start,
                    phase=phase,
                    gpu_util_pct=gpu_util,
                    sm_util_pct=sm_util,
                    mem_util_pct=mem_util,
                    mem_used_mib=mem_used,
                )
            )
            time.sleep(self.sample_s)


def summarize_phase(samples: list[GpuSample], phase: str) -> PhaseStats:
    rows = [sample for sample in samples if sample.phase == phase]
    if not rows:
        return PhaseStats()

    def _values(key: str) -> list[float]:
        return [getattr(row, key) for row in rows if getattr(row, key) is not None]

    gpu_vals = _values("gpu_util_pct")
    sm_vals = _values("sm_util_pct")
    mem_vals = _values("mem_util_pct")
    return PhaseStats(
        count=len(rows),
        gpu_util_pct_mean=mean(gpu_vals) if gpu_vals else 0.0,
        gpu_util_pct_max=max(gpu_vals) if gpu_vals else 0.0,
        sm_util_pct_mean=mean(sm_vals) if sm_vals else 0.0,
        sm_util_pct_max=max(sm_vals) if sm_vals else 0.0,
        mem_util_pct_mean=mean(mem_vals) if mem_vals else 0.0,
        mem_util_pct_max=max(mem_vals) if mem_vals else 0.0,
    )


def verdict(stats: RunStats) -> str:
    sm = stats.forward.sm_util_pct_mean
    gpu = stats.forward.gpu_util_pct_mean
    score = sm if sm > 0 else gpu
    if score >= 80:
        return "well utilized"
    if score >= 50:
        return "moderately utilized"
    if score > 0:
        return "underutilized"
    return "metrics unavailable"


@torch.no_grad()
def run_monitored_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list,
    batch_size: int,
    block_size: int,
    num_blocks: int,
    max_steps: int,
    confidence_threshold: float,
    acceptance_policy: str,
    mask_token_id: int | None,
    device: torch.device,
    monitor: GpuMonitor,
    warmup_steps: int,
    target_prompt_tokens: int,
) -> RunStats:
    mask_token_id = choose_mask_token_id(tokenizer, mask_token_id)
    input_ids, attention_mask, prompt_lengths = build_prompt_batch(tokenizer, prompts, device)
    seq_len = int(input_ids.shape[1])

    forward_ms: list[float] = []
    post_ms: list[float] = []
    forward_steps = 0

    for _block_index in range(num_blocks):
        input_ids, attention_mask, (block_start, block_end) = append_mask_block(
            input_ids, attention_mask, block_size, mask_token_id
        )
        token_steps = torch.zeros((batch_size, block_size), dtype=torch.long, device=device)
        block_mask_index = torch.ones((batch_size, block_size), dtype=torch.bool, device=device)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, max_steps)

        for step in range(1, max_steps + 1):
            if not bool(token_steps.eq(0).any()):
                break

            monitor.set_phase("forward")
            cuda_sync()
            start = time.perf_counter()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            cuda_sync()
            forward_ms.append((time.perf_counter() - start) * 1000.0)
            forward_steps += 1

            monitor.set_phase("post")
            post_start = time.perf_counter()
            for request_id in range(batch_size):
                masked_positions = token_steps[request_id].eq(0)
                if not bool(masked_positions.any()):
                    continue
                block_logits = logits[request_id, block_start:block_end]
                probs = torch.softmax(block_logits, dim=-1)
                confidence, predicted = torch.max(probs, dim=-1)
                force_accept = torch.zeros_like(masked_positions)
                threshold_accept = torch.zeros_like(masked_positions)
                masked_conf = confidence.masked_fill(~masked_positions, -float("inf"))
                if acceptance_policy == "topk":
                    k = int(num_transfer_tokens[request_id, step - 1].item())
                    k = min(k, int(masked_positions.sum().item()))
                    if k > 0:
                        _, select_index = torch.topk(masked_conf, k=k)
                        threshold_accept[select_index] = True
                    accept = threshold_accept
                elif acceptance_policy == "confidence_cutoff":
                    sorted_conf, sorted_index = torch.sort(masked_conf, descending=True)
                    keep = sorted_conf.ge(confidence_threshold)
                    if bool(keep.any()):
                        threshold_accept[sorted_index[keep]] = True
                    accept = threshold_accept.clone()
                    if not bool(accept.any()):
                        force_accept[torch.argmax(masked_conf)] = True
                        accept = force_accept
                else:
                    threshold_accept = masked_positions & confidence.ge(confidence_threshold)
                    accept = threshold_accept.clone()
                    if not bool(accept.any()):
                        force_accept[torch.argmax(masked_conf)] = True
                        accept = force_accept
                if not bool(accept.any()):
                    continue
                absolute_positions = torch.arange(block_start, block_end, device=device)[accept]
                input_ids[request_id, absolute_positions] = predicted[accept]
                token_steps[request_id, accept] = step
            post_ms.append((time.perf_counter() - post_start) * 1000.0)

            if forward_steps <= warmup_steps:
                forward_ms.clear()
                post_ms.clear()

    monitor.set_phase("idle")
    return RunStats(
        batch_size=batch_size,
        block_size=block_size,
        target_prompt_tokens=target_prompt_tokens,
        prompt_tokens_mean=mean(prompt_lengths) if prompt_lengths else 0.0,
        seq_len=seq_len,
        forward_steps=max(0, forward_steps - warmup_steps),
        forward_ms_mean=mean(forward_ms) if forward_ms else 0.0,
        forward_ms_max=max(forward_ms) if forward_ms else 0.0,
        post_ms_mean=mean(post_ms) if post_ms else 0.0,
        idle_ms_mean=0.0,
        forward=summarize_phase(monitor.samples, "forward"),
        post=summarize_phase(monitor.samples, "post"),
        idle=summarize_phase(monitor.samples, "idle"),
    )


@torch.inference_mode()
def maybe_profile_one_forward(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    print("[gpu util] running one-step torch profiler (top CUDA kernels)")
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        model(input_ids=input_ids, attention_mask=attention_mask)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))


def print_run_stats(stats: RunStats) -> None:
    print(
        f"\n=== batch_size={stats.batch_size} block_size={stats.block_size} "
        f"prompt_tokens~{stats.target_prompt_tokens} (mean={stats.prompt_tokens_mean:.0f}) "
        f"seq_len={stats.seq_len} forward_steps={stats.forward_steps} ==="
    )
    print(
        f"  forward latency: mean={stats.forward_ms_mean:.2f}ms max={stats.forward_ms_max:.2f}ms"
    )
    print(f"  postproc latency: mean={stats.post_ms_mean:.2f}ms")
    print(
        "  GPU during forward: "
        f"mean={stats.forward.gpu_util_pct_mean:.1f}% max={stats.forward.gpu_util_pct_max:.1f}% "
        f"(samples={stats.forward.count})"
    )
    print(
        "  SM active during forward: "
        f"mean={stats.forward.sm_util_pct_mean:.1f}% max={stats.forward.sm_util_pct_max:.1f}%"
    )
    print(
        "  Mem during forward: "
        f"mean={stats.forward.mem_util_pct_mean:.1f}% max={stats.forward.mem_util_pct_max:.1f}%"
    )
    print(
        "  GPU during postproc: "
        f"mean={stats.post.gpu_util_pct_mean:.1f}% max={stats.post.gpu_util_pct_max:.1f}%"
    )
    print(f"  verdict: {verdict(stats)}")


def write_csv(path: Path, rows: list[RunStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch_size",
        "block_size",
        "target_prompt_tokens",
        "prompt_tokens_mean",
        "seq_len",
        "forward_steps",
        "forward_ms_mean",
        "forward_ms_max",
        "post_ms_mean",
        "gpu_util_forward_mean",
        "gpu_util_forward_max",
        "sm_util_forward_mean",
        "sm_util_forward_max",
        "mem_util_forward_mean",
        "mem_util_forward_max",
        "gpu_util_post_mean",
        "verdict",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "batch_size": row.batch_size,
                    "block_size": row.block_size,
                    "target_prompt_tokens": row.target_prompt_tokens,
                    "prompt_tokens_mean": f"{row.prompt_tokens_mean:.1f}",
                    "seq_len": row.seq_len,
                    "forward_steps": row.forward_steps,
                    "forward_ms_mean": f"{row.forward_ms_mean:.3f}",
                    "forward_ms_max": f"{row.forward_ms_max:.3f}",
                    "post_ms_mean": f"{row.post_ms_mean:.3f}",
                    "gpu_util_forward_mean": f"{row.forward.gpu_util_pct_mean:.1f}",
                    "gpu_util_forward_max": f"{row.forward.gpu_util_pct_max:.1f}",
                    "sm_util_forward_mean": f"{row.forward.sm_util_pct_mean:.1f}",
                    "sm_util_forward_max": f"{row.forward.sm_util_pct_max:.1f}",
                    "mem_util_forward_mean": f"{row.forward.mem_util_pct_mean:.1f}",
                    "mem_util_forward_max": f"{row.forward.mem_util_pct_max:.1f}",
                    "gpu_util_post_mean": f"{row.post.gpu_util_pct_mean:.1f}",
                    "verdict": verdict(row),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--prompts", type=Path, default=None, help="Unused; kept for CLI compatibility")
    parser.add_argument("--target-prompt-tokens", type=int, default=300)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--acceptance-policy", choices=["confidence_cutoff", "topk", "threshold"], default="confidence_cutoff")
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", choices=["none", "auto"], default="none")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--sample-ms", type=int, default=50, help="GPU/SM polling interval in milliseconds")
    parser.add_argument("--warmup-steps", type=int, default=2, help="Drop first N forward steps per batch")
    parser.add_argument("--profile", action="store_true", help="Run one torch profiler step for the first batch")
    parser.add_argument("--out", type=Path, default=Path("outputs/gpu_util/gpu_util_summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        print("[gpu util] ERROR: CUDA is required")
        sys.exit(1)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device(f"cuda:{args.device_index}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map == "auto":
        model_kwargs["device_map"] = "auto"
    with patch_llada_transformers_compat():
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    ensure_all_tied_weights_keys(model)
    disable_use_cache(model)
    if args.device_map == "none":
        model = model.to(device)
    model.eval()

    print(
        "[gpu util] config "
        f"batch_sizes={args.batch_sizes} block_size={args.block_size} "
        f"target_prompt_tokens={args.target_prompt_tokens} num_blocks={args.num_blocks} "
        f"max_steps={args.max_steps} sample_ms={args.sample_ms} warmup_steps={args.warmup_steps}"
    )

    if args.profile:
        profile_bs = args.batch_sizes[0]
        probe_prompts = build_fixed_length_prompts(tokenizer, profile_bs, args.target_prompt_tokens)
        profile_input_ids, profile_attention_mask, _ = build_prompt_batch(tokenizer, probe_prompts, device)
        profile_input_ids, profile_attention_mask, _ = append_mask_block(
            profile_input_ids,
            profile_attention_mask,
            args.block_size,
            choose_mask_token_id(tokenizer, args.mask_token_id),
        )
        maybe_profile_one_forward(model, profile_input_ids, profile_attention_mask)

    all_stats: list[RunStats] = []
    for batch_size in args.batch_sizes:
        batch_prompts = build_fixed_length_prompts(tokenizer, batch_size, args.target_prompt_tokens)
        monitor = GpuMonitor(device_index=args.device_index, sample_ms=args.sample_ms)
        monitor.start()
        try:
            stats = run_monitored_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                batch_size=batch_size,
                block_size=args.block_size,
                num_blocks=args.num_blocks,
                max_steps=args.max_steps,
                confidence_threshold=args.confidence_threshold,
                acceptance_policy=args.acceptance_policy,
                mask_token_id=args.mask_token_id,
                device=device,
                monitor=monitor,
                warmup_steps=args.warmup_steps,
                target_prompt_tokens=args.target_prompt_tokens,
            )
        finally:
            monitor.stop()

        all_stats.append(stats)
        print_run_stats(stats)

    write_csv(args.out, all_stats)
    print(f"\n[gpu util] wrote {args.out}")
    print(
        "[gpu util] note: SM active % uses NVML kernel-active time over each sample window "
        f"({args.sample_ms}ms). nvidia-smi dmon cannot sample faster than 1s."
    )


if __name__ == "__main__":
    main()
