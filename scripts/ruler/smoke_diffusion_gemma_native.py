"""One-H100 native DiffusionGemma text-generation smoke and memory report."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from dllm.models import GenerationRequest, create_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/diffusiongemma-26B-A4B-it")
    parser.add_argument("--revision")
    parser.add_argument("--prompt", default="Explain why the sky is blue in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this smoke test requires one CUDA GPU")
    adapter = create_adapter(
        "diffusion_gemma",
        args.model,
        revision=args.revision,
        device="cuda",
        precision="bfloat16",
    ).load()
    torch.cuda.reset_peak_memory_stats()
    result = adapter.generate(
        GenerationRequest(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            block_size=256,
            steps=args.steps,
            seed=42,
        )
    )
    if not result.completion_tokens:
        raise RuntimeError("native smoke produced no completion tokens")
    parameter_devices = sorted({str(value.device) for value in adapter.model.parameters()})
    payload = {
        "model": args.model,
        "revision": args.revision,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        **adapter.runtime_metadata(),
        "prompt_tokens": len(result.prompt_tokens),
        "completion_tokens": len(result.completion_tokens),
        "completion": result.text,
        "elapsed_seconds": result.elapsed_seconds,
        "model_evaluations": result.model_evaluations,
        "termination_reason": result.termination_reason,
        "generation_metadata": result.metadata,
        "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "model_parameter_devices": parameter_devices,
        "full_checkpoint_on_cuda": bool(parameter_devices)
        and all(value.startswith("cuda") for value in parameter_devices),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
