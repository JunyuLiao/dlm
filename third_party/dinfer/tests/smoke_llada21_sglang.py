#!/usr/bin/env python3
"""Full-checkpoint, single-H100 smoke test for dInfer LLaDA2.1-mini."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="inclusionAI/LLaDA2.1-mini")
    parser.add_argument("--prompt", default="Write one short sentence about Paris.")
    parser.add_argument("--gen-length", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--editing-threshold", type=float, default=0.0)
    parser.add_argument("--max-post-steps", type=int, default=16)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = (
        args.model_path
        if Path(args.model_path).is_dir()
        else snapshot_download(
            args.model_path, local_files_only=args.local_files_only
        )
    )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(free_port()))

    from sglang.srt import distributed
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.server_args import ServerArgs

    from dinfer import SamplingParams
    from dinfer.decoding.diffusion_runner import ModelRunner
    from dinfer.decoding.serving import init_generator
    from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SGLangLM

    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, 1, 1, backend="nccl")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    server_args = ServerArgs(
        model_path=model_path,
        enable_dp_attention=True,
        trust_remote_code=True,
        tp_size=1,
        dp_size=1,
        pp_size=1,
    )
    try:
        from sglang.srt.server_args import set_global_server_args_for_scheduler
    except ImportError:
        pass
    else:
        set_global_server_args_for_scheduler(server_args)
    initialize_dp_attention(server_args=server_args, model_config=config)
    initialize_moe_config(server_args)

    load_start = time.perf_counter()
    model = LLaDA2SGLangLM(config=config, expert_map_path=".").eval()
    model.load_weights(model_path, device=device)
    model = model.to(device)
    load_seconds = time.perf_counter() - load_start

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(device)

    position_ids = torch.arange(prompt.shape[1], device=device).unsqueeze(0)
    attention_mask = torch.ones(
        (1, prompt.shape[1], prompt.shape[1]), dtype=torch.bool, device=device
    )
    with torch.inference_mode():
        forward = model(
            prompt,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    assert forward.logits.shape == (1, prompt.shape[1], config.vocab_size)
    assert forward.past_key_values is not None

    params = SamplingParams(
        threshold=args.threshold,
        editing_threshold=args.editing_threshold,
        max_post_steps=args.max_post_steps,
        cache="prefix",
        use_bd=True,
        enable_torch_compile=False,
        use_naive_batching=True,
        max_length=max(256, prompt.shape[1] + args.gen_length + args.block_length),
        prefilling_limit=256,
        mask_id=tokenizer.mask_token_id,
        eos_id=tokenizer.eos_token_id,
    )
    runner = ModelRunner(
        model,
        device=str(device),
        enable_cuda_graph=False,
        enable_compile=False,
        supported_batch_sizes=[1],
        server_args=server_args,
        max_length=params.max_length,
        block_length=args.block_length,
        prefill_lengths=[args.block_length],
        decoding_lengths=[args.block_length],
        cache_lengths=[128, 256],
        use_cross_block=True,
    )
    generator = init_generator(
        runner, params, backend="sglang", max_length=params.max_length
    )
    torch.cuda.reset_peak_memory_stats(device)
    generation_start = time.perf_counter()
    with torch.inference_mode():
        generated = generator.generate(
            prompt,
            gen_length=args.gen_length,
            block_length=args.block_length,
        )
    torch.cuda.synchronize(device)
    generation_seconds = time.perf_counter() - generation_start
    completion = generated[:, prompt.shape[1] :]
    text = tokenizer.decode(completion[0], skip_special_tokens=True)
    assert generated.ndim == 2 and generated.shape[0] == 1
    assert completion.shape[1] > 0

    print(
        json.dumps(
            {
                "model_path": model_path,
                "device": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "load_seconds": load_seconds,
                "prompt_tokens": prompt.shape[1],
                "output_tokens_including_prompt": generated.shape[1],
                "completion_tokens": completion.shape[1],
                "generation_seconds": generation_seconds,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "forward_logits_shape": list(forward.logits.shape),
                "kv_layers": len(forward.past_key_values),
                "text": text,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
