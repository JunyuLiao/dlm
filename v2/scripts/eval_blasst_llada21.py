#!/usr/bin/env python3
"""Paired dense/BLASST evaluation for inclusionAI/LLaDA2.1-mini.

This intentionally reuses only v2's dense-QK BLASST reference mechanism and
paired evaluation metrics. Generation is the model's native block-diffusion
algorithm (including token editing); no Fast-dLLM cache or acceleration path is
imported.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

from scripts.blasst_common import (  # noqa: E402
    cosine,
    load_examples,
    normalized_token_difference,
    relative_l2,
    resolve_dtype,
    set_seed,
)
from sparse_attention import Blasst2DConfig, Blasst2DStats, install_blasst_2d  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="inclusionAI/LLaDA2.1-mini")
    parser.add_argument(
        "--dataset-path",
        default=str(V2_ROOT / "data/alpaca/test/test_252.json"),
    )
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--mask-ratios", default="0.15,0.30,0.50,0.70,0.90")
    parser.add_argument("--blasst-lambda", type=float, default=0.5)
    parser.add_argument("--q-tile-size", type=int, default=128)
    parser.add_argument("--kv-tile-size", type=int, default=64)
    parser.add_argument("--collect-blasst-stats", action="store_true", default=True)
    parser.add_argument("--dump-blasst-trace", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--denoising-threshold", type=float, default=0.5)
    parser.add_argument("--editing-threshold", type=float, default=0.0)
    parser.add_argument("--max-post-steps", type=int, default=16)
    return parser.parse_args()


def load_model(args):
    # The published checkpoint declares transformers 4.57.1 but imports the
    # newer create_bidirectional_mask helper. This experiment always supplies
    # the explicit 4-D block mask, so preserving that mask is the compatible
    # behavior on 4.57.x. Newer Transformers releases take the native path.
    import transformers.masking_utils as masking_utils

    if not hasattr(masking_utils, "create_bidirectional_mask"):
        def create_bidirectional_mask(*, attention_mask=None, **kwargs):
            return attention_mask

        masking_utils.create_bidirectional_mask = create_bidirectional_mask

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    if not hasattr(config, "rope_parameters"):
        config.rope_parameters = {
            "rope_type": "default",
            "rope_theta": config.rope_theta,
            "partial_rotary_factor": config.partial_rotary_factor,
        }
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=resolve_dtype(args.precision),
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    return model, tokenizer


def chat_ids(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    return ids.to(device)


def block_causal_mask(length: int, block_size: int, device: torch.device) -> torch.Tensor:
    blocks = (length + block_size - 1) // block_size
    block_mask = torch.ones((blocks, blocks), dtype=torch.bool, device=device).tril()
    return block_mask.repeat_interleave(block_size, 0).repeat_interleave(
        block_size, 1
    )[:length, :length][None, None]


class ForwardCapture:
    def __init__(self, model) -> None:
        self.attention: list[torch.Tensor] = []
        self.hidden: list[torch.Tensor] = []
        self.top1: list[torch.Tensor] = []
        self.handles = [
            model.model.norm.register_forward_hook(self._hidden_hook),
            model.lm_head.register_forward_hook(self._logit_hook),
        ]
        for layer in model.model.layers:
            self.handles.append(layer.attention.register_forward_hook(self._attention_hook))

    def _attention_hook(self, module, args, output) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        self.attention.append(tensor.detach().float().cpu())

    def _hidden_hook(self, module, args, output) -> None:
        self.hidden.append(output.detach().float().cpu())

    def _logit_hook(self, module, args, output) -> None:
        self.top1.append(output.detach().argmax(-1).cpu())

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def forward_state(model, input_ids: torch.Tensor, block_size: int):
    length = input_ids.shape[1]
    positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
    return model(
        input_ids=input_ids,
        attention_mask=block_causal_mask(length, block_size, input_ids.device),
        position_ids=positions,
        use_cache=False,
    )


def masked_state(model, tokenizer, example, ratio, block_size, seed):
    prompt = chat_ids(tokenizer, example["input"], model.device)
    target = tokenizer(example["output"], return_tensors="pt")["input_ids"].to(model.device)
    prompt_length = prompt.shape[1]
    window_end = ((prompt_length + block_size - 1) // block_size) * block_size
    capacity = window_end - prompt_length
    if capacity == 0:
        window_end += block_size
        capacity = block_size
    clean_target = target[:, :capacity]
    if clean_target.shape[1] < capacity:
        clean_target = F.pad(
            clean_target,
            (0, capacity - clean_target.shape[1]),
            value=tokenizer.eos_token_id,
        )
    clean = torch.cat((prompt, clean_target), dim=1)
    candidates = torch.arange(prompt_length, window_end, device=model.device)
    count = max(1, round(candidates.numel() * ratio))
    generator = torch.Generator().manual_seed(seed)
    selected = candidates[torch.randperm(candidates.numel(), generator=generator)[:count]]
    noisy = clean.clone()
    noisy[:, selected] = tokenizer.mask_token_id
    return noisy, clean, selected


def capture_forward(model, ids, block_size):
    capture = ForwardCapture(model)
    output = forward_state(model, ids, block_size)
    capture.close()
    return output, capture


def one_step_metrics(model, tokenizer, runtime, sparse_config, examples, ratios, args):
    rows = []
    dense_config = replace(sparse_config, enable_blasst_2d=False)
    for example_index, example in enumerate(examples):
        for ratio_index, ratio in enumerate(ratios):
            noisy, clean, selected = masked_state(
                model,
                tokenizer,
                example,
                ratio,
                args.block_size,
                args.seed + 100 * example_index + ratio_index,
            )
            runtime.config = dense_config
            dense, dense_capture = capture_forward(model, noisy.clone(), args.block_size)
            runtime.config = sparse_config
            sparse, sparse_capture = capture_forward(model, noisy.clone(), args.block_size)
            dense_logits = dense.logits[:, selected]
            sparse_logits = sparse.logits[:, selected]
            dense_logp = F.log_softmax(dense_logits.float(), dim=-1)
            sparse_logp = F.log_softmax(sparse_logits.float(), dim=-1)
            labels = clean[:, selected]
            dense_top1 = dense_logits.argmax(-1)
            sparse_top1 = sparse_logits.argmax(-1)
            dense_attention = torch.cat([value.flatten() for value in dense_capture.attention])
            sparse_attention = torch.cat([value.flatten() for value in sparse_capture.attention])
            rows.append(
                {
                    "example": example_index,
                    "mask_ratio": ratio,
                    "masked_tokens": selected.numel(),
                    "query_length": noisy.shape[1],
                    "kv_length": noisy.shape[1],
                    "attention_relative_l2": relative_l2(dense_attention, sparse_attention),
                    "attention_cosine": cosine(dense_attention, sparse_attention),
                    "hidden_relative_l2": relative_l2(
                        dense_capture.hidden[-1], sparse_capture.hidden[-1]
                    ),
                    "logit_relative_l2": relative_l2(dense_logits, sparse_logits),
                    "logit_cosine": cosine(dense_logits, sparse_logits),
                    "top1_agreement": float((dense_top1 == sparse_top1).float().mean()),
                    "dense_token_accuracy": float((dense_top1 == labels).float().mean()),
                    "sparse_token_accuracy": float((sparse_top1 == labels).float().mean()),
                    "kl_dense_to_sparse": float(
                        (dense_logp.exp() * (dense_logp - sparse_logp)).sum(-1).mean()
                    ),
                }
            )
    return rows


@torch.no_grad()
def generate_one(model, tokenizer, prompt, args):
    inputs = chat_ids(tokenizer, prompt, model.device)
    completion = model.generate(
        inputs=inputs,
        gen_length=args.max_new_tokens,
        block_length=args.block_size,
        threshold=args.denoising_threshold,
        editing_threshold=args.editing_threshold,
        max_post_steps=args.max_post_steps,
        temperature=0.0,
        eos_early_stop=True,
        mask_id=tokenizer.mask_token_id,
        eos_id=tokenizer.eos_token_id,
    )[0]
    return completion.detach().cpu().tolist(), tokenizer.decode(
        completion, skip_special_tokens=True
    )


def end_to_end_metrics(model, tokenizer, runtime, sparse_config, examples, args):
    rows = []
    dense_config = replace(sparse_config, enable_blasst_2d=False)
    for index, example in enumerate(examples):
        runtime.config = dense_config
        set_seed(args.seed + index)
        dense_ids, dense_text = generate_one(model, tokenizer, example["input"], args)
        runtime.config = sparse_config
        set_seed(args.seed + index)
        sparse_ids, sparse_text = generate_one(model, tokenizer, example["input"], args)
        overlap = min(len(dense_ids), len(sparse_ids))
        agreement = sum(
            dense_ids[i] == sparse_ids[i] for i in range(overlap)
        ) / max(len(dense_ids), len(sparse_ids), 1)
        normalize = lambda text: " ".join(text.lower().split())
        reference = normalize(example["output"])
        rows.append(
            {
                "example": index,
                "dense_completion": dense_text,
                "sparse_completion": sparse_text,
                "reference": example["output"],
                "dense_task_correct": int(normalize(dense_text) == reference),
                "sparse_task_correct": int(normalize(sparse_text) == reference),
                "final_token_agreement": agreement,
                "final_exact_match_dense": int(dense_ids == sparse_ids),
                "normalized_token_difference": normalized_token_difference(
                    dense_ids, sparse_ids
                ),
            }
        )
    return rows


def mean(rows, key):
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def environment() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=V2_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "repository_commit": commit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def main() -> None:
    args = parse_args()
    ratios = [float(value) for value in args.mask_ratios.split(",")]
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args)
    examples = load_examples(args.dataset_path, args.num_samples)
    sparse_config = Blasst2DConfig(
        enable_blasst_2d=True,
        blasst_lambda=args.blasst_lambda,
        q_tile_size=args.q_tile_size,
        kv_tile_size=args.kv_tile_size,
        collect_blasst_stats=args.collect_blasst_stats,
        dump_blasst_trace=args.dump_blasst_trace,
    )
    stats = Blasst2DStats()
    probe, _, _ = masked_state(
        model, tokenizer, examples[0], ratios[0], args.block_size, args.seed
    )
    original = forward_state(model, probe, args.block_size).logits.detach()
    runtime = install_blasst_2d(
        model,
        replace(sparse_config, enable_blasst_2d=False),
        stats,
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=None,
        attention_class_names=("LLaDA2MoeAttention",),
    )

    runtime.config = replace(sparse_config, enable_blasst_2d=False)
    dispatched = forward_state(model, probe, args.block_size).logits.detach()
    dense_regression = {
        "exact_match": bool(torch.equal(original, dispatched)),
        "max_abs_error": float((original - dispatched).abs().max()),
    }
    if not dense_regression["exact_match"]:
        raise AssertionError(f"disabled BLASST changed dense logits: {dense_regression}")

    one_step = one_step_metrics(
        model, tokenizer, runtime, sparse_config, examples, ratios, args
    )
    end_to_end = end_to_end_metrics(
        model, tokenizer, runtime, sparse_config, examples, args
    )
    env = environment()
    run_config = {**vars(args), **env, "mask_ratios": ratios}
    stats.export(output_dir, sparse_config, run_config)
    write_csv(output_dir / "same_state.csv", one_step)
    (output_dir / "end_to_end.json").write_text(
        json.dumps(end_to_end, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    accuracy = {
        "dense_task_accuracy": mean(end_to_end, "dense_task_correct"),
        "sparse_task_accuracy": mean(end_to_end, "sparse_task_correct"),
        "final_token_agreement": mean(end_to_end, "final_token_agreement"),
        "same_state_top1_agreement": mean(one_step, "top1_agreement"),
        "same_state_logit_relative_l2": mean(one_step, "logit_relative_l2"),
    }
    (output_dir / "accuracy.json").write_text(
        json.dumps(accuracy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "dense_regression.json").write_text(
        json.dumps(dense_regression, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sparsity = stats.summary()
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# LLaDA2.1-mini reference 2D-BLASST evaluation",
                "",
                "This is dense-QK score-mask emulation, not a sparse performance kernel. ",
                "It uses LLaDA2.1's native block-causal attention and token-editing generation.",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Physical tile sparsity | {sparsity['physical_tile_sparsity']:.6f} |",
                f"| Row-vote sparsity | {sparsity['row_vote_sparsity']:.6f} |",
                f"| Valid-element sparsity | {sparsity['valid_element_sparsity']:.6f} |",
                f"| Dense task exact match | {accuracy['dense_task_accuracy']:.6f} |",
                f"| Sparse task exact match | {accuracy['sparse_task_accuracy']:.6f} |",
                f"| Final token agreement | {accuracy['final_token_agreement']:.6f} |",
                f"| Same-state top-1 agreement | {accuracy['same_state_top1_agreement']:.6f} |",
                f"| Same-state logit relative L2 | {accuracy['same_state_logit_relative_l2']:.6f} |",
                "",
                "Detailed query/KV-length, layer, and head breakdowns are in ",
                "`per_step.csv`, `per_layer.csv`, and `per_head.csv`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "config": asdict(sparse_config),
                "sparsity": sparsity,
                "accuracy": accuracy,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
