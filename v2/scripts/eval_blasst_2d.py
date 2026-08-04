#!/usr/bin/env python3
"""Paired dense/sparse accuracy and sparsity evaluation for 2D-BLASST."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

from scripts.blasst_common import (  # noqa: E402
    cosine,
    generate_one,
    load_examples,
    load_model,
    normalized_token_difference,
    relative_l2,
    set_seed,
)
from sparse_attention import (  # noqa: E402
    Blasst2DConfig,
    Blasst2DStats,
    install_blasst_2d,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="Efficient-Large-Model/Fast_dLLM_v2_7B",
    )
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
    parser.add_argument(
        "--output-dir",
        default=str(V2_ROOT.parent / "results/blasst_2d_lambda_0p5"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--denoising-threshold", type=float, default=0.9)
    return parser.parse_args()


class ForwardCapture:
    def __init__(self, model) -> None:
        self.attention: list[torch.Tensor] = []
        self.hidden: list[torch.Tensor] = []
        self.top1: list[torch.Tensor] = []
        self.inputs: list[torch.Tensor] = []
        self.handles = []
        base_model = model.model
        self.handles.append(
            base_model.register_forward_pre_hook(self._input_hook, with_kwargs=True)
        )
        self.handles.append(base_model.norm.register_forward_hook(self._hidden_hook))
        self.handles.append(model.lm_head.register_forward_hook(self._logit_hook))
        for layer in base_model.layers:
            self.handles.append(
                layer.self_attn.register_forward_hook(self._attention_hook)
            )

    def _input_hook(self, module, args, kwargs) -> None:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is not None:
            self.inputs.append(input_ids.detach().cpu())

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


def _masked_state(
    tokenizer,
    example: dict[str, str],
    mask_ratio: float,
    block_size: int,
    mask_token_id: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt = tokenizer(example["input"], return_tensors="pt")["input_ids"][0]
    target = tokenizer(example["output"], return_tensors="pt")["input_ids"][0]
    prefix_length = (len(prompt) // block_size) * block_size
    prefix = prompt[:prefix_length]
    current_clean = torch.cat((prompt[prefix_length:], target))[:block_size]
    active = torch.ones(block_size, dtype=torch.bool)
    if len(current_clean) < block_size:
        pad = torch.full(
            (block_size - len(current_clean),),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )
        active[len(current_clean) :] = False
        current_clean = torch.cat((current_clean, pad))
    generator = torch.Generator().manual_seed(seed)
    active_indices = active.nonzero().flatten()
    count = max(1, round(len(active_indices) * mask_ratio))
    chosen = active_indices[
        torch.randperm(len(active_indices), generator=generator)[:count]
    ]
    noisy = current_clean.clone()
    noisy[chosen] = mask_token_id
    return (
        prefix[None].to(device) if len(prefix) else None,
        noisy[None].to(device),
        current_clean[None].to(device),
        chosen.to(device),
    )


@torch.no_grad()
def _forward_state(model, input_ids, prefix_cache, block_size):
    kwargs = {
        "input_ids": input_ids,
        "use_cache": True,
        "update_past_key_values": False,
        "use_block_cache": False,
        "block_size": block_size,
    }
    if prefix_cache is not None:
        kwargs["past_key_values"] = prefix_cache
    return model(**kwargs)


def _one_step_metrics(
    model,
    tokenizer,
    runtime,
    sparse_config,
    examples,
    ratios,
    args,
) -> list[dict[str, Any]]:
    rows = []
    mask_token_id = getattr(model.config, "mask_token_id", 151665)
    dense_config = replace(sparse_config, enable_blasst_2d=False)
    for example_index, example in enumerate(examples):
        for ratio_index, ratio in enumerate(ratios):
            prefix, noisy, clean, selected = _masked_state(
                tokenizer,
                example,
                ratio,
                args.block_size,
                mask_token_id,
                model.device,
                args.seed + example_index * 100 + ratio_index,
            )
            prefix_cache = None
            if prefix is not None:
                runtime.config = dense_config
                prefix_cache = model(
                    input_ids=prefix,
                    use_cache=True,
                    update_past_key_values=True,
                    block_size=args.block_size,
                ).past_key_values

            runtime.config = dense_config
            dense_capture = ForwardCapture(model)
            dense_output = _forward_state(
                model, noisy.clone(), prefix_cache, args.block_size
            )
            dense_capture.close()

            runtime.config = sparse_config
            runtime.forward_call_index = 0
            sparse_capture = ForwardCapture(model)
            sparse_output = _forward_state(
                model, noisy.clone(), prefix_cache, args.block_size
            )
            sparse_capture.close()

            dense_logits = dense_output.logits[:, : noisy.shape[1]]
            sparse_logits = sparse_output.logits[:, : noisy.shape[1]]
            dense_selected = dense_logits[:, selected]
            sparse_selected = sparse_logits[:, selected]
            dense_log_prob = F.log_softmax(dense_selected.float(), dim=-1)
            sparse_log_prob = F.log_softmax(sparse_selected.float(), dim=-1)
            dense_prob = dense_log_prob.exp()
            kl = (dense_prob * (dense_log_prob - sparse_log_prob)).sum(-1).mean()
            dense_top1 = dense_selected.argmax(-1)
            sparse_top1 = sparse_selected.argmax(-1)
            labels = clean[:, selected]

            dense_attention = torch.cat(
                [tensor.flatten() for tensor in dense_capture.attention]
            )
            sparse_attention = torch.cat(
                [tensor.flatten() for tensor in sparse_capture.attention]
            )
            dense_hidden = dense_capture.hidden[-1]
            sparse_hidden = sparse_capture.hidden[-1]
            rows.append(
                {
                    "example": example_index,
                    "mask_ratio": ratio,
                    "masked_tokens": len(selected),
                    "query_length": noisy.shape[1],
                    "kv_length": (
                        noisy.shape[1]
                        + (prefix.shape[1] if prefix is not None else 0)
                    ),
                    "attention_relative_l2": relative_l2(
                        dense_attention, sparse_attention
                    ),
                    "attention_cosine": cosine(
                        dense_attention, sparse_attention
                    ),
                    "attention_max_abs_error": float(
                        (dense_attention - sparse_attention).abs().max()
                    ),
                    "hidden_relative_l2": relative_l2(
                        dense_hidden, sparse_hidden
                    ),
                    "logit_relative_l2": relative_l2(
                        dense_selected, sparse_selected
                    ),
                    "logit_cosine": cosine(dense_selected, sparse_selected),
                    "top1_agreement": float(
                        (dense_top1 == sparse_top1).float().mean()
                    ),
                    "dense_token_accuracy": float(
                        (dense_top1 == labels).float().mean()
                    ),
                    "sparse_token_accuracy": float(
                        (sparse_top1 == labels).float().mean()
                    ),
                    "kl_dense_to_sparse": float(kl),
                }
            )
    return rows


def _trajectory_agreement(
    dense: ForwardCapture, sparse: ForwardCapture
) -> tuple[list[float], float]:
    agreements = []
    for dense_top1, sparse_top1 in zip(dense.top1, sparse.top1):
        rows = min(dense_top1.shape[-2], sparse_top1.shape[-2])
        agreements.append(
            float(
                (
                    dense_top1[..., :rows] == sparse_top1[..., :rows]
                ).float().mean()
            )
        )
    return agreements, sum(agreements) / len(agreements) if agreements else 0.0


def _end_to_end_metrics(
    model,
    tokenizer,
    runtime,
    sparse_config,
    examples,
    args,
) -> list[dict[str, Any]]:
    rows = []
    dense_config = replace(sparse_config, enable_blasst_2d=False)
    for index, example in enumerate(examples):
        runtime.config = dense_config
        set_seed(args.seed + index)
        dense_capture = ForwardCapture(model)
        dense = generate_one(
            model,
            tokenizer,
            example["input"],
            block_size=args.block_size,
            max_new_tokens=args.max_new_tokens,
            threshold=args.denoising_threshold,
        )
        dense_capture.close()

        runtime.config = sparse_config
        runtime.forward_call_index = 0
        set_seed(args.seed + index)
        sparse_capture = ForwardCapture(model)
        sparse = generate_one(
            model,
            tokenizer,
            example["input"],
            block_size=args.block_size,
            max_new_tokens=args.max_new_tokens,
            threshold=args.denoising_threshold,
        )
        sparse_capture.close()

        dense_ids = dense["completion_tokens"]
        sparse_ids = sparse["completion_tokens"]
        overlap = min(len(dense_ids), len(sparse_ids))
        token_agreement = (
            sum(dense_ids[i] == sparse_ids[i] for i in range(overlap))
            / max(len(dense_ids), len(sparse_ids), 1)
        )
        per_step, average_step = _trajectory_agreement(
            dense_capture, sparse_capture
        )
        normalize = lambda text: " ".join(text.lower().split())
        reference = normalize(example["output"])
        rows.append(
            {
                "example": index,
                "dense_completion": dense["completion"],
                "sparse_completion": sparse["completion"],
                "reference": example["output"],
                "dense_task_correct": int(
                    normalize(dense["completion"]) == reference
                ),
                "sparse_task_correct": int(
                    normalize(sparse["completion"]) == reference
                ),
                "final_token_agreement": token_agreement,
                "final_exact_match_dense": int(dense_ids == sparse_ids),
                "normalized_token_difference": normalized_token_difference(
                    dense_ids, sparse_ids
                ),
                "per_step_top1_agreement": per_step,
                "average_per_step_top1_agreement": average_step,
            }
        )
    return rows


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _environment() -> dict[str, Any]:
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


def _report(
    path: Path,
    args,
    environment,
    stats,
    one_step,
    end_to_end,
    dense_regression,
) -> None:
    dense_accuracy = _mean(end_to_end, "dense_task_correct")
    sparse_accuracy = _mean(end_to_end, "sparse_task_correct")
    summary = stats.summary()
    count_fields = (
        "eligible_tiles",
        "skipped_tiles",
        "retained_tiles",
        "structurally_masked_tiles",
        "skippable_row_votes",
        "valid_row_votes",
        "skipped_valid_elements",
        "valid_elements",
    )

    def aggregate(rows, keys):
        grouped = {}
        for row in rows:
            key = tuple(row[name] for name in keys)
            target = grouped.setdefault(
                key,
                {name: row[name] for name in keys}
                | {name: 0 for name in count_fields},
            )
            for name in count_fields:
                target[name] += int(row[name])
        for target in grouped.values():
            eligible = target["eligible_tiles"]
            votes = target["valid_row_votes"]
            elements = target["valid_elements"]
            target["physical"] = (
                target["skipped_tiles"] / eligible if eligible else 0.0
            )
            target["row_vote"] = (
                target["skippable_row_votes"] / votes if votes else 0.0
            )
            target["elements"] = (
                target["skipped_valid_elements"] / elements if elements else 0.0
            )
        return list(grouped.values())

    step_rows = aggregate(
        stats.per_step.values(),
        ("denoising_step", "mask_ratio", "query_length"),
    )
    layer_rows = aggregate(stats.per_layer.values(), ("layer",))
    head_rows = aggregate(stats.per_head.values(), ("layer", "head"))
    by_ratio = {}
    for ratio in sorted({row["mask_ratio"] for row in one_step}):
        subset = [row for row in one_step if row["mask_ratio"] == ratio]
        by_ratio[ratio] = {
            "top1": _mean(subset, "top1_agreement"),
            "logit_l2": _mean(subset, "logit_relative_l2"),
        }
    verdict = (
        "acceptable on this smoke set"
        if sparse_accuracy >= dense_accuracy
        and _mean(end_to_end, "final_token_agreement") >= 0.95
        else "not accuracy-preserving on this evaluation"
    )
    lines = [
        "# Reference 2D-BLASST λ=0.5 evaluation",
        "",
        "This is dense-QK score-mask emulation. It measures physical sparsity and "
        "theoretical skipped softmax/PV work; it does not measure or claim kernel speedup.",
        "",
        "## Setup",
        "",
        f"- Commit: `{environment['repository_commit']}`",
        f"- Device: {environment['device']}; PyTorch {environment['torch']}; CUDA {environment['cuda']}",
        f"- Model: `{args.model_path}`; dataset: `{args.dataset_path}`; precision: {args.precision}",
        f"- λ: {args.blasst_lambda}; Q tile: {args.q_tile_size}; KV tile: {args.kv_tile_size}; seed: {args.seed}",
        f"- Generation block: {args.block_size}; sub-block optimization and dual block cache: disabled.",
        f"- Recorded BLASST query length: {args.block_size} tokens; prompt prefill/cache updates use original dense SDPA.",
        "- Integration: the Fast_dLLM_QwenAttention SDPA interface after RoPE, GQA expansion, scaling, and the model block/padding mask.",
        f"- Dense regression: original vs disabled-dispatch maximum absolute logit difference "
        f"{dense_regression['max_abs_error']:.8g}; exact match: {dense_regression['exact_match']}.",
        "",
        "## Aggregate results",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Eligible physical tiles | {summary['eligible_tiles']} |",
        f"| BLASST-skipped physical tiles | {summary['skipped_tiles']} |",
        f"| Physical tile sparsity | {summary['physical_tile_sparsity']:.6f} |",
        f"| Row-vote sparsity | {summary['row_vote_sparsity']:.6f} |",
        f"| Valid-element sparsity | {summary['valid_element_sparsity']:.6f} |",
        f"| Dense task exact-match accuracy | {dense_accuracy:.6f} |",
        f"| Sparse task exact-match accuracy | {sparse_accuracy:.6f} |",
        f"| Task accuracy delta | {sparse_accuracy - dense_accuracy:+.6f} |",
        f"| Final token agreement with dense | {_mean(end_to_end, 'final_token_agreement'):.6f} |",
        f"| Final sequence exact-match with dense | {_mean(end_to_end, 'final_exact_match_dense'):.6f} |",
        f"| Average per-step top-1 agreement | {_mean(end_to_end, 'average_per_step_top1_agreement'):.6f} |",
        f"| Same-state logit relative L2 | {_mean(one_step, 'logit_relative_l2'):.6f} |",
        "",
        "## Same-state breakdown by mask ratio",
        "",
        "| Mask ratio | Top-1 agreement | Logit relative L2 |",
        "|---:|---:|---:|",
    ]
    for ratio, values in by_ratio.items():
        lines.append(
            f"| {ratio:.2f} | {values['top1']:.6f} | {values['logit_l2']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Physical sparsity by denoising step",
            "",
            "| Step | Mask ratio | Query length | Eligible | Skipped | Physical | Row vote | Valid element |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        step_rows,
        key=lambda item: (
            item["denoising_step"],
            item["mask_ratio"],
            item["query_length"],
        ),
    ):
        lines.append(
            f"| {row['denoising_step']} | {row['mask_ratio']:.4f} | "
            f"{row['query_length']} | "
            f"{row['eligible_tiles']} | {row['skipped_tiles']} | "
            f"{row['physical']:.4f} | {row['row_vote']:.4f} | "
            f"{row['elements']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Physical sparsity by layer",
            "",
            "| Layer | Eligible | Skipped | Physical | Row vote | Valid element |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(layer_rows, key=lambda item: item["layer"]):
        lines.append(
            f"| {row['layer']} | {row['eligible_tiles']} | "
            f"{row['skipped_tiles']} | {row['physical']:.4f} | "
            f"{row['row_vote']:.4f} | {row['elements']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Highest-sparsity layer/head pairs",
            "",
            "| Layer | Head | Eligible | Skipped | Physical | Row vote |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        head_rows, key=lambda item: item["physical"], reverse=True
    )[:20]:
        lines.append(
            f"| {row['layer']} | {row['head']} | {row['eligible_tiles']} | "
            f"{row['skipped_tiles']} | {row['physical']:.4f} | "
            f"{row['row_vote']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The complete query/KV-length-resolved breakdowns are in `per_step.csv`, "
            "`per_layer.csv`, and `per_head.csv`.",
            "",
            "## Interpretation and limitations",
            "",
            "Row-vote sparsity counts individual valid query-row/KV-tile votes. "
            "Physical 2D sparsity is stricter: a Q×KV tile is skipped only when every "
            "active query row votes to skip, so it cannot be inferred from row sparsity.",
            "",
            "The local Alpaca set is open-ended, making exact reference-string accuracy "
            "a deliberately strict task metric. The paired dense agreement metrics are "
            "reported separately and do not replace that labeled metric. The reference "
            "path still computes dense QK and is expected to be slower than SDPA.",
            "",
            "The host uses Python 3.13, on which PyTorch 2.5 disables Dynamo. The "
            "checkpoint's import-time compile decorator on its training-only "
            "FlexAttention helper was therefore loaded eagerly; inference uses SDPA and "
            "is unaffected.",
            "",
            f"**Verdict:** λ=0.5 is {verdict}; observed physical sparsity is "
            f"{summary['physical_tile_sparsity']:.2%}.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ratios = [float(value) for value in args.mask_ratios.split(",")]
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args.model_path, args.device, args.precision)
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
    probe_ids = tokenizer(
        examples[0]["input"], return_tensors="pt"
    )["input_ids"][:, -args.block_size :].to(model.device)
    with torch.no_grad():
        original_dense_logits = model(
            input_ids=probe_ids,
            use_cache=False,
            block_size=args.block_size,
        ).logits.detach()
    runtime = install_blasst_2d(
        model,
        replace(sparse_config, enable_blasst_2d=False),
        stats,
        mask_token_id=getattr(model.config, "mask_token_id", 151665),
        pad_token_id=tokenizer.pad_token_id,
        dual_cache_only=False,
    )
    with torch.no_grad():
        dispatched_dense_logits = model(
            input_ids=probe_ids,
            use_cache=False,
            block_size=args.block_size,
        ).logits.detach()
    dense_regression = {
        "exact_match": bool(
            torch.equal(original_dense_logits, dispatched_dense_logits)
        ),
        "allclose": bool(
            torch.allclose(
                original_dense_logits,
                dispatched_dense_logits,
                rtol=0.0,
                atol=0.0,
            )
        ),
        "max_abs_error": float(
            (original_dense_logits - dispatched_dense_logits).abs().max()
        ),
    }
    if not dense_regression["exact_match"]:
        raise AssertionError(
            f"disabled BLASST changed dense logits: {dense_regression}"
        )
    runtime.config = sparse_config
    one_step = _one_step_metrics(
        model, tokenizer, runtime, sparse_config, examples, ratios, args
    )
    end_to_end = _end_to_end_metrics(
        model, tokenizer, runtime, sparse_config, examples, args
    )
    environment = _environment()
    run_config = {**vars(args), **environment, "mask_ratios": ratios}
    stats.export(output_dir, sparse_config, run_config)
    _write_csv(output_dir / "same_state.csv", one_step)
    (output_dir / "end_to_end.json").write_text(
        json.dumps(end_to_end, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    accuracy = {
        "dense_task_accuracy": _mean(end_to_end, "dense_task_correct"),
        "sparse_task_accuracy": _mean(end_to_end, "sparse_task_correct"),
        "final_token_agreement": _mean(end_to_end, "final_token_agreement"),
        "final_exact_match_rate": _mean(end_to_end, "final_exact_match_dense"),
        "average_per_step_top1_agreement": _mean(
            end_to_end, "average_per_step_top1_agreement"
        ),
        "same_state_logit_relative_l2": _mean(
            one_step, "logit_relative_l2"
        ),
    }
    (output_dir / "accuracy.json").write_text(
        json.dumps(accuracy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "dense_regression.json").write_text(
        json.dumps(dense_regression, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _report(
        output_dir / "report.md",
        args,
        environment,
        stats,
        one_step,
        end_to_end,
        dense_regression,
    )
    print(json.dumps({"sparsity": stats.summary(), "accuracy": accuracy}, indent=2))


if __name__ == "__main__":
    main()
