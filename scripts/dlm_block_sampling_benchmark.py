#!/usr/bin/env python3
"""模拟并绘制 DLM 动态 block sampling 实验。

核心点：一个 request 的一个 block 里，每个 token/位置都有自己的完成 step。
如果 prompt 更难，token_steps 会整体更大、尾部更重；如果 batch 里混入 hard
request，同步调度会让 easy request 陪跑，从而出现明显 straggler / waste。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median

DIFFICULTY_SCALE = {
    "easy": 0.55,
    "medium": 1.00,
    "hard": 1.85,
    "extreme": 3.00,
}
DIFFICULTY_SIGMA = {
    "easy": 0.32,
    "medium": 0.48,
    "hard": 0.65,
    "extreme": 0.82,
}


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: int
    difficulty: str
    prompt: str


@dataclass(frozen=True)
class RequestRow:
    batch_size: int
    block_size: int
    scheduler: str
    trial: int
    request_id: int
    block_index: int
    prompt_id: int
    prompt_difficulty: str
    steps_needed: int
    steps_executed: int
    useful_token_steps: int
    executed_token_steps: int
    token_step_min: int
    token_step_mean: float
    token_step_max: int
    block_internal_waste_ratio: float
    latency_ms: float
    tokens_generated: int


def infer_difficulty(prompt: str) -> str:
    """没有显式 difficulty 时，用关键词粗略推断 prompt 难度。"""

    lower = prompt.lower()
    extreme_words = ["证明", "prove", "复杂", "logic", "推理题", "step by step"]
    hard_words = ["sql", "algorithm", "算法", "代码", "python", "cuda", "数学", "推理"]
    easy_words = ["翻译", "translate", "改写", "email", "两句话"]
    if any(word in lower for word in extreme_words):
        return "extreme"
    if any(word in lower for word in hard_words):
        return "hard"
    if any(word in lower for word in easy_words):
        return "easy"
    return "medium"


def read_prompt_specs(path: Path | None) -> list[PromptSpec]:
    if path is None:
        return [
            PromptSpec(0, "easy", "简单翻译：今天天气很好。"),
            PromptSpec(1, "medium", "解释 DLM dynamic sampling。"),
            PromptSpec(2, "hard", "写 Python 质数判断函数并解释复杂度。"),
            PromptSpec(3, "extreme", "构造并求解复杂逻辑推理题。"),
        ]

    specs: list[PromptSpec] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                item = json.loads(line)
                prompt = str(item["prompt"])
                difficulty = str(item.get("difficulty") or infer_difficulty(prompt))
            else:
                prompt = line
                difficulty = infer_difficulty(prompt)
            if difficulty not in DIFFICULTY_SCALE:
                raise ValueError(f"unknown difficulty {difficulty!r}; use {sorted(DIFFICULTY_SCALE)}")
            specs.append(PromptSpec(index, difficulty, prompt))
    if not specs:
        raise ValueError(f"prompt file is empty: {path}")
    return specs


def choose_prompt_batch(prompt_specs: list[PromptSpec], batch_size: int, trial: int) -> list[PromptSpec]:
    """构造 mixed-complexity batch，让 easy/hard 差距尽量明显。"""

    by_level = {level: [p for p in prompt_specs if p.difficulty == level] for level in DIFFICULTY_SCALE}
    ordered_levels = ["easy", "medium", "hard", "extreme"]
    batch: list[PromptSpec] = []
    cursor = trial
    while len(batch) < batch_size:
        for level in ordered_levels:
            candidates = by_level[level]
            if not candidates:
                continue
            batch.append(candidates[cursor % len(candidates)])
            if len(batch) == batch_size:
                break
        cursor += 1
    return batch


def forward_time_ms(active_token_count: int, batch_size: int, block_size: int) -> float:
    """H100-like toy latency for one DLM denoising/refinement forward pass."""

    launch_overhead = 0.10
    batch_cost = 0.10 * (batch_size**0.70)
    token_cost = 0.010 * (max(active_token_count, 1) ** 0.62)
    block_cost = 0.006 * (block_size**0.55)
    return launch_overhead + batch_cost + token_cost + block_cost


def sample_token_steps(block_size: int, prompt: PromptSpec, rng: random.Random) -> list[int]:
    """根据 prompt 难度，为一个 block 内每个 token 采样完成 step。

    easy prompt 的 token_steps 较小且集中；hard/extreme prompt 的 token_steps 更大、
    方差更大，所以同一个 batch 内会出现明显 straggler。
    """

    scale = DIFFICULTY_SCALE[prompt.difficulty]
    sigma = DIFFICULTY_SIGMA[prompt.difficulty]
    base = max(2.0, block_size / 12.0) * scale
    prompt_length_factor = min(1.45, 1.0 + len(prompt.prompt) / 600.0)
    steps = []
    for token_index in range(block_size):
        position_factor = 1.0 + 0.45 * (token_index / max(block_size - 1, 1))
        local_burst = 1.0
        if prompt.difficulty in {"hard", "extreme"} and rng.random() < 0.12:
            local_burst = rng.uniform(1.5, 2.6)
        value = rng.lognormvariate(
            mu=math.log(base * prompt_length_factor * position_factor * local_burst),
            sigma=sigma,
        )
        steps.append(max(1, min(128, int(round(value)))))
    return steps


def estimate_latency_ms(
    all_token_steps: list[list[int]],
    scheduler: str,
    batch_size: int,
    block_size: int,
    request_id: int,
    rng: random.Random,
) -> tuple[int, int, float]:
    request_steps = max(all_token_steps[request_id])
    if scheduler == "sync":
        batch_steps = max(max(token_steps) for token_steps in all_token_steps)
        executed_steps = batch_steps
        executed_token_steps = batch_steps * block_size
        latency = sum(
            forward_time_ms(batch_size * block_size, batch_size, block_size)
            for _ in range(batch_steps)
        )
    elif scheduler == "dynamic":
        executed_steps = request_steps
        executed_token_steps = sum(all_token_steps[request_id])
        latency = 0.0
        for step in range(1, request_steps + 1):
            active_tokens = sum(
                1
                for token_steps in all_token_steps
                for token_step in token_steps
                if token_step >= step
            )
            latency += forward_time_ms(active_tokens, batch_size, block_size)
        latency *= 1.05
    else:
        raise ValueError(f"unknown scheduler: {scheduler}")
    return executed_steps, executed_token_steps, latency * rng.uniform(0.97, 1.03)


def simulate_trial(
    batch_size: int,
    block_size: int,
    scheduler: str,
    trial: int,
    prompt_specs: list[PromptSpec],
    rng: random.Random,
) -> list[RequestRow]:
    prompt_batch = choose_prompt_batch(prompt_specs, batch_size, trial)
    all_token_steps = [sample_token_steps(block_size, prompt, rng) for prompt in prompt_batch]
    rows: list[RequestRow] = []
    for request_id, (prompt, token_steps) in enumerate(zip(prompt_batch, all_token_steps)):
        steps_needed = max(token_steps)
        useful_token_steps = sum(token_steps)
        steps_executed, executed_token_steps, latency_ms = estimate_latency_ms(
            all_token_steps,
            scheduler,
            batch_size,
            block_size,
            request_id,
            rng,
        )
        rows.append(
            RequestRow(
                batch_size=batch_size,
                block_size=block_size,
                scheduler=scheduler,
                trial=trial,
                request_id=request_id,
                block_index=0,
                prompt_id=prompt.prompt_id,
                prompt_difficulty=prompt.difficulty,
                steps_needed=steps_needed,
                steps_executed=steps_executed,
                useful_token_steps=useful_token_steps,
                executed_token_steps=executed_token_steps,
                token_step_min=min(token_steps),
                token_step_mean=mean(token_steps),
                token_step_max=max(token_steps),
                block_internal_waste_ratio=(steps_needed * block_size) / useful_token_steps,
                latency_ms=latency_ms,
                tokens_generated=block_size,
            )
        )
    return rows


def simulate(args: argparse.Namespace) -> list[RequestRow]:
    rng = random.Random(args.seed)
    prompt_specs = read_prompt_specs(args.prompt_file)
    rows: list[RequestRow] = []
    for block_size in args.block_sizes:
        for batch_size in args.batch_sizes:
            for scheduler in args.schedulers:
                for trial in range(args.trials):
                    rows.extend(
                        simulate_trial(
                            batch_size,
                            block_size,
                            scheduler,
                            trial,
                            prompt_specs,
                            rng,
                        )
                    )
    return rows


def read_rows(path: Path) -> list[RequestRow]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            block_size = int(row["block_size"])
            steps_needed = int(row["steps_needed"])
            steps_executed = int(row["steps_executed"])
            useful_token_steps = int(row.get("useful_token_steps") or steps_needed)
            executed_token_steps = int(row.get("executed_token_steps") or steps_executed)
            token_step_min = int(row.get("token_step_min") or steps_needed)
            token_step_mean = float(row.get("token_step_mean") or steps_needed)
            token_step_max = int(row.get("token_step_max") or steps_needed)
            block_internal_waste_ratio = float(
                row.get("block_internal_waste_ratio")
                or ((token_step_max * block_size) / useful_token_steps)
            )
            rows.append(
                RequestRow(
                    batch_size=int(row["batch_size"]),
                    block_size=block_size,
                    scheduler=row["scheduler"],
                    trial=int(row["trial"]),
                    request_id=int(row["request_id"]),
                    block_index=int(row.get("block_index") or 0),
                    prompt_id=int(row.get("prompt_id") or row.get("request_id") or 0),
                    prompt_difficulty=row.get("prompt_difficulty") or "unknown",
                    steps_needed=steps_needed,
                    steps_executed=steps_executed,
                    useful_token_steps=useful_token_steps,
                    executed_token_steps=executed_token_steps,
                    token_step_min=token_step_min,
                    token_step_mean=token_step_mean,
                    token_step_max=token_step_max,
                    block_internal_waste_ratio=block_internal_waste_ratio,
                    latency_ms=float(row["latency_ms"]),
                    tokens_generated=int(row["tokens_generated"]),
                )
            )
        return rows


def write_rows(rows: list[RequestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    index = (len(values) - 1) * pct / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[int(index)]
    return values[lower] * (upper - index) + values[upper] * (index - lower)


def summarize(rows: list[RequestRow], path: Path) -> None:
    groups: dict[tuple[int, int, str], list[RequestRow]] = {}
    for row in rows:
        groups.setdefault((row.batch_size, row.block_size, row.scheduler), []).append(row)

    fieldnames = [
        "batch_size",
        "block_size",
        "scheduler",
        "mean_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "throughput_tokens_per_s",
        "straggler_ratio",
        "batch_waste_ratio",
        "block_internal_waste_ratio",
        "mean_steps_needed",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (batch_size, block_size, scheduler), group in sorted(groups.items()):
            latencies = [row.latency_ms for row in group]
            useful = sum(row.useful_token_steps for row in group)
            executed = sum(row.executed_token_steps for row in group)
            tokens = sum(row.tokens_generated for row in group)
            total_time_s = sum(latencies) / batch_size / 1000.0
            writer.writerow(
                {
                    "batch_size": batch_size,
                    "block_size": block_size,
                    "scheduler": scheduler,
                    "mean_latency_ms": f"{mean(latencies):.3f}",
                    "p50_latency_ms": f"{median(latencies):.3f}",
                    "p95_latency_ms": f"{percentile(latencies, 95):.3f}",
                    "throughput_tokens_per_s": f"{tokens / total_time_s:.3f}",
                    "straggler_ratio": f"{max(latencies) / mean(latencies):.3f}",
                    "batch_waste_ratio": f"{executed / useful:.3f}",
                    "block_internal_waste_ratio": f"{mean([row.block_internal_waste_ratio for row in group]):.3f}",
                    "mean_steps_needed": f"{mean([row.steps_needed for row in group]):.3f}",
                }
            )


def write_difficulty_summary(rows: list[RequestRow], path: Path) -> None:
    groups: dict[tuple[str, str], list[RequestRow]] = {}
    for row in rows:
        groups.setdefault((row.prompt_difficulty, row.scheduler), []).append(row)
    fieldnames = ["prompt_difficulty", "scheduler", "mean_steps_needed", "p95_steps_needed", "mean_latency_ms"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (difficulty, scheduler), group in sorted(groups.items()):
            steps = [row.steps_needed for row in group]
            writer.writerow(
                {
                    "prompt_difficulty": difficulty,
                    "scheduler": scheduler,
                    "mean_steps_needed": f"{mean(steps):.3f}",
                    "p95_steps_needed": f"{percentile(steps, 95):.3f}",
                    "mean_latency_ms": f"{mean([row.latency_ms for row in group]):.3f}",
                }
            )


def write_prompt_block_steps(rows: list[RequestRow], path: Path) -> None:
    """Write a compact table: each prompt/request block needs how many steps."""

    fieldnames = [
        "trial",
        "batch_size",
        "block_size",
        "scheduler",
        "request_id",
        "prompt_id",
        "prompt_difficulty",
        "block_index",
        "steps_needed",
        "steps_executed",
        "token_step_min",
        "token_step_mean",
        "token_step_max",
        "useful_token_steps",
        "executed_token_steps",
        "latency_ms",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def scheduler_label(scheduler: str) -> str:
    labels = {
        "sync": "simulated sync batch",
        "dynamic": "simulated dynamic token skip",
        "real_llada_confidence_cutoff": "real LLaDA confidence-cutoff batch",
        "real_llada_topk": "real LLaDA fixed top-k batch",
        "real_llada_threshold": "real LLaDA threshold batch",
    }
    return labels.get(scheduler, scheduler)

def actual_step_schedulers(rows: list[RequestRow]) -> set[str]:
    schedulers = {row.scheduler for row in rows}
    real = {scheduler for scheduler in schedulers if scheduler.startswith("real_llada")}
    if real:
        return real
    if "dynamic" in schedulers:
        return {"dynamic"}
    return schedulers


def plot(rows: list[RequestRow], outdir: Path, data_label: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipped PNG plots. Install with: pip install matplotlib")
        return False

    outdir.mkdir(parents=True, exist_ok=True)
    block_for_exp1 = 32 if any(row.block_size == 32 for row in rows) else rows[0].block_size
    batch_for_exp2 = 8 if any(row.batch_size == 8 for row in rows) else rows[0].batch_size
    batch_for_exp3 = batch_for_exp2

    fig, ax = plt.subplots(figsize=(7, 4))
    for scheduler in sorted({row.scheduler for row in rows}):
        xs, ys = [], []
        for batch_size in sorted({row.batch_size for row in rows}):
            group = [
                row
                for row in rows
                if row.block_size == block_for_exp1
                and row.batch_size == batch_size
                and row.scheduler == scheduler
            ]
            if group:
                xs.append(batch_size)
                ys.append(sum(row.executed_token_steps for row in group) / sum(row.useful_token_steps for row in group))
        ax.plot(xs, ys, marker="o", label=scheduler_label(scheduler))
    ax.set_title(f"Exp1: batch size vs wasted work (block={block_for_exp1})\n{data_label}")
    ax.set_xlabel("batch size")
    ax.set_ylabel("waste ratio (1.0 = no wasted token-steps)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "exp1_batch_speed_gap.png", dpi=180)
    plt.close(fig)

    preferred_schedulers = actual_step_schedulers(rows)
    example = [
        row
        for row in rows
        if row.batch_size == batch_for_exp2
        and row.block_size == block_for_exp1
        and row.trial == 0
        and row.scheduler in preferred_schedulers
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    difficulties = ["easy", "medium", "hard", "extreme", "unknown"]
    for difficulty in difficulties:
        group = [row for row in example if row.prompt_difficulty == difficulty]
        if not group:
            continue
        ax.scatter(
            [row.steps_needed for row in group],
            [row.latency_ms for row in group],
            label=difficulty,
            s=55,
            alpha=0.75,
        )
    ax.set_title(f"Exp2: actual block steps vs latency (dynamic rows only)\n{data_label}")
    ax.set_xlabel("actual steps used by this request block")
    ax.set_ylabel("request/block latency (ms)")
    if ax.has_data():
        ax.legend(title="prompt difficulty")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "exp2_steps_vs_perf.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for scheduler in sorted({row.scheduler for row in rows}):
        xs, ys = [], []
        for block_size in sorted({row.block_size for row in rows}):
            group = [
                row
                for row in rows
                if row.batch_size == batch_for_exp3
                and row.block_size == block_size
                and row.scheduler == scheduler
            ]
            if group:
                xs.append(block_size)
                ys.append(mean(row.latency_ms for row in group))
        ax.plot(xs, ys, marker="o", label=scheduler_label(scheduler))
    ax.set_title(f"Exp3: block size vs mean latency (batch={batch_for_exp3})\n{data_label}")
    ax.set_xlabel("block size")
    ax.set_ylabel("mean request/block latency (ms)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "exp3_block_size_latency.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    order = ["easy", "medium", "hard", "extreme", "unknown"]
    labels, values = [], []
    for difficulty in order:
        group = [row for row in rows if row.prompt_difficulty == difficulty and row.scheduler in actual_step_schedulers(rows)]
        if group:
            labels.append(difficulty)
            values.append(mean(row.steps_needed for row in group))
    ax.bar(labels, values)
    ax.set_title(f"Exp4: prompt difficulty vs actual steps used\n{data_label}")
    ax.set_xlabel("prompt difficulty")
    ax.set_ylabel("mean actual block steps")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "exp4_prompt_difficulty_steps.png", dpi=180)
    plt.close(fig)
    return True


def write_plot_guide(outdir: Path, num_blocks: int | None = None, data_label: str = "simulated data only") -> None:
    block_text = "unknown" if num_blocks is None else str(num_blocks)
    guide = f"""# Plot guide

数据来源：`{data_label}`。H100 真实实验只消费 LLaDA real probe 写出的 CSV；模拟模式只用于 smoke/debug。

- `real LLaDA confidence-cutoff batch`: H100 主实验结果；每一步按 confidence 从高到低接受达到阈值的前缀 token，所以不同 block/request 可以用不同步数。
- `real LLaDA fixed top-k batch`: 官方固定步数 top-k unmasking 对照；如果 `max_steps_per_block > block_size`，`steps_used` 很可能等于 `block_size`，不适合作为动态步数主图。
- `simulated sync batch` / `simulated dynamic token skip`: 只会出现在 simulator 输出里，用于 debug 图表，不作为 H100 真实实验结论。

## 输出文件

- `per_request_rows.csv`: 完整分析表；H100 真实 runner 中只包含真实 `real_llada_*` 行，用于画真实 latency/step 图。
- `prompt_block_steps.csv`: 从完整表整理出的 per request/block step 表。
- H100 真实模型 runner 还会写 `../block_steps.csv`: 一行一个 prompt/request/block，字段包括 `steps_used`, `mean_confidence`, `min_confidence`。

## 图怎么读

1. `exp1_batch_speed_gap.png`: batch size 越大，sync 的 waste ratio 通常越高，说明容易被 hardest request 拖住；真实 LLaDA 曲线展示实际 batch/block 的同步 cost。
2. `exp2_steps_vs_perf.png`: 只画真实/动态行，横轴是这个 block 实际用了多少 denoising steps，纵轴是 latency。
3. `exp3_block_size_latency.png`: block size 16/32/64 对 mean latency 的影响。
4. `exp4_prompt_difficulty_steps.png`: 不同 difficulty 的 prompt 平均实际 block steps；真实模型下规律不一定单调，要以 `block_steps.csv` 为准。

当前默认 NUM_BLOCKS={block_text}；每个 request 会有多个 `block_index`。
"""
    (outdir / "plot_guide.md").write_text(guide, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["simulate", "plot-csv"], default="simulate")
    parser.add_argument("--input-csv", type=Path, help="Per-request CSV from a real DLM run")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/dlm_block_sampling"))
    parser.add_argument("--prompt-file", type=Path, default=Path("data/prompts_heterogeneous.jsonl"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--schedulers", nargs="+", default=["sync", "dynamic"])
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data-label", default=None, help="Label printed in plot titles and plot_guide.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "plot-csv":
        if not args.input_csv:
            raise SystemExit("--input-csv is required with --mode plot-csv")
        rows = read_rows(args.input_csv)
    else:
        rows = simulate(args)
    data_label = args.data_label or ("real LLaDA H100 data" if args.mode == "plot-csv" else "simulated data only")

    raw_csv = args.outdir / "per_request_rows.csv"
    summary_csv = args.outdir / "summary.csv"
    difficulty_csv = args.outdir / "difficulty_summary.csv"
    prompt_block_csv = args.outdir / "prompt_block_steps.csv"
    write_rows(rows, raw_csv)
    summarize(rows, summary_csv)
    write_difficulty_summary(rows, difficulty_csv)
    write_prompt_block_steps(rows, prompt_block_csv)
    plotted = plot(rows, args.outdir, data_label)
    num_blocks = max((row.block_index for row in rows), default=0) + 1
    write_plot_guide(args.outdir, num_blocks, data_label)
    print(f"wrote {raw_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {difficulty_csv}")
    print(f"wrote {prompt_block_csv}")
    print(f"wrote {args.outdir / 'plot_guide.md'}")
    if plotted:
        print(f"wrote figures under {args.outdir}")


if __name__ == "__main__":
    main()
