"""Native uncertainty-adaptive sweep for the preserved multi-benchmark manifests.

This runner deliberately creates a new result bundle.  Dense/reference outputs and
frozen full-centered policies are read from audited predecessor bundles; no final
scores are used to select thresholds.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import torch

from dllm.models import create_adapter
from experiments.diffusion_gemma_jl_output_aware import runner
from experiments.diffusion_gemma_value_aware.run import shard_path


BASE = Path("results")
SOURCES = {
    "aime_longbench": BASE / "diffusion_gemma_jl_lowrank_aime30_longbench100_v11",
    "ruler8k": BASE / "diffusion_gemma_ruler8k_gaussian_rank_sweep_v15",
}
ROOTS = {
    "aime_longbench": BASE / "diffusion_gemma_adaptive_aime30_longbench100_v22",
    "ruler8k": BASE / "diffusion_gemma_ruler8k_adaptive_v22",
}
TARGETS = {"aime_longbench": (0.5,), "ruler8k": (0.5, 0.75)}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    tmp.replace(path)


def load_policies(kind, source):
    data = json.loads((source / "thresholds.json").read_text())
    out = {}
    for row in data:
        name = row.get("name")
        target = float(row.get("target", -1))
        if name != "full_centered" or target not in TARGETS[kind]:
            continue
        if kind == "ruler8k":
            out.setdefault(target, {})[row["attention_type"]] = {
                "log_threshold": float(row["log_threshold"]),
                "source": str(source / "thresholds.json"),
            }
        else:
            # v11 stores one policy object per benchmark/target.
            out.setdefault(target, {}).update({k: dict(v) for k, v in row["policy"].items()})
    for t in TARGETS[kind]:
        if set(out.get(t, {})) != {"local", "global"}:
            raise ValueError(f"missing full-centered policy for {kind} target {t}")
    return out


def aggregate(out):
    records = out.get("records", [])
    def sums(kind=None):
        rr = records if kind is None else [r for r in records if r.get("attention_type") == kind]
        return sum(int(r.get("eligible", 0)) for r in rr), sum(int(r.get("skipped", 0)) for r in rr)
    eligible, skipped = sums()
    ge, gs = sums("global"); le, ls = sums("local")
    return dict(id=out["id"], benchmark=out.get("benchmark"), score=out.get("score"),
                eligible=eligible, skipped=skipped, sparsity=skipped/max(1, eligible),
                global_eligible=ge, global_skipped=gs, global_sparsity=gs/max(1, ge),
                local_eligible=le, local_skipped=ls, local_sparsity=ls/max(1, le),
                completion_tokens=out.get("completion_tokens", []), finite_calls=out.get("finite_calls"),
                work=out.get("work_accounting", {}))


def run(kind, root=None, source=None, max_examples=None):
    source = Path(source or SOURCES[kind]); root = Path(root or ROOTS[kind])
    setup = json.loads((source / "setup.json").read_text())
    rows = list(setup["final"])
    if kind == "aime_longbench":
        rows = [r for r in rows if r.get("benchmark") in ("aime26", "longbench_v2")]
    if max_examples is not None:
        rows = rows[: int(max_examples)]
    frozen = load_policies(kind, source)
    config = dict(method="adaptive", family="gaussian", rank=64,
                  cascade_stages=(2, 8, 32, 64), projection_seed=1729,
                  interval_delta=1e-3, exact_fallback=True)
    manifest = dict(schema=f"{kind}_adaptive_v22", source=str(source), targets=list(TARGETS[kind]),
                    config=config, examples=len(rows), dense_reused=True,
                    final_scores_not_used_for_thresholds=True,
                    threshold_source=str(source / "thresholds.json"),
                    note="Native adaptive route keeps unresolved max-rank tiles; exact fallback is only a frozen-history diagnostic.")
    write_json(root / "setup.json", manifest); write_json(root / "thresholds.json", frozen)
    adapter = create_adapter("diffusion_gemma", setup["model"], device="cuda",
                             precision="bfloat16", revision=setup["revision"]).load()
    old_score = runner.score
    if kind == "ruler8k":
        from experiments import diffusion_gemma_ruler8k_gaussian_sweep as ruler
        runner.score = ruler.base.score
    results = []
    total = len(rows) * len(TARGETS[kind]); done = 0
    try:
        for target in TARGETS[kind]:
            label = f"adaptive_analytic_s{int(target*100)}"
            policy = {k: dict(v) for k, v in frozen[target].items()}
            for row in sorted(rows, key=lambda r: (len(r["prompt_tokens"]), r["id"])):
                path = shard_path(root, "final", label, row["id"])
                if path.exists():
                    out = runner.load_output(path)
                else:
                    out = runner.generate(adapter, row, "adaptive", config, policy)
                    out.update(target=target, condition=label, benchmark=row.get("benchmark"))
                    runner.write_output(path, out)
                item = aggregate(out); item.update(target=target, condition=label)
                results.append(item); done += 1
                hb = dict(stage="final", condition=label, completed=done, total=total,
                          target=target, id=row["id"], updated=time.time())
                write_json(root / "progress.json", hb)
                with (root / "progress.jsonl").open("a") as f: f.write(json.dumps(hb) + "\n")
                (root / "progress.md").write_text(
                    f"# Adaptive {kind} heartbeat\n\nCompleted {done}/{total}; condition `{label}`; latest `{row['id']}`.\n"
                    f"Updated Unix time {hb['updated']:.3f}.\n")
    finally:
        runner.score = old_score
        del adapter; torch.cuda.empty_cache()
    write_json(root / "results.json", results)
    summaries = []
    for target in TARGETS[kind]:
        group = [r for r in results if r["target"] == target]
        def weighted(key):
            return sum(r[key] * r["eligible"] for r in group) / max(1, sum(r["eligible"] for r in group))
        ge = sum(r["global_eligible"] for r in group); gs = sum(r["global_skipped"] for r in group)
        le = sum(r["local_eligible"] for r in group); ls = sum(r["local_skipped"] for r in group)
        summaries.append(dict(condition=f"adaptive_analytic_s{int(target*100)}", examples=len(group),
            accuracy=sum(float(r.get("score") or 0) for r in group) / max(1, len(group)),
            actual_sparsity=weighted("sparsity"), global_sparsity=gs/max(1, ge),
            local_sparsity=ls/max(1, le), finite_calls=sum(r.get("finite_calls") or 0 for r in group)))
    write_json(root / "summary.json", summaries)
    if summaries:
        with (root / "summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0])); w.writeheader(); w.writerows(summaries)
    lines = [f"# Native uncertainty-adaptive {kind} extension", "",
             f"Examples: {len(rows)}; dense outputs reused from `{source}`.",
             "Thresholds are frozen full-centered policies from the audited predecessor; no final score selected them.", "",
             "| condition | examples | accuracy | actual sparsity | global | local |",
             "|---|---:|---:|---:|---:|---:|"]
    for x in summaries:
        lines.append(f"| {x['condition']} | {x['examples']} | {100*x['accuracy']:.2f}% | {100*x['actual_sparsity']:.2f}% | {100*x['global_sparsity']:.2f}% | {100*x['local_sparsity']:.2f}% |")
    (root / "report.md").write_text("\n".join(lines) + "\n")
    files = [root / n for n in ("setup.json", "thresholds.json", "results.json", "summary.json", "summary.csv", "report.md") if (root / n).exists()]
    audit = dict(passed=len(results) == total, complete=len(results) == total,
                 completed=len(results), expected=total, dense_reused=True,
                 artifacts={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    write_json(root / "audit.json", audit)
    return audit


def main():
    p = argparse.ArgumentParser(); p.add_argument("kind", choices=tuple(SOURCES)); p.add_argument("--output", type=Path); p.add_argument("--source", type=Path); p.add_argument("--max-examples", type=int); a = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    print(json.dumps(run(a.kind, a.output, a.source, a.max_examples), indent=2))


if __name__ == "__main__": main()
