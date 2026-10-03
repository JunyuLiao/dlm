"""Generate paired diagnostics for a completed HumanEval v2 run."""
from __future__ import annotations

import argparse, collections, difflib, json, math, re
from pathlib import Path

METHODS = ("dense", "gaussian32", "blasst_aggressive", "C_gate", "T_prior")


def read_rows(root: Path, method: str):
    return json.loads((root / f"{method}_evaluation.json").read_text())


def paired(rows):
    ids = [r["id"] for r in rows["blasst_aggressive"]]
    score = {m: {r["id"]: int(bool(r["score"])) for r in rows[m]} for m in METHODS}
    out = {}
    for other in ("gaussian32", "C_gate", "T_prior"):
        d = [score["blasst_aggressive"][i] - score[other][i] for i in ids]
        out[f"BLASST_minus_{other}"] = dict(
            wins=sum(x > 0 for x in d), losses=sum(x < 0 for x in d), ties=sum(x == 0 for x in d),
            delta=sum(d) / len(d), differences=d,
        )
    return out


def quality(rows):
    result = {}
    for m in METHODS:
        rr = rows[m]
        statuses = collections.Counter(r.get("humaneval_execution", {}).get("status", "unknown") for r in rr)
        by_status = {}
        for status in statuses:
            group = [r for r in rr if r.get("humaneval_execution", {}).get("status") == status]
            by_status[status] = dict(n=len(group), mean_steps=sum(r["steps"] for r in group) / len(group),
                                     mean_chars=sum(len(r.get("prediction", "")) for r in group) / len(group))
        by_stratum = {}
        for stratum in sorted({r.get("stratum", "unknown") for r in rr}):
            group = [r for r in rr if r.get("stratum") == stratum]
            by_stratum[stratum] = dict(n=len(group), accuracy=sum(r["score"] for r in group) / len(group),
                                       mean_steps=sum(r["steps"] for r in group) / len(group),
                                       syntax_errors=sum(r.get("humaneval_execution", {}).get("status") == "syntax_error" for r in group))
        result[m] = dict(statuses=dict(statuses), by_status=by_status, by_stratum=by_stratum)
    return result


def similarity(rows):
    dense = {r["id"]: r["prediction"] for r in rows["dense"]}
    result = {}
    for m in METHODS[1:]:
        values = []
        for r in rows[m]:
            values.append(difflib.SequenceMatcher(None, dense[r["id"]], r["prediction"]).ratio())
        result[m] = dict(mean=sum(values) / len(values), median=sorted(values)[len(values) // 2],
                         p10=sorted(values)[max(0, len(values) // 10 - 1)])
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args(); root = args.root
    rows = {m: read_rows(root, m) for m in METHODS}
    report = dict(paired=paired(rows), quality=quality(rows), similarity=similarity(rows))
    thresholds = {}
    for m in METHODS[1:]:
        p = root / "thresholds" / f"{m}_s50.json"
        if p.exists(): thresholds[m] = json.loads(p.read_text())
    report["thresholds"] = thresholds
    (root / "paired_diagnostics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# HumanEval v2 paired diagnostics", "", "## BLASST paired differences", "", "| Comparison | BLASST wins | losses | ties | accuracy delta |", "|---|---:|---:|---:|---:|"]
    for k, v in report["paired"].items():
        lines.append(f"| {k} | {v['wins']} | {v['losses']} | {v['ties']} | {v['delta']:.1%} |")
    lines += ["", "## Failure and trajectory status", "", "| Method | Status counts |", "|---|---|"]
    for m, v in report["quality"].items():
        lines.append(f"| {m} | `{json.dumps(v['statuses'], sort_keys=True)}` |")
    lines += ["", "## Output similarity to dense", "", "| Method | Mean | Median | P10 |", "|---|---:|---:|---:|"]
    for m, v in report["similarity"].items():
        lines.append(f"| {m} | {v['mean']:.3f} | {v['median']:.3f} | {v['p10']:.3f} |")
    (root / "paired_diagnostics.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
