"""Reduce a v23 bootstrap6 redacted scorer CSV into per-dataset arm tables and
paired contrasts (JSON + Markdown). Stdlib-only.

Usage:
    python -m scripts.v23_bootstrap6_reduce --csv PATH --out-json PATH --out-md PATH

Input: one row per (dataset, id, seed, arm) cell, six fixed arms. A
(dataset, id, seed) "block" is only used if all six arms have a non-empty
decoder_calls value (rows for blocks not yet executed are excluded).
"""

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict

ARMS = [
    "D_native",
    "T_scope",
    "M3_R3_A8_incumbent",
    "M1_native_bootstrap2_observe1",
    "M3_native_bootstrap2_observe1",
    "B_native_bootstrap2_observe1",
]

ALIAS = {
    "D_native": "D_native",
    "T_scope": "T_scope",
    "M3_R3_A8_incumbent": "M3_R3_A8_incumbent",
    "M1_native_bootstrap2_observe1": "M1_boot",
    "M3_native_bootstrap2_observe1": "M3_boot",
    "B_native_bootstrap2_observe1": "B_boot",
}
ALIAS_TO_ARM = {v: k for k, v in ALIAS.items()}

# Arms that do not use the router at all (blank router columns by design).
NO_ROUTER_ARMS = {"D_native", "T_scope"}

CONTRASTS = [
    ("M3_boot", "D_native"),
    ("M3_boot", "M3_R3_A8_incumbent"),
    ("M3_boot", "B_boot"),
    ("M3_boot", "T_scope"),
    ("M1_boot", "D_native"),
    ("B_boot", "D_native"),
    ("M3_R3_A8_incumbent", "D_native"),
    ("T_scope", "D_native"),
    ("M1_boot", "M3_boot"),
]

BOOTSTRAP_SEED = 23
BOOTSTRAP_RESAMPLES = 2000

HEADER_NOTE = (
    "First-generation outputs only were scored for quality; warm-repeat wall "
    "times are timing measurements, not independent samples; all intervals "
    "below are exploratory question-cluster bootstraps on an exposed "
    "development panel, not noninferiority tests."
)


def parse_float(s):
    if s is None or s == "":
        return None
    return float(s)


def parse_int(s):
    if s is None or s == "":
        return None
    return int(float(s))


def parse_bool(s):
    return s == "True"


def load_rows(csv_path):
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [r for r in reader if r.get("arm") in ARMS]


def group_blocks(rows):
    """dataset -> (id, seed) -> arm -> row"""
    by_dataset = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        key = (r["id"], r["seed"])
        by_dataset[r["dataset"]][key][r["arm"]] = r
    return by_dataset


def complete_blocks_for_dataset(blocks_by_key):
    complete = {}
    for key, arm_rows in blocks_by_key.items():
        ok = True
        for arm in ARMS:
            row = arm_rows.get(arm)
            if row is None or row.get("decoder_calls", "") == "":
                ok = False
                break
        if ok:
            complete[key] = arm_rows
    return complete


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def build_arm_table(dataset, complete, arm):
    n = len(complete)
    rows = [arm_rows[arm] for arm_rows in complete.values()]

    strict_correct_n = sum(1 for r in rows if parse_bool(r.get("strict_correct", "")))
    capped_n = sum(1 for r in rows if parse_bool(r.get("capped", "")))
    unparsed_n = sum(1 for r in rows if r.get("parsed", "") != "True")

    total_decoder_calls = sum(parse_int(r["decoder_calls"]) for r in rows)
    total_canvases = sum(parse_int(r["canvases"]) or 0 for r in rows)
    calls_per_canvas = (
        total_decoder_calls / total_canvases if total_canvases else None
    )
    total_output_tokens = sum(parse_int(r.get("output_tokens", "")) or 0 for r in rows)

    if arm in NO_ROUTER_ARMS:
        router_a = router_d = router_h = None
    else:
        router_a = sum(parse_int(r.get("router_A", "")) or 0 for r in rows)
        router_d = sum(parse_int(r.get("router_D", "")) or 0 for r in rows)
        router_h = sum(parse_int(r.get("router_H", "")) or 0 for r in rows)

    warm_accepted_rows = [r for r in rows if r.get("warm_status") == "accepted"]
    warm_accepted_count = len(warm_accepted_rows)
    sum_accepted_warm_wall = sum(
        parse_float(r["accepted_warm_request_wall_s"]) for r in warm_accepted_rows
    )
    sum_calls_warm_accepted = sum(
        parse_int(r["decoder_calls"]) for r in warm_accepted_rows
    )
    amortized_warm_wall_per_call_ms = (
        (sum_accepted_warm_wall / sum_calls_warm_accepted) * 1000.0
        if sum_calls_warm_accepted
        else None
    )

    per_host = defaultdict(list)
    for r in warm_accepted_rows:
        per_host[r["host"]].append(parse_float(r["accepted_warm_request_wall_s"]))
    per_host_mean_warm_wall = {
        host: (sum(vals) / len(vals)) for host, vals in per_host.items()
    }

    return {
        "n": n,
        "strict_correct": {"count": strict_correct_n, "n": n},
        "capped_count": capped_n,
        "unparsed_count": unparsed_n,
        "total_decoder_calls": total_decoder_calls,
        "total_canvases": total_canvases,
        "calls_per_canvas": calls_per_canvas,
        "total_output_tokens": total_output_tokens,
        "router_A_total": router_a,
        "router_D_total": router_d,
        "router_H_total": router_h,
        "warm_accepted_count": warm_accepted_count,
        "sum_accepted_warm_wall_s": sum_accepted_warm_wall,
        "amortized_warm_wall_per_call_ms": amortized_warm_wall_per_call_ms,
        "per_host_mean_warm_wall_s": per_host_mean_warm_wall,
    }


def build_contrast(complete, arm1, arm2):
    """arm1, arm2 are the real arm names (not aliases)."""
    S = []
    for (qid, seed), arm_rows in complete.items():
        r1 = arm_rows[arm1]
        r2 = arm_rows[arm2]
        if r1.get("warm_status") == "accepted" and r2.get("warm_status") == "accepted":
            wall1 = parse_float(r1["accepted_warm_request_wall_s"])
            wall2 = parse_float(r2["accepted_warm_request_wall_s"])
            calls1 = parse_int(r1["decoder_calls"])
            calls2 = parse_int(r2["decoder_calls"])
            S.append(
                {
                    "id": qid,
                    "seed": seed,
                    "wall1": wall1,
                    "wall2": wall2,
                    "calls1": calls1,
                    "calls2": calls2,
                    "correct1": parse_bool(r1.get("strict_correct", "")),
                    "correct2": parse_bool(r2.get("strict_correct", "")),
                }
            )

    n = len(S)
    if n == 0:
        return {
            "n": 0,
            "warm_wall_geo_ratio": None,
            "warm_wall_ratio_min": None,
            "warm_wall_ratio_max": None,
            "decoder_call_geo_ratio": None,
            "strict_correct_diff": {"arm1_correct_arm2_wrong": 0, "arm2_correct_arm1_wrong": 0},
            "bootstrap_ci_log_warm_wall_ratio": None,
        }

    wall_log_ratios = [math.log(b["wall1"] / b["wall2"]) for b in S]
    wall_raw_ratios = [b["wall1"] / b["wall2"] for b in S]
    call_log_ratios = [math.log(b["calls1"] / b["calls2"]) for b in S]

    warm_wall_geo_ratio = math.exp(statistics.mean(wall_log_ratios))
    decoder_call_geo_ratio = math.exp(statistics.mean(call_log_ratios))

    a1_correct_a2_wrong = sum(1 for b in S if b["correct1"] and not b["correct2"])
    a2_correct_a1_wrong = sum(1 for b in S if b["correct2"] and not b["correct1"])

    distinct_ids = sorted(set(b["id"] for b in S))
    if len(distinct_ids) < 2:
        bootstrap_ci = None
    else:
        by_id = defaultdict(list)
        for b, lr in zip(S, wall_log_ratios):
            by_id[b["id"]].append(lr)
        rng = random.Random(BOOTSTRAP_SEED)
        n_ids = len(distinct_ids)
        resample_stats = []
        for _ in range(BOOTSTRAP_RESAMPLES):
            sampled_ids = [rng.choice(distinct_ids) for _ in range(n_ids)]
            vals = []
            for sid in sampled_ids:
                vals.extend(by_id[sid])
            if vals:
                resample_stats.append(statistics.mean(vals))
        resample_stats.sort()
        lo = percentile(resample_stats, 2.5)
        hi = percentile(resample_stats, 97.5)
        bootstrap_ci = {
            "low": math.exp(lo),
            "high": math.exp(hi),
            "n_questions": n_ids,
            "n_resamples": len(resample_stats),
        }

    return {
        "n": n,
        "warm_wall_geo_ratio": warm_wall_geo_ratio,
        "warm_wall_ratio_min": min(wall_raw_ratios),
        "warm_wall_ratio_max": max(wall_raw_ratios),
        "decoder_call_geo_ratio": decoder_call_geo_ratio,
        "strict_correct_diff": {
            "arm1_correct_arm2_wrong": a1_correct_a2_wrong,
            "arm2_correct_arm1_wrong": a2_correct_a1_wrong,
        },
        "bootstrap_ci_log_warm_wall_ratio": bootstrap_ci,
    }


def fmt(x, digits=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{digits}g}"
    return str(x)


def render_markdown(result):
    lines = []
    lines.append("# v23 bootstrap6 reduction")
    lines.append("")
    lines.append(HEADER_NOTE)
    lines.append("")
    for dataset, dres in result["datasets"].items():
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(f"Complete blocks used: {dres['complete_block_count']}")
        lines.append("")
        lines.append(
            "| arm | n | strict_correct | capped | unparsed | total_calls | "
            "total_canvases | calls/canvas | total_out_tok | router A/D/H | "
            "warm_accepted | sum_warm_wall_s | amortized_ms/call |"
        )
        lines.append(
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
        )
        for arm in ARMS:
            t = dres["arm_table"][arm]
            router_str = (
                "n/a"
                if t["router_A_total"] is None
                else f"{t['router_A_total']}/{t['router_D_total']}/{t['router_H_total']}"
            )
            lines.append(
                "| {arm} | {n} | {sc}/{n} | {capped} | {unparsed} | {calls} | "
                "{canv} | {cpc} | {tok} | {router} | {warm} | {sumwall} | {amort} |".format(
                    arm=ALIAS[arm],
                    n=t["n"],
                    sc=t["strict_correct"]["count"],
                    capped=t["capped_count"],
                    unparsed=t["unparsed_count"],
                    calls=t["total_decoder_calls"],
                    canv=t["total_canvases"],
                    cpc=fmt(t["calls_per_canvas"]),
                    tok=t["total_output_tokens"],
                    router=router_str,
                    warm=t["warm_accepted_count"],
                    sumwall=fmt(t["sum_accepted_warm_wall_s"]),
                    amort=fmt(t["amortized_warm_wall_per_call_ms"]),
                )
            )
            hosts = t["per_host_mean_warm_wall_s"]
            if hosts:
                host_str = ", ".join(f"{h}: {fmt(v)}" for h, v in sorted(hosts.items()))
                lines.append(f"  - per-host mean warm wall (s): {host_str}")
        lines.append("")
        lines.append(
            "| contrast (arm1/arm2) | n | warm geo ratio | min | max | "
            "calls geo ratio | arm1✓arm2✗ | arm2✓arm1✗ | 95% CI (log warm ratio, exp) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for key, c in dres["contrasts"].items():
            if c["bootstrap_ci_log_warm_wall_ratio"] is None:
                ci_str = "null (fewer than 2 distinct questions)" if c["n"] > 0 else "n/a"
            else:
                ci = c["bootstrap_ci_log_warm_wall_ratio"]
                ci_str = f"[{fmt(ci['low'])}, {fmt(ci['high'])}] (n_q={ci['n_questions']})"
            lines.append(
                "| {key} | {n} | {geo} | {mn} | {mx} | {cgeo} | {a1} | {a2} | {ci} |".format(
                    key=key,
                    n=c["n"],
                    geo=fmt(c["warm_wall_geo_ratio"]),
                    mn=fmt(c["warm_wall_ratio_min"]),
                    mx=fmt(c["warm_wall_ratio_max"]),
                    cgeo=fmt(c["decoder_call_geo_ratio"]),
                    a1=c["strict_correct_diff"]["arm1_correct_arm2_wrong"],
                    a2=c["strict_correct_diff"]["arm2_correct_arm1_wrong"],
                    ci=ci_str,
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    for out_path in (args.out_json, args.out_md):
        if os.path.exists(out_path):
            print(f"refusing to overwrite existing output: {out_path}", file=sys.stderr)
            sys.exit(1)

    rows = load_rows(args.csv)
    by_dataset = group_blocks(rows)

    result = {"csv": args.csv, "arms": ARMS, "header_note": HEADER_NOTE, "datasets": {}}

    for dataset in sorted(by_dataset.keys()):
        blocks_by_key = by_dataset[dataset]
        complete = complete_blocks_for_dataset(blocks_by_key)

        arm_table = {arm: build_arm_table(dataset, complete, arm) for arm in ARMS}

        contrasts = {}
        for alias1, alias2 in CONTRASTS:
            arm1 = ALIAS_TO_ARM[alias1]
            arm2 = ALIAS_TO_ARM[alias2]
            contrasts[f"{alias1}/{alias2}"] = build_contrast(complete, arm1, arm2)

        result["datasets"][dataset] = {
            "complete_block_count": len(complete),
            "total_block_count": len(blocks_by_key),
            "arm_table": arm_table,
            "contrasts": contrasts,
        }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    md = render_markdown(result)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
