"""Reproducible CPU-only audit of published v20 request trajectories.

The metadata-only selection is frozen before completion text is accessed.
Qualified first-cell scores may be supplied with --quality-json; absent scores
remain null rather than being reconstructed with a different scorer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("D_native", "D_matched", "T_scope", "B_A8_matched", "M1_R1_A8_current_output",
        "M3_R2_A8_current_output", "M3_R3_A8_current_output", "G75L30_nativeQ128")
REFERENCE_ARMS = ("D_native", "T_scope", "B_A8_matched")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS = ROOT / "results/fan_m1_m3_multidataset_20260927/generation_records"
DEFAULT_OUT = ROOT / "results/m3_numeric_trajectory_bridge_20260927"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def quantiles(values: list[float]) -> dict | None:
    if not values:
        return None
    s = sorted(values)
    def q(p):
        x = (len(s) - 1) * p
        a, b = int(x), min(int(x) + 1, len(s) - 1)
        return s[a] + (s[b] - s[a]) * (x - a)
    return {"n": len(s), "mean": statistics.mean(s), "median": q(.5), "p90": q(.9), "max": s[-1]}


def verify_manifest(records_root: Path) -> tuple[dict, dict[str, dict], str]:
    raw = (records_root / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema") != "v20_public_generation_export_manifest_v1":
        raise ValueError("wrong published manifest schema")
    inventory = manifest["record_inventory"]
    if len(inventory) != 800 or manifest["records"] != 800 or manifest["first_records"] != 400 or manifest["warm_records"] != 400:
        raise ValueError("published inventory/count mismatch")
    if manifest["by_stage"] != {"initial": 84, "remainder": 616, "historical": 100}:
        raise ValueError("stage inventory mismatch")
    archive_catalog = {(a["stage"], a["host"], a["sha256"]) for a in manifest["source_archives"]}
    if len(archive_catalog) != 6:
        raise ValueError("six source archive identities required")
    seen, rows = set(), {}
    for item in inventory:
        path = records_root / item["relative_path"]
        if not path.resolve().is_relative_to(records_root.resolve()):
            raise ValueError("record path escapes published directory")
        data = path.read_bytes()
        if digest(data) != item["sha256"]:
            raise ValueError(f"record SHA drift: {item['relative_path']}")
        row = json.loads(data)
        identity = row["identity"]
        if row.get("schema") != "v20_public_generation_record_v1" or any(
            identity.get(k) != item[k] for k in ("execution_key", "arm", "dataset", "host", "role", "seed", "stage")
        ):
            raise ValueError("record identity drift")
        if (row["provenance"].get("original_receipt_sha256") != item["original_receipt_sha256"] or
            row["provenance"].get("source_archive_sha256") != item["source_archive_sha256"]):
            raise ValueError("source archive/receipt identity drift")
        if (item["stage"], item["host"], item["source_archive_sha256"]) not in archive_catalog:
            raise ValueError("record references an unlisted source archive")
        tokens = row["generated"]["completion_tokens"]
        if digest(json.dumps(tokens, separators=(",", ":")).encode()) != row["provenance"]["completion_token_sha256"]:
            raise ValueError("completion token hash drift")
        key = identity["execution_key"]
        if key in seen:
            raise ValueError("duplicate execution key")
        seen.add(key)
        rows[key] = row
    if Counter(r["identity"]["role"] for r in rows.values()) != {"attempt0": 400, "warm": 400}:
        raise ValueError("first/warm role count mismatch")
    if Counter(r["identity"]["stage"] for r in rows.values()) != manifest["by_stage"]:
        raise ValueError("stage count drift")
    by_dataset_arm = {f"{dataset}/{arm}": n for (dataset, arm), n in
                      Counter((r["identity"]["dataset"], r["identity"]["arm"]) for r in rows.values()).items()}
    if by_dataset_arm != manifest["by_dataset_arm"]:
        raise ValueError("dataset/arm inventory drift")
    return manifest, rows, digest(raw)


def build_cells(records: dict[str, dict]) -> dict[tuple, dict]:
    cells = defaultdict(dict)
    for row in records.values():
        i = row["identity"]
        cells[(i["dataset"], i["id"], i["seed"], i["arm"])][i["role"]] = row
    if len(cells) != 400:
        raise ValueError("expected 400 first/warm cells")
    for key, pair in cells.items():
        if set(pair) != {"attempt0", "warm"}:
            raise ValueError(f"incomplete first/warm cell {key}")
        first, warm = pair["attempt0"], pair["warm"]
        fi, wi = first["identity"], warm["identity"]
        if fi["cell_id"] != wi["cell_id"] or fi["host"] != wi["host"] or fi["gpu_uuid"] != wi["gpu_uuid"]:
            raise ValueError("first/warm GPU/cell identity mismatch")
        if warm.get("acceptance") != {"accepted": True, "reasons": []}:
            raise ValueError("warm not accepted")
        for field in ("completion_tokens", "output_tokens", "termination_reason", "total_decoder_calls", "per_canvas"):
            if first["generated"][field] != warm["generated"][field]:
                raise ValueError(f"warm {field} differs")
        for field in ("phase_evidence", "router_phase_evidence"):
            a = {k: v for k, v in first[field].items() if k != "prefill_end_to_finish_gpu_s"}
            b = {k: v for k, v in warm[field].items() if k != "prefill_end_to_finish_gpu_s"}
            if a != b:
                raise ValueError(f"warm {field} differs")
    return dict(cells)


def select_text_cells(cells: dict[tuple, dict]) -> list[dict]:
    """Freeze at most four LB and two AIME q/seed selections from counts only."""
    selected = []
    for dataset, limit in (("longbench_v2", 4), ("aime26", 2)):
        candidates = []
        questions = sorted({(d, q, s) for d, q, s, _ in cells if d == dataset})
        for d, q, s in questions:
            get = lambda arm: cells[(d, q, s, arm)]["attempt0"]["generated"]["total_decoder_calls"]
            native = get("D_native")
            m3_delta = get("M3_R2_A8_current_output") - native
            matched_delta = get("D_matched") - native
            candidates.append({"dataset": d, "id": q, "seed": s, "m3_r2_minus_native_calls": m3_delta,
                               "d_matched_minus_native_calls": matched_delta})
        orders = [
            ("largest_positive_M3R2_call_delta", sorted(candidates, key=lambda x: (-x["m3_r2_minus_native_calls"], x["id"], x["seed"]))),
            ("largest_negative_M3R2_call_delta", sorted(candidates, key=lambda x: (x["m3_r2_minus_native_calls"], x["id"], x["seed"]))),
        ]
        if limit == 4:
            orders.append(("largest_Dmatched_discordance", sorted(candidates, key=lambda x: (-abs(x["d_matched_minus_native_calls"]), x["id"], x["seed"]))))
            orders.append(("second_Dmatched_discordance", orders[-1][1][1:]))
        taken = set()
        for reason, order in orders:
            for x in order:
                identity = (x["id"], x["seed"])
                if identity not in taken:
                    selected.append({**x, "selection_reason": reason})
                    taken.add(identity)
                    break
    return selected


def stop_category(canvas: dict) -> str:
    a, b = bool(canvas["native_stop_final_call"]), bool(canvas["iteration_cap_final_call"])
    return "both" if a and b else "native_stop_only" if a else "cap_only" if b else "neither"


def repetition(tokens: list[int], width: int = 16) -> float | None:
    if len(tokens) < width:
        return None
    seen, repeated = set(), 0
    for j in range(len(tokens) - width + 1):
        chunk = tuple(tokens[j:j + width])
        repeated += chunk in seen
        seen.add(chunk)
    return repeated / (len(tokens) - width + 1)


def similarity(tokens: list[int], reference: list[int], chunk_size: int = 64) -> dict:
    prefix = 0
    for a, b in zip(tokens, reference):
        if a != b: break
        prefix += 1
    n = min(len(tokens), len(reference)) // chunk_size
    equal = sum(tokens[j * chunk_size:(j + 1) * chunk_size] ==
                reference[j * chunk_size:(j + 1) * chunk_size] for j in range(n))
    return {"common_prefix_tokens": prefix, "aligned_64_token_chunks_equal": equal,
            "aligned_64_token_chunks_compared": n}


def load_quality(path: Path | None, cells: dict, protocol_id: str) -> tuple[dict, str | None, str | None]:
    if path is None:
        return {}, None, None
    raw = path.read_bytes()
    obj = json.loads(raw)
    if obj.get("schema") != "v21_v20_qualified_first_scores_v1" or obj.get("first_cells") != 400:
        raise ValueError("wrong qualified score export")
    public_protocol = ROOT / "results/fan_m1_m3_multidataset_20260927/frozen_protocol.json"
    if obj.get("protocol_id") != protocol_id or obj.get("protocol_sha256") != digest(public_protocol.read_bytes()):
        raise ValueError("qualified score protocol identity drift")
    expected_ids = {v["attempt0"]["identity"]["cell_id"]: k for k, v in cells.items()}
    scores = {}
    for row in obj["cells"]:
        cid = row["cell_id"]
        if cid in scores or cid not in expected_ids or tuple(row[k] for k in ("dataset", "id", "seed", "arm")) != expected_ids[cid]:
            raise ValueError("quality cell identity mismatch")
        scores[cid] = row
    if set(scores) != set(expected_ids):
        raise ValueError("qualified score export incomplete")
    return scores, digest(raw), obj["protocol_sha256"]


def audit(records_root: Path, quality_path: Path | None = None) -> dict:
    manifest, records, manifest_sha = verify_manifest(records_root)
    protocol_bytes = (ROOT / "results/fan_m1_m3_multidataset_20260927/frozen_protocol.json").read_bytes()
    lf_sha = digest(protocol_bytes)
    crlf_sha = digest(protocol_bytes.replace(b"\n", b"\r\n"))
    if b"\r\n" in protocol_bytes or manifest["protocol_sha256"] != crlf_sha:
        raise ValueError("published manifest protocol differs beyond documented LF/CRLF expansion")
    cells = build_cells(records)
    scores, score_sha, score_protocol_sha = load_quality(quality_path, cells, manifest["protocol_id"])
    # Selection accesses only metadata, never raw_completion, prediction or tokens.
    selection = select_text_cells(cells)
    selected_ids = {(x["dataset"], x["id"], x["seed"]) for x in selection}
    groups = defaultdict(list)
    output_rows = []
    for (dataset, qid, seed, arm), pair in sorted(cells.items()):
        first, warm = pair["attempt0"], pair["warm"]
        i, g = first["identity"], first["generated"]
        canvases = g["per_canvas"]
        calls = [c["decoder_calls"] for c in canvases]
        if sum(calls) != g["total_decoder_calls"]:
            raise ValueError("canvas call conservation failed")
        categories = Counter(stop_category(c) for c in canvases)
        router = first.get("router_phase_evidence") or {}
        if all(k in router for k in ("A", "D", "H")) and sum(router[k] for k in ("A", "D", "H")) != router["attention_calls"]:
            raise ValueError("router A/D/H conservation failed")
        ref = cells[(dataset, qid, seed, "D_native")]
        ri = ref["attempt0"]["identity"]
        if (i["host"], i["gpu_uuid"]) != (ri["host"], ri["gpu_uuid"]):
            raise ValueError("same-question native comparator crossed GPU")
        sim = similarity(g["completion_tokens"], ref["attempt0"]["generated"]["completion_tokens"])
        q = scores.get(i["cell_id"])
        row = {"dataset": dataset, "id": qid, "seed": seed, "arm": arm, "stage": i["stage"],
               "host": i["host"], "gpu_uuid": i["gpu_uuid"], "cell_id": i["cell_id"],
               "first_score": q["score"] if q else None,
               "strict_correct": q["strict_correct"] if q else None,
               "task_correct": q["task_correct"] if q else None,
               "task_at_cap": bool(q["task_correct"] and q["capped"]) if q else None,
               "parsed": q["parsed"] if q else None, "eos": q["eos"] if q else None,
               "capped": q["capped"] if q else None,
               "termination": g["termination_reason"], "output_tokens": g["output_tokens"],
               "canvases": len(canvases), "total_decoder_calls": g["total_decoder_calls"],
               "mean_calls_per_canvas": g["total_decoder_calls"] / len(canvases),
               "calls_per_canvas": quantiles(calls),
               "native_stop_only": categories["native_stop_only"], "cap_only": categories["cap_only"],
               "both": categories["both"], "neither": categories["neither"],
               "router_A": router.get("A"), "router_D": router.get("D"), "router_H": router.get("H"),
               "decision_interval": (first.get("router_metrics") or {}).get("decision_interval"),
               "decision_age_distribution": None,
               "warm_request_wall_s": warm["timing"]["request_wall_seconds"],
               "warm_gpu_timeline_s": warm["timing"]["generation_gpu_timeline_seconds"],
               **sim,
               "repeated_16_token_window_fraction": repetition(g["completion_tokens"]) if
                   (dataset, qid, seed) in selected_ids else None}
        output_rows.append(row)
        groups[(dataset, i["host"], arm)].append(row)
    by_group = []
    for (dataset, host, arm), rows in sorted(groups.items()):
        qkeys = ("output_tokens", "canvases", "total_decoder_calls", "warm_request_wall_s", "warm_gpu_timeline_s")
        by_group.append({"dataset": dataset, "host": host, "arm": arm, "cells": len(rows),
                         **{k: quantiles([r[k] for r in rows if r[k] is not None]) for k in qkeys},
                         "exact_within_host_totals": {
                             "output_tokens": sum(r["output_tokens"] for r in rows),
                             "canvases": sum(r["canvases"] for r in rows),
                             "decoder_calls": sum(r["total_decoder_calls"] for r in rows),
                             "pooled_calls_per_canvas": sum(r["total_decoder_calls"] for r in rows) /
                                                         sum(r["canvases"] for r in rows),
                             "accepted_warm_request_wall_s": sum(r["warm_request_wall_s"] for r in rows),
                             "warm_wall_s_per_first_decoder_call_amortized":
                                 sum(r["warm_request_wall_s"] for r in rows) / sum(r["total_decoder_calls"] for r in rows)},
                         "calls_per_canvas_pooled": quantiles([c["decoder_calls"] for (d, _, _, a), pair in cells.items()
                                                               if d == dataset and a == arm and pair["attempt0"]["identity"]["host"] == host
                                                               for c in pair["attempt0"]["generated"]["per_canvas"]]),
                         "stop_categories": {k: sum(r[k] for r in rows) for k in ("native_stop_only", "cap_only", "both", "neither")},
                         "router_phase_totals": {k: sum(r[k] for r in rows if r[k] is not None)
                                                 for k in ("router_A", "router_D", "router_H")},
                         "score_mean": statistics.mean(r["first_score"] for r in rows) if scores else None,
                         "strict_correct": sum(bool(r["strict_correct"]) for r in rows) if scores else None,
                         "task_correct_at_cap": sum(bool(r["task_correct"] and r["capped"]) for r in rows) if scores else None})
    ratios = []
    index = {(r["dataset"], r["id"], r["seed"], r["arm"]): r for r in output_rows}
    for row in output_rows:
        for refarm in REFERENCE_ARMS:
            if row["arm"] == refarm: continue
            ref = index[(row["dataset"], row["id"], row["seed"], refarm)]
            if (row["host"], row["gpu_uuid"]) != (ref["host"], ref["gpu_uuid"]):
                raise ValueError("paired reference crossed GPU")
            ratios.append({"dataset": row["dataset"], "id": row["id"], "seed": row["seed"], "host": row["host"],
                           "arm": row["arm"], "reference": refarm,
                           "warm_wall_ratio": row["warm_request_wall_s"] / ref["warm_request_wall_s"],
                           "call_ratio": row["total_decoder_calls"] / ref["total_decoder_calls"],
                           "warm_wall_delta_s": row["warm_request_wall_s"] - ref["warm_request_wall_s"],
                           "first_call_delta": row["total_decoder_calls"] - ref["total_decoder_calls"]})
    ratio_groups = defaultdict(list)
    for row in ratios:
        ratio_groups[(row["dataset"], row["host"], row["arm"], row["reference"])].append(row)
    paired_group_ratios = [
        {"dataset": d, "host": h, "arm": a, "reference": ref, "paired_cells": len(rows),
         "warm_wall_ratio_of_totals": (sum(r["warm_wall_ratio"] * index[(r["dataset"], r["id"], r["seed"], ref)]["warm_request_wall_s"] for r in rows) /
                                      sum(index[(r["dataset"], r["id"], r["seed"], ref)]["warm_request_wall_s"] for r in rows)),
         "warm_wall_geometric_ratio": math.exp(statistics.mean(math.log(r["warm_wall_ratio"]) for r in rows)),
         "calls_ratio_of_totals": (sum(r["call_ratio"] * index[(r["dataset"], r["id"], r["seed"], ref)]["total_decoder_calls"] for r in rows) /
                                   sum(index[(r["dataset"], r["id"], r["seed"], ref)]["total_decoder_calls"] for r in rows))}
        for (d, h, a, ref), rows in sorted(ratio_groups.items())]
    selected_text = []
    for s in selection:
        d, q, seed = s["dataset"], s["id"], s["seed"]
        arms = []
        for arm in ARMS:
            rec = cells[(d, q, seed, arm)]["attempt0"]
            text = rec["generated"]["raw_completion"]
            arms.append({"arm": arm, "cell_id": rec["identity"]["cell_id"],
                         "repeated_16_token_window_fraction": repetition(rec["generated"]["completion_tokens"]),
                         "final_channel_delimiter_present": "<channel|>" in text,
                         "termination": rec["generated"]["termination_reason"],
                         "iteration_cap_canvases": sum(c["iteration_cap_final_call"] for c in rec["generated"]["per_canvas"])})
        selected_text.append({**s, "arms": arms})
    return {"schema": "v21_trajectory_audit_v1", "manifest_sha256": manifest_sha,
            "protocol_id": manifest["protocol_id"], "protocol_sha256": manifest["protocol_sha256"],
            "qualified_score_export_sha256": score_sha, "qualified_score_status": "complete" if scores else "pending",
            "qualified_scorer_protocol_sha256": score_protocol_sha,
            "manifest_vs_scorer_protocol_byte_sha_equal": manifest["protocol_sha256"] == score_protocol_sha,
            "protocol_line_ending_identity": {"deployed_git_lf_sha256": lf_sha,
                                              "published_export_crlf_sha256": crlf_sha,
                                              "crlf_expansion_only": True},
            "first_records": 400, "warm_records": 400, "source_archives": manifest["source_archives"],
            "selection_rule": "Before text access: for LB select largest positive/negative M3R2-minus-native call delta and two largest absolute Dmatched-minus-native discordances without duplicate q/seed; for AIME select positive/negative M3R2 extremes, tie by id then seed.",
            "selected_text_cells": selected_text, "cells": output_rows, "groups": by_group,
            "paired_ratios": ratios, "paired_group_ratios": paired_group_ratios,
            "limitations": ["No full per-step QKV/attention tensors or per-decision age distribution were recorded.",
                            "The export manifest protocol SHA is exactly the CRLF expansion of the deployed/Git LF protocol bytes; parsed JSON and all cell identities are unchanged. The qualified scorer binds the LF bytes.",
                            "Token-prefix and aligned-chunk identity is descriptive; after divergence positional tokens need not represent the same semantic step.",
                            "The device timeline is first encoder-forward end to final CUDA event, including host gaps and later work; it is not synchronized decode wall time.",
                            "Historical G75L30 follows core execution and may have temporal drift."]}


def write_outputs(result: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "trajectory_audit.json").write_bytes((json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    fields = ["dataset", "id", "seed", "arm", "stage", "host", "gpu_uuid", "cell_id", "first_score", "strict_correct",
              "task_correct", "task_at_cap", "parsed", "eos", "capped", "termination", "output_tokens", "canvases", "total_decoder_calls", "mean_calls_per_canvas",
              "native_stop_only", "cap_only", "both", "neither", "router_A", "router_D", "router_H", "decision_interval",
              "warm_request_wall_s", "warm_gpu_timeline_s", "common_prefix_tokens", "aligned_64_token_chunks_equal",
              "aligned_64_token_chunks_compared", "repeated_16_token_window_fraction",
              "warm_wall_ratio_vs_D_native", "calls_ratio_vs_D_native",
              "warm_wall_ratio_vs_T_scope", "calls_ratio_vs_T_scope",
              "warm_wall_ratio_vs_B_A8_matched", "calls_ratio_vs_B_A8_matched"]
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    ratio_index = {(r["dataset"], r["id"], r["seed"], r["arm"], r["reference"]): r
                   for r in result["paired_ratios"]}
    for row in result["cells"]:
        flat = {k: row[k] for k in fields if k in row}
        for ref in REFERENCE_ARMS:
            comparison = ratio_index.get((row["dataset"], row["id"], row["seed"], row["arm"], ref))
            if row["arm"] == ref:
                flat[f"warm_wall_ratio_vs_{ref}"] = 1.0
                flat[f"calls_ratio_vs_{ref}"] = 1.0
            elif comparison:
                flat[f"warm_wall_ratio_vs_{ref}"] = comparison["warm_wall_ratio"]
                flat[f"calls_ratio_vs_{ref}"] = comparison["call_ratio"]
        writer.writerow(flat)
    (out / "trajectory_audit.csv").write_bytes(stream.getvalue().encode())
    lines = ["# v21 trajectory audit", "", f"Verified 800 published requests (400 first, 400 accepted warm), manifest SHA-256 `{result['manifest_sha256']}`.",
             f"Qualified first-cell score status: **{result['qualified_score_status']}**. The score source is the frozen v20 `score_firsts` export; no new scoring rule is used.",
             f"The export manifest binds CRLF protocol bytes `{result['protocol_sha256']}`; the scorer binds deployed/Git LF bytes `{result['protocol_line_ending_identity']['deployed_git_lf_sha256']}`. Replacing each LF with CRLF gives the manifest hash exactly, with the same parsed JSON.",
             "", "The CSV has one row per first cell, with its accepted warm timing. Stop categories are disjoint per canvas. Calls equal the sum of canvas calls; `calls_per_canvas` distributions and pooled group distributions are in JSON.",
             "", "| Dataset | Host | Arm | Cells | Calls mean | Canvases mean | Warm wall mean (s) | Strict correct | Cap-only canvases | Native-stop-only canvases |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in result["groups"]:
        lines.append(f"| {g['dataset']} | {g['host']} | {g['arm']} | {g['cells']} | {g['total_decoder_calls']['mean']:.1f} | {g['canvases']['mean']:.1f} | {g['warm_request_wall_s']['mean']:.2f} | {g['strict_correct'] if g['strict_correct'] is not None else 'N/A'} | {g['stop_categories']['cap_only']} | {g['stop_categories']['native_stop_only']} |")
    lines += ["", "Same-GPU cell ratios to D_native, T_scope, and B_A8_matched are in JSON and CSV; a ratio below 1 means less warm request wall time or fewer first decoder calls. Absolute times are shown separately by host.",
              "", "| Dataset | Host | Arm / reference | Paired cells | Warm wall ratio of totals | First call ratio of totals |",
              "|---|---|---|---:|---:|---:|"]
    for r in result["paired_group_ratios"]:
        if r["arm"] in ("M3_R2_A8_current_output", "M3_R3_A8_current_output", "D_matched"):
            lines.append(f"| {r['dataset']} | {r['host']} | {r['arm']} / {r['reference']} | {r['paired_cells']} | {r['warm_wall_ratio_of_totals']:.3f} | {r['calls_ratio_of_totals']:.3f} |")
    lines += [
              "", "## Predeclared text-inspection cells", "", result["selection_rule"], ""]
    for s in result["selected_text_cells"]:
        lines.append(f"- {s['dataset']} `{s['id']}` seed {s['seed']}: {s['selection_reason']}; M3R2−native {s['m3_r2_minus_native_calls']:+d} calls, Dmatched−native {s['d_matched_minus_native_calls']:+d} calls.")
    lines += ["", "| Selected arm | First completions | Mean repeated 16-token window fraction | No final-channel delimiter | Length-capped | No delimiter and capped |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        selected = [a for s in result["selected_text_cells"] for a in s["arms"] if a["arm"] == arm]
        repeats = [a["repeated_16_token_window_fraction"] for a in selected
                   if a["repeated_16_token_window_fraction"] is not None]
        lines.append(f"| {arm} | {len(selected)} | {statistics.mean(repeats) if repeats else 0:.3f} | "
                     f"{sum(not a['final_channel_delimiter_present'] for a in selected)} | "
                     f"{sum(a['termination'] == 'length' for a in selected)} | "
                     f"{sum(not a['final_channel_delimiter_present'] and a['termination'] == 'length' for a in selected)} |")
    lines += ["", "Repetition is the fraction of 16-token sliding windows previously seen within that same first completion; `final_channel_delimiter_present` only checks the literal `<channel|>` delimiter used by the existing AIME parser and is not a new correctness rule. Selected-cell numbers are in JSON.",
              "", "## Limits", ""] + [f"- {x}" for x in result["limitations"]]
    (out / "trajectory_audit.md").write_bytes(("\n".join(lines) + "\n").encode())
    addendum = ["# v20 evidence addendum for v21 CP0", "", "The published generation-record manifest and every record SHA-256 were checked before this CPU audit. There are 400 first and 400 accepted warm receipts across the three frozen datasets; the separately timed historical arm contributes 100 executions.",
                "", "Qualified task correctness and strict EOS come from unchanged v20 offline `score_firsts` in the pinned CP5 environment; the allowlisted 400-cell output is copied as `qualified_first_scores.json`. No text-derived score substitutes for it.",
                "", f"Source score-export SHA-256: `{result['qualified_score_export_sha256']}`. Export manifest protocol SHA is the exact CRLF expansion of the deployed/Git LF protocol bytes; this is a byte-line-ending difference, not a changed parsed protocol.",
                "", "Request records preserve completion tokens, per-canvas native stopping/cap flags, schedule-step arrays, calls, A/D/H totals when available, and request/device timing. They do not preserve per-step attention tensors, decision-age distributions, or a synchronized prefill-excluded generation wall time. Historical G75L30_nativeQ128 is a native-Q128 transplant rather than the exact older vLLM grouping."]
    (out / "v20_evidence_addendum.md").write_bytes(("\n".join(addendum) + "\n").encode())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    p.add_argument("--quality-json", type=Path)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    result = audit(args.records, args.quality_json)
    write_outputs(result, args.out)
    print(json.dumps({"cells": len(result["cells"]), "selected": len(result["selected_text_cells"]),
                      "quality": result["qualified_score_status"], "manifest_sha256": result["manifest_sha256"]}))


if __name__ == "__main__":
    main()
