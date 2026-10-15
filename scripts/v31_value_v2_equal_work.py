"""Resumable V2 threshold sweep and equal-physical-work audit.

This campaign deliberately treats physical GLOBAL kept tiles as the matching
quantity.  It uses the existing native sampler and selector-only adapter, and
publishes only sanitized aggregates.  The runner writes raw completions under
private attempt directories, as in the existing LongBench harness.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results/v31_value_v2_equal_work_20261011"
PYTHON = sys.executable
MODEL = ""  # Supplied explicitly; private host/cache locations are not portable.
POOL = ""
OLD_STUDY = ROOT / "results/v31_value_selectors_20261008"
DEV_SOURCE = OLD_STUDY / "private/cells_development_original_s1.json"
TARGET_SOURCE = OLD_STUDY / "private/cells_target_full503.json"
THRESHOLDS = [0.001, 0.002, 0.005, 0.009, 0.015, 0.021, 0.027, 0.033, 0.045, 0.0675, 0.108]
DEV_SHARDS = 4
TARGET_SHARDS = 4
EARLY_PREFERRED = 20
EARLY_MIN_ITEMS = 8

SOURCE_FILES = [
    "experiments/numerical_qk_reuse/vllm_adapter.py",
    "experiments/numerical_qk_reuse/v31_value_selectors.py",
    "experiments/numerical_qk_reuse/v31_value_kernels.py",
    "experiments/numerical_qk_reuse/v31_value_summary.py",
    "experiments/numerical_qk_reuse/v31_value_workspace.py",
    "experiments/numerical_qk_reuse/v31_value_projection.py",
    "experiments/numerical_qk_reuse/v31_value_snapshots.py",
    "experiments/numerical_qk_reuse/v31_value_native_audit.py",
    "experiments/numerical_qk_reuse/v27_fa4.py",
    "experiments/numerical_qk_reuse/v31_fa4_observe.py",
    "experiments/numerical_qk_reuse/v29_paged_copy.py",
    "experiments/numerical_qk_reuse/v29_lse_merge.py",
    "experiments/numerical_qk_reuse/v31_logit_stats.py",
    "experiments/diffusion_gemma_jl_output_aware/projections.py",
    "experiments/value_direction_hopper/projection.py",
    "scripts/v31_vllm_paired_bench.py",
    "scripts/v31_value_campaign.py",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def threshold_name(t: float) -> str:
    return "threshold_" + format(t, ".10g").replace(".", "p")


def key(record):
    return f"{record['dataset']}|{record['index']}|{record['panel_seed']}|{record['repeat']}"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCE_FILES}


def verify_source_freeze(protocol):
    now = source_hashes()
    if now != protocol["source_sha256"]:
        changed = sorted(k for k in set(now) | set(protocol["source_sha256"]) if now.get(k) != protocol["source_sha256"].get(k))
        raise RuntimeError(f"execution source changed after freeze: {changed}")


def wait_for_gpu_idle(timeout_s=180):
    """Avoid the runner's short post-CUDA teardown occupancy race between shards."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                text=True, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            time.sleep(2)
            continue
        if not out.strip():
            return
        time.sleep(2)
    raise RuntimeError("H100 did not become idle before the next resumable shard")


def setup():
    STUDY.mkdir(parents=True, exist_ok=True)
    for d in ("sweep", "matched_calibration", "official_audit", "receipts", "private"):
        (STUDY / d).mkdir(parents=True, exist_ok=True)
    for source, dest in ((DEV_SOURCE, STUDY / "private/cells_development_original_s1.json"),
                         (TARGET_SOURCE, STUDY / "private/cells_target_full503.json")):
        if dest.exists() and dest.read_bytes() != source.read_bytes():
            raise RuntimeError(f"immutable schedule changed: {dest}")
        if not dest.exists():
            shutil.copyfile(source, dest)
    model_inventory = OLD_STUDY / "model_source_fingerprints.json"
    model_dest = STUDY / "model_source_fingerprints.json"
    if model_dest.exists() and model_dest.read_bytes() != model_inventory.read_bytes():
        raise RuntimeError("model source inventory changed")
    if not model_dest.exists():
        shutil.copyfile(model_inventory, model_dest)
    dev = load_json(STUDY / "private/cells_development_original_s1.json")
    target = load_json(STUDY / "private/cells_target_full503.json")
    manifest_names = sorted({c["dataset"] for c in dev + target})
    manifest_hashes = {name: sha(Path(POOL) / f"{name}_generation_manifest.json") for name in manifest_names}
    protocol = dict(
        schema="v31_value_v2_equal_work_protocol_v1",
        study="v31_value_v2_equal_work_20261011",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256=source_hashes(),
        development_cells_sha256=sha(STUDY / "private/cells_development_original_s1.json"),
        target_cells_sha256=sha(STUDY / "private/cells_target_full503.json"),
        manifest_sha256=manifest_hashes,
        model_revision="f7f5b7f5fa82ffc52addd066915886d497f5517b",
        runtime=dict(python=PYTHON, model=MODEL, pool=POOL, vllm="0.30.0",
                     torch="2.13.0+cu130", gpu="NVIDIA H100 80GB HBM3",
                     fix_51994=True, fa4_local_fix=True, global_layers=[5, 11, 17, 23, 29],
                     local_scope="native_dense", q_tile=128, kv_tile=64, mage_k=8192,
                     select_step=1, carry_first=True, settle_trigger=0.15,
                     trigger_signal="settle", sticky=1.386, projection_rank=32, projection_seed=1729),
        selector="value_v2_online_preserve_mass", control="current_v31_control",
        thresholds=THRESHOLDS, exact_greedy="excluded",
        matching_quantity="absolute physical GLOBAL kept tiles",
        development_schedule="original S1 64 cells, seeds 1 and 2",
        target_schedule="official 503 cells, seed 1",
        timing="paused; audit wall time is descriptive only",
    )
    p = STUDY / "protocol.json"
    if p.exists() and load_json(p) != protocol:
        raise RuntimeError("frozen equal-work protocol already differs")
    if not p.exists():
        p.write_text(json.dumps(protocol, indent=2) + "\n")
    (STUDY / "receipts/source_fingerprints.json").write_text(json.dumps(dict(
        status="frozen", source_sha256=protocol["source_sha256"], source_commit=protocol["source_commit"],
        model_source_inventory_sha256=sha(model_dest), protocol_sha256=sha(p)), indent=2) + "\n")
    verify_source_freeze(protocol)
    readme = STUDY / "README.md"
    if not readme.exists():
        readme.write_text("""# V2 equal physical GLOBAL work campaign

This campaign sweeps the unchanged `value_v2_online_preserve_mass` threshold on the original 64-cell development schedule, then runs an official 503-cell audit at threshold(s) selected by absolute physical GLOBAL kept tiles. Timing is paused; audit instrumentation is not a clean latency result.

The source and schedule identity are frozen in `protocol.json`. Raw completions remain under private attempt directories and are excluded from published summaries.
""")
    return protocol


def make_shards(source: Path, root: Path, count: int):
    cells = load_json(source)
    if count <= 0:
        raise ValueError(count)
    paths = []
    for i in range(count):
        start = len(cells) * i // count
        end = len(cells) * (i + 1) // count
        path = root / f"shard{i + 1:03d}.json"
        content = json.dumps(cells[start:end], indent=1) + "\n"
        if path.exists() and path.read_text() != content:
            raise RuntimeError(f"immutable shard changed: {path}")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        paths.append(path)
    return paths


def complete_attempts(stage_root: Path, arm: str, shards, purpose: str, threshold):
    out = []
    for shard in shards:
        base = stage_root / arm / shard.stem
        attempts = sorted(base.glob("attempt[0-9][0-9][0-9]"))
        found = None
        for attempt in attempts:
            marker = attempt / "attempt_complete.json"
            if not marker.exists():
                continue
            config = load_json(attempt / "config.json")
            receipt = load_json(attempt / "receipt.json")
            if config.get("arm") == arm and config.get("purpose") == purpose and config.get("threshold") == threshold:
                if receipt.get("status") != "complete" or receipt.get("validation_errors"):
                    raise RuntimeError(f"invalid completed attempt: {attempt}")
                found = attempt
                break
        if found is None:
            return None
        out.append(found)
    return out


def run_arm(stage_root: Path, arm: str, shards, purpose: str, threshold, protocol):
    verify_source_freeze(protocol)
    existing = complete_attempts(stage_root, arm, shards, purpose, threshold)
    if existing is not None:
        return existing
    for shard in shards:
        base = stage_root / arm / shard.stem
        base.mkdir(parents=True, exist_ok=True)
        attempts = sorted(base.glob("attempt[0-9][0-9][0-9]"))
        done = None
        for attempt in attempts:
            marker = attempt / "attempt_complete.json"
            if marker.exists():
                config = load_json(attempt / "config.json")
                receipt = load_json(attempt / "receipt.json")
                if config.get("arm") == arm and config.get("purpose") == purpose and config.get("threshold") == threshold:
                    if receipt.get("status") == "complete" and not receipt.get("validation_errors"):
                        done = attempt
                        break
        if done is not None:
            continue
        ordinal = max((int(p.name[-3:]) for p in attempts), default=0) + 1
        attempt = base / f"attempt{ordinal:03d}"
        cmd = [PYTHON, str(ROOT / "scripts/v31_value_campaign.py"), "--python", PYTHON,
               "--model", MODEL, "--manifests", POOL, "--cells", str(shard),
               "--attempt", str(attempt), "--arm", arm, "--purpose", purpose,
               "--result-root", str(STUDY)]
        if threshold is not None:
            cmd += ["--threshold", str(threshold)]
        wait_for_gpu_idle()
        launch_log = base / "latest_launch.log"
        print(json.dumps(dict(event="launch", arm=arm, purpose=purpose, threshold=threshold,
                              shard=shard.stem, attempt=str(attempt))), flush=True)
        with launch_log.open("a") as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        verify_source_freeze(protocol)
    final = complete_attempts(stage_root, arm, shards, purpose, threshold)
    if final is None:
        raise RuntimeError(f"arm did not complete: {arm} {purpose} {threshold}")
    return final


def records_from_attempts(attempts):
    rows = []
    for attempt in attempts:
        marker = load_json(attempt / "attempt_complete.json")
        if marker.get("status") != "complete":
            raise RuntimeError(f"incomplete attempt {attempt}")
        rows.extend(load_json_line(attempt / "records.jsonl"))
    keys = [key(r) for r in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate cell in completed shards")
    return rows


def load_json_line(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def flatten_attempt_records(attempts):
    rows = []
    for attempt in attempts:
        marker = load_json(attempt / "attempt_complete.json")
        if marker.get("status") != "complete":
            raise RuntimeError(f"incomplete attempt {attempt}")
        rows.extend(json.loads(line) for line in (attempt / "records.jsonl").read_text().splitlines() if line.strip())
    if len({key(r) for r in rows}) != len(rows):
        raise RuntimeError("duplicate cell in completed shards")
    return rows


def quantiles(values):
    if not values:
        return dict(n=0, mean=None, median=None, p10=None, p25=None, p75=None, p90=None, min=None, max=None)
    a = np.asarray(values, dtype=float)
    return dict(n=int(a.size), mean=float(a.mean()), median=float(np.median(a)),
                p10=float(np.quantile(a, .10)), p25=float(np.quantile(a, .25)),
                p75=float(np.quantile(a, .75)), p90=float(np.quantile(a, .90)),
                min=float(a.min()), max=float(a.max()))


def aggregate(rows):
    if not rows:
        raise ValueError("empty records")
    adapters = [r.get("receipts", {}).get("adapter", {}) for r in rows]
    for r, a in zip(rows, adapters):
        if r.get("cuda_graph_captures", 0) != 0:
            raise RuntimeError("unexpected CUDA graph capture")
        if a.get("value_nonfinite_attention_calls", 0):
            raise RuntimeError("nonfinite attention output")
        if a.get("value_local_router_installed", False):
            raise RuntimeError("LOCAL value router installed")
        phases = a.get("value_phase_tiles")
        canvas = a.get("value_canvas_phase_tiles")
        if phases is None or canvas is None:
            raise RuntimeError("missing phase/canvas physical accounting")
        if a.get("value_selector") and a.get("value_invalid_rows", 0):
            raise RuntimeError("invalid selector rows")
        if a.get("value_selector") and a.get("value_output_path") != "unchanged_fa4":
            raise RuntimeError("selector output path changed")
        if sum(p.get("eligible_tiles", 0) for p in phases.values()) != a.get("global_eligible_tiles"):
            raise RuntimeError("phase eligible mismatch")
        if sum(p.get("kept_tiles", 0) for p in phases.values()) != a.get("global_kept_tiles"):
            raise RuntimeError("phase kept mismatch")
    n = sum(r["denoise_forwards"] for r in rows)
    c = sum(r["canvases"] for r in rows)
    t = sum(r["output_tokens"] for r in rows)
    eligible = sum(a.get("global_eligible_tiles", 0) for a in adapters)
    kept = sum(a.get("global_kept_tiles", 0) for a in adapters)
    phases_out = {}
    for phase in ("dense", "initial", "refresh", "held", "carried"):
        vals = [a.get("value_phase_tiles", {}).get(phase, {}) for a in adapters]
        e = sum(v.get("eligible_tiles", 0) for v in vals)
        k = sum(v.get("kept_tiles", 0) for v in vals)
        phases_out[phase] = dict(calls=sum(v.get("calls", 0) for v in vals), eligible_tiles=e,
                                 kept_tiles=k, skipped_tiles=e-k, sparsity=(1-k/e if e else 0.0))
    canvas_by_item = {}
    for r, a in zip(rows, adapters):
        cmap = {}
        for cidx, byphase in a.get("value_canvas_phase_tiles", {}).items():
            e = sum(v.get("eligible_tiles", 0) for v in byphase.values())
            k = sum(v.get("kept_tiles", 0) for v in byphase.values())
            cmap[int(cidx)] = dict(eligible_tiles=e, kept_tiles=k, skipped_tiles=e-k)
        canvas_by_item[key(r)] = cmap
    max_canvas = max(r["canvases"] for r in rows)
    cumulative = {}
    for cidx in range(1, max_canvas + 1):
        vals = []
        for r in rows:
            if r["canvases"] < cidx:
                continue
            cmap = canvas_by_item[key(r)]
            vals.append((sum(cmap.get(i, {}).get("eligible_tiles", 0) for i in range(cidx)),
                         sum(cmap.get(i, {}).get("kept_tiles", 0) for i in range(cidx))))
        cumulative[str(cidx)] = dict(items=len(vals), eligible=quantiles([x[0] for x in vals]),
                                      kept=quantiles([x[1] for x in vals]),
                                      total_eligible=sum(x[0] for x in vals), total_kept=sum(x[1] for x in vals))
    return dict(items=len(rows), N=n, C=c, T=t,
                N_per_problem=float(n / len(rows)), N_median=float(statistics.median(r["denoise_forwards"] for r in rows)),
                C_per_problem=float(c / len(rows)), N_per_canvas=float(n / c) if c else None,
                output_tokens_mean=float(t / len(rows)), output_tokens_median=float(statistics.median(r["output_tokens"] for r in rows)),
                global_eligible_tiles=eligible, global_kept_tiles=kept,
                global_skipped_tiles=eligible-kept, global_kept_fraction=float(kept/eligible) if eligible else None,
                global_sparsity=float(1-kept/eligible) if eligible else None,
                overall_eligible_rectangles=sum(a.get("overall_eligible_rectangles", 0) for a in adapters),
                overall_skipped_rectangles=sum(a.get("overall_skipped_rectangles", 0) for a in adapters),
                overall_sparsity=float(sum(a.get("overall_skipped_rectangles", 0) for a in adapters) /
                                      sum(a.get("overall_eligible_rectangles", 0) for a in adapters)) if sum(a.get("overall_eligible_rectangles", 0) for a in adapters) else None,
                phases=phases_out, cumulative_canvas=cumulative, canvas_by_item=canvas_by_item,
                value_selected_prefix_tiles=sum(a.get("value_selected_prefix_tiles", 0) or 0 for a in adapters),
                selector_discovery_qk_rectangles=sum(a.get("value_statistics_qk_rectangles", 0) or 0 for a in adapters),
                selector_candidate_evaluations=sum(a.get("value_candidate_evaluations", 0) or 0 for a in adapters),
                selector_discovery_ms=sum((a.get("value_audit_timings") or {}).get("discovery_ms", 0) or 0 for a in adapters),
                selector_selection_ms=sum((a.get("value_audit_timings") or {}).get("selection_ms", 0) or 0 for a in adapters),
                fa4_global_tile_work=kept, native_denoising_forwards=n,
                item_global_kept={key(r): a.get("global_kept_tiles", 0) for r, a in zip(rows, adapters)},
                item_N={key(r): r["denoise_forwards"] for r in rows},
                finish_reasons=dict(__import__("collections").Counter(r.get("finish_reason") for r in rows)),
                capped=sum(r.get("finish_reason") == "length" for r in rows),
                unparsed=None, rows_meta={key(r): {k: r.get(k) for k in ("output_hash", "output_tokens", "denoise_forwards", "canvases", "finish_reason")} for r in rows})


def scorer(prefix: Path, cells: Path, private_files):
    summary = Path(str(prefix) + ".summary.json")
    if summary.exists():
        return load_json(summary)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON, str(ROOT / "scripts/v31_score_longbench_official.py"), str(prefix), POOL, POOL,
           "--cells", str(cells), *[str(x) for x in private_files]]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return load_json(summary)


def private_inputs(attempts, label=None, staging=None):
    paths = []
    for i, attempt in enumerate(attempts):
        src = attempt / "private" / ("run_" + load_json(attempt / "config.json")["arm"] + ".private.jsonl")
        # The runner's arm is current_v31_control or value_v2...; custom
        # official labels use immutable symlinks so scorer labels stay distinct.
        if label:
            staging.mkdir(parents=True, exist_ok=True)
            # The official scorer derives an arm label from the filename after
            # the first underscore.  Keep the shard ordinal before the
            # underscore so all shards contribute to one arm.
            dst = staging / f"{i+1:03d}_{label}.private.jsonl"
            if dst.exists() or dst.is_symlink():
                if dst.resolve() != src.resolve():
                    raise RuntimeError(f"private input alias changed: {dst}")
            else:
                dst.symlink_to(src)
            paths.append(dst)
        else:
            paths.append(src)
    return paths


def score_pair(stage_dir: Path, cells: Path, control_attempts, candidate_attempts, candidate_label=None):
    staging = stage_dir / "private_inputs"
    control_files = private_inputs(control_attempts, "current_v31_control" if candidate_label else None, staging)
    cand_files = private_inputs(candidate_attempts, candidate_label, staging) if candidate_label else private_inputs(candidate_attempts, None, staging)
    prefix = stage_dir / "scoring" / "official"
    return scorer(prefix, cells, control_files + cand_files)


def score_pair_data(summary_path: Path, official_path: Path, control_label="current_v31_control", candidate_label=None):
    summary = load_json(summary_path)
    official = load_json(official_path)
    labels = list(official)
    if candidate_label is None:
        candidate_label = next(x for x in labels if x != control_label)
    c = official[control_label]
    v = official[candidate_label]
    if set(c) != set(v):
        raise RuntimeError("scorer cell sets differ")
    a = np.asarray([bool(c[k]) for k in c], dtype=float)
    b = np.asarray([bool(v[k]) for k in c], dtype=float)
    delta = b - a
    b10 = int(((a == 1) & (b == 0)).sum())
    c01 = int(((a == 0) & (b == 1)).sum())
    ndisc = b10 + c01
    if ndisc:
        p = 2 * sum(math.comb(ndisc, k) for k in range(min(b10, c01) + 1)) / (2 ** ndisc)
        p = min(1.0, p)
    else:
        p = 1.0
    rng = np.random.default_rng(1729)
    draws = rng.integers(len(delta), size=(10000, len(delta)))
    boot = delta[draws].mean(axis=1) * 100.0
    return dict(candidate=candidate_label, correct_control=int(a.sum()), correct_candidate=int(b.sum()),
                items=len(a), accuracy_control=float(a.mean()), accuracy_candidate=float(b.mean()),
                delta_pp=float(delta.mean() * 100), discordant_control_only=b10, discordant_candidate_only=c01,
                mcnemar_exact_p=float(p), paired_item_bootstrap_ci95_pp=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                output_hash_mismatches=None, summary=summary)


def mismatch(control_rows, candidate_rows):
    c = {key(r): r for r in control_rows}
    v = {key(r): r for r in candidate_rows}
    if c.keys() != v.keys():
        raise RuntimeError("trajectory keys differ")
    return dict(output_hash_mismatches=sum(c[k].get("output_hash") != v[k].get("output_hash") for k in c),
                token_mismatches=sum(c[k].get("output_tokens") != v[k].get("output_tokens") for k in c),
                denoising_call_mismatches=sum(c[k].get("denoise_forwards") != v[k].get("denoise_forwards") for k in c),
                canvas_mismatches=sum(c[k].get("canvases") != v[k].get("canvases") for k in c),
                finish_reason_mismatches=sum(c[k].get("finish_reason") != v[k].get("finish_reason") for k in c))


def cumulative_item(agg, cidx):
    return {k: v["kept"]["mean"] if False else None for k, v in agg["cumulative_canvas"].items() if int(k) == cidx}


def early_work(control_agg, candidate_agg, cidx):
    c = {k: v for k, v in control_agg["canvas_by_item"].items()}
    v = {k: x for k, x in candidate_agg["canvas_by_item"].items()}
    common = [k for k in c if k in v and max(c[k], default=-1) >= cidx - 1 and max(v[k], default=-1) >= cidx - 1]
    cv = {k: sum(c[k].get(i, {}).get("kept_tiles", 0) for i in range(cidx)) for k in common}
    vv = {k: sum(v[k].get(i, {}).get("kept_tiles", 0) for i in range(cidx)) for k in common}
    diffs = [vv[k] - cv[k] for k in common]
    return dict(canvas=cidx, items=len(common), control_kept=sum(cv.values()), candidate_kept=sum(vv.values()),
                abs_mismatch=abs(sum(diffs)), rel_mismatch=abs(sum(diffs))/sum(cv.values()) if sum(cv.values()) else None,
                per_item_delta=quantiles(diffs))


def choose_early_prefix(control_rows, candidate_rows_by_threshold):
    for cidx in range(EARLY_PREFERRED, 0, -1):
        counts = []
        for rows in [control_rows] + list(candidate_rows_by_threshold.values()):
            counts.append(sum(r["canvases"] >= cidx for r in rows))
        if min(counts, default=0) >= EARLY_MIN_ITEMS:
            return cidx, counts
    return 1, [sum(r["canvases"] >= 1 for r in rows) for rows in [control_rows] + list(candidate_rows_by_threshold.values())]


def write_curve(rows, control_agg, control_score, path_csv, path_json):
    fields = ["threshold", "accuracy", "correct", "items", "global_eligible_tiles", "global_kept_tiles",
              "global_sparsity", "overall_sparsity", "denoising_forwards", "mean_N_per_problem",
              "median_N_per_problem", "mean_C_per_problem", "mean_N_per_canvas", "mean_output_tokens",
              "median_output_tokens", "initial_kept_fraction", "refresh_kept_fraction", "held_kept_fraction",
              "carried_kept_fraction", "first_C_cumulative_kept_tiles", "first_C_items", "total_work_delta",
              "total_work_delta_fraction", "early_work_delta", "early_work_delta_fraction",
              "output_hash_mismatches", "token_mismatches", "denoising_call_mismatches", "unparsed",
              "no_final_response", "capped", "truncated", "selector_discovery_qk_rectangles",
              "selector_candidate_evaluations", "selector_discovery_ms", "selector_selection_ms",
              "fa4_global_tile_work"]
    with path_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})
    path_json.write_text(json.dumps(dict(status="complete_sweep_rows", control=control_agg,
                                         control_score=control_score, thresholds=rows), indent=2) + "\n")


def sweep(protocol):
    dev_cells = STUDY / "private/cells_development_original_s1.json"
    dev_shards = make_shards(dev_cells, STUDY / "private/development_shards", DEV_SHARDS)
    control_root = STUDY / "sweep" / "control"
    control_attempts = run_arm(control_root, "current_v31_control", dev_shards, "development", None, protocol)
    control_rows = flatten_attempt_records(control_attempts)
    control_agg = aggregate(control_rows)
    control_score = scorer(control_root / "scoring" / "official", dev_cells,
                           private_inputs(control_attempts, None, control_root / "private_inputs"))
    candidate_rows = {}
    curve_rows = []
    staged = {}
    for t in THRESHOLDS:
        name = threshold_name(t)
        root = STUDY / "sweep" / name
        attempts = run_arm(root, "value_v2_online_preserve_mass", dev_shards, "development", t, protocol)
        rows = flatten_attempt_records(attempts)
        candidate_rows[t] = rows
        staged[t] = aggregate(rows)
    early_c, counts = choose_early_prefix(control_rows, candidate_rows)
    for t in THRESHOLDS:
        name = threshold_name(t)
        root = STUDY / "sweep" / name
        score = score_pair(root, dev_cells, control_attempts, complete_attempts(root, "value_v2_online_preserve_mass", dev_shards, "development", t), None)
        pair = score_pair_data(root / "scoring/official.summary.json", root / "scoring/official.official.json")
        m = mismatch(control_rows, candidate_rows[t])
        a = staged[t]
        ep = early_work(control_agg, a, early_c)
        phases = a["phases"]
        diag = score["arms"]["value_v2_online_preserve_mass"]["longbench_v2_0shot_think"]
        row = dict(threshold=t, accuracy=pair["accuracy_candidate"], correct=pair["correct_candidate"], items=pair["items"],
                   global_eligible_tiles=a["global_eligible_tiles"], global_kept_tiles=a["global_kept_tiles"],
                   global_sparsity=a["global_sparsity"], overall_sparsity=a["overall_sparsity"],
                   denoising_forwards=a["N"], mean_N_per_problem=a["N_per_problem"], median_N_per_problem=statistics.median(r["denoise_forwards"] for r in candidate_rows[t]),
                   mean_C_per_problem=a["C_per_problem"], mean_N_per_canvas=a["N_per_canvas"], mean_output_tokens=a["output_tokens_mean"], median_output_tokens=a["output_tokens_median"],
                   initial_kept_fraction=phases["initial"]["kept_tiles"] / phases["initial"]["eligible_tiles"] if phases["initial"]["eligible_tiles"] else None,
                   refresh_kept_fraction=phases["refresh"]["kept_tiles"] / phases["refresh"]["eligible_tiles"] if phases["refresh"]["eligible_tiles"] else None,
                   held_kept_fraction=phases["held"]["kept_tiles"] / phases["held"]["eligible_tiles"] if phases["held"]["eligible_tiles"] else None,
                   carried_kept_fraction=phases["carried"]["kept_tiles"] / phases["carried"]["eligible_tiles"] if phases["carried"]["eligible_tiles"] else None,
                   first_C_cumulative_kept_tiles=ep["candidate_kept"], first_C_items=ep["items"],
                   total_work_delta=a["global_kept_tiles"] - control_agg["global_kept_tiles"],
                   total_work_delta_fraction=(a["global_kept_tiles"] - control_agg["global_kept_tiles"]) / control_agg["global_kept_tiles"],
                   early_work_delta=ep["candidate_kept"] - ep["control_kept"],
                   early_work_delta_fraction=(ep["candidate_kept"] - ep["control_kept"]) / ep["control_kept"] if ep["control_kept"] else None,
                   output_hash_mismatches=m["output_hash_mismatches"], token_mismatches=m["token_mismatches"],
                   denoising_call_mismatches=m["denoising_call_mismatches"],
                   unparsed=diag.get("unparsed"), no_final_response=diag.get("no_final_response"),
                   capped=diag.get("capped"), truncated=diag.get("truncated_items"),
                   selector_discovery_qk_rectangles=a["selector_discovery_qk_rectangles"],
                   selector_candidate_evaluations=a["selector_candidate_evaluations"],
                   selector_discovery_ms=a["selector_discovery_ms"], selector_selection_ms=a["selector_selection_ms"],
                   fa4_global_tile_work=a["fa4_global_tile_work"],
                   paired=pair, score=score, aggregate=a, early=ep)
        curve_rows.append(row)
        (root / "config.json").write_text(json.dumps(dict(status="frozen", threshold=t,
                                                            arm="value_v2_online_preserve_mass",
                                                            purpose="development", protocol_sha256=sha(STUDY / "protocol.json"),
                                                            cells_sha256=sha(dev_cells)), indent=2) + "\n")
        (root / "receipt.json").write_text(json.dumps(dict(status="complete", threshold=t, records=len(candidate_rows[t]),
                                                             source_commit=protocol["source_commit"],
                                                             source_sha256=protocol["source_sha256"],
                                                             selector="value_v2_online_preserve_mass",
                                                             output_path="unchanged_fa4", local_scope="native_dense",
                                                             validation_errors=[]), indent=2) + "\n")
        (root / "summary.json").write_text(json.dumps(dict(status="complete", threshold=t, aggregate=a, score=score,
                                                            paired=pair, mismatches=m, early=ep), indent=2) + "\n")
        (root / "final_complete.json").write_text(json.dumps(dict(status="complete", threshold=t,
                                                                    records=len(candidate_rows[t]), scorer="official"), indent=2) + "\n")
        write_curve(curve_rows, control_agg, control_score, STUDY / "sweep/curve.csv", STUDY / "sweep/curve.json")
        (STUDY / "sweep/summary.md").write_text(sweep_markdown(control_agg, control_score, curve_rows, early_c, counts))
    return protocol, control_rows, control_attempts, control_agg, control_score, candidate_rows, staged, curve_rows, early_c, counts


def sweep_markdown(control, score, rows, early_c, counts):
    lines = ["# V2 threshold sweep", "", f"Development schedule: original 64 cells; selected early prefix C={early_c}.",
             f"Items reaching C={early_c}: control={counts[0]}, thresholds={min(counts[1:]) if len(counts)>1 else 0} minimum.", "",
             "| threshold | accuracy | kept GLOBAL tiles | GLOBAL sparsity | N/problem | early kept tiles | total work delta | early work delta |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    lines.append(f"| V31 control | {score.get('arms',{}).get('current_v31_control',{}).get('longbench_v2_0shot_think',{}).get('runs',{}).get('seed1_repeat0',{}).get('Overall','n/a')} | {control['global_kept_tiles']} | {control['global_sparsity']:.4f} | {control['N_per_problem']:.2f} | {control['cumulative_canvas'].get(str(early_c),{}).get('total_kept','n/a')} | 0 | 0 |")
    for r in rows:
        lines.append(f"| {r['threshold']} | {r['correct']}/{r['items']} | {r['global_kept_tiles']} | {r['global_sparsity']:.4f} | {r['mean_N_per_problem']:.2f} | {r['first_C_cumulative_kept_tiles']} | {r['total_work_delta']} ({r['total_work_delta_fraction']:+.2%}) | {r['early_work_delta']} ({r['early_work_delta_fraction']:+.2%}) |")
    lines += ["", "All work is physical GLOBAL FA4 consumer work. Selector discovery QK/PV work and audit instrumentation are recorded separately in each threshold summary. No wall-time speed claim is made.", ""]
    return "\n".join(lines)


def choose_targets(control_agg, curve_rows, candidate_rows, early_c):
    def total_key(r):
        return (abs(r["total_work_delta"]), abs(r["early_work_delta"]), -r["accuracy"])
    def early_key(r):
        return (abs(r["early_work_delta"]), abs(r["total_work_delta"]), -r["accuracy"])
    total = min(curve_rows, key=total_key)
    early = min(curve_rows, key=early_key)
    total_ok = abs(total["total_work_delta_fraction"]) <= .02
    early_ok = abs(total["early_work_delta_fraction"]) <= .03
    one = total if total["threshold"] == early["threshold"] and total_ok and early_ok else None
    selected = {"tau_total": total["threshold"], "tau_early": early["threshold"],
                "single_threshold": one["threshold"] if one else None,
                "early_canvas_prefix": early_c, "total_ok": total_ok, "early_ok_at_total": early_ok,
                "total_row": total, "early_row": early,
                "policy": "absolute physical GLOBAL kept tiles first; development accuracy only tie-breaks",
                "tolerances": {"total_fraction": .02, "early_fraction": .03},
                "candidate_thresholds": [r["threshold"] for r in curve_rows]}
    out = STUDY / "matched_calibration/selected_thresholds.json"
    out.write_text(json.dumps(selected, indent=2) + "\n")
    (STUDY / "matched_calibration/target.json").write_text(json.dumps(dict(status="complete", target="current_v31_control",
                                                                           total_kept_tiles=control_agg["global_kept_tiles"],
                                                                           early_canvas_prefix=early_c,
                                                                           early_target_kept_tiles=control_agg["cumulative_canvas"].get(str(early_c), {}).get("total_kept"),
                                                                           selected=selected), indent=2) + "\n")
    return selected


def official(protocol, control_dev_rows, selected):
    target_cells = STUDY / "private/cells_target_full503.json"
    shards = make_shards(target_cells, STUDY / "private/target_shards", TARGET_SHARDS)
    root = STUDY / "official_audit"
    control_attempts = run_arm(root, "current_v31_control", shards, "audit", None, protocol)
    selected_thresholds = []
    if selected["single_threshold"] is not None:
        selected_thresholds = [("v2_work_match_total_and_early", selected["single_threshold"])]
    else:
        selected_thresholds = [("v2_work_match_total", selected["tau_total"]), ("v2_work_match_early", selected["tau_early"])]
    all_results = []
    for label, threshold in selected_thresholds:
        actual = run_arm(root, "value_v2_online_preserve_mass", shards, "audit", threshold, protocol)
        rows_c = flatten_attempt_records(control_attempts)
        rows_v = flatten_attempt_records(actual)
        stage_dir = root / label
        stage_dir.mkdir(parents=True, exist_ok=True)
        # Make scorer labels unique while retaining the immutable runner arm.
        score = score_pair(stage_dir, target_cells, control_attempts, actual, label)
        pair = score_pair_data(stage_dir / "scoring/official.summary.json", stage_dir / "scoring/official.official.json", candidate_label=label)
        ac = aggregate(rows_c)
        av = aggregate(rows_v)
        mm = mismatch(rows_c, rows_v)
        cidx = int(selected["early_canvas_prefix"])
        ep = early_work(ac, av, cidx)
        items = sorted(set(ac["item_global_kept"]) & set(av["item_global_kept"]))
        diffs = [av["item_global_kept"][k] - ac["item_global_kept"][k] for k in items]
        work = dict(control=ac, candidate=av, total_kept_ratio=av["global_kept_tiles"] / ac["global_kept_tiles"],
                    total_kept_delta=av["global_kept_tiles"] - ac["global_kept_tiles"],
                    per_item_total_kept_delta=quantiles(diffs), early=ep,
                    phase_delta={p: dict(eligible=av["phases"][p]["eligible_tiles"] - ac["phases"][p]["eligible_tiles"],
                                        kept=av["phases"][p]["kept_tiles"] - ac["phases"][p]["kept_tiles"])
                                 for p in ac["phases"]})
        (stage_dir / "work_accounting.json").write_text(json.dumps(work, indent=2) + "\n")
        (stage_dir / "official_scorer_summary.json").write_text(json.dumps(score, indent=2) + "\n")
        (stage_dir / "paired_accuracy.json").write_text(json.dumps(pair, indent=2) + "\n")
        (stage_dir / "summary.json").write_text(json.dumps(dict(status="complete", label=label, threshold=threshold,
                                                                  work=work, score=score, paired=pair, mismatches=mm), indent=2) + "\n")
        all_results.append(dict(label=label, threshold=threshold, work=work, score=score, paired=pair, mismatches=mm))
    # Consolidated sanitized tables.
    write_official_tables(all_results, root)
    (root / "summary.md").write_text(official_markdown(all_results, selected))
    (root / "final_complete.json").write_text(json.dumps(dict(status="complete", arms=len(all_results),
                                                               official_cells=503, scorer="official", timing="paused"), indent=2) + "\n")
    return all_results


def write_official_tables(results, root):
    work_fields = ["arm", "threshold", "accuracy", "correct", "items", "N", "C", "N_per_problem", "N_per_canvas",
                   "T", "global_eligible_tiles", "global_kept_tiles", "global_skipped_tiles", "global_sparsity",
                   "overall_eligible_rectangles", "overall_skipped_rectangles", "overall_sparsity", "total_kept_ratio",
                   "total_kept_delta", "early_canvas", "early_items", "early_control_kept", "early_candidate_kept",
                   "early_kept_delta", "output_hash_mismatches", "token_mismatches", "denoising_call_mismatches",
                   "selector_discovery_qk_rectangles", "selector_candidate_evaluations", "selector_discovery_ms",
                   "selector_selection_ms", "fa4_global_tile_work"]
    with (root / "work_accounting.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=work_fields)
        w.writeheader()
        for x in results:
            a = x["work"]["candidate"]
            e = x["work"]["early"]
            w.writerow(dict(arm=x["label"], threshold=x["threshold"], accuracy=x["paired"]["accuracy_candidate"],
                            correct=x["paired"]["correct_candidate"], items=x["paired"]["items"], N=a["N"], C=a["C"],
                            N_per_problem=a["N_per_problem"], N_per_canvas=a["N_per_canvas"], T=a["T"],
                            global_eligible_tiles=a["global_eligible_tiles"], global_kept_tiles=a["global_kept_tiles"],
                            global_skipped_tiles=a["global_skipped_tiles"], global_sparsity=a["global_sparsity"],
                            overall_eligible_rectangles=a["overall_eligible_rectangles"], overall_skipped_rectangles=a["overall_skipped_rectangles"],
                            overall_sparsity=a["overall_sparsity"], total_kept_ratio=x["work"]["total_kept_ratio"],
                            total_kept_delta=x["work"]["total_kept_delta"], early_canvas=e["canvas"], early_items=e["items"],
                            early_control_kept=e["control_kept"], early_candidate_kept=e["candidate_kept"],
                            early_kept_delta=e["candidate_kept"]-e["control_kept"],
                            output_hash_mismatches=x["mismatches"]["output_hash_mismatches"], token_mismatches=x["mismatches"]["token_mismatches"],
                            denoising_call_mismatches=x["mismatches"]["denoising_call_mismatches"],
                            selector_discovery_qk_rectangles=a["selector_discovery_qk_rectangles"],
                            selector_candidate_evaluations=a["selector_candidate_evaluations"],
                            selector_discovery_ms=a["selector_discovery_ms"], selector_selection_ms=a["selector_selection_ms"],
                            fa4_global_tile_work=a["fa4_global_tile_work"]))
    phase_fields = ["arm", "threshold", "phase", "control_calls", "candidate_calls", "control_eligible_tiles", "candidate_eligible_tiles",
                    "control_kept_tiles", "candidate_kept_tiles", "control_sparsity", "candidate_sparsity"]
    with (root / "phase_accounting.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=phase_fields); w.writeheader()
        for x in results:
            for phase in ("dense", "initial", "refresh", "held", "carried"):
                c, a = x["work"]["control"]["phases"][phase], x["work"]["candidate"]["phases"][phase]
                w.writerow(dict(arm=x["label"], threshold=x["threshold"], phase=phase,
                                control_calls=c["calls"], candidate_calls=a["calls"], control_eligible_tiles=c["eligible_tiles"], candidate_eligible_tiles=a["eligible_tiles"],
                                control_kept_tiles=c["kept_tiles"], candidate_kept_tiles=a["kept_tiles"], control_sparsity=c["sparsity"], candidate_sparsity=a["sparsity"]))
    with (root / "cumulative_canvas_work.csv").open("w", newline="") as f:
        fields = ["arm", "threshold", "canvas", "control_items", "candidate_items", "control_total_kept", "candidate_total_kept", "control_total_eligible", "candidate_total_eligible"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for x in results:
            csum, asum = x["work"]["control"]["cumulative_canvas"], x["work"]["candidate"]["cumulative_canvas"]
            for cidx in sorted(set(csum) & set(asum), key=int):
                w.writerow(dict(arm=x["label"], threshold=x["threshold"], canvas=cidx,
                                control_items=csum[cidx]["items"], candidate_items=asum[cidx]["items"],
                                control_total_kept=csum[cidx]["total_kept"], candidate_total_kept=asum[cidx]["total_kept"],
                                control_total_eligible=csum[cidx]["total_eligible"], candidate_total_eligible=asum[cidx]["total_eligible"]))
    (root / "official_scorer_summary.json").write_text(json.dumps({x["label"]: x["score"] for x in results}, indent=2) + "\n")


def official_markdown(results, selected):
    lines = ["# Equal-work V2 versus V31 official audit", "", f"Selected early canvas prefix: C={selected['early_canvas_prefix']}.",
             "All physical work is from adapter receipts. Audit wall time is descriptive and is not clean latency.", "",
             "| arm | threshold | accuracy | N/problem | GLOBAL kept tiles | GLOBAL sparsity | kept ratio vs V31 | early kept delta | paired delta pp | bootstrap 95% CI pp | McNemar p |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for x in results:
        p = x["paired"]; a = x["work"]["candidate"]; e = x["work"]["early"]
        lines.append(f"| {x['label']} | {x['threshold']} | {p['correct_candidate']}/{p['items']} ({p['accuracy_candidate']*100:.1f}%) | {a['N_per_problem']:.2f} | {a['global_kept_tiles']} | {a['global_sparsity']:.4f} | {x['work']['total_kept_ratio']:.4f} | {e['candidate_kept']-e['control_kept']} | {p['delta_pp']:+.3f} | [{p['paired_item_bootstrap_ci95_pp'][0]:+.3f}, {p['paired_item_bootstrap_ci95_pp'][1]:+.3f}] | {p['mcnemar_exact_p']:.4f} |")
    lines += ["", "The report separates physical FA4 consumer tiles, selector discovery QK/PV work, selection candidate evaluations, native denoising forwards, and audit instrumentation. Equal total kept tiles do not imply equal early-canvas work, equal stopping, or equal end-to-end throughput.", ""]
    return "\n".join(lines)


def main():
    global STUDY, PYTHON, MODEL, POOL, OLD_STUDY, DEV_SOURCE, TARGET_SOURCE
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("all", "sweep", "official"), default="all")
    p.add_argument("--study", type=Path, required=True,
                   help="new campaign directory under results; never an existing frozen run")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--model", required=True)
    p.add_argument("--manifests", required=True)
    p.add_argument("--reference-study", type=Path, default=OLD_STUDY)
    p.add_argument("--development-cells", type=Path)
    p.add_argument("--target-cells", type=Path)
    args = p.parse_args()
    STUDY = args.study.resolve()
    if not STUDY.is_relative_to(ROOT / "results"):
        raise ValueError("study must be under the canonical results root")
    PYTHON, MODEL, POOL = args.python, args.model, args.manifests
    OLD_STUDY = args.reference_study.resolve()
    DEV_SOURCE = args.development_cells or OLD_STUDY / "private/cells_development_original_s1.json"
    TARGET_SOURCE = args.target_cells or OLD_STUDY / "private/cells_target_full503.json"
    protocol = setup()
    if args.stage in ("all", "sweep"):
        result = sweep(protocol)
        if args.stage == "sweep":
            return
        _, control_rows, control_attempts, control_agg, _, candidate_rows, _, curve_rows, early_c, _ = result
        selected = choose_targets(control_agg, curve_rows, candidate_rows, early_c)
        official(protocol, control_rows, selected)
    else:
        selected = load_json(STUDY / "matched_calibration/selected_thresholds.json")
        official(protocol, [], selected)
    print(json.dumps(dict(status="complete", study=str(STUDY))), flush=True)


if __name__ == "__main__":
    main()
