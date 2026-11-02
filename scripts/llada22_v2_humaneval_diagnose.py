"""Bounded source-pinned replay and V2 selector-route audit.

The cohort is previously examined benchmark data. This is mechanism diagnosis,
not a benchmark result or calibration panel.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import types

import torch


BASE_ARM = "value_v2_gaussian32_reuse_r4"
EXPECTED = {
    "runtime_sha256": "489b4abf466c40e44fd11d18af41e4c4a69cf7f418c1dd6eb972a76eb43f17dd",
    "attention_source_sha256": "619ec332ed70247d0a72996f496fbb34ae7e7debac6637e3c236eac53a099aab",
    "threshold": 0.8944271909999159,
    "model_revision": "e4f77d25ab5ca86f2223c6cdc6a72b887e20c8a5",
    "manifest_sha256": "0c900f9e48cd8b311e9889b487802e393258409bfb7e37c917a5804bf637ba5b",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_hash(tokens):
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()


def score_rows(root, arm):
    records = {}
    for path in sorted((root / "runs/llada22/humaneval" / arm).glob(
            "shard*/attempt001/private/generation.private.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            records[row["id"]] = row
    return records


def token_metrics(text, entry_point):
    fences = re.findall(r"```([^\n`]*)\n(.*?)```", text, re.S)
    bodies = [body.strip() for lang, body in fences
              if lang.strip().lower() == "" or lang.strip().lower().startswith("py")]
    if not bodies:
        bodies = [text]
    import ast
    syntax = []
    present = substantive = docstring_only = stub_only = False
    for body in bodies:
        try:
            tree = ast.parse(body)
        except SyntaxError:
            continue
        syntax.append(True)
        fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == entry_point), None)
        if fn is None:
            continue
        present = True
        statements = [n for n in fn.body if not (
            isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str))]
        docstring_only |= not statements
        is_stub = bool(statements) and all(
            isinstance(n, ast.Pass) or isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Constant) and n.value.value is Ellipsis
            for n in statements)
        stub_only |= is_stub
        substantive |= bool(statements) and not is_stub
    return {"fence_count": len(fences), "python_fence_count": len(bodies),
            "any_fenced_body_parses": bool(syntax), "entry_point_present": present,
            "entry_point_docstring_only": docstring_only,
            "entry_point_stub_only": stub_only,
            "entry_point_has_substantive_body": substantive}


def move_stats(stats, device):
    from experiments.numerical_qk_reuse.v31_value_selectors import Stats
    return Stats(*(x.to(device) if isinstance(x, torch.Tensor) else x
                   for x in (stats.log_mass, stats.mean, stats.nu, stats.invalid,
                             stats.prefix_tiles, stats.heads, stats.blocks)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    base, out = args.base.resolve(), args.out.resolve()
    if not (out / "protocol.json").exists() or (out / "receipt.json").exists():
        raise RuntimeError("expected a fresh frozen protocol directory without a receipt")
    (out / "private").mkdir(exist_ok=True)
    snap = base / "private/execution/224f8f11173bd16e5cf10054a963c74233758cf2"
    exp_root = snap / "experiments/numerical_qk_reuse"
    script_root = snap / "scripts"
    sys.path.insert(0, str(snap))
    sys.path.insert(0, str(script_root))
    import experiments.numerical_qk_reuse.multi_model_attention as attention
    import experiments.numerical_qk_reuse.multi_model_runtime as runtime
    import multi_model_worker as worker
    if sha(exp_root / "multi_model_runtime.py") != EXPECTED["runtime_sha256"]:
        raise RuntimeError("snapshot runtime does not match benchmark pin")
    if sha(exp_root / "multi_model_attention.py") != EXPECTED["attention_source_sha256"]:
        raise RuntimeError("snapshot attention source does not match benchmark pin")
    if Path(runtime.__file__).resolve() != (exp_root / "multi_model_runtime.py").resolve():
        raise RuntimeError("runtime imported from outside pinned snapshot")
    if Path(attention.__file__).resolve() != (exp_root / "multi_model_attention.py").resolve():
        raise RuntimeError("attention imported from outside pinned snapshot")

    full = worker.ModelHarness(base, "llada22").load()
    if full.key != "llada22":
        raise RuntimeError("wrong model harness")
    dataset = json.loads((base / "private/data/llada22/humaneval.json").read_text())
    task = next(row for row in dataset if row["id"] == "humaneval/0")
    expected_row = score_rows(base, BASE_ARM)[task["id"]]
    config_path = (base / "runs/llada22/humaneval" / BASE_ARM /
                   "shard001/attempt001/config.json")
    benchmark_config = json.loads(config_path.read_text())
    if (benchmark_config["threshold"] != EXPECTED["threshold"] or
            benchmark_config["runtime_sha256"] != EXPECTED["runtime_sha256"] or
            benchmark_config["attention_source_sha256"] != EXPECTED["attention_source_sha256"]):
        raise RuntimeError("selected benchmark receipt differs from pinned inputs")
    scored = [json.loads(line) for line in
              (base / "analysis_scope30_20261102_attempt001/scored_records.jsonl").read_text().splitlines()]
    baseline_score = next(r["quality"] for r in scored if
                          (r["model"], r["suite"], r["arm"], r["id"]) ==
                          ("llada22", "humaneval", BASE_ARM, task["id"]))

    report = {"status": "running", "model_revision": EXPECTED["model_revision"],
              "numerical_worker_revision": "224f8f11173bd16e5cf10054a963c74233758cf2",
              "runtime_sha256": sha(exp_root / "multi_model_runtime.py"),
              "attention_source_sha256": sha(exp_root / "multi_model_attention.py"),
              "cohort": [task["id"]], "previously_examined_input": True,
              "official_calibration_or_benchmark": False,
              "gpu_worker_seconds": None, "replay": {}, "same_state_eos": None,
              "selector_route_oracle": None}

    primary_runtime = full.runtime
    primary_runtime.arm = BASE_ARM
    primary_runtime.threshold = EXPECTED["threshold"]
    capture = {"latest_v2_refresh": None}
    original_route = primary_runtime.attention

    def attention_hook(q, k, v, layer, scale, **kwargs):
        result = original_route(q, k, v, layer, scale, **kwargs)
        rt = full.runtime
        if (rt.arm == BASE_ARM and rt.phase == "denoise" and layer == 10 and
                rt.call_in_block and (rt.call_in_block - 1) % 4 == 0):
            key = (layer, q.shape[2], k.shape[2])
            keep = rt.maps[key][0].detach().cpu().clone()
            capture["latest_v2_refresh"] = {
                "layer": layer, "query": q.detach().cpu().clone(),
                "key": k.detach().cpu().clone(), "value": v.detach().cpu().clone(),
                "actual_keep": keep, "threshold": rt.threshold,
                "block_start": rt.block_start, "block_end": rt.block_end,
                "call_in_block": rt.call_in_block, "offset": kwargs.get("offset", 0),
                "scale": scale}
        return result

    primary_runtime.attention = attention_hook
    try:
        replay = full.generate(task["prompt"], worker.CAPS["llada22"]["humaneval"],
                               42, answer_prefix=task.get("answer_prefix", ""))
    finally:
        primary_runtime.attention = original_route
    output_hash = file_hash(replay["tokens"])
    report["replay"] = {
        "id": task["id"], "seed": 42, "threshold": EXPECTED["threshold"],
        "token_count": len(replay["tokens"]), "saved_token_count": expected_row["output_tokens"],
        "output_sha256": output_hash, "saved_output_sha256": expected_row["output_sha256"],
        "exact_saved_output_reproduction": output_hash == expected_row["output_sha256"] and
            len(replay["tokens"]) == expected_row["output_tokens"],
        "official_score": baseline_score,
        "token_diagnostics": token_metrics(replay["text"], task["gold"]["entry_point"]),
        "native_counts": replay["counts"]}
    repeat = full.generate(task["prompt"], worker.CAPS["llada22"]["humaneval"],
                           42, answer_prefix=task.get("answer_prefix", ""))
    report["replay"]["within_process_repeat"] = {
        "token_count": len(repeat["tokens"]), "output_sha256": file_hash(repeat["tokens"]),
        "matches_first_replay": file_hash(repeat["tokens"]) == output_hash,
        "matches_saved_benchmark": file_hash(repeat["tokens"]) == expected_row["output_sha256"]}

    snapshot = capture["latest_v2_refresh"]
    if snapshot is None:
        raise RuntimeError("no layer-10 V2 refresh captured")
    q = snapshot["query"].to("cuda")
    k = snapshot["key"].to("cuda")
    v = snapshot["value"].to("cuda")
    selector_stats = importlib.import_module("experiments.numerical_qk_reuse.v31_value_kernels")
    oracle_module = importlib.import_module("experiments.numerical_qk_reuse.v31_value_selectors")
    projection_runtime = attention.AttentionRuntime(BASE_ARM, snapshot["threshold"])
    z, nu = projection_runtime.project(v, snapshot["layer"])
    scale = snapshot["scale"] or q.shape[-1] ** -.5
    kt = math.ceil(k.shape[2] / 64)
    stats = selector_stats.statistics_cuda(q, k, z, nu, scale, kt)
    mandatory = torch.zeros((q.shape[1] * stats.blocks, kt), device="cuda", dtype=torch.bool)
    mandatory[:, snapshot["block_start"] // 64:
              math.ceil(snapshot["block_end"] / 64)] = True
    eligible = stats.valid.any(-1)
    mandatory &= eligible
    gpu_keep, _, _, _ = selector_stats.online_cuda(
        stats, snapshot["threshold"], True, mandatory,
        torch.zeros_like(mandatory), 0.)
    cpu_stats = move_stats(stats, "cpu")
    cpu_keep, _, _, _ = oracle_module.online_reference(
        cpu_stats, snapshot["threshold"], preserve_mass=True,
        mandatory=mandatory.cpu(), held=torch.zeros_like(mandatory.cpu()), sticky=0.)
    actual = snapshot["actual_keep"].reshape_as(gpu_keep.cpu())
    report["selector_route_oracle"] = {
        "layer": snapshot["layer"], "call_in_block": snapshot["call_in_block"],
        "refresh_point": (snapshot["call_in_block"] - 1) % 4 == 0,
        "threshold": snapshot["threshold"], "query_rows": int(q.shape[2]),
        "key_tokens": int(k.shape[2]), "eligible_tile_units": int(eligible.sum()),
        "mandatory_tiles": int(mandatory.sum()),
        "actual_map_equals_cuda_recompute": torch.equal(actual, gpu_keep.cpu()),
        "cuda_route_equals_cpu_online_recurrence": torch.equal(gpu_keep.cpu(), cpu_keep),
        "actual_tiles": int(actual.sum()), "recomputed_tiles": int(gpu_keep.sum()),
        "cpu_recurrence_tiles": int(cpu_keep.sum()),
        "tile_mismatches_actual_vs_cuda": int((actual != gpu_keep.cpu()).sum()),
        "tile_mismatches_cuda_vs_cpu": int((gpu_keep.cpu() != cpu_keep).sum())}
    if not report["selector_route_oracle"]["actual_map_equals_cuda_recompute"] or not report["selector_route_oracle"]["cuda_route_equals_cpu_online_recurrence"]:
        raise RuntimeError("V2 map differs from the exact CUDA recompute or CPU recurrence")

    torch.cuda.synchronize()
    report["status"] = "pass_diagnostic_complete"
    report["gpu_worker_seconds"] = None
    report["replay"].pop("native_counts", None)
    (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "reproduced": True,
                      "same_state_eos": report["same_state_eos"],
                      "selector_route_oracle": report["selector_route_oracle"]}, indent=2))


if __name__ == "__main__":
    main()
