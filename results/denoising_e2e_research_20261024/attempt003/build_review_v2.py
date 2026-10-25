"""Public, numerical review supplements for the completed frozen screen."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shlex
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "review002"
    out.mkdir(exist_ok=False)
    assert read(root / "final_complete.json")["passed"]
    assert read(root / "binding_review001/verification.json")["passed"]
    scores = read(root / "scoring001/official_scores.json")
    quality = {(x["arm"], x["id"], x["seed"]): x for x in scores["outcomes"]}
    protocol = read(root / "e2e_protocol.json")
    bindings = read(root / "result_directory_bindings.json")
    counters = defaultdict(Counter)
    rows = []
    inputs = {}
    for suite in ["short", "long"]:
        for arm in protocol["arms"]:
            leaf = root / bindings["result_paths"].get(suite, {}).get(arm, f"runs/panel/{suite}/{arm}")
            for path in sorted(leaf.glob("clean[0-9][0-9][0-9].json")):
                clean = read(path)
                audit_path = path.with_name(path.name.replace("clean", "audit"))
                audit = read(audit_path)
                outcome = quality[arm, clean["id"], clean["seed"]]
                group = f"{clean['benchmark']}|{arm}"
                selector = audit["selector"]
                if selector:
                    for key in ["selector_discovery_qk_rectangles", "selector_discovery_pv_rectangles",
                                "certificate_current_tail_qk_rectangles", "target_eligible_tiles",
                                "target_kept_tiles", "target_skipped_tiles", "selector_refresh_calls",
                                "held_mask_calls", "carried_mask_calls", "certificate_failures", "fallback_units"]:
                        counters[group][key] += selector[key]
                    for name, event in selector["selector_phase_gpu"].items():
                        counters[group][name + "_calls"] += event["calls"]
                        counters[group][name + "_gpu_ms"] += event["gpu_ms"]
                rows.append(dict(benchmark=clean["benchmark"], id=clean["id"], seed=clean["seed"], arm=arm,
                                 correct=outcome["correct"], N=clean["N"], C=clean["C"], N_per_C=clean["N_per_C"],
                                 T=clean["T"], S=clean["S"], W=clean["W"], S_per_N=clean["S_per_N"],
                                 finish_reason=clean["finish_reason"], clean_timing_valid=clean["timing_valid"],
                                 no_final_response=outcome["no_final_response"],
                                 global_eligible=audit["work"]["global_eligible_tiles"],
                                 global_kept=audit["work"]["global_kept_tiles"],
                                 global_skipped=audit["work"]["global_eligible_tiles"] - audit["work"]["global_kept_tiles"],
                                 local_eligible=audit["work"]["native_local_eligible_rectangles"],
                                 local_skipped=audit["work"]["native_local_skipped_tiles"]))
                inputs[str(path.relative_to(root))] = sha(path)
                inputs[str(audit_path.relative_to(root))] = sha(audit_path)
    assert len(rows) == 450
    with (out / "per_cell_metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(out / "selector_accounting.json", dict(groups={key: dict(value) for key, value in counters.items()},
          scope="Audit CUDA-event costs and actual discovery work; request latency comes from clean records",
          rectangle_units="Q128xKV64 dense-GEMM/consumer rectangles, not CUDA CTA counts",
          overlapping_events="selector_discovery is nested in complete_support_management_refresh; never add them"))
    commands = []
    base = root.parent
    for path in sorted(base.rglob("launch_receipt.json")):
        if any(x in path.parts for x in ["private", "cache", "runtime"]):
            continue
        receipt = read(path)
        if "command" in receipt:
            commands.append(dict(receipt=str(path.relative_to(base)), command=receipt["command"],
                                 environment_overrides=receipt.get("environment_overrides"), receipt_sha256=sha(path)))
    for path in sorted((base / "attempt009").glob("*_receipt.json")):
        receipt = read(path)
        if "command" in receipt:
            commands.append(dict(receipt=str(path.relative_to(base)), command=receipt["command"],
                                 CPU_only=receipt.get("cpu_only"), receipt_sha256=sha(path)))
    write(out / "execution_commands.json", dict(commands=commands, same_frozen_scope=True))
    text = ["# Recorded execution commands", "", "The JSON catalog records exact worker/readout argument vectors and environment overrides from receipts.",
            "Do not rerun commands into immutable existing result directories. A new run needs a new source/protocol binding and private inputs.", ""]
    for item in commands:
        text.extend(["Receipt: " + item["receipt"], "", "```bash", shlex.join(item["command"]), "```", ""])
    text.extend(["Focused CPU follow-up (36 passed):", "", "```bash",
        "CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /home/exouser/dyh/v31_env_20261005/bin/python -m pytest -q results/denoising_e2e_research_20261024/attempt003/tests/test_online_selector.py results/denoising_e2e_research_20261024/attempt001/tests/test_pulsehold_inherited.py tests/test_trajectory_dynamics.py tests/test_dynamics_screen_protocol.py results/denoising_e2e_research_20261024/attempt005/tests/test_v31_pass_binding.py --junitxml=results/denoising_e2e_research_20261024/attempt003/tests/final_cpu001/results.xml",
        "```", "", "Supplemental execution-source verification (450 cells, two bindings, seven runtime files passed):", "", "```bash",
        "CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/exouser/dyh/v31_env_20261005/bin/python results/denoising_e2e_research_20261024/attempt003/verify_execution_bindings.py --root results/denoising_e2e_research_20261024/attempt003", "```", "",
        "Completion orchestration: attempt006/launch_long.py then attempt009/finish.py; the latter ran attempt005/launch_v31.py --suite long and the CPU jobs listed above.",
        "The earlier test command with tests/test_online_selector.py failed before collection because that path does not exist; explicit correct paths passed22 tests, then the final suite passed36."])
    (out / "COMMANDS.md").write_text("\n".join(text) + "\n")
    verification = read(root / "closeout001/verification.json")
    qualification = read(base / "attempt001/qualification001/exit_receipt.json")
    dense_preflight = read(base / "attempt002/launches/short/dense_piecewise/exit_receipt.json")
    allkept_preflight = read(base / "attempt002/private/e2e_live_resource.json")
    failed_v31 = read(root / "launches/short/v31_optimized/exit_receipt.json")
    preflight_markers = {arm: read(base / f"attempt002/runs/panel/short/{arm}/final_complete.json")
                         for arm in ["dense_piecewise", "allkept_layer29"]}
    preflight_forwards = {arm: marker["model_executions"] + marker["startup_model_forwards"]
                          for arm, marker in preflight_markers.items()}
    earlier = [dict(scope="attempt001 replay qualification failed", allocated_worker_seconds=qualification["allocated_worker_seconds"],
                    counted_forwards=qualification["resource"]["forwards"], allocation_status="exit receipt"),
               dict(scope="attempt002 superseded dense preflight", allocated_worker_seconds=dense_preflight["allocated_worker_seconds"],
                    counted_forwards=preflight_forwards["dense_piecewise"], allocation_status="exit receipt; engine+startup from final marker"),
               dict(scope="attempt002 superseded all-kept preflight", allocated_worker_seconds=allkept_preflight["allocated_worker_seconds"],
                    counted_forwards=preflight_forwards["allkept_layer29"], allocation_status="lower bound through final measured cell; exact process-exit allocation unavailable"),
               dict(scope="attempt003 failed V31 mixed-mode warm-up", allocated_worker_seconds=failed_v31["allocated_worker_seconds"],
                    counted_forwards=None, allocation_status="exit receipt; counter was not persisted before failure")]
    write(out / "resources.json", dict(completed_panel_allocated_worker_seconds=verification["allocated_worker_seconds"],
          completed_panel_counted_model_forwards=verification["counted_model_forwards"],
          earlier_attempts=earlier, earlier_allocation_seconds_lower_bound=sum(x["allocated_worker_seconds"] for x in earlier),
          earlier_known_counted_forwards=sum(x["counted_forwards"] or 0 for x in earlier),
          global_campaign_ceiling=None, timing_definition="W is synchronized engine.add_request-to-finish request time on frozen token inputs; excludes model startup, tokenization preparation, audit and task scoring. S excludes prefill.",
          accounting_limit="The failed mixed-mode V31 counter and superseded all-kept exact process-exit allocation are unavailable; do not present whole-campaign forward/allocation totals as exact."))
    write(out / "review_inputs.json", dict(created_unix=time.time(), builder_sha256=sha(Path(__file__)),
          protocol_sha256=sha(root / "e2e_protocol.json"), inputs=inputs,
          scoring_sha256=sha(root / "scoring001/official_scores.json"), completed_cells=450, no_generation_changes=True))
    print(json.dumps(dict(cells=len(rows), selector_groups=len(counters), commands=len(commands))))


if __name__ == "__main__":
    main()
