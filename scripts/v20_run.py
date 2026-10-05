"""Fail-closed v20 paired request worker; imports CUDA only after CPU identity checks.

The method binding and host-local configs are separate from the immutable selection
protocol. A binding must be frozen before this worker can launch any request.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from scripts.v13_seed_runs import (arm_config_hash, execution_key, is_device_error,
                                   receipt_path, run_schedule, validate_resume)
from scripts.v18_evaluate import native_phase_evidence, run_complete_blocks, strict_warm
from scripts.v18_protocol import GOLD_KEYS, read_rows, sha
from scripts.v20_panel import ARMS, HISTORICAL

SCOPES = ("ALL_NATIVE_LEGAL", "GLOBAL_ONLY_NATIVE_LOCAL")


def router_phase_evidence(arm: str, counters: dict | None) -> dict | None:
    """Identity of actual A/D/H router work; separate from decoder-call phases."""
    if arm == "D_native":
        return {"phase": "native_dense"}
    if not isinstance(counters, dict):
        return None
    if arm in ("D_matched", HISTORICAL):
        counts = counters.get("phase_counts")
        calls = counters.get("attention_calls")
        if not isinstance(counts, dict) or any(type(counts.get(k)) is not int for k in ("A", "D", "H")):
            return None
        if sum(counts.values()) != calls:
            return None
        return {"phase": "matched_dense" if arm == "D_matched" else "nativeQ128_held",
                "A": counts["A"], "D": counts["D"], "H": counts["H"], "attention_calls": calls}
    if arm == "T_scope":
        calls = counters.get("attention_calls")
        return {"phase": "fresh_T", "attention_calls": calls} if type(calls) is int and calls > 0 else None
    calls, a, decisions, held = (counters.get(k) for k in
                                 ("attention_calls", "score_refresh_calls", "decision_refresh_calls", "held_decision_calls"))
    if any(type(v) is not int or v < 0 for v in (calls, a, decisions, held)) or decisions < a or decisions + held != calls:
        return None
    return {"phase": "M1_M3_cached", "A": a, "D": decisions - a, "H": held, "attention_calls": calls}


def strict_v20_warm(first: dict | None, warm: dict) -> dict:
    result = strict_warm(first, warm)
    reasons = list(result["reasons"])
    if not first or not first.get("router_phase_evidence") or not warm.get("router_phase_evidence"):
        reasons.append("router_phase_evidence_absent")
    elif first["router_phase_evidence"] != warm["router_phase_evidence"]:
        reasons.append("router_phase_evidence_mismatch")
    return {"accepted": not reasons, "reasons": sorted(set(reasons))}


def _json(path: Path):
    return json.loads(path.read_text())


def validate_inputs(protocol_path: Path, binding_path: Path, manifests_dir: Path,
                    host: str, gpu_uuid: str, *, stage: str) -> tuple[dict, dict, dict, dict]:
    """Pure-CPU validation; returns protocol, binding, id->row, arm->config."""
    protocol = _json(protocol_path)
    binding = _json(binding_path)
    if protocol.get("schema") != "v20_fan_panel_v1" or protocol.get("planned_executions") != 700:
        raise ValueError("wrong frozen panel")
    if binding.get("status") != "frozen" or binding.get("panel_protocol_sha256") != sha(protocol_path.read_bytes()):
        raise ValueError("method binding not frozen against exact panel bytes")
    scope = binding.get("scope")
    if scope not in SCOPES or not binding.get("policy_sha256"):
        raise ValueError("qualified common scope/policy missing")
    if stage == "historical" and binding.get("historical_qualified") is not True:
        raise ValueError("historical G75L30 transplant not qualified")
    expected_scope = scope
    if stage == "historical":
        expected_scope = binding.get("historical_scope")
        if expected_scope != "ALL_NATIVE_LEGAL":
            raise ValueError("G75L30_nativeQ128 requires independently qualified ALL_NATIVE_LEGAL scope")
    if stage not in ("initial", "remainder", "all", "historical"):
        raise ValueError("unknown stage")
    assignments = protocol["block_assignments"]
    if not any(a["host"] == host and a["gpu_uuid"] == gpu_uuid for a in assignments.values()):
        raise ValueError("host/GPU not in frozen block map")
    config_spec = binding.get("host_configs", {}).get(host, {})
    model = binding.get("host_models", {}).get(host)
    if not model or not Path(model).is_dir():
        raise ValueError("bound host model missing")
    configs = {}
    for dataset in protocol["ids"]:
        configs[dataset] = {}
        for arm in (HISTORICAL,) if stage == "historical" else ARMS:
            item = config_spec.get(dataset, {}).get(arm)
            if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
                raise ValueError(f"missing host-local config: {dataset}/{arm}")
            path = Path(item["path"])
            if sha(path.read_bytes()) != item["sha256"]:
                raise ValueError(f"config byte drift: {dataset}/{arm}")
            config = _json(path)
            if arm != "D_native" and config.get("v20_scope") != expected_scope:
                raise ValueError(f"scope drift: {dataset}/{arm}")
            if arm != "D_native" and config.get("support_geometry", "native_legal") != "native_legal":
                raise ValueError(f"support geometry drift: {dataset}/{arm}")
            if config.get("policy_sha256") != binding["policy_sha256"]:
                raise ValueError(f"policy drift: {dataset}/{arm}")
            if config.get("manifest_sha256") != protocol["generation_manifest_sha256"][dataset]:
                raise ValueError(f"manifest config drift: {dataset}/{arm}")
            if config.get("model") != model:
                raise ValueError(f"model config drift: {dataset}/{arm}")
            if not config.get("source_hashes") or not config.get("model_metadata_hashes"):
                raise ValueError("source/model hashes required")
            for source, expected in config["source_hashes"].items():
                if sha(Path(source).read_bytes()) != expected:
                    raise ValueError(f"source/binary drift: {source}")
            for filename, expected in config["model_metadata_hashes"].items():
                if sha((Path(config["model"]) / filename).read_bytes()) != expected:
                    raise ValueError(f"model metadata drift: {filename}")
            configs[dataset][arm] = config
    rows = {}
    for dataset, ids in protocol["ids"].items():
        path = manifests_dir / f"{dataset}_generation_manifest.json"
        if sha(path.read_bytes()) != protocol["generation_manifest_sha256"][dataset]:
            raise ValueError(f"generation manifest drift: {dataset}")
        loaded = read_rows(path)
        if len(loaded) != len(ids) or {r["id"] for r in loaded} != set(ids):
            raise ValueError(f"manifest ID coverage drift: {dataset}")
        for row in loaded:
            if any(k in row for k in GOLD_KEYS) or row.get("prompt_hash") != sha(row["prompt"]):
                raise ValueError("gold or prompt identity drift")
            if not isinstance(row.get("prompt_tokens"), list) or row.get("prompt_token_count") != len(row["prompt_tokens"]):
                raise ValueError("exact prompt tokens missing")
            if row.get("thinking") is not (dataset != "ruler4k"):
                raise ValueError("task thinking contract drift")
            if dataset == "ruler4k" and row.get("generation_budget") not in (30, 32, 50, 120, 128):
                raise ValueError("RULER task budget drift")
            if dataset != "ruler4k" and row.get("generation_budget") != 8192:
                raise ValueError("task budget drift")
            rows[row["id"]] = row
    return protocol, binding, rows, configs


def stage_entries(protocol: dict, host: str, gpu_uuid: str, stage: str) -> list[dict]:
    schedule = (protocol["historical_extension"]["schedule"] if stage == "historical" else protocol["schedule"])
    if stage == "initial":
        schedule = [e for e in schedule if e["block"] < 6]
    elif stage == "remainder":
        schedule = [e for e in schedule if e["block"] >= 6]
    return [e for e in schedule if e["host"] == host and e["gpu_uuid"] == gpu_uuid]


def run(protocol_path: Path, binding_path: Path, manifests_dir: Path, private: Path,
        ledger: Path, lock_path: Path, *, host: str, stage: str, deadline_epoch: float,
        gpu_budget_s: float, remaining_requests: int, block_guard_s: float,
        timeout: int = 900) -> str:
    import fcntl
    uuid = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()[0].strip()
    protocol, binding, rows, configs = validate_inputs(protocol_path, binding_path, manifests_dir, host, uuid, stage=stage)
    import torch
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse.runner import _atomic, _one
    from scripts.v9_clean_request_timing import append, disk_cache_entries, redacted
    from scripts.v15_seed_runs import mapped_shared_objects
    from triton.runtime.jit import JITFunction
    entries = stage_entries(protocol, host, uuid, stage)
    if not entries:
        return "no_assigned_blocks"
    source_hashes = {}
    for family in configs.values():
        for config in family.values():
            for path, digest in config["source_hashes"].items():
                if path in source_hashes and source_hashes[path] != digest:
                    raise ValueError("conflicting source identities")
                source_hashes[path] = digest
    identity = dict(protocol_id=protocol["protocol_id"] + ("/historical" if stage == "historical" else ""),
                    model_revision=protocol["model_revision"],
                    arm_hashes={f"{dataset}/{arm}": arm_config_hash(c)
                                for dataset, family in configs.items() for arm, c in family.items()},
                    source_hashes=source_hashes,
                    binding_sha256=sha(binding_path.read_bytes()), private_root=str(private.resolve()))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        events = [json.loads(s) for s in ledger.read_text().splitlines() if s.strip()] if ledger.exists() else []
        # v13 validates the old identity fields; the binding hash is checked on every start too.
        for event in events:
            if event.get("event") == "start" and event.get("binding_sha256") != identity["binding_sha256"]:
                raise ValueError("binding changed across resume")
        done = validate_resume(events, identity, entries)
        for e in entries:
            if execution_key(e) not in done and receipt_path(private, e).exists():
                raise ValueError("orphan receipt: preserve immutable first output")
        if all(execution_key(e) in done for e in entries):
            return "complete_prefix"
        started = time.perf_counter()
        append(ledger, dict(event="start", when=time.time(), pid=os.getpid(), host=host, gpu_uuid=uuid, **identity))
        try:
            adapter = create_adapter("diffusion_gemma", binding["host_models"][host], device="cuda",
                                     precision="bfloat16", revision=protocol["model_revision"]).load()
        except BaseException as exc:
            append(ledger, dict(event="worker_end", status="load_error", error=f"{type(exc).__name__}: {exc}"[:500],
                                when=time.time(), host=host, gpu_uuid=uuid,
                                gpu_process_seconds=time.perf_counter() - started))
            raise
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            compiles = []
            JITFunction.cache_hook = lambda **kw: (compiles.append(time.perf_counter()), False)[1]
            first = {e["cell_id"]: e for e in events if e.get("event") == "run" and e.get("role") == "attempt0"}
        except BaseException as exc:
            JITFunction.cache_hook = None
            append(ledger, dict(event="worker_end", status="setup_error", error=f"{type(exc).__name__}: {exc}"[:500],
                                when=time.time(), host=host, gpu_uuid=uuid,
                                gpu_process_seconds=time.perf_counter() - started))
            raise

        class Timeout(Exception):
            pass

        def alarm(*_):
            raise Timeout()

        try:
            signal.signal(signal.SIGALRM, alarm)
        except BaseException as exc:
            JITFunction.cache_hook = None
            append(ledger, dict(event="worker_end", status="setup_error", error=f"{type(exc).__name__}: {exc}"[:500],
                                when=time.time(), host=host, gpu_uuid=uuid,
                                gpu_process_seconds=time.perf_counter() - started))
            raise

        def execute_one(row, seed, config, entry):
            misses, cache_before, so_before = len(compiles), disk_cache_entries(), mapped_shared_objects()
            receipt, error = None, None
            before = time.perf_counter()
            signal.alarm(timeout)
            try:
                receipt = _one(adapter, row, seed, config)
            except Timeout:
                error = f"timeout>{timeout}s"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:500]
            finally:
                signal.alarm(0)
            record = dict(ok=receipt is not None, error=error, generation_seed=seed,
                          arm_config_hash=identity["arm_hashes"][entry["dataset"] + "/" + entry["arm"]],
                          triton_misses=len(compiles) - misses,
                          triton_disk_entries_added=disk_cache_entries() - cache_before,
                          new_shared_objects=sorted(mapped_shared_objects() - so_before),
                          outer_wall_s=time.perf_counter() - before, host=host, gpu_uuid=uuid)
            if receipt is not None:
                if receipt.get("seed") != seed:
                    raise ValueError("actual generation seed drift")
                record.update(redacted(receipt))
                record["phase_evidence"] = native_phase_evidence(receipt, sparse=entry["arm"] not in ("D_native", "D_matched"))
                if record["phase_evidence"] is not None:
                    record["phase_evidence"]["phase"] = ("native_dense_decoder" if entry["arm"] == "D_native" else
                                                          "matched_dense_decoder" if entry["arm"] == "D_matched" else
                                                          "fresh_T_decoder" if entry["arm"] == "T_scope" else
                                                          "held_bitmap_decoder" if entry["arm"] == HISTORICAL else
                                                          "M1_M3_decoder")
                record["router_phase_evidence"] = router_phase_evidence(entry["arm"], receipt.get("counters"))
            if entry["role"] == "warm":
                record["acceptance"] = strict_v20_warm(first.get(entry["cell_id"]), record)
                if receipt is not None:
                    path = receipt_path(private, entry)
                    _atomic(path, receipt)
                    record["private_receipt"] = str(path)
            else:
                first[entry["cell_id"]] = record
            return dict(record=record, receipt=receipt, fatal=bool(error and is_device_error(error)))

        status = "error"
        failure = None
        try:
            def execute_block(block_entries):
                dataset = block_entries[0]["dataset"]
                if any(e["dataset"] != dataset for e in block_entries):
                    raise ValueError("mixed-dataset block")
                return run_schedule(block_entries, done=done, execute_one=execute_one, rows=rows,
                                    configs=configs[dataset], ledger_append=lambda r: append(ledger, r),
                                    private=private, save_receipt=_atomic)
            status = run_complete_blocks(entries, done, deadline_epoch=deadline_epoch,
                                         gpu_budget_s=gpu_budget_s, remaining_requests=remaining_requests,
                                         block_guard_s=block_guard_s, process_started=started,
                                         execute_block=execute_block)
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"[:500]
            raise
        finally:
            JITFunction.cache_hook = None
            append(ledger, dict(event="worker_end", status=status, error=failure, when=time.time(), host=host,
                                gpu_uuid=uuid, gpu_process_seconds=time.perf_counter() - started))
    return status


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "binding", "manifests-dir", "private", "ledger", "lock"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--host", required=True)
    p.add_argument("--stage", choices=("initial", "remainder", "all", "historical"), required=True)
    p.add_argument("--deadline-epoch", type=float, required=True)
    p.add_argument("--gpu-budget-s", type=float, required=True)
    p.add_argument("--remaining-requests", type=int, required=True)
    p.add_argument("--block-guard-s", type=float, required=True)
    p.add_argument("--timeout", type=int, default=900)
    a = p.parse_args()
    print(run(a.protocol, a.binding, a.manifests_dir, a.private, a.ledger, a.lock,
              host=a.host, stage=a.stage, deadline_epoch=a.deadline_epoch,
              gpu_budget_s=a.gpu_budget_s, remaining_requests=a.remaining_requests,
              block_guard_s=a.block_guard_s, timeout=a.timeout))


if __name__ == "__main__":
    main()
