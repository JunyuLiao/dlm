"""Deterministic, fail-closed export of v20 generated outputs and request telemetry.

The six private tarballs are read in memory. No archive member is extracted to disk.
The export intentionally contains generated text/tokens but no prompt or gold fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path


STAGES = (("initial", "collect_initial_001", "initial_001"),
          ("remainder", "collect_remainder_002", "remainder_001"),
          ("historical", "collect_historical_001", "historical_001"))
HOSTS = {"mpk": "149.165.151.254", "dllm": "149.165.159.64"}
TOP_KEYS = frozenset("attempt completion_tokens condition counters diagnostics fingerprint "
    "generation_excluding_initial_prefill_seconds generation_gpu_timeline_note "
    "generation_gpu_timeline_seconds generation_time_note id max_new_tokens metadata "
    "output_tokens per_canvas phase prediction prompt_hash prompt_token_hash raw_completion "
    "request_wall_seconds routing schema seed source_id termination_reason total_decoder_calls".split())
METADATA_KEYS = frozenset("actual_denoising_step_count block_size_applied denoising_configuration "
    "native_canvas_length requested_block_size sampling special_token_ids thinking tokens_per_forward".split())
CANVAS_KEYS = frozenset("canvas_index completion_slice_tokens decoder_calls "
    "iteration_cap_final_call native_stop_final_call schedule_steps".split())
COUNTER_NUMERIC = frozenset("attention_calls bitmap_observation_calls bootstrap_calls current_qk_elements "
    "decision_interval decision_refresh_calls held_decision_calls history_summary_peak_bytes "
    "history_summary_transient_upper_bound_bytes peak_score_bytes preqk_consumer_calls "
    "projected_current_v_tokens retained_score_cap_bytes reused_current_v_tokens "
    "reused_qk_elements routing_only_extra_qk_elements score_budget_bytes score_live_bytes "
    "score_peak_bytes score_peak_transient_bytes score_period score_refresh_calls "
    "score_refresh_period summary_budget_bytes summary_budget_declines summary_builds "
    "summary_cap_bytes summary_hits summary_live_bytes summary_misses summary_peak_bytes "
    "summary_prefix_tiles_recomputed summary_prefix_tiles_served summary_resident_bytes "
    "unsupported_mask_refreshes".split())
COUNTER_BOOLEAN = frozenset("cuda_graph_qualified current_output fast_t".split())
COUNTER_TAGS = frozenset("condition consumer guard_mode kernel_variant method output_mode scope selector "
    "selector_layers support support_geometry telemetry v20_arm v20_scope".split())
COUNTER_LISTS = frozenset("active_layers history_layers native_local_layers sketch_layers summary_layers".split())
COUNTER_NULL = frozenset("actual_qk_pv_counts support_build".split())
COUNTER_OMITTED = frozenset("bound_modules counters_note memory_note per_tile_statistics publication "
    "work_counter_scope".split())
COUNTER_KEYS = (COUNTER_NUMERIC | COUNTER_BOOLEAN | COUNTER_TAGS | COUNTER_LISTS |
                COUNTER_NULL | COUNTER_OMITTED | {"phase_counts"})
MEMBER = re.compile(r"private/cells/([0-9a-f]{64})/(attempt00|warm1)\.json\Z")


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def encoded(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def check_keys(value: object, expected: frozenset[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} schema differs; unexpected or missing field")
    return value


def safe_counters(value: object) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not set(value) <= COUNTER_KEYS:
        raise ValueError("unexpected router counter field")
    result = {}
    for key, item in value.items():
        if key in COUNTER_OMITTED or key in COUNTER_NULL:
            if key in COUNTER_NULL and item is not None:
                raise ValueError("unexpected counter source/path payload")
            continue
        if key in COUNTER_NUMERIC and (type(item) is not int or item < 0):
            raise ValueError(f"invalid numeric counter {key}")
        if key in COUNTER_BOOLEAN and type(item) is not bool:
            raise ValueError(f"invalid boolean counter {key}")
        if key in COUNTER_TAGS:
            slash_forbidden = key != "method" and "/" in item if isinstance(item, str) else True
            if (not isinstance(item, str) or slash_forbidden or "\\" in item or
                item.startswith("/") or len(item) > 160):
                raise ValueError(f"path-like/invalid router tag {key}")
        if key in COUNTER_LISTS and (not isinstance(item, list) or
                                     any(type(x) is not int or x < 0 for x in item)):
            raise ValueError(f"invalid layer list {key}")
        if key == "phase_counts" and (not isinstance(item, dict) or set(item) != {"A", "D", "H"} or
                                       any(type(x) is not int or x < 0 for x in item.values())):
            raise ValueError("invalid A/D/H counts")
        result[key] = item
    return result


def public_record(receipt: dict, event: dict, spec: dict, *, stage: str, archive_hash: str,
                  source_record_hash: str, protocol_id: str) -> dict:
    check_keys(receipt, TOP_KEYS, "private receipt")
    if receipt["diagnostics"] is not None or receipt["routing"] not in (None, []):
        raise ValueError("unexpected unreviewed diagnostic/routing payload")
    metadata = check_keys(receipt["metadata"], METADATA_KEYS, "metadata")
    if set(metadata["denoising_configuration"]) != {"confidence_threshold", "entropy_bound",
        "max_denoising_steps", "stability_threshold", "t_max", "t_min"} or set(metadata["sampling"]) != {
        "native_temperature_schedule", "temperature", "top_p"} or set(metadata["special_token_ids"]) != {
        "boi_token_id", "eoi_token_id", "eos_token_ids", "image_token_id", "mask_token_id", "pad_token_id"}:
        raise ValueError("unexpected nested generation metadata")
    canvases = receipt["per_canvas"]
    if not isinstance(canvases, list) or not canvases:
        raise ValueError("missing canvas trajectory")
    for i, canvas in enumerate(canvases):
        check_keys(canvas, CANVAS_KEYS, "per-canvas")
        if (canvas["canvas_index"] != i or type(canvas["decoder_calls"]) is not int or
            canvas["decoder_calls"] != len(canvas["schedule_steps"]) or
            any(type(x) is not int for x in canvas["schedule_steps"]) or
            type(canvas["native_stop_final_call"]) is not bool or
            type(canvas["iteration_cap_final_call"]) is not bool):
            raise ValueError("per-canvas native trajectory drift")
    tokens = receipt["completion_tokens"]
    if (not isinstance(tokens, list) or any(type(x) is not int or x < 0 for x in tokens) or
        len(tokens) != receipt["output_tokens"] or
        sum(c["completion_slice_tokens"] for c in canvases) != len(tokens) or
        sum(c["decoder_calls"] for c in canvases) != receipt["total_decoder_calls"]):
        raise ValueError("generated token/canvas/call conservation failed")
    if (receipt["id"], receipt["seed"], receipt["termination_reason"]) != (
            spec["id"], spec["seed"], event["termination"]):
        raise ValueError("receipt identity/termination differs from immutable ledger")
    if (receipt["total_decoder_calls"], receipt["output_tokens"],
        [c["decoder_calls"] for c in canvases]) != (
            event["decoder_calls"], event["output_tokens"], event["per_canvas_calls"]):
        raise ValueError("receipt work differs from ledger")
    if receipt["fingerprint"] != event["fingerprint"] or receipt["request_wall_seconds"] != event["api_wall_s"]:
        raise ValueError("receipt configuration/timing differs from ledger")
    token_hash = sha(json.dumps(tokens, separators=(",", ":")).encode())
    if token_hash != event["completion_token_hash"]:
        raise ValueError("generated token hash differs from ledger")
    if event.get("ok") is not True or event.get("error") is not None:
        raise ValueError("failed request cannot be exported as generated output")
    if spec["role"] == "warm" and event.get("acceptance") != {"accepted": True, "reasons": []}:
        raise ValueError("warm record was not strictly accepted")
    return dict(schema="v20_public_generation_record_v1",
                identity={key: spec[key] for key in ("arm", "block", "cell_id", "dataset", "gpu_uuid",
                                                   "host", "id", "index", "repeat", "role", "seed")}
                         | dict(protocol_id=protocol_id, stage=stage, execution_key=event["execution_key"]),
                generated=dict(completion_tokens=tokens, raw_completion=receipt["raw_completion"],
                               prediction=receipt["prediction"], output_tokens=receipt["output_tokens"],
                               termination_reason=receipt["termination_reason"],
                               per_canvas=canvases, total_decoder_calls=receipt["total_decoder_calls"],
                               max_new_tokens=receipt["max_new_tokens"], metadata=metadata),
                timing=dict(request_wall_seconds=receipt["request_wall_seconds"],
                            generation_gpu_timeline_seconds=receipt["generation_gpu_timeline_seconds"],
                            generation_gpu_timeline_note=receipt["generation_gpu_timeline_note"],
                            generation_excluding_initial_prefill_seconds=receipt["generation_excluding_initial_prefill_seconds"],
                            generation_time_note=receipt["generation_time_note"],
                            outer_wall_seconds=event["outer_wall_s"]),
                router_metrics=safe_counters(receipt["counters"]),
                phase_evidence=event.get("phase_evidence"),
                router_phase_evidence=event.get("router_phase_evidence"),
                acceptance=event.get("acceptance") if spec["role"] == "warm" else None,
                compile_evidence=dict(triton_misses=event["triton_misses"],
                                      triton_disk_entries_added=event["triton_disk_entries_added"],
                                      new_shared_objects_count=len(event["new_shared_objects"])),
                provenance=dict(source_archive_sha256=archive_hash,
                                original_receipt_sha256=source_record_hash,
                                completion_token_sha256=token_hash,
                                arm_config_hash=event["arm_config_hash"]))


def export(private_root: Path, protocol_path: Path, output: Path) -> dict:
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    core = protocol["schedule"]
    history = protocol["historical_extension"]["schedule"]
    if len(core) != 700 or len(history) != 100:
        raise ValueError("frozen 700+100 schedule changed")
    expected = {}
    for stage, schedule in (("initial", core[:84]), ("remainder", core[84:]), ("historical", history)):
        for spec in schedule:
            key = f"{spec['cell_id']}:{spec['role']}:{spec['repeat']}"
            if key in expected:
                raise ValueError("duplicate frozen execution key")
            expected[key] = (stage, spec)
    if len(expected) != 800:
        raise ValueError("expected exactly 800 frozen request identities")
    prepared, archive_index = {}, []
    for stage, folder, label in STAGES:
        for host_alias, host in HOSTS.items():
            folder_path = private_root / folder
            archives = list(folder_path.glob(f"{host_alias}.{label}.private.tar.gz"))
            if len(archives) != 1:
                raise ValueError(f"missing or duplicate {stage}/{host_alias} archive")
            archive = archives[0]
            archive_hash = sha(archive.read_bytes())
            archive_index.append(dict(stage=stage, host=host, archive_basename=archive.name,
                                      sha256=archive_hash))
            with tarfile.open(archive, "r:gz") as tar:
                members = tar.getmembers()
                ledger_members = [m for m in members if m.name.startswith("ledger/")]
                if len(ledger_members) != 1 or not ledger_members[0].isfile():
                    raise ValueError("archive ledger inventory differs")
                events = [json.loads(line) for line in tar.extractfile(ledger_members[0]).read().splitlines()]
                runs = {}
                for event in events:
                    if event.get("event") == "run":
                        key = event.get("execution_key")
                        if key in runs or key not in expected:
                            raise ValueError("duplicate/foreign ledger run")
                        frozen_stage, spec = expected[key]
                        if (stage, host) != (frozen_stage, spec["host"]):
                            raise ValueError("ledger run assigned to wrong stage/host")
                        if any(event.get(field) != spec[field] for field in (
                                "arm", "block", "cell_id", "gpu_uuid", "host", "id", "index", "repeat", "role", "seed")):
                            raise ValueError("ledger request differs from frozen schedule")
                        runs[key] = event
                    elif event.get("event") not in ("start", "worker_end"):
                        raise ValueError("unexpected archive ledger event")
                if len(runs) != (42 if stage == "initial" else 308 if stage == "remainder" else 50):
                    raise ValueError("per-host stage request count differs")
                seen = set()
                for member in members:
                    if member is ledger_members[0]:
                        continue
                    match = MEMBER.fullmatch(member.name)
                    if not member.isfile() or not match:
                        raise ValueError("unexpected private archive member")
                    cell_id, filename_role = match.groups()
                    role, repeat = ("attempt0", 0) if filename_role == "attempt00" else ("warm", 1)
                    key = f"{cell_id}:{role}:{repeat}"
                    if key not in runs or key in prepared or key in seen:
                        raise ValueError("orphan/duplicate private receipt")
                    seen.add(key)
                    event = runs[key]
                    if not str(event.get("private_receipt", "")).endswith(
                            f"/cells/{cell_id}/{filename_role}.json"):
                        raise ValueError("receipt member differs from ledger path identity")
                    raw = tar.extractfile(member).read()
                    receipt = json.loads(raw)
                    _, spec = expected[key]
                    public = public_record(receipt, event, spec, stage=stage,
                                           archive_hash=archive_hash, source_record_hash=sha(raw),
                                           protocol_id=protocol["protocol_id"])
                    relative = f"{stage}/{host_alias}/{cell_id}_{filename_role}.json"
                    prepared[key] = (relative, encoded(public), public)
                if seen != set(runs):
                    raise ValueError("ledger and receipt members do not match")
    if set(prepared) != set(expected) or len(prepared) != 800:
        raise ValueError("the six archives do not cover all 800 frozen executions")
    roles = Counter(item[2]["identity"]["role"] for item in prepared.values())
    if roles != {"attempt0": 400, "warm": 400}:
        raise ValueError("first/warm counts differ from frozen design")
    stages = Counter(item[2]["identity"]["stage"] for item in prepared.values())
    if stages != {"initial": 84, "remainder": 616, "historical": 100}:
        raise ValueError("stage execution counts differ")
    by_task_arm = Counter((item[2]["identity"]["dataset"], item[2]["identity"]["arm"])
                          for item in prepared.values())
    manifest_rows = []
    for key, (relative, contents, public) in sorted(prepared.items(), key=lambda x: x[1][0]):
        ident = public["identity"]
        manifest_rows.append(dict(relative_path=relative, sha256=sha(contents), execution_key=key,
                                  stage=ident["stage"], host=ident["host"], dataset=ident["dataset"],
                                  arm=ident["arm"], role=ident["role"], seed=ident["seed"],
                                  source_archive_sha256=public["provenance"]["source_archive_sha256"],
                                  original_receipt_sha256=public["provenance"]["original_receipt_sha256"]))
    manifest = dict(schema="v20_public_generation_export_manifest_v1",
                    protocol_id=protocol["protocol_id"], protocol_sha256=sha(protocol_bytes),
                    records=800, first_records=roles["attempt0"], warm_records=roles["warm"],
                    by_stage=dict(sorted(stages.items())),
                    by_dataset_arm={f"{dataset}/{arm}": count for (dataset, arm), count in sorted(by_task_arm.items())},
                    source_archives=archive_index, record_inventory=manifest_rows,
                    excluded_private_fields=["prompt_hash", "prompt_token_hash", "source_id", "private_receipt",
                                             "absolute_private_paths", "gold/answer_source_fields"],
                    limitation="No full per-step QKV/attention tensors were captured in request receipts.")
    readme = f"""# v20 generation records (800 requests)

These are generated outputs and request-level telemetry for the frozen v20 Fan panel: 700 core executions plus 100 separately named G75L30_nativeQ128 executions. Each JSON record contains generated completion tokens, raw generated completion, prediction, per-canvas native stopping flags and schedule steps, decoder calls, timing, and available router counters. There are 400 first and 400 accepted warm records. The core first84 and remainder retain their original stage labels.

The export excludes prompts, prompt hashes, gold/answer-source data, credentials, private environment details and absolute private receipt paths. Question IDs, seeds, arm names, host/GPU IDs and source SHA-256 identities remain for reproducibility. Source tarballs were read without extraction or modification. The manifest binds every exported byte to its source archive and original receipt hash.

Per-canvas schedule_steps are the actual native adaptive decoder-call trajectory. Full per-step QKV/attention tensors, individual attention masks and intermediate activations were not captured in these request receipts and are not reconstructible from this export. The GPU timeline starts after the first actual encoder forward and includes host launch gaps and later encoder/commit work; it is not synchronized prefill-excluded generation wall. Request wall/call is amortized, not direct forward timing.

The historical G75L30_nativeQ128 arm is a native-Q128/K64 transplant, not identical to the older vLLM physical grouping. Its runs were later than the core panel; timing contrasts may include temporal drift. No gold or scoring input is needed to inspect these records. See manifest.json for exact counts and hashes.
"""
    planned = {output / relative: contents for relative, contents, _ in prepared.values()}
    planned[output / "manifest.json"] = encoded(manifest)
    planned[output / "README.md"] = readme.encode("utf-8")
    for path, contents in planned.items():
        if path.exists() and path.read_bytes() != contents:
            raise ValueError(f"refusing to replace different export: {path}")
    for path, contents in planned.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.private_root, args.protocol, args.out)
    print(json.dumps(dict(records=result["records"], first=result["first_records"],
                          warm=result["warm_records"], stages=result["by_stage"]), sort_keys=True))


if __name__ == "__main__":
    main()
