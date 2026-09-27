"""CPU-only freeze and host bind for the 12-request v21 zero-pruning diagnostic.

``freeze`` needs no Torch. It selects the first two IDs already frozen by v20,
assigns both seeds of the first ID to mpk and both seeds of the second to dllm,
and writes one gold-free LB request manifest plus an immutable public protocol.
``bind`` runs on each host after the new source commit is fixed. It derives the
three task configs from that host's exact old LB D_native P0 config and emits a
host fragment; the coordinator must merge both fragments into a final binding.
Neither subcommand runs a model or a GPU request.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from scripts.v18_protocol import GOLD_KEYS, read_rows, sha
from scripts.v20_bind import (CONTROLS, control_config, model_hashes, sha_json,
                              source_hashes)
from scripts.v21_run import validate_protocol

ARMS = ("D_native", "D_matched_legacy", "D_matched_new_numeric")
HOSTS = (
    {"host": "149.165.151.254", "gpu_uuid": "GPU-6139046a-b005-8fe5-a837-f8270472ab72"},
    {"host": "149.165.159.64", "gpu_uuid": "GPU-fc12ad5c-5334-5509-8fc6-465498fd3915"},
)


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def atomic_same(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"immutable output differs: {path}")
        return
    with path.open("xb") as stream:
        stream.write(data)


def host_map(v20: dict, descriptor: Path | None) -> list[dict]:
    hosts = json.loads(descriptor.read_text()) if descriptor else list(HOSTS)
    if not isinstance(hosts, list) or len(hosts) != 2 or hosts != list(HOSTS):
        raise ValueError("diagnostic host map must be the qualified mpk/dllm host+GPU UUID pair in that order")
    qualified = {(x["host"], x["gpu_uuid"]) for x in v20["block_assignments"].values()}
    if qualified != {(h["host"], h["gpu_uuid"]) for h in hosts}:
        raise ValueError("diagnostic host/GPU differs from frozen v20 panel")
    return hosts


def freeze(v20_protocol_path: Path, v20_binding_path: Path, v20_lb_manifest_path: Path, out_dir: Path,
           hosts_json: Path | None = None) -> dict:
    old_bytes = v20_protocol_path.read_bytes()
    old = json.loads(old_bytes)
    if old.get("schema") != "v20_fan_panel_v1" or old.get("planned_executions") != 700:
        raise ValueError("wrong frozen v20 panel")
    binding_bytes = v20_binding_path.read_bytes()
    old_binding = json.loads(binding_bytes)
    old_sha = hashlib.sha256(old_bytes).hexdigest()
    if (old_binding.get("status") != "frozen" or old_binding.get("panel_protocol_sha256") != old_sha or
            old_binding.get("scope") not in ("ALL_NATIVE_LEGAL", "GLOBAL_ONLY_NATIVE_LOCAL") or
            not isinstance(old_binding.get("policy_sha256"), str) or len(old_binding["policy_sha256"]) != 64):
        raise ValueError("qualified v20 binding/protocol/scope/policy drift")
    hosts = host_map(old, hosts_json)
    ids = old["ids"]["longbench_v2"][:2]
    if len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError("v20 first two LB IDs invalid")
    if sha(v20_lb_manifest_path.read_bytes()) != old["generation_manifest_sha256"]["longbench_v2"]:
        raise ValueError("old LB request manifest SHA drift")
    old_rows = read_rows(v20_lb_manifest_path)
    rows_by_id = {row["id"]: row for row in old_rows}
    if len(rows_by_id) != len(old_rows) or not set(ids) <= set(rows_by_id):
        raise ValueError("old LB manifest IDs duplicated or missing")
    selected = [rows_by_id[q] for q in ids]
    for row in selected:
        if any(k in row for k in GOLD_KEYS) or row.get("prompt_hash") != sha(row["prompt"]):
            raise ValueError("gold or prompt identity drift")
        if row.get("thinking") is not True or row.get("generation_budget") != 8192:
            raise ValueError("LB native request settings drift")
        if not isinstance(row.get("prompt_tokens"), list) or row.get("prompt_token_count") != len(row["prompt_tokens"]):
            raise ValueError("exact LB prompt tokens absent")
    manifest_bytes = (json.dumps(selected, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    old_binding_sha = hashlib.sha256(binding_bytes).hexdigest()
    protocol_id = "v21_zero_pruning_" + hashlib.sha256(canonical([old_sha, old_binding_sha, manifest_sha, hosts])).hexdigest()[:16]
    contracts = {
        "D_native": {"kind": "native"},
        "D_matched_legacy": {"kind": "v20_legacy", "parent_v20_arm": "D_matched",
                             "scope": old_binding["scope"]},
        "D_matched_new_numeric": {"kind": "v21_control", "parent_v20_arm": "D_matched",
                                  "scope": old_binding["scope"],
                                  "output_score_precision": "fp32_scores_bf16_pv", "output_layout": "head_major"},
    }
    old_native_hashes = {h["host"]: old_binding["host_configs"][h["host"]]["longbench_v2"]["D_native"]["sha256"]
                         for h in hosts}
    schedule, assignments = [], {}
    for block, (qid, seed, host) in enumerate((q, s, hosts[number]) for number, q in enumerate(ids)
                                              for s in (101, 202)):
        assignments[str(block)] = dict(dataset="longbench_v2", id=qid, seed=seed, **host)
        offset = block % len(ARMS)
        order = ARMS[offset:] + ARMS[:offset]
        for arm in order:
            cell = hashlib.sha256(json.dumps([protocol_id, qid, seed, arm], separators=(",", ":")).encode()).hexdigest()[:32]
            schedule.append(dict(index=len(schedule), block=block, dataset="longbench_v2", id=qid,
                                 seed=seed, arm=arm, role="attempt0", repeat=0, cell_id=cell, **host))
    protocol = dict(schema="v21_conditional_panel_v1", status="frozen", execution_ready=True,
                    panel_kind="zero_pruning_diagnostic", protocol_id=protocol_id,
                    model_revision=old["model_revision"], v20_protocol_sha256=old_sha,
                    v20_binding_sha256=old_binding_sha,
                    v20_policy_sha256=old_binding["policy_sha256"],
                    old_native_config_sha256_by_host=old_native_hashes,
                    quality_eligible=False, timing_eligible=False,
                    source_identity=old["source_identity"], ids={"longbench_v2": ids}, seeds=[101, 202],
                    arms=list(ARMS), arm_contracts=contracts,
                    generation_manifest_sha256={"longbench_v2": manifest_sha},
                    block_assignments=assignments, stages={"diagnostic": list(range(4))},
                    schedule=schedule, planned_executions=12,
                    intent="CP1 natural no-pruning numeric diagnostic; no quality/timing promotion from answers")
    # validate_protocol reads the checked-in v20 protocol for exact ID/byte binding.
    if hashlib.sha256(Path(__file__).resolve().parents[1].joinpath(
            "results/fan_m1_m3_multidataset_20260927/frozen_protocol.json").read_bytes()).hexdigest() != old_sha:
        raise ValueError("freeze requires exact checked-in deployed v20 protocol bytes")
    validate_protocol(protocol)
    atomic_same(out_dir / "manifests/longbench_v2_generation_manifest.json", manifest_bytes)
    atomic_same(out_dir / "zero_pruning_diagnostic_protocol.json", canonical(protocol))
    return protocol


def bind(protocol_path: Path, old_native_config: Path, old_native_sha256: str, manifests_dir: Path,
         host: str, scope: str, source_commit: str, out: Path) -> dict:
    from experiments.numerical_qk_reuse import v21

    protocol = json.loads(protocol_path.read_text())
    validate_protocol(protocol)
    if protocol["panel_kind"] != "zero_pruning_diagnostic" or host not in [h["host"] for h in HOSTS]:
        raise ValueError("bind requires frozen diagnostic and assigned host")
    if scope != protocol["arm_contracts"]["D_matched_legacy"]["scope"]:
        raise ValueError("scope differs from frozen same-consumer pair")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be exact 40-hex deployed commit")
    if sha(old_native_config.read_bytes()) != old_native_sha256:
        raise ValueError("old D_native config SHA differs from qualified binding")
    if old_native_sha256 != protocol["old_native_config_sha256_by_host"][host]:
        raise ValueError("old native config differs from frozen v20 host binding")
    old = json.loads(old_native_config.read_text())
    if (old.get("condition") != "native_dense" or old.get("plugin") is not None or
            old.get("policy_name") != "P0" or old.get("thinking") is not True or
            old.get("max_new_tokens") != 8192 or old.get("native_adaptive") is not True or
            old.get("diagnostic") is not False or old.get("revision") != protocol["model_revision"]):
        raise ValueError("old native P0/runtime/task contract drift")
    if (old.get("policy_sha256") != sha_json(old["policy"]) or
            old["policy_sha256"] != protocol["v20_policy_sha256"]):
        raise ValueError("old P0 policy content/hash drift")
    manifest = manifests_dir / "longbench_v2_generation_manifest.json"
    if sha(manifest.read_bytes()) != protocol["generation_manifest_sha256"]["longbench_v2"]:
        raise ValueError("two-ID diagnostic manifest byte drift")
    model = Path(old["model"])
    metadata = model_hashes(model)
    if metadata != old["model_metadata_hashes"]:
        raise ValueError("old model metadata identity drift")
    source = source_hashes(CONTROLS, old["library"], old["torch_library"], old.get("support_build"))
    for helper in (Path(__file__).resolve(), Path(__file__).with_name('v21_run.py').resolve()):
        source[str(helper)] = sha(helper.read_bytes())
    base = dict(old)
    base.update(phase=protocol["protocol_id"], ids=protocol["ids"]["longbench_v2"], seeds=[101, 202],
                manifest=str(manifest.resolve()),
                manifest_sha256=protocol["generation_manifest_sha256"]["longbench_v2"],
                source_commit=source_commit, source_hashes=source, model_metadata_hashes=metadata,
                timing_events=False, diagnostic=False)
    for k in ("condition", "plugin", "v20_scope", "output_mode", "fingerprint"):
        base.pop(k, None)
    native = control_config(base, "native_dense", scope)
    legacy = control_config(dict(base, control='D_matched'), "v20_dense_consumer", scope)
    newer = v21.effective_control_config(base, scope,
                                         output_score_precision="fp32_scores_bf16_pv",
                                         output_layout="head_major")
    family = dict(zip(ARMS, (native, legacy, newer)))
    config_dir = out.with_name(out.stem + "_configs")
    if out.exists() or config_dir.exists():
        raise FileExistsError("host fragment/config output already exists")
    paths = {}
    for arm, config in family.items():
        path = config_dir / "longbench_v2" / (arm + ".json")
        data = canonical(config)
        atomic_same(path, data)
        paths[arm] = {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()}
    fragment = {"schema": "v21_conditional_binding_fragment_v1", "status": "host_fragment",
                "panel_protocol_sha256": sha(protocol_path.read_bytes()),
                "host": host, "gpu_uuid": next(h["gpu_uuid"] for h in HOSTS if h["host"] == host),
                "host_models": {host: str(model.resolve())},
                "host_configs": {host: {"longbench_v2": paths}},
                "policy_sha256": old["policy_sha256"], "policy_point": "P0",
                "source_commit": source_commit, "old_native_config_sha256": old_native_sha256,
                "quality_eligible": False, "timing_eligible": False}
    atomic_same(out, canonical(fragment))
    return fragment


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--v20-protocol", type=Path, required=True)
    f.add_argument("--v20-binding", type=Path, required=True)
    f.add_argument("--v20-lb-manifest", type=Path, required=True)
    f.add_argument("--hosts-json", type=Path, help="optional exact qualified mpk/dllm descriptor list")
    f.add_argument("--out-dir", type=Path, required=True)
    b = sub.add_parser("bind")
    for name in ("protocol", "old-native-config", "manifests-dir", "out"):
        b.add_argument("--" + name, type=Path, required=True)
    b.add_argument("--old-native-sha256", required=True)
    b.add_argument("--host", required=True)
    b.add_argument("--scope", choices=("ALL_NATIVE_LEGAL", "GLOBAL_ONLY_NATIVE_LOCAL"), required=True)
    b.add_argument("--source-commit", required=True)
    a = p.parse_args()
    if a.action == "freeze":
        result = freeze(a.v20_protocol, a.v20_binding, a.v20_lb_manifest, a.out_dir, a.hosts_json)
        print(json.dumps({"protocol_id": result["protocol_id"], "executions": result["planned_executions"],
                          "hosts": HOSTS}))
    else:
        result = bind(a.protocol, a.old_native_config, a.old_native_sha256,
                      a.manifests_dir, a.host, a.scope, a.source_commit, a.out)
        print(json.dumps({"host": result["host"], "protocol_sha256": result["panel_protocol_sha256"],
                          "configs": len(result["host_configs"][a.host]["longbench_v2"])}))


if __name__ == "__main__":
    main()
