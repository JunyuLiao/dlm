"""Fail-closed worker for a separately frozen conditional v21 panel.

This module does not freeze a panel. CPU validation precedes CUDA imports and
no missing execution is retried after an uncertain worker launch or partial
question-seed block. Original first receipts and failed run rows are immutable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import time
from collections import Counter
from pathlib import Path

from scripts.v13_seed_runs import arm_config_hash, execution_key, is_device_error, receipt_path, run_schedule
from scripts.v18_evaluate import native_phase_evidence, run_complete_blocks, strict_warm
from scripts.v18_protocol import GOLD_KEYS, read_rows, sha
from scripts.v20_run import router_phase_evidence

SCHEMA = "v21_conditional_panel_v1"
BINDING_SCHEMA = "v21_conditional_binding_v1"
PANEL_COUNTS = {"numeric7": (7, 448), "layout_pair": (2, 96), "bootstrap6": (6, 240),
                "bootstrap6_ruler": (6, 90), "aligned6_pilot": (6, 90), "aligned_bridge": (2, 16), "v26_seven": (7, 84),
                "zero_pruning_diagnostic": (3, 12)}
V20_PROTOCOL = Path(__file__).resolve().parents[1] / "results/fan_m1_m3_multidataset_20260927/frozen_protocol.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_task_rows(path: Path, dataset: str, ids: list[str], expected_sha: str) -> list[dict]:
    raw = path.read_bytes()
    if sha(raw) != expected_sha:
        raise ValueError(f"generation manifest byte drift: {dataset}")
    if not ids:
        if raw != b"[]\n":
            raise ValueError(f"empty task manifest is not canonical []: {dataset}")
        return []
    loaded = read_rows(path)
    if len(loaded) != len(ids) or {r["id"] for r in loaded} != set(ids):
        raise ValueError(f"manifest selected ID coverage drift: {dataset}")
    return loaded


def validate_protocol(protocol: dict) -> None:
    if protocol.get("schema") != SCHEMA or protocol.get("status") != "frozen" or protocol.get("execution_ready") is not True:
        raise ValueError("v21 conditional panel is not frozen and execution-ready")
    v27 = protocol.get("panel_kind") == "v27_panel"
    if protocol.get("panel_kind") not in PANEL_COUNTS and not v27:
        raise ValueError("unknown conditional panel kind")
    arms, schedule = protocol.get("arms"), protocol.get("schedule")
    if v27:
        # v27 generic panel: counts follow from the frozen ids x seeds x arms x roles.
        blocks_expected = sum(len(v) for v in (protocol.get("ids") or {}).values()) * len(protocol.get("seeds") or [])
        arm_count = len(arms) if isinstance(arms, list) else -1
        expected = blocks_expected * arm_count * (2 if protocol.get("warm") else 1)
    else:
        arm_count, expected = PANEL_COUNTS[protocol["panel_kind"]]
    if (not isinstance(arms, list) or len(arms) != arm_count or len(set(arms)) != len(arms) or
            not all(isinstance(a, str) and a for a in arms)):
        raise ValueError("explicit frozen arm inventory differs from panel kind")
    if not isinstance(schedule, list) or len(schedule) != expected or protocol.get("planned_executions") != expected:
        raise ValueError("conditional panel execution count differs from fixed design")
    if not protocol.get("protocol_id") or not protocol.get("model_revision"):
        raise ValueError("protocol/model identity missing")
    contracts = protocol.get("arm_contracts")
    if not isinstance(contracts, dict) or set(contracts) != set(arms):
        raise ValueError("arm contracts must match explicit arm list")
    if sum(c.get("kind") == "native" for c in contracts.values()) > 1:
        raise ValueError("more than one native arm")
    from experiments.numerical_qk_reuse.v27_substrate import SUBSTRATES
    if protocol.get("substrate", "eager") not in SUBSTRATES or ("substrate" in protocol and not v27):
        raise ValueError("unknown execution substrate")
    for arm, c in contracts.items():
        if c.get("kind") == "native":
            if arm != "D_native":
                raise ValueError("native contract must use D_native name")
        elif c.get("kind") == "v20_legacy":
            if c.get("parent_v20_arm") not in ("T_scope", "D_matched") or c.get("scope") not in (
                    "ALL_NATIVE_LEGAL", "GLOBAL_ONLY_NATIVE_LOCAL"):
                raise ValueError("legacy task/control identity drift")
        elif c.get("kind") == "v27_dense":
            pass
        elif c.get("kind") == "v27_sparsed":
            if (not v27 or c.get("parent_v20_arm") != "D_matched" or c.get("keep") not in (0.1, 0.2, 0.3, 0.5, 0.6, 0.7)
                    or c.get("skip_steps") not in (1, 10) or c.get("consumer") != "fa4"
                    or c.get("scope") != "GLOBAL_ONLY_NATIVE_LOCAL"):
                raise ValueError(f"v27 SparseD port contract drift: {arm}")
        elif c.get("kind") == "v27_g75":
            if (not v27 or c.get("parent_v20_arm") != "D_matched" or c.get("local_fraction") not in (0.0, 0.15, 0.3)
                    or c.get("consumer", "triton64") not in ("triton64", "fa4")
                    or c.get("scope") != ("GLOBAL_ONLY_NATIVE_LOCAL" if c.get("local_fraction") == 0.0
                                          else "ALL_NATIVE_LEGAL")):
                raise ValueError(f"v27 G75 port contract drift: {arm}")
        elif c.get("kind") in ("v21_method", "v21_control"):
            if (c.get("scope") not in ("ALL_NATIVE_LEGAL", "GLOBAL_ONLY_NATIVE_LOCAL") or
                    c.get("output_score_precision") not in ("legacy_bf16_scores", "fp32_scores_bf16_pv") or
                    c.get("output_layout") not in ("head_major", "model_major")):
                raise ValueError(f"incomplete v21 arm contract: {arm}")
            if c["kind"] == "v21_method" and c.get("parent_v20_arm") not in (
                    "M1_R1_A8_current_output", "M3_R2_A8_current_output",
                    "M3_R3_A8_current_output", "B_A8_matched"):
                raise ValueError("unrecognized v20 parent arm")
            if c["kind"] == "v21_control" and c.get("parent_v20_arm") != "D_matched":
                raise ValueError("v21 control must be D_matched parent")
            if c.get("route_storage", "aligned16") not in ("aligned16", "aligned16_odd") or \
                    c.get("mu_mode", "pooled") not in ("pooled", "pooled_compact") or \
                    c.get("score_period", 16) not in (16, 64):
                raise ValueError(f"unknown v25 route storage: {arm}")
            if c.get("decision_interval", 6) not in (6, 12) or c.get("hold_only", True) is not True or \
                    c.get("threshold_shift", "plus_ln2") not in (
                        "minus_3ln2", "minus_2ln2", "minus_ln2", "plus_ln2", "plus_2ln2", "plus_3ln2", "plus_4ln2"):
                raise ValueError(f"unknown v27 clock/threshold field: {arm}")
            if c.get("min_route_keys", 8192) not in (2048, 4096, 8192, 16384, 32768):
                raise ValueError(f"unknown v27 length gate: {arm}")
            if (c.get("share_layers", "one") not in ("pairs", "one", "mid3_one", "local_blocks") or
                    c.get("route_layers", "mid3") not in ("mid3", "no_first", "no_last", "late3", "global_plus_local_mid") or
                    c.get("consumer64", 2) not in (1, 2, 4) or c.get("memory_caps", "long") != "long" or
                    c.get("fused_observe", True) is not True or c.get("fresh_fused", True) is not True or
                    c.get("route_pipeline", True) is not True or c.get("risk_state", "dense_prefix") != "dense_prefix" or
                    c.get("density_gate", "ent0.05") not in ("ent0.05", "ent0.02", "cap12", "cap24", "stall2", "stable1",
                                                              "ent0.05_stall2") or
                    c.get("risk_budget", "b0") not in ("b0", "b2ln2", "b4ln2", "b6ln2", "b8ln2") or
                    c.get("risk_topk", "k50") not in ("k70", "k60", "k50", "k30", "k20", "k12", "k5") or
                    c.get("carry_canvases", 2) not in (2, 3, 4, 8) or
                    c.get("observe_step", 2) not in (2, 3) or c.get("protect_output", True) is not True or
                    c.get("carry_first", True) is not True or
                    c.get("proj_rank", 8) not in (4, 8, 16) or c.get("risk_value", "mass") != "mass" or
                    ("risk_topk" in c and (c.get("risk_state") != "dense_prefix" or "risk_budget" in c)) or
                    ("risk_budget" in c and c.get("risk_state") != "dense_prefix")):
                raise ValueError(f"unknown v27 execution field: {arm}")
            if not v27 and any(k in c for k in ("decision_interval", "hold_only", "threshold_shift",
                                                "min_route_keys", "share_layers", "route_layers",
                                                "consumer64", "memory_caps", "fused_observe", "fresh_fused",
                                                "route_pipeline", "risk_state", "density_gate", "risk_budget",
                                                "risk_topk", "carry_canvases", "observe_step", "protect_output",
                                                "carry_first", "proj_rank", "risk_value")):
                raise ValueError(f"v27 fields outside a v27 panel: {arm}")
            if c.get("bootstrap_policy", "native_bootstrap2_observe1") != "native_bootstrap2_observe1" or \
                    c.get("observation_producer", "grouped_q") not in ("grouped_q", "repeat_interleave"):
                raise ValueError(f"unknown v23 method field: {arm}")
        else:
            raise ValueError(f"unknown arm contract: {arm}")
        if c.get("kind") == "v27_dense" and (not v27 or c.get("control") not in ("D_c64", "D_fast", "D_fa4",
                                                                                 "D_fa4_allkept")):
            raise ValueError(f"v27 dense control outside a v27 panel: {arm}")
    diagnostic = protocol["panel_kind"] == "zero_pruning_diagnostic"
    ids = protocol.get("ids")
    required_tasks = (set(ids) if v27 and isinstance(ids, dict) and ids and
                      set(ids) <= set(__import__("scripts.v27_datasets", fromlist=["x"]).DATASETS) else
                      {"longbench_v2"} if diagnostic else {"aime26", "longbench_v2"}
                      if protocol["panel_kind"] == "bootstrap6" else {"ruler4k"}
                      if protocol["panel_kind"] == "bootstrap6_ruler" else {"longbench_v2"}
                      if protocol["panel_kind"] == "aligned_bridge" else {"ruler4k", "aime26", "longbench_v2"})
    pilot = protocol["panel_kind"] == "aligned6_pilot"
    if pilot and protocol.get("seeds_by_dataset") != {"longbench_v2": [101, 202], "aime26": [101, 202],
                                                       "ruler4k": [101]}:
        raise ValueError("aligned6 pilot seed contract drift")
    if not isinstance(ids, dict) or set(ids) != required_tasks:
        raise ValueError("task ID inventory differs from frozen panel kind")
    if 'humaneval' in ids:
        from scripts.v27_humaneval import validate_contract
        validate_contract(protocol.get('task_contracts', {}).get('humaneval'))
    if any(not isinstance(v, list) or len(v) != len(set(v)) for v in ids.values()):
        raise ValueError("task IDs missing or duplicated")
    ruler_stage = protocol["panel_kind"] == "bootstrap6_ruler"
    single_seed = ruler_stage or protocol["panel_kind"] == "v26_seven"
    if v27:
        from scripts.v27_datasets import V27_SEEDS
        seeds = protocol.get("seeds")
        if (not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds) or
                not set(seeds) <= V27_SEEDS):   # 404: final AIME panel; 505-909: E4 large-seed confirmation
            raise ValueError("v27 seeds outside the allowed set")
    elif protocol.get("seeds") != ([101] if single_seed else [101, 202]):
        raise ValueError("frozen v20 generation seeds changed")
    if diagnostic:
        frozen_v20 = _json(V20_PROTOCOL)
        if (protocol.get("v20_protocol_sha256") != sha(V20_PROTOCOL.read_bytes()) or
                ids["longbench_v2"] != frozen_v20["ids"]["longbench_v2"][:2] or
                protocol.get("quality_eligible") is not False or
                protocol.get("timing_eligible") is not False):
            raise ValueError("CP1 natural diagnostic must bind first two frozen LB IDs and forbid panel scoring/timing")
        kinds = Counter(c.get("kind") for c in contracts.values())
        if kinds != {"native": 1, "v20_legacy": 1, "v21_control": 1}:
            raise ValueError("CP1 natural diagnostic requires native, legacy dense and v21 dense control")
        legacy = next(c for c in contracts.values() if c["kind"] == "v20_legacy")
        updated = next(c for c in contracts.values() if c["kind"] == "v21_control")
        if (legacy["parent_v20_arm"] != "D_matched" or updated["parent_v20_arm"] != "D_matched" or
                legacy["scope"] != updated["scope"] or
                updated["output_score_precision"] != "fp32_scores_bf16_pv" or
                updated["output_layout"] != "head_major"):
            raise ValueError("CP1 must compare identical-scope all-kept legacy/new numeric consumers")
    assignments = protocol.get("block_assignments")
    if not isinstance(assignments, dict):
        raise ValueError("block assignments missing")
    blocks = {}
    keys, cell_ids = set(), {}
    for index, entry in enumerate(schedule):
        if not isinstance(entry, dict) or entry.get("index") != index:
            raise ValueError("schedule index drift")
        allowed_roles = ("attempt0",) if diagnostic else ("attempt0", "warm")
        if entry.get("role") not in allowed_roles or entry.get("repeat") != (0 if entry["role"] == "attempt0" else 1):
            raise ValueError("first/warm role or repeat drift")
        if entry.get("arm") not in contracts or entry.get("dataset") not in ids or entry.get("id") not in ids[entry["dataset"]] or entry.get("seed") not in protocol["seeds"]:
            raise ValueError("foreign task/arm/seed in schedule")
        block = entry.get("block")
        if type(block) is not int or block < 0:
            raise ValueError("invalid block index")
        assignment = assignments.get(str(block))
        if not assignment or any(entry.get(k) != assignment.get(k) for k in
                                 ("dataset", "id", "seed", "host", "gpu_uuid")):
            raise ValueError("entry differs from frozen whole-block host/GPU assignment")
        key = execution_key(entry)
        if key in keys:
            raise ValueError("duplicate execution key")
        keys.add(key)
        ckey = (entry["dataset"], entry["id"], entry["seed"], entry["arm"])
        if ckey in cell_ids and cell_ids[ckey] != entry.get("cell_id"):
            raise ValueError("first/warm cell identity differs")
        cell_ids[ckey] = entry.get("cell_id")
        blocks.setdefault(block, []).append(entry)
    if sorted(blocks) != list(range(len(blocks))) or set(assignments) != {str(b) for b in blocks}:
        raise ValueError("blocks not contiguous or assignments foreign")
    seeds_by = protocol.get("seeds_by_dataset") if pilot else None
    expected_qseeds = {(d, q, seed) for d, questions in ids.items() for q in questions
                       for seed in (seeds_by[d] if seeds_by else protocol["seeds"])}
    actual_qseeds = {(a["dataset"], a["id"], a["seed"]) for a in assignments.values()}
    if len(actual_qseeds) != len(assignments) or actual_qseeds != expected_qseeds:
        raise ValueError("frozen panel omits or duplicates a task question-seed block")
    for block, entries in blocks.items():
        warm = (bool(protocol.get("warm")) if v27 else
                not diagnostic and (not (ruler_stage or pilot) or block in protocol.get("warm_blocks", [])))
        required_roles = ["attempt0"] * len(arms) + (["warm"] * len(arms) if warm else [])
        if len(entries) != len(required_roles) or [e["role"] for e in entries] != required_roles:
            raise ValueError("incomplete or interleaved question-seed block")
        first_arms = [e["arm"] for e in entries[:len(arms)]]
        warm_arms = [e["arm"] for e in entries[len(arms):]]
        if set(first_arms) != set(arms) or (warm and warm_arms != first_arms[::-1]):
            raise ValueError("frozen balanced first/reverse warm order drift")
    stages = protocol.get("stages")
    if not isinstance(stages, dict) or not stages:
        raise ValueError("explicit stage-to-block map required")
    stage_blocks = [b for values in stages.values() for b in values]
    if sorted(stage_blocks) != list(range(len(blocks))) or any(not isinstance(v, list) for v in stages.values()):
        raise ValueError("stages must partition all complete blocks")
    if set(protocol.get("generation_manifest_sha256", {})) != set(ids):
        raise ValueError("generation manifests not bound for all tasks")


def validate_arm_config(config: dict, contract: dict, *, model: str, manifest_sha: str,
                        policy_sha: str) -> None:
    from experiments.numerical_qk_reuse import v21
    if contract["kind"] == "native":
        if config.get("condition") != "native_dense" or config.get("plugin"):
            raise ValueError("native config is not native_dense")
        parent = config
    elif contract["kind"] == "v27_sparsed":
        if (config.get("plugin") != "experiments.numerical_qk_reuse.v20_controls:install" or
                config.get("condition") != "v27_sparsed_fa4" or config.get("consumer") != "fa4" or
                config.get("v20_scope") != contract["scope"] or
                config.get("sparsed_keep") != contract["keep"] or
                config.get("sparsed_skip_steps") != contract["skip_steps"]):
            raise ValueError("v27 SparseD port config differs from its contract")
        parent = config
    elif contract["kind"] == "v27_g75":
        g75_consumer = contract.get("consumer", "triton64")
        if (config.get("plugin") != "experiments.numerical_qk_reuse.v20_controls:install" or
                config.get("condition") != ("v27_G75_fa4" if g75_consumer == "fa4" else "v27_G75_c64") or
                config.get("consumer") != g75_consumer or
                config.get("v20_scope") != contract["scope"] or
                config.get("g75_local_fraction") != contract["local_fraction"]):
            raise ValueError("v27 G75 port config differs from its contract")
        parent = config
    elif contract["kind"] == "v27_dense":
        from experiments.numerical_qk_reuse import v27_fast_dense as fast
        if (config.get("plugin") != fast.PLUGIN or config.get("condition") != fast.CONDITION or
                config.get("control") != contract["control"]):
            raise ValueError("v27 dense control config differs from its contract")
        parent = config
    elif contract["kind"] == "v20_legacy":
        if (config.get("v20_scope") != contract["scope"] or
                config.get("v20_arm", config.get("control")) != contract["parent_v20_arm"]):
            raise ValueError("legacy v20 arm/scope drift")
        if (contract["parent_v20_arm"] == "D_matched" and
                (config.get("condition") != "v20_dense_consumer" or
                 config.get("plugin") != "experiments.numerical_qk_reuse.v20_controls:install")):
            raise ValueError("legacy D_matched must use unchanged all-kept v20 control")
        parent = config
    else:
        if config.get("parent_kind") != ("v20_method" if contract["kind"] == "v21_method" else "v20_control"):
            raise ValueError("v21 parent kind drift")
        if (config.get("output_score_precision") != contract["output_score_precision"] or
                config.get("output_layout") != contract["output_layout"]):
            raise ValueError("v21 numeric/layout mode differs from arm contract")
        if config.get("route_storage", "logical") != contract.get("route_storage", "logical"):
            raise ValueError("v25 route storage differs from arm contract")
        if (config.get("mu_mode", "exact") != contract.get("mu_mode", "exact") or
                config.get("score_period", 8) != contract.get("score_period", 8)):
            raise ValueError("v26 mu mode / score period differs from arm contract")
        if (config.get("decision_interval") != contract.get("decision_interval") or
                bool(config.get("hold_only", False)) != bool(contract.get("hold_only", False)) or
                config.get("threshold_shift") != contract.get("threshold_shift") or
                config.get("min_route_keys") != contract.get("min_route_keys") or
                any(config.get(k) != contract.get(k) for k in ("share_layers", "route_layers", "consumer64",
                                                                "memory_caps")) or
                bool(config.get("fused_observe", False)) != bool(contract.get("fused_observe", False)) or
                bool(config.get("fresh_fused", False)) != bool(contract.get("fresh_fused", False)) or
                bool(config.get("route_pipeline", False)) != bool(contract.get("route_pipeline", False)) or
                config.get("risk_state") != contract.get("risk_state") or
                config.get("density_gate") != contract.get("density_gate") or
                config.get("risk_budget") != contract.get("risk_budget") or
                config.get("risk_topk") != contract.get("risk_topk") or
                config.get("carry_canvases") != contract.get("carry_canvases") or
                config.get("observe_step") != contract.get("observe_step") or
                bool(config.get("protect_output", False)) != bool(contract.get("protect_output", False)) or
                bool(config.get("carry_first", False)) != bool(contract.get("carry_first", False)) or
                config.get("proj_rank") != contract.get("proj_rank") or
                config.get("risk_value") != contract.get("risk_value") or
                bool(config.get("fa4_consumer", False)) != bool(contract.get("fa4_consumer", False)) or
                bool(config.get("async_route", False)) != bool(contract.get("async_route", False))):
            raise ValueError("v27 clock/threshold differs from arm contract")
        if (config.get("bootstrap_policy") != contract.get("bootstrap_policy") or
                config.get("observation_producer", "repeat_interleave") !=
                contract.get("observation_producer", "repeat_interleave")):
            raise ValueError("v23 bootstrap/producer differs from arm contract")
        v21.validate_effective(config, config["condition"])
        parent = config["parent_config"]
        if (parent.get("v20_scope") != contract["scope"] or
                parent.get("v20_arm", "D_matched") != contract["parent_v20_arm"]):
            raise ValueError("nested v20 parent identity drift")
    if (parent.get("model") != model or parent.get("manifest_sha256") != manifest_sha or
            parent.get("policy_sha256") != policy_sha):
        raise ValueError("nested parent model/manifest/policy drift")
    if not parent.get("source_hashes") or not parent.get("model_metadata_hashes"):
        raise ValueError("source/model metadata hashes missing")
    for source, expected in parent["source_hashes"].items():
        if sha(Path(source).read_bytes()) != expected:
            raise ValueError(f"parent source byte drift: {source}")
    for filename, expected in parent["model_metadata_hashes"].items():
        if sha((Path(model) / filename).read_bytes()) != expected:
            raise ValueError(f"model metadata drift: {filename}")


def check_task_row(dataset: str, row: dict) -> None:
    """Task contract per row: RULER (any length) runs without thinking at its task budget;
    AIME/LB (any length) run with thinking at 8192."""
    from scripts.v27_datasets import base_task
    ruler = base_task(dataset) == "ruler4k"
    if row.get("thinking") is not (not ruler):
        raise ValueError("task thinking setting drift")
    if ruler and row.get("generation_budget") not in (30, 32, 50, 120, 128):
        raise ValueError("RULER task generation budget drift")
    if not ruler and row.get("generation_budget") != 8192:
        raise ValueError("AIME/LB generation budget drift")


def validate_inputs(protocol_path: Path, binding_path: Path, manifests_dir: Path,
                    host: str, gpu_uuid: str, *, stage: str) -> tuple[dict, dict, dict, dict]:
    protocol, binding = _json(protocol_path), _json(binding_path)
    validate_protocol(protocol)
    if (binding.get("schema") != BINDING_SCHEMA or binding.get("status") != "frozen" or
            binding.get("panel_protocol_sha256") != sha(protocol_path.read_bytes())):
        raise ValueError("v21 binding not frozen to exact panel bytes")
    if stage != "all" and stage not in protocol["stages"]:
        raise ValueError("unknown frozen stage")
    if not any(a["host"] == host and a["gpu_uuid"] == gpu_uuid for a in protocol["block_assignments"].values()):
        raise ValueError("host/GPU absent from frozen assignments")
    model = binding.get("host_models", {}).get(host)
    if not model or not Path(model).is_dir():
        raise ValueError("bound model directory missing")
    policy_sha = binding.get("policy_sha256")
    if not isinstance(policy_sha, str) or len(policy_sha) != 64:
        raise ValueError("frozen policy hash missing")
    configs = {}
    for dataset in protocol["ids"]:
        configs[dataset] = {}
        if not protocol["ids"][dataset]:
            continue
        for arm in protocol["arms"]:
            item = binding.get("host_configs", {}).get(host, {}).get(dataset, {}).get(arm)
            if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
                raise ValueError(f"missing host config {dataset}/{arm}")
            path = Path(item["path"])
            if sha(path.read_bytes()) != item["sha256"]:
                raise ValueError(f"host config byte drift {dataset}/{arm}")
            config = _json(path)
            validate_arm_config(config, protocol["arm_contracts"][arm], model=model,
                                manifest_sha=protocol["generation_manifest_sha256"][dataset], policy_sha=policy_sha)
            configs[dataset][arm] = config
    rows = {}
    for dataset, ids in protocol["ids"].items():
        path = manifests_dir / f"{dataset}_generation_manifest.json"
        loaded = read_task_rows(path, dataset, ids, protocol["generation_manifest_sha256"][dataset])
        for row in loaded:
            if any(k in row for k in GOLD_KEYS) or row.get("prompt_hash") != sha(row["prompt"]):
                raise ValueError("gold/prompt identity drift")
            if not isinstance(row.get("prompt_tokens"), list) or row.get("prompt_token_count") != len(row["prompt_tokens"]):
                raise ValueError("exact prompt token list missing")
            check_task_row(dataset, row)
            rows[row["id"]] = row
    return protocol, binding, rows, configs


def stage_entries(protocol: dict, host: str, gpu_uuid: str, stage: str) -> list[dict]:
    blocks = set(range(len(protocol["block_assignments"]))) if stage == "all" else set(protocol["stages"][stage])
    return [e for e in protocol["schedule"] if e["block"] in blocks and e["host"] == host and e["gpu_uuid"] == gpu_uuid]


def validate_resume_events(events: list[dict], identity: dict, full_entries: list[dict], private: Path) -> set[str]:
    """Require closed launches and only whole complete local blocks before continuation."""
    expected = {execution_key(e): e for e in full_entries}
    done, starts, ends = set(), 0, 0
    for event in events:
        kind = event.get("event")
        if kind == "start":
            starts += 1
            if starts != ends + 1 or any(event.get(k) != v for k, v in identity.items()):
                raise ValueError("uncertain or identity-drifted worker launch")
        elif kind == "worker_end":
            ends += 1
            if ends != starts or event.get("host") != identity["host"] or event.get("gpu_uuid") != identity["gpu_uuid"]:
                raise ValueError("unmatched worker terminal receipt")
        elif kind == "run":
            key = event.get("execution_key")
            spec = expected.get(key)
            if starts != ends + 1 or spec is None or key in done:
                raise ValueError("foreign, duplicate, or writerless execution")
            if any(event.get(k) != spec[k] for k in ("index", "block", "dataset", "arm", "id", "seed", "role", "repeat", "cell_id", "host", "gpu_uuid")):
                raise ValueError("run receipt differs from frozen schedule")
            if event.get("generation_seed") != spec["seed"]:
                raise ValueError("actual generation seed drift")
            done.add(key)
    if starts != ends:
        raise ValueError("prior worker launch has no terminal receipt; do not retry uncertain cell")
    if done and not starts:
        raise ValueError("run rows lack source/identity start receipt")
    seen_incomplete = False
    for block in dict.fromkeys(e["block"] for e in full_entries):
        entries = [e for e in full_entries if e["block"] == block]
        present = [execution_key(e) in done for e in entries]
        if any(present) and not all(present):
            raise ValueError("partial question-seed block: preserve first/failure and stop")
        # v23: frozen stages may run in their declared order (e.g. lb_preview,
        # aime, lb_rest), so whole completed blocks may follow a gap. Partial
        # blocks, duplicates, foreign rows and unclosed launches still fail.
        if not any(present):
            seen_incomplete = True
    for entry in full_entries:
        if execution_key(entry) not in done and receipt_path(private, entry).exists():
            raise ValueError("orphan immutable receipt; uncertain execution")
    return done


def strict_v21_warm(first: dict | None, warm: dict) -> dict:
    result = strict_warm(first, warm)
    reasons = list(result["reasons"])
    if not first or first.get("router_phase_evidence") is None or warm.get("router_phase_evidence") is None:
        reasons.append("router_phase_evidence_absent")
    elif first["router_phase_evidence"] != warm["router_phase_evidence"]:
        reasons.append("router_phase_evidence_mismatch")
    return {"accepted": not reasons, "reasons": sorted(set(reasons))}


def per_canvas_stopping(receipt: dict) -> list[dict] | None:
    canvases = receipt.get("per_canvas")
    if not isinstance(canvases, list) or not canvases:
        return None
    out = []
    for canvas in canvases:
        native, cap = canvas.get("native_stop_final_call"), canvas.get("iteration_cap_final_call")
        if type(native) is not bool or type(cap) is not bool:
            return None
        out.append({"native_stop": native, "iteration_cap": cap})
    return out


def phase_for(config: dict, receipt: dict, contract: dict) -> dict | None:
    parent = config.get("parent_config", config)
    arm = contract.get("parent_v20_arm", "D_native")
    counters = receipt.get("counters")
    if arm == "D_native":
        return {"phase": "native_dense"}
    evidence = router_phase_evidence(arm, counters)
    if evidence is not None and contract.get("bootstrap_policy"):
        dense, observed = counters.get("bootstrap_dense_calls"), counters.get("bootstrap_observation_calls")
        if type(dense) is not int or type(observed) is not int or counters.get("score_clock_origin") != 1:
            return None
        evidence = dict(evidence, bootstrap_dense=dense, bootstrap_observe=observed)
    return evidence


def run(protocol_path: Path, binding_path: Path, manifests_dir: Path, private: Path, ledger: Path, lock_path: Path,
        *, host: str, stage: str, deadline_epoch: float, gpu_budget_s: float, remaining_requests: int,
        block_guard_s: float, timeout: int = 900) -> str:
    import fcntl
    uuid = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()[0].strip()
    protocol, binding, rows, configs = validate_inputs(protocol_path, binding_path, manifests_dir, host, uuid, stage=stage)
    full_entries = stage_entries(protocol, host, uuid, "all")
    entries = stage_entries(protocol, host, uuid, stage)
    if not entries:
        return "no_assigned_blocks"
    if (any(not math.isfinite(v) or v <= 0 for v in (deadline_epoch, gpu_budget_s, block_guard_s)) or
            remaining_requests < 1 or timeout < 1):
        raise ValueError("finite positive resource guards required")
    source_hashes = {}
    for family in configs.values():
        for config in family.values():
            for owner in (config, config.get("parent_config", {})):
                for path, value in owner.get("source_hashes", {}).items():
                    if path in source_hashes and source_hashes[path] != value:
                        raise ValueError("conflicting nested source hash")
                    source_hashes[path] = value
    identity = {"protocol_id": protocol["protocol_id"], "model_revision": protocol["model_revision"],
                "arm_hashes": {f"{d}/{a}": arm_config_hash(c) for d, family in configs.items() for a, c in family.items()},
                "source_hashes": source_hashes, "binding_sha256": sha(binding_path.read_bytes()),
                "private_root": str(private.resolve()), "host": host, "gpu_uuid": uuid}
    substrate = protocol.get("substrate", "eager")
    if substrate != "eager":
        # execution substrate is part of the launch identity (resume refuses a drifted substrate module)
        from experiments.numerical_qk_reuse import v27_substrate
        identity["substrate"] = substrate
        identity["substrate_source_sha256"] = sha(Path(v27_substrate.__file__).resolve().read_bytes())
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        events = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()] if ledger.exists() else []
        done = validate_resume_events(events, identity, full_entries, private)
        if all(execution_key(e) in done for e in entries):
            return "complete_prefix"
        import torch
        from dllm.models import create_adapter
        from experiments.numerical_qk_reuse.runner import _atomic, _one
        from scripts.v9_clean_request_timing import append, disk_cache_entries, redacted
        from scripts.v15_seed_runs import mapped_shared_objects
        from triton.runtime.jit import JITFunction
        started = time.perf_counter()
        append(ledger, {"event": "start", "when": time.time(), "pid": os.getpid(), **identity})
        status, error = "error", None
        try:
            adapter = create_adapter("diffusion_gemma", binding["host_models"][host], device="cuda",
                                     precision="bfloat16", revision=protocol["model_revision"]).load()
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if substrate != "eager":
                append(ledger, {"event": "substrate", "when": time.time(), "host": host,
                                **v27_substrate.install(adapter.model, name=substrate)})
            compiles = []
            JITFunction.cache_hook = lambda **kw: (compiles.append(time.perf_counter()), False)[1]
            first = {e["cell_id"]: e for e in events if e.get("event") == "run" and e.get("role") == "attempt0"}

            class Timeout(Exception):
                pass

            def alarm(*_):
                raise Timeout()

            signal.signal(signal.SIGALRM, alarm)

            def execute_one(row, seed, config, entry):
                misses, cache_before, so_before = len(compiles), disk_cache_entries(), mapped_shared_objects()
                receipt, execution_error = None, None
                before = time.perf_counter()
                signal.alarm(timeout)
                graphs_before = None
                try:
                    from experiments.numerical_qk_reuse.v27_long import prefill_dense64
                    if substrate != "eager":
                        v27_substrate.set_local(adapter.model, v27_substrate.local_mode_for(config))
                        graphs_before = v27_substrate.identity(adapter.model).get("dynamo_unique_graphs")
                    with prefill_dense64(adapter.model, os.environ.get("V27_PREFILL_DENSE64") == "1",
                                        kernel=__import__("experiments.numerical_qk_reuse.v27_substrate", fromlist=["x"]).prefill_kernel(substrate)):
                        receipt = _one(adapter, row, seed, config)
                except Timeout:
                    execution_error = f"timeout>{timeout}s"
                except Exception as exc:
                    execution_error = f"{type(exc).__name__}: {exc}"[:500]
                finally:
                    signal.alarm(0)
                record = {"ok": receipt is not None, "error": execution_error, "generation_seed": seed,
                          "arm_config_hash": identity["arm_hashes"][entry["dataset"] + "/" + entry["arm"]],
                          "triton_misses": len(compiles) - misses,
                          "triton_disk_entries_added": disk_cache_entries() - cache_before,
                          "new_shared_objects": sorted(mapped_shared_objects() - so_before),
                          "outer_wall_s": time.perf_counter() - before, "host": host, "gpu_uuid": uuid,
                          "dataset": entry["dataset"],
                          "quality_eligible": protocol.get("quality_eligible", True),
                          "timing_eligible": protocol.get("timing_eligible", True)}
                if substrate != "eager":
                    record["substrate"] = v27_substrate.identity(adapter.model)
                    after = record["substrate"].get("dynamo_unique_graphs")
                    # a timed request must not compile: new graphs here would put compile time into its wall
                    record["substrate_new_graphs"] = (None if graphs_before is None or after is None
                                                      else after - graphs_before)
                if receipt is not None:
                    if receipt.get("seed") != seed:
                        raise ValueError("actual generation seed differs")
                    record.update(redacted(receipt))
                    record["per_canvas_stopping"] = per_canvas_stopping(receipt)
                    parent = config.get("parent_config", config)
                    native = parent.get("condition") == "native_dense"
                    record["phase_evidence"] = native_phase_evidence(receipt, sparse=not native)
                    if record["phase_evidence"] is not None:
                        record["phase_evidence"]["phase"] = "native_dense_decoder" if native else "v21_current_output_decoder"
                    record["router_phase_evidence"] = phase_for(config, receipt,
                                                                 protocol["arm_contracts"][entry["arm"]])
                if entry["role"] == "warm":
                    record["acceptance"] = strict_v21_warm(first.get(entry["cell_id"]), record)
                    if receipt is not None:
                        path = receipt_path(private, entry)
                        _atomic(path, receipt)
                        record["private_receipt"] = str(path)
                else:
                    first[entry["cell_id"]] = record
                return {"record": record, "receipt": receipt,
                        "fatal": bool(execution_error and is_device_error(execution_error))}

            def execute_block(block_entries):
                dataset = block_entries[0]["dataset"]
                if any(e["dataset"] != dataset for e in block_entries):
                    raise ValueError("mixed-dataset block")
                return run_schedule(block_entries, done=done, execute_one=execute_one, rows=rows,
                                    configs=configs[dataset], ledger_append=lambda r: append(ledger, r),
                                    private=private, save_receipt=_atomic)

            if substrate != "eager":
                # untimed warm-up: one short request per arm compiles/captures every graph variant it needs
                # (compile time must not land in a timed request). Outputs are discarded; only a ledger event.
                from experiments.numerical_qk_reuse.v27_long import prefill_dense64
                # two passes on two different prompts (when the stage has them): graph variants that depend on the
                # previous arm's LOCAL mode or on the prompt are then compiled here, not in a timed request
                warm_ids = list(dict.fromkeys(e["id"] for e in entries))[:2]
                for warm_id in warm_ids:
                    for arm in protocol["arms"]:
                        entry = next((e for e in entries if e["arm"] == arm and e["id"] == warm_id), None)
                        if entry is None:
                            continue
                        config = configs[entry["dataset"]][arm]
                        row = dict(rows[entry["id"]])
                        row["generation_budget"] = min(512, int(row.get("generation_budget", 8192)))
                        v27_substrate.set_local(adapter.model, v27_substrate.local_mode_for(config))
                        graphs_before = v27_substrate.identity(adapter.model).get("dynamo_unique_graphs")
                        before, warm_error = time.perf_counter(), None
                        try:
                            with prefill_dense64(adapter.model, os.environ.get("V27_PREFILL_DENSE64") == "1",
                                        kernel=__import__("experiments.numerical_qk_reuse.v27_substrate", fromlist=["x"]).prefill_kernel(substrate)):
                                _one(adapter, row, entry["seed"], config)
                        except Exception as exc:
                            warm_error = f"{type(exc).__name__}: {exc}"[:500]
                        ident = v27_substrate.identity(adapter.model)
                        append(ledger, {"event": "substrate_warmup", "arm": arm, "dataset": entry["dataset"],
                                        "warm_id_index": warm_ids.index(warm_id),
                                        "when": time.time(), "wall_s": time.perf_counter() - before,
                                        "new_graphs": (None if graphs_before is None else
                                                       ident.get("dynamo_unique_graphs") - graphs_before),
                                        "error": warm_error, "host": host, **ident})
                        if warm_error and is_device_error(warm_error):
                            raise RuntimeError(warm_error)
            status = run_complete_blocks(entries, done, deadline_epoch=deadline_epoch, gpu_budget_s=gpu_budget_s,
                                         remaining_requests=remaining_requests, block_guard_s=block_guard_s,
                                         process_started=started, execute_block=execute_block)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"[:500]
            raise
        finally:
            try:
                from triton.runtime.jit import JITFunction
                JITFunction.cache_hook = None
            except ImportError:
                pass
            append(ledger, {"event": "worker_end", "status": status, "error": error,
                            "when": time.time(), "host": host, "gpu_uuid": uuid,
                            "gpu_process_seconds": time.perf_counter() - started})
    return status


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "binding", "manifests-dir", "private", "ledger", "lock"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--host", required=True)
    p.add_argument("--stage", required=True, help="frozen protocol stage name or all")
    p.add_argument("--deadline-epoch", type=float, required=True)
    p.add_argument("--gpu-budget-s", type=float, required=True)
    p.add_argument("--remaining-requests", type=int, required=True)
    p.add_argument("--block-guard-s", type=float, required=True)
    p.add_argument("--timeout", type=int, default=900)
    a = p.parse_args()
    print(run(a.protocol, a.binding, a.manifests_dir, a.private, a.ledger, a.lock, host=a.host,
              stage=a.stage, deadline_epoch=a.deadline_epoch, gpu_budget_s=a.gpu_budget_s,
              remaining_requests=a.remaining_requests, block_guard_s=a.block_guard_s, timeout=a.timeout))


if __name__ == "__main__":
    main()
