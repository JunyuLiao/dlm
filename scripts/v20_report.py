"""Redacted v20 direct-profile cost table and measured-status Markdown (CPU only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


METHOD_ARMS = frozenset(("M1_R1_A8_current_output", "M3_R2_A8_current_output",
                         "M3_R3_A8_current_output", "B_A8_matched"))
CONTROL_CONDITIONS = {"v20_dense_consumer": "D_matched", "v20_fresh_T": "T_scope",
                      "v20_G75L30_nativeQ128": "G75L30_nativeQ128"}


def canonical_arm(arm: dict | str) -> str | None:
    """Use frozen execution identity, never a scope/policy-prefixed display name."""
    if isinstance(arm, str):
        return arm if arm in METHOD_ARMS | {"D_native", "D_matched", "T_scope", "G75L30_nativeQ128"} else None
    config = arm.get("config", {})
    condition = arm.get("condition")
    if condition == "native_dense":
        return "D_native"
    if condition in CONTROL_CONDITIONS and arm.get("plugin") == "experiments.numerical_qk_reuse.v20_controls:install":
        return CONTROL_CONDITIONS[condition]
    method = config.get("v20_arm")
    if (method in METHOD_ARMS and arm.get("plugin") == "experiments.numerical_qk_reuse.v20:install"
            and condition == config.get("condition")):
        return method
    return None


def _phase(delta: dict, arm: dict | str) -> str | None:
    """Classify a complete model call from all of its routed-layer counters."""
    method = canonical_arm(arm)
    a = delta.get("score_refresh_calls")
    d = delta.get("decision_refresh_calls")
    h = delta.get("held_decision_calls")
    calls = delta.get("attention_calls")
    if method in METHOD_ARMS and all(type(x) is int and x >= 0 for x in (a, d, h, calls)):
        if a > 0 and d == a and h == 0 and calls == a:
            return "A"
        if a == 0 and d > 0 and h == 0 and calls == d:
            return "D"
        if a == 0 and d == 0 and h > 0 and calls == h:
            return "H"
        if calls > 0 or a + d + h > 0:
            return "MIXED"
    if type(calls) is not int or calls <= 0:
        return None
    if method == "T_scope":
        return "FRESH"
    if method in ("D_matched", "G75L30_nativeQ128"):
        bootstrap = delta.get("bootstrap_calls")
        observation = delta.get("bitmap_observation_calls")
        held = delta.get("held_decision_calls")
        if all(type(x) is int and x >= 0 for x in (bootstrap, observation, held)):
            if method == "D_matched" and bootstrap == calls and observation == held == 0:
                return "FRESH"
            if method == "G75L30_nativeQ128":
                if bootstrap == calls and observation == held == 0:
                    return "FRESH"
                if observation == calls and bootstrap == held == 0:
                    return "A"
                if held == calls and bootstrap == observation == 0:
                    return "H"
            return "MIXED"
    return None


def _phase_costs(summary: dict, arm: dict | str) -> dict:
    phases = summary.get("phase_deltas", [])
    costs = summary.get("per_call_event_median_ms", [])
    if len(phases) != len(costs):
        raise ValueError("phase/call timing length mismatch")
    grouped = defaultdict(list)
    for delta, cost in zip(phases, costs):
        label = _phase(delta, arm)
        if label:
            grouped[label].append((cost, delta))
    return {name: dict(calls=len(grouped[name]), median_complete_call_event_ms=(
                statistics.median(x[0] for x in grouped[name]) if grouped[name] else None),
                counter_totals={key: sum(x[1].get(key, 0) for x in grouped[name])
                                for key in ("attention_calls", "score_refresh_calls", "decision_refresh_calls",
                                            "held_decision_calls", "bootstrap_calls", "bitmap_observation_calls")
                                if name == "MIXED" and any(key in x[1] for x in grouped[name])})
            for name in ("A", "D", "H", "FRESH", "MIXED")}


def extract_profile(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if data.get("schema") != "v20_direct_full_forward_v1" or not isinstance(data.get("targets"), dict):
        raise ValueError("not a completed v20 direct profile")
    identity = sha(path.read_bytes())
    environment = data.get("runtime_identity", {})
    host_identity = dict(host=environment.get("hostname"), gpu_uuid=environment.get("gpu_uuid"),
                         gpu_name=environment.get("gpu"), torch_version=environment.get("torch"))
    seen = set()
    rows = []
    for target in data["targets"].values():
        state = (target["dataset"], target["id"], target["canvas"])
        state_hash = hashlib.sha256(json.dumps(state, separators=(",", ":")).encode()).hexdigest()[:12]
        resolution = target.get("resolution", {})
        if resolution.get("missing"):
            rows.append(dict(host_identity, profile_sha256=identity, dataset=state[0], state=state_hash,
                             canvas=state[2], requested_call=target.get("requested_call"),
                             status="missing_native_state", reason="native stopped before requested/fallback call"))
            continue
        if state in seen:
            continue  # profiler aliases every reached target on a canvas to the same timed sequence
        seen.add(state)
        available = target.get("boundaries", {})
        for boundary in data.get("selected_boundaries", ("model_forward", "denoising_step")):
            for length in data.get("selected_sequence_lengths", (4, 16)):
                sequence = f"N{length}"
                if sequence not in available.get(boundary, {}):
                    rows.append(dict(host_identity, profile_sha256=identity, dataset=state[0], state=state_hash,
                                     boundary=boundary, sequence=sequence, status="missing_timing",
                                     reason="native sequence too short or profile checkpoint incomplete"))
        for boundary, sequences in available.items():
            for sequence, result in sequences.items():
                bracket = result.get("native_bracket_drift", [])
                drift = [abs(b["close_median_ms"] / b["open_median_ms"] - 1)
                         for b in bracket if b.get("open_median_ms", 0) > 0]
                native_name = data["arms"][0]["name"]
                native_item = result.get("arms", {}).get(native_name, {})
                native = native_item.get("summary")
                native_epoch = native_item.get("direct_epoch")
                native_blocks = native_item.get("blocks", [])
                native_valid = bool(native_blocks) and all(b.get("triton_misses") == 0 and
                    b.get("triton_specializations_before") == b.get("triton_specializations_after") and
                    b.get("triton_disk_entries_before") == b.get("triton_disk_entries_after") for b in native_blocks)
                for arm in data["arms"]:
                    name = arm["name"]
                    item = result.get("arms", {}).get(name)
                    if item is None or item.get("summary") is None:
                        rows.append(dict(host_identity, profile_sha256=identity, dataset=state[0], state=state_hash,
                                         boundary=boundary, sequence=sequence, arm=name,
                                         status="missing_timing", reason="no accepted complete-call summary"))
                        continue
                    summary = item["summary"]
                    blocks = item.get("blocks", [])
                    jit_ok = bool(blocks) and all(b.get("triton_misses") == 0 and
                        b.get("triton_specializations_before") == b.get("triton_specializations_after") and
                        b.get("triton_disk_entries_before") == b.get("triton_disk_entries_after") for b in blocks)
                    direct_epoch = item.get("direct_epoch", {})
                    value = summary.get("event_sum_median_ms")
                    nvalue = None if native is None else native.get("event_sum_median_ms")
                    epoch = direct_epoch.get("event_median_ms")
                    nepoch = None if native_epoch is None else native_epoch.get("event_median_ms")
                    scope = arm.get("config", {}).get("v20_scope", "NATIVE" if arm.get("condition") == "native_dense" else "unknown")
                    policy = arm.get("config", {}).get("policy_sha256")
                    rows.append(dict(host_identity, profile_sha256=identity, dataset=state[0], state=state_hash,
                                     canvas=state[2], requested_call=target.get("requested_call"),
                                     selected_call=resolution.get("selected_call"), fallback_used=resolution.get("fallback_used"),
                                     scope=scope, policy_sha256=policy, arm=name,
                                     canonical_arm=canonical_arm(arm), condition=arm.get("condition"),
                                     plugin=arm.get("plugin"), consumer=arm.get("config", {}).get("consumer"),
                                     boundary=boundary, sequence=sequence,
                                     reached_calls=result.get("reached_calls"), requested_calls=result.get("requested_calls"),
                                     status="measured" if jit_ok else "invalid_new_jit_or_missing_blocks",
                                     complete_call_event_sum_median_ms=value,
                                     direct_epoch_event_median_ms=epoch,
                                     complete_call_wall_sum_median_ms=summary.get("wall_sum_median_ms"),
                                     direct_epoch_wall_median_ms=direct_epoch.get("wall_median_ms"),
                                     within_gpu_complete_call_ratio_to_native=(value / nvalue if jit_ok and native_valid and nvalue and native_name != name else
                                                                                1.0 if jit_ok and native_valid and native_name == name else None),
                                     within_gpu_direct_epoch_ratio_to_native=(epoch / nepoch if jit_ok and native_valid and nepoch and native_name != name else
                                                                               1.0 if jit_ok and native_valid and native_name == name else None),
                                     phase_complete_call_costs=_phase_costs(summary, arm),
                                     native_bracket_max_abs_drift= max(drift) if drift else None,
                                     peak_allocated_bytes=max((b["peak_allocated_bytes"] for b in blocks
                                                               if isinstance(b.get("peak_allocated_bytes"), int)), default=None),
                                     accepted_repetitions=len(blocks), jit_valid=jit_ok,
                                     timing_definition="CUDA stream elapsed span including host launch gaps; model load and snapshot prep excluded"))
    return rows


def render(rows: list[dict], scored: dict | None = None) -> str:
    lines = ["# Fan M1/M3 same-day measurement draft", "",
             "Direct profile rows are same-state teacher-forced calls, not natural request speedups. Ratios are method/native within one profile/GPU; <1 is faster. CUDA event spans include host launch gaps. Missing values stay N/A.", ""]
    if scored is None:
        lines += ["Scored first84 and full-panel status: N/A (no frozen scored summary supplied).", ""]
    else:
        for subset in ("first84", "full"):
            value = scored.get(subset, {})
            lines += [f"{subset}: {value.get('recorded_executions', 0)}/{value.get('planned_executions', '?')} execution rows recorded (including failures); "
                      f"{value.get('recorded_all14_blocks', 'N/A')}/{value.get('planned_blocks', '?')} blocks have all 14 rows, "
                      f"{value.get('successful_first_all7_blocks', 'N/A')} have seven successful first outputs, "
                      f"{value.get('strictwarm_all7_blocks', 'N/A')} have seven accepted warm runs, "
                      f"and {value.get('complete_valid_pair_blocks', 'N/A')} are complete valid paired blocks. "
                      f"Failed blocks: {value.get('failed_block_ids', 'N/A')}; partial blocks: {value.get('partial_block_ids', 'N/A')}. "
                      "Scored results remain dataset-separated in the bound summary.", ""]
    lines += ["| Host | GPU UUID | Dataset | State | Scope | Policy SHA | Arm | Boundary | Sequence | Phase complete-call ms (count) | Sum of calls ms | Direct epoch ms | Sum ratio/native | Epoch ratio/native | Bracket drift | Peak GiB | Validity |",
              "|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    def fmt(v, digits=3):
        return "N/A" if v is None else f"{v:.{digits}f}"
    for r in rows:
        if r["status"] != "measured":
            lines.append(f"| {r.get('host') or 'N/A'} | {r.get('gpu_uuid') or 'N/A'} | {r['dataset']} | {r['state']} | N/A | N/A | {r.get('arm', 'N/A')} | {r.get('boundary', 'N/A')} | {r.get('sequence', 'N/A')} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | {r['status']} |")
            continue
        phases = r["phase_complete_call_costs"]
        phase_parts = []
        for phase in ("A", "D", "H", "FRESH", "MIXED"):
            item = phases[phase]
            if item["calls"]:
                part = f"{phase}:{fmt(item['median_complete_call_event_ms'])} ({item['calls']})"
                if phase == "MIXED":
                    counts = item["counter_totals"]
                    part += " [" + ", ".join(f"{key}={value}" for key, value in counts.items()) + "]"
                phase_parts.append(part)
        phase_text = "; ".join(phase_parts) or "N/A"
        peak = r["peak_allocated_bytes"] / 1024**3 if r["peak_allocated_bytes"] is not None else None
        lines.append(f"| {r.get('host') or 'N/A'} | {r.get('gpu_uuid') or 'N/A'} | {r['dataset']} | {r['state']} | {r['scope']} | {(r['policy_sha256'] or 'N/A')[:12]} | {r['arm']} | {r['boundary']} | {r['sequence']} ({r['reached_calls']}/{r['requested_calls']}) | {phase_text} | {fmt(r['complete_call_event_sum_median_ms'])} | {fmt(r['direct_epoch_event_median_ms'])} | {fmt(r['within_gpu_complete_call_ratio_to_native'])} | {fmt(r['within_gpu_direct_epoch_ratio_to_native'])} | {fmt(r['native_bracket_max_abs_drift'])} | {fmt(peak)} | valid, no new JIT |")
    lines += ["", "Phase entries are medians of directly timed complete model calls classified by aggregate routed-layer counter deltas. MIXED shows the measured layer counts rather than assigning a pure phase; FRESH marks T or dense-consumer full-QK calls. N4/N16 sums and direct epochs are distinct measurements; neither is an observed whole-request time. N/A means the profiler did not expose classifying counters. No winner is selected from task scores or missing rows.", ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", type=Path, action="append", default=[])
    p.add_argument("--scored", type=Path)
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--markdown", type=Path, required=True)
    a = p.parse_args()
    rows = [r for path in a.profile for r in extract_profile(path)]
    scored = json.loads(a.scored.read_text()) if a.scored else None
    table = json.dumps(dict(schema="v20_cost_selection_input_v1", profiles=[sha(x.read_bytes()) for x in a.profile],
                            rows=rows, no_quality_based_selection=True), indent=2, sort_keys=True) + "\n"
    markdown = render(rows, scored)
    for path, payload in ((a.table, table), (a.markdown, markdown)):
        if path.exists() and path.read_bytes() != payload.encode():
            raise ValueError(f"refusing to overwrite different report: {path}")
    for path, payload in ((a.table, table), (a.markdown, markdown)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload.encode())
    print(json.dumps(dict(profiles=len(a.profile), rows=len(rows), measured=sum(r["status"] == "measured" for r in rows))))


if __name__ == "__main__":
    main()
