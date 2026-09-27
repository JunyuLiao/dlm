"""Gold-free per-window physical and operator evidence for selected001 profiles."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

from scripts.v20_physical_report import COUNT_FIELDS, SEGMENTS, _add, _counts, _rates, digest


def _identity(value: object) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()[:16]


def _range(values: list[int]) -> dict | None:
    return dict(min=min(values), max=max(values)) if values else None


def _operator(probe: dict | None) -> dict:
    if not probe or probe.get("status") != "qualified":
        return dict(status=(probe or {}).get("status", "N/A"), reason=(probe or {}).get("reason"))
    rows = probe.get("rows", [])
    if any(row.get("layer") not in (0, 5) for row in rows):
        raise ValueError("operator probe left predeclared layers 0/5")
    output = []
    for layer in (0, 5):
        selected = [row for row in rows if row["layer"] == layer]
        if not selected:
            continue
        output.append(dict(layer=layer, observations=len(selected),
                           max_actual_output_abs_error=max(row["actual_output_max_abs_error"] for row in selected),
                           max_actual_output_relative_l2_error=max(row["actual_output_relative_l2_error"] for row in selected),
                           prepared_consumer_event_median_ms_range=_range([
                               row["prepared_consumer_event_median_ms"] for row in selected]),
                           v11_envelope_all=all(row.get("v11_envelope_pass") is True for row in selected)))
    return dict(status="qualified", by_layer=output,
                definition="operator only: same QKV/support consumer, excludes projection, route, model forward and generation")


def _physical(diagnostic: dict, timed: dict) -> dict:
    if diagnostic.get("status") == "N/A":
        return dict(status="unavailable", reason=diagnostic.get("reason"))
    if diagnostic.get("status") != "qualified":
        raise ValueError("unqualified selected counter twin")
    if diagnostic.get("input_output_digests") != timed.get("summary", {}).get("input_output_digests"):
        raise ValueError("selected counter twin input/output digest mismatch")
    physical = diagnostic.get("physical", {})
    rows = physical.get("rows", [])
    if not rows or physical.get("calls") != len(rows) or physical.get("accepted_timing") is not False:
        raise ValueError("selected physical rows/timing provenance mismatch")
    if physical.get("projection_skipping_measured") is not False:
        raise ValueError("projection skipping must remain unmeasured")
    kinds = {kind: {segment: _counts() for segment in SEGMENTS} for kind in ("local", "global")}
    phases = Counter()
    phases_by_kind = {kind: Counter() for kind in kinds}
    ages = {key: [] for key in ("numeric_age", "decision_age")}
    layers = set()
    for row in rows:
        kind, phase = row.get("kind"), row.get("phase")
        if kind not in kinds or phase not in ("A", "D", "H"):
            raise ValueError("unknown kind/phase in physical row")
        layers.add(row["layer"])
        phases[phase] += 1
        phases_by_kind[kind][phase] += 1
        for key in ages:
            value = row.get(key)
            if type(value) is int:
                ages[key].append(value)
        for segment in SEGMENTS:
            _add(kinds[kind][segment], row["by_segment"][segment])
    # The profiler's independent phase/kind aggregate must equal its row-level evidence.
    aggregate = {kind: {segment: _counts() for segment in SEGMENTS} for kind in kinds}
    for name, segments in physical.get("by_phase_kind", {}).items():
        phase, kind = name.split("/")
        if kind not in kinds or phases_by_kind[kind][phase] <= 0:
            raise ValueError("phase/kind aggregate lacks rows")
        for segment in SEGMENTS:
            _add(aggregate[kind][segment], segments[segment])
    if aggregate != kinds:
        raise ValueError("phase/kind aggregate differs from physical rows")
    all_kinds = {segment: _counts() for segment in SEGMENTS}
    for segment in SEGMENTS:
        for kind in kinds:
            _add(all_kinds[segment], kinds[kind][segment])
    return dict(status="qualified", exact_input_output_match=True, attention_layer_calls=len(rows),
                routed_layers=sorted(layers), routed_layer_count=len(layers),
                phase_attention_calls={phase: phases[phase] for phase in ("A", "D", "H")},
                phase_attention_calls_by_kind={kind: {phase: phases_by_kind[kind][phase]
                                                      for phase in ("A", "D", "H")} for kind in kinds},
                numeric_age=_range(ages["numeric_age"]), decision_age=_range(ages["decision_age"]),
                by_kind={kind: {segment: _rates(value) for segment, value in segments.items()}
                         for kind, segments in kinds.items()},
                all_kinds={segment: _rates(value) for segment, value in all_kinds.items()},
                accepted_timing=False, projection_skipping_measured=False)


def summarize(profile_paths: dict[tuple[str, str], Path]) -> dict:
    if set(profile_paths) != {(stage, host) for stage in ("selected_001", "ruler_selected_001")
                             for host in ("mpk", "dllm")}:
        raise ValueError("both hosts and both selected stages required")
    profiles, rows = [], []
    datasets_by_stage = {"selected_001": {"aime26", "longbench_v2", "ruler4k"},
                         "ruler_selected_001": {"ruler4k"}}
    for (stage, host), path in sorted(profile_paths.items()):
        data = json.loads(path.read_text())
        runtime = data.get("runtime_identity", {})
        if (data.get("schema") != "v20_direct_full_forward_v1" or not data.get("counter_twins") or
                runtime.get("hostname") != host or not runtime.get("gpu_uuid")):
            raise ValueError("selected profile identity/counter twin mismatch")
        if set(data.get("selected_boundaries", [])) != {"model_forward", "denoising_step"}:
            raise ValueError("selected boundaries changed")
        expected_n = 4 if stage == "ruler_selected_001" else 16
        if data.get("selected_sequence_lengths") != [expected_n]:
            raise ValueError("selected sequence length changed")
        profile_sha = digest(path)
        profiles.append(dict(stage=stage, host=host, gpu_uuid=runtime["gpu_uuid"],
                             sha256=profile_sha, source_sha256=data["source_sha256"],
                             source_commits=sorted({arm["config"].get("source_commit") for arm in data["arms"]})))
        arms = {arm["name"]: arm for arm in data["arms"]}
        if len(arms) != 8:
            raise ValueError("selected canonical arm inventory changed")
        for target in data["targets"].values():
            dataset = target["dataset"]
            if dataset not in datasets_by_stage[stage]:
                raise ValueError("target dataset differs from selected stage")
            state = _identity([dataset, target["id"], target["canvas"]])
            window = _identity([dataset, target["id"], target["canvas"], target["requested_call"]])
            resolution = target.get("resolution", {})
            for boundary in ("model_forward", "denoising_step"):
                sequence = target.get("boundaries", {}).get(boundary, {}).get(f"N{expected_n}")
                for arm_name, arm in arms.items():
                    config = arm["config"]
                    base = dict(stage=stage, host=host, gpu_uuid=runtime["gpu_uuid"],
                                profile_sha256=profile_sha, dataset=dataset, state=state,
                                canvas=target["canvas"], window=window,
                                requested_call=target["requested_call"],
                                selected_call=resolution.get("selected_call"),
                                fallback_used=resolution.get("fallback_used"),
                                boundary=boundary, sequence=f"N{expected_n}",
                                canonical_arm=arm_name, scope=config.get("v20_scope") or "NATIVE",
                                policy_name=config.get("policy_name"),
                                policy_sha256=config.get("policy_sha256"))
                    if sequence is None:
                        rows.append(dict(base, status="missing_sequence", reason=(
                            "native RULER canvas reached fewer than 16 calls" if stage == "selected_001" and
                            dataset == "ruler4k" else "selected sequence absent")))
                        continue
                    diagnostic = sequence["diagnostic_replays"][arm_name]
                    physical = _physical(diagnostic, sequence["arms"][arm_name])
                    rows.append(dict(base, status=physical["status"], reached_calls=sequence["reached_calls"],
                                     requested_calls=sequence["requested_calls"],
                                     sequence_complete=sequence["reached_calls"] == sequence["requested_calls"],
                                     sequence_status=("complete" if sequence["reached_calls"] == sequence["requested_calls"]
                                                      else "native_shortened"), physical=physical,
                                     operator_only=_operator(diagnostic.get("operator_probe"))))
    # The RULER call0/call2 aliases replay the same actual N4 sequence starting at
    # call0. Count it once; retain both requested labels, and fail if evidence differs.
    deduplicated = {}
    for row in rows:
        key = (row["stage"], row["host"], row["dataset"], row["state"], row["canvas"],
               row["boundary"], row["sequence"], row["canonical_arm"], row["scope"],
               row["policy_sha256"])
        alias = dict(requested_call=row["requested_call"], selected_call=row["selected_call"],
                     window=row["window"])
        if key not in deduplicated:
            row["requested_call_aliases"] = [alias]
            deduplicated[key] = row
            continue
        prior = deduplicated[key]
        evidence = ("status", "reason", "reached_calls", "requested_calls", "sequence_status",
                    "physical", "operator_only", "profile_sha256", "fallback_used")
        if any(prior.get(field) != row.get(field) for field in evidence):
            raise ValueError("same-canvas alias has different diagnostic evidence")
        prior["requested_call_aliases"].append(alias)
    rows = list(deduplicated.values())
    state_keys = {(row["host"], row["dataset"], row["state"]) for row in rows}
    for row in rows:
        row["requested_call_aliases"].sort(key=lambda alias: alias["requested_call"])
        row["same_canvas_window_count"] = len(row["requested_call_aliases"])
    return dict(schema="v20_selected_physical_v1", profiles=profiles, rows=rows,
                measured_rows=sum(r["status"] == "qualified" for r in rows),
                missing_sequence_rows=sum(r["status"] == "missing_sequence" for r in rows),
                distinct_captured_state_keys=len(state_keys),
                interpretation="Each row counts one distinct selected canvas sequence per boundary and arm. RULER requested-call aliases replay the same call0 N4 sequence and are not summed. The two boundaries repeat the captured state and are not independent requests; global-only scope routes five layers. Operator errors/times cover layers 0/5 only and are not model costs.")


def csv_text(result: dict) -> str:
    columns = ("stage", "host", "dataset", "state", "canvas", "window", "same_canvas_window_count",
               "requested_call_aliases",
               "requested_call", "selected_call", "boundary", "sequence", "reached_calls", "requested_calls",
               "sequence_status", "canonical_arm", "scope",
               "policy_name", "policy_sha256", "status", "kind", "segment", "routed_layer_count",
               "A_layer_calls", "D_layer_calls", "H_layer_calls", "numeric_age_min", "numeric_age_max",
               "decision_age_min", "decision_age_max") + COUNT_FIELDS + ("qk_skipped_fraction", "pv_skipped_fraction")
    columns += ("operator_only_l0_max_abs_error", "operator_only_l0_prepared_consumer_ms_min",
                "operator_only_l0_prepared_consumer_ms_max", "operator_only_l5_max_abs_error",
                "operator_only_l5_prepared_consumer_ms_min", "operator_only_l5_prepared_consumer_ms_max")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in result["rows"]:
        physical = row.get("physical", {})
        for kind in ("local", "global") if row["status"] == "qualified" else ("N/A",):
            for segment in SEGMENTS if row["status"] == "qualified" else ("N/A",):
                counts = physical.get("by_kind", {}).get(kind, {}).get(segment, {})
                phases = physical.get("phase_attention_calls_by_kind", {}).get(kind, {})
                ages = {key: physical.get(key) or {} for key in ("numeric_age", "decision_age")}
                operators = {item["layer"]: item for item in row.get("operator_only", {}).get("by_layer", [])}
                writer.writerow({**{key: (",".join(str(x["requested_call"]) for x in row[key])
                                          if key == "requested_call_aliases" else row.get(key))
                                    for key in columns if key in row},
                                 "kind": kind, "segment": segment,
                                 "routed_layer_count": physical.get("routed_layer_count"),
                                 **{f"{phase}_layer_calls": phases.get(phase) for phase in ("A", "D", "H")},
                                 **{f"{key}_{edge}": ages[key].get(edge) for key in ages for edge in ("min", "max")},
                                 **{key: counts.get(key) for key in COUNT_FIELDS},
                                 "qk_skipped_fraction": counts.get("qk_skipped_fraction"),
                                 "pv_skipped_fraction": counts.get("pv_skipped_fraction"),
                                 **{f"operator_only_l{layer}_max_abs_error": operators.get(layer, {}).get("max_actual_output_abs_error")
                                    for layer in (0, 5)},
                                 **{f"operator_only_l{layer}_prepared_consumer_ms_{edge}":
                                    (operators.get(layer, {}).get("prepared_consumer_event_median_ms_range") or {}).get(edge)
                                    for layer in (0, 5) for edge in ("min", "max")}})
    return stream.getvalue()


def render(result: dict) -> str:
    lines = ["# Selected001 physical work and phase evidence", "",
             f"{result['measured_rows']} qualified arm/canvas-sequence/boundary rows; {result['missing_sequence_rows']} missing-sequence rows (RULER N16). RULER requested calls 0 and 2 alias the same N4 replay beginning at call0 and are counted once. The two boundaries replay the same captured state and are not independent requests.", "",
             "Whole/static-prefix/current-canvas and LOCAL/GLOBAL legal-pair counts are in the JSON and CSV. GLOBAL_ONLY_NATIVE_LOCAL counts cover five routed GLOBAL layers, not all 30 layers. Operator errors and prepared-consumer timings cover only observed layer 0/5 QKV and support; they exclude route, projection, model forward and generation.", ""]
    for profile in result["profiles"]:
        lines.append(f"- {profile['stage']} {profile['host']}: profile `{profile['sha256']}`, GPU `{profile['gpu_uuid']}`, source commits {', '.join(str(x) for x in profile['source_commits'])}.")
        lines.extend(f"  - `{path.rsplit('/', 1)[-1]}` `{value}`" for path, value in sorted(profile["source_sha256"].items()))
    lines += ["", "| Stage | Host | Dataset | Canvas/window | Boundary/N reached | Arm | Scope | A/D/H layers | Numeric age | Decision age | Whole legal QK/PV skipped | Operator only L0/L5 max abs error, prepared consumer ms | Status |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    def pct(x):
        return "N/A" if x is None else f"{100 * x:.1f}%"
    for row in result["rows"]:
        physical = row.get("physical", {})
        counts = physical.get("all_kinds", {}).get("whole", {})
        phases = physical.get("phase_attention_calls", {})
        age = lambda key: (f"{physical[key]['min']}..{physical[key]['max']}" if physical.get(key) else "N/A")
        ops = row.get("operator_only", {}).get("by_layer", [])
        op = "/".join(f"L{x['layer']}:e={x['max_actual_output_abs_error']:.4g},t="
                      f"{x['prepared_consumer_event_median_ms_range']['min']:.3g}.."
                      f"{x['prepared_consumer_event_median_ms_range']['max']:.3g}ms" for x in ops) or "N/A"
        lines.append(f"| {row['stage']} | {row['host']} | {row['dataset']} | {row['canvas']}/{row['window'][:6]} "
                     f"(requested {','.join(str(x['requested_call']) for x in row['requested_call_aliases'])}) | {row['boundary']}/{row['sequence']} "
                     f"{row.get('reached_calls', 'N/A')}/{row.get('requested_calls', 'N/A')} | "
                     f"{row['canonical_arm']} | {row['scope']} | "
                     f"{phases.get('A', 0)}/{phases.get('D', 0)}/{phases.get('H', 0)} | "
                     f"{age('numeric_age')} | {age('decision_age')} | "
                     f"{pct(counts.get('qk_skipped_fraction'))}/{pct(counts.get('pv_skipped_fraction'))} | {op} | "
                     f"{row['status']}{' (' + row['sequence_status'] + ')' if row.get('sequence_status') == 'native_shortened' else ''} |")
    lines += ["", "Counter twins matched accepted input/output digests when qualified and are untimed. Physical QK/PV pairs are native-legal eligible pairs; projection skipping is unmeasured. The table makes no answer-quality or request-speed claim.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", required=True, help="stage:host=path")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    profiles = {}
    for item in args.profile:
        key, value = item.split("=", 1)
        stage, host = key.split(":", 1)
        if (stage, host) in profiles:
            raise ValueError("duplicate profile")
        profiles[stage, host] = Path(value)
    report = summarize(profiles)
    payloads = ((args.json, json.dumps(report, sort_keys=True, indent=2) + "\n"),
                (args.markdown, render(report)), (args.csv, csv_text(report)))
    for path, payload in payloads:
        encoded = payload.encode("utf-8")
        if path.exists() and path.read_bytes() != encoded:
            raise ValueError(f"refusing to overwrite different report: {path}")
    for path, payload in payloads:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload.encode("utf-8"))
    print(json.dumps(dict(measured=report["measured_rows"], missing=report["missing_sequence_rows"])))


if __name__ == "__main__":
    main()
