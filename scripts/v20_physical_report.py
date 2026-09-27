"""Gold-free screen002 physical-counter summary; untimed twins only."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


COUNT_FIELDS = ("eligible_pairs", "skipped_qk_pairs", "executed_qk_pairs",
                "skipped_pv_pairs", "executed_pv_pairs",
                "executed_qk_multiply_accumulates", "executed_pv_multiply_accumulates")
SEGMENTS = ("whole", "static_prefix", "current_canvas")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _counts() -> dict[str, int]:
    return {key: 0 for key in COUNT_FIELDS}


def _add(destination: dict, source: dict) -> None:
    for key in COUNT_FIELDS:
        value = source.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid physical counter {key}")
        destination[key] += value


def _rates(counts: dict) -> dict:
    eligible = counts["eligible_pairs"]
    for operation in ("qk", "pv"):
        if counts[f"skipped_{operation}_pairs"] + counts[f"executed_{operation}_pairs"] != eligible:
            raise ValueError(f"physical {operation} pair conservation failed")
    return dict(counts, qk_skipped_fraction=counts["skipped_qk_pairs"] / eligible if eligible else None,
                pv_skipped_fraction=counts["skipped_pv_pairs"] / eligible if eligible else None)


def summarize(profile_paths: dict[str, Path], cost_path: Path) -> dict:
    cost = json.loads(cost_path.read_text())
    profiles = {host: digest(path) for host, path in profile_paths.items()}
    if cost.get("no_quality_based_selection") is not True or set(cost.get("profiles", [])) != set(profiles.values()):
        raise ValueError("cost table is not bound to exactly these screen profiles")
    grouped = {}
    sources = []
    for host, path in sorted(profile_paths.items()):
        data = json.loads(path.read_text())
        identity = data.get("runtime_identity", {})
        if data.get("schema") != "v20_direct_full_forward_v1" or not data.get("counter_twins"):
            raise ValueError("profile is not an explicit counter-twin screen")
        if identity.get("hostname") != host or not identity.get("gpu_uuid"):
            raise ValueError("host/GPU identity mismatch")
        if data.get("selected_boundaries") != ["model_forward"] or data.get("selected_sequence_lengths") != [4]:
            raise ValueError("screen002 boundary/sequence changed")
        arms = {arm["name"]: arm for arm in data["arms"]}
        sources.append(dict(host=host, gpu_uuid=identity["gpu_uuid"], profile_sha256=profiles[host],
                            source_sha256=data["source_sha256"], source_commits=sorted({
                                arm["config"].get("source_commit") for arm in arms.values()})))
        for target in data["targets"].values():
            dataset = target["dataset"]
            sequence = target.get("boundaries", {}).get("model_forward", {}).get("N4")
            if sequence is None:
                raise ValueError("screen002 target lacks model_forward/N4")
            diagnostics = sequence.get("diagnostic_replays", {})
            if set(diagnostics) != set(arms) or set(sequence.get("arms", {})) != set(arms):
                raise ValueError("target arm inventory differs from frozen profile")
            for name, arm in arms.items():
                config = arm["config"]
                key = (host, dataset, config.get("v20_scope") or "NATIVE", name,
                       config.get("policy_name"), config.get("policy_sha256"))
                bucket = grouped.setdefault(key, dict(states=0, qualified=0, unavailable=0,
                    unavailable_reasons=Counter(), phase_calls=Counter(),
                    physical={kind: {segment: _counts() for segment in SEGMENTS}
                              for kind in ("local", "global")}, numerical_calls=0,
                    projection_skipping_measured=False))
                bucket["states"] += 1
                diagnostic = diagnostics[name]
                if diagnostic.get("status") == "N/A":
                    bucket["unavailable"] += 1
                    bucket["unavailable_reasons"][diagnostic.get("reason", "unspecified")] += 1
                    continue
                if diagnostic.get("status") != "qualified":
                    raise ValueError("counter twin is neither qualified nor explicitly unavailable")
                measured = sequence["arms"][name].get("summary")
                if measured is None or diagnostic.get("input_output_digests") != measured.get("input_output_digests"):
                    raise ValueError("untimed twin does not match accepted input/output digests")
                physical = diagnostic.get("physical")
                if not physical or physical.get("accepted_timing") is not False or physical.get("projection_skipping_measured") is not False:
                    raise ValueError("physical twin provenance changed")
                rows = physical.get("rows", [])
                if not rows or physical.get("calls") != len(rows):
                    raise ValueError("physical attention-call rows mismatch")
                bucket["qualified"] += 1
                bucket["numerical_calls"] += len(rows)
                bucket["phase_calls"].update(row["phase"] for row in rows)
                for phase_kind, segments in physical["by_phase_kind"].items():
                    phase, kind = phase_kind.split("/")
                    if kind not in ("local", "global") or phase not in ("A", "D", "H"):
                        raise ValueError("unexpected physical phase/kind")
                    for segment in SEGMENTS:
                        _add(bucket["physical"][kind][segment], segments[segment])
    rows = []
    for (host, dataset, scope, arm, policy, policy_sha), bucket in sorted(grouped.items(), key=lambda x: str(x[0])):
        status = "qualified" if bucket["qualified"] == bucket["states"] else (
            "unavailable" if bucket["qualified"] == 0 else "partial_qualification")
        kinds = {kind: {segment: _rates(counts) for segment, counts in segments.items()}
                 for kind, segments in bucket["physical"].items()}
        whole = {segment: _counts() for segment in SEGMENTS}
        for segment in SEGMENTS:
            for kind in kinds:
                _add(whole[segment], kinds[kind][segment])
        whole = {segment: _rates(counts) for segment, counts in whole.items()}
        calls = bucket["numerical_calls"]
        rows.append(dict(host=host, dataset=dataset, scope=scope, arm=arm,
                         policy_name=policy, policy_sha256=policy_sha, sampled_states=bucket["states"],
                         qualified_twin_states=bucket["qualified"], unavailable_twin_states=bucket["unavailable"],
                         unavailable_reasons=dict(bucket["unavailable_reasons"]), status=status,
                         twin_input_output_match=(status == "qualified"), numerical_attention_calls=calls,
                         phase_attention_calls=dict(sorted(bucket["phase_calls"].items())),
                         phase_attention_call_fractions={phase: bucket["phase_calls"][phase] / calls if calls else None
                                                         for phase in ("A", "D", "H")},
                         all_kinds=whole, by_kind=kinds, accepted_timing=False,
                         projection_skipping_measured=False))
    cost_counts = Counter((row.get("profile_sha256"), row.get("dataset"), row.get("arm"))
                          for row in cost.get("rows", []) if row.get("status") == "measured")
    for row in rows:
        row["measured_cost_rows"] = cost_counts[(profiles[row["host"]], row["dataset"], row["arm"])]
    return dict(schema="v20_screen002_physical_v1", profiles=sources,
                cost_table_sha256=digest(cost_path), cost_table_source_profiles=sorted(profiles.values()),
                group_count=len(rows), groups=rows,
                interpretation="Untimed, token/input-output-matched numerical counter twins; pairs count legal physical work, not GPU latency. No projection skipping or answer quality inferred.")


def render(result: dict) -> str:
    lines = ["# Screen002 physical-work counters", "",
             "Untimed counter twins replayed the accepted model-forward N4 inputs. Qualified twins match every accepted input/output digest. Pair fractions are over native-legal eligible pairs; they are physical QK/PV work, not latency or task quality. Projection skipping was not measured.", "",
             f"Cost table SHA-256: `{result['cost_table_sha256']}`.", ""]
    for source in result["profiles"]:
        lines.append(f"- {source['host']} GPU `{source['gpu_uuid']}`; profile `{source['profile_sha256']}`; source commits {', '.join(str(x) for x in source['source_commits'])}.")
        lines.extend(f"  - `{path.rsplit('/', 1)[-1]}`: `{hash_value}`" for path, hash_value in sorted(source["source_sha256"].items()))
    lines += ["", "| Host | Dataset | Scope | Arm | Policy | Twin | A/D/H layer calls | QK skipped whole/static/current | PV skipped whole/static/current | Local QK/PV whole | Global QK/PV whole | Cost rows |",
              "|---|---|---|---|---|---|---|---|---|---|---|---:|"]
    def pct(value):
        return "N/A" if value is None else f"{100 * value:.1f}%"
    for row in result["groups"]:
        all_kinds, kinds = row["all_kinds"], row["by_kind"]
        qk = "/".join(pct(all_kinds[s]["qk_skipped_fraction"]) for s in SEGMENTS)
        pv = "/".join(pct(all_kinds[s]["pv_skipped_fraction"]) for s in SEGMENTS)
        local = "/".join(pct(kinds["local"]["whole"][f"{op}_skipped_fraction"]) for op in ("qk", "pv"))
        global_ = "/".join(pct(kinds["global"]["whole"][f"{op}_skipped_fraction"]) for op in ("qk", "pv"))
        phases = "/".join(str(row["phase_attention_calls"].get(p, 0)) for p in ("A", "D", "H"))
        lines.append(f"| {row['host']} | {row['dataset']} | {row['scope']} | {row['arm']} | {row['policy_name'] or 'N/A'} | "
                     f"{row['status']} {row['qualified_twin_states']}/{row['sampled_states']} | {phases} | {qk} | {pv} | {local} | {global_} | {row['measured_cost_rows']} |")
    lines += ["", "The JSON preserves raw eligible/skipped/executed pair and multiply-accumulate counts by LOCAL/GLOBAL and whole/static-prefix/current-canvas. A/D/H counts are attention-layer calls, not decoder calls. N/A twins carry no physical-work claim. This report makes no policy selection or answer-quality claim.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", required=True, help="host=profile.json")
    parser.add_argument("--cost", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    profiles = dict(item.split("=", 1) for item in args.profile)
    if len(profiles) != len(args.profile):
        raise ValueError("duplicate host profile")
    result = summarize({host: Path(path) for host, path in profiles.items()}, args.cost)
    for path, payload in ((args.json, json.dumps(result, sort_keys=True, indent=2) + "\n"),
                          (args.markdown, render(result))):
        encoded = payload.encode("utf-8")
        if path.exists() and path.read_bytes() != encoded:
            raise ValueError(f"refusing to replace different report: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
    print(json.dumps(dict(groups=result["group_count"], profiles=len(profiles))))


if __name__ == "__main__":
    main()
