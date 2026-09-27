"""CPU-only v20 question selection and paired schedule freeze.

Inputs are exact already-tokenized task manifests. Gold/scorer identity is supplied as
hashes, and answers are never written into generation manifests or public protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.v18_protocol import GOLD_KEYS, read_rows

SEEDS = (101, 202)
AIME_IDS = tuple(f"aime26/{n}" for n in (2, 8, 14, 20, 23, 30))
ARMS = (
    "D_native", "D_matched", "T_scope", "M1_R1_A8_current_output",
    "M3_R2_A8_current_output", "M3_R3_A8_current_output", "B_A8_matched",
)
HISTORICAL = "G75L30_first_bitmap_native_adaptive"
REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"
PANEL_SEED = "v20/fan-panel/20260927"
SCHEDULE_SEED = "v20/fan-schedule/20260927"


def sha(value: bytes | str) -> str:
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def key(*parts: object) -> str:
    return sha("/".join(map(str, parts)))


def _rows(path: Path, kind: str) -> list[dict]:
    rows = read_rows(path)
    if len({r.get("id") for r in rows}) != len(rows):
        raise ValueError(f"duplicate {kind} ID")
    clean = []
    for r in rows:
        if not isinstance(r, dict) or not isinstance(r.get("id"), str):
            raise ValueError(f"invalid {kind} row")
        if not isinstance(r.get("prompt"), str) or not r["prompt"] or r.get("prompt_hash") != sha(r["prompt"]):
            raise ValueError(f"{kind} prompt identity mismatch: {r.get('id')}")
        tokens = r.get("prompt_tokens")
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError(f"{kind} exact prompt tokens required: {r['id']}")
        n = r.get("prompt_token_count", r.get("prompt_tokens_n"))
        if n != len(tokens):
            raise ValueError(f"{kind} prompt token count mismatch: {r['id']}")
        # Only known request fields cross the gold firewall; all task metadata stays public.
        fields = ("id", "benchmark", "source_id", "task", "domain", "sub_domain", "bin",
                  "prompt", "prompt_hash", "prompt_tokens", "generation_budget")
        x = {k: r[k] for k in fields if k in r}
        x["prompt_token_count"] = len(tokens)
        if any(k in x for k in GOLD_KEYS):
            raise AssertionError("gold leaked")
        clean.append(x)
    return clean


def _select_ruler(rows: list[dict]) -> list[dict]:
    tasks = sorted({r.get("task") for r in rows})
    if len(rows) != 130 or len(tasks) != 13 or None in tasks or Counter(r["task"] for r in rows) != {t: 10 for t in tasks}:
        raise ValueError("RULER source must be 10 questions in each of 13 tasks")
    chosen = [min((r for r in rows if r["task"] == task), key=lambda r: key(PANEL_SEED, "ruler", task, r["id"]))
              for task in tasks]
    for r in chosen:
        if r.get("generation_budget") not in (30, 32, 50, 120, 128):
            raise ValueError("RULER task budget missing")
        r["thinking"] = False
    return chosen


def _select_aime(rows: list[dict]) -> list[dict]:
    by_id = {r["id"]: r for r in rows}
    if len(rows) != 30 or set(by_id) != {f"aime26/{i}" for i in range(1, 31)}:
        raise ValueError("AIME source must cover all 30 existing IDs")
    chosen = [by_id[i] for i in AIME_IDS]
    for r in chosen:
        if r.get("generation_budget", 8192) != 8192:
            raise ValueError("AIME budget must be 8192")
        r["generation_budget"] = 8192
        r["thinking"] = True
    return chosen


def _select_lb(rows: list[dict]) -> list[dict]:
    if len(rows) != 12 or any(not r["id"].startswith("longbench_v2/") for r in rows):
        raise ValueError("LB source must be the existing 12")
    strata: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        n = r["prompt_token_count"]
        expected_bin = "10-15K" if 10000 <= n < 15000 else "15-20K" if 15000 <= n <= 20000 else None
        if expected_bin is None or r.get("bin") != expected_bin or not r.get("domain"):
            raise ValueError("LB untruncated length/domain stratum invalid")
        if r.get("generation_budget", 8192) != 8192:
            raise ValueError("LB budget must be 8192")
        strata.setdefault((r["domain"], r["bin"]), []).append(r)
    for pair, group in strata.items():
        group.sort(key=lambda r: key(PANEL_SEED, "lb", *pair, r["id"]))
    # One pass per domain/length cell, then the next depth; independent of outcomes.
    order = sorted(strata)
    chosen = []
    depth = 0
    while len(chosen) < 6 and any(len(strata[s]) > depth for s in order):
        for s in order:
            if len(strata[s]) > depth and len(chosen) < 6:
                chosen.append(strata[s][depth])
        depth += 1
    if len(chosen) != 6 or len({r["domain"] for r in chosen}) < min(4, len({r["domain"] for r in rows})):
        raise ValueError("LB strata insufficient for six balanced questions")
    for r in chosen:
        r["generation_budget"] = 8192
        r["thinking"] = True
    return chosen


def _identity(identity: dict, source_paths: dict[str, Path]) -> dict:
    if identity.get("model_revision") != REVISION:
        raise ValueError("model revision identity missing or changed")
    if identity.get("scope") not in ("all_native_legal", "global_only", "pending"):
        raise ValueError("one qualified common scope must be frozen")
    if identity.get("policy_sha256") != "pending" and (
        not isinstance(identity.get("policy_sha256"), str) or len(identity["policy_sha256"]) != 64
    ):
        raise ValueError("one common policy SHA256 must be frozen")
    if (identity["scope"] == "pending") != (identity["policy_sha256"] == "pending"):
        raise ValueError("scope and policy must be bound together")
    for dataset, path in source_paths.items():
        item = identity.get("datasets", {}).get(dataset, {})
        actual = sha(path.read_bytes())
        if item.get("source_manifest_sha256") != actual:
            raise ValueError(f"{dataset} source manifest SHA mismatch")
        for name in ("gold_sha256", "scorer_sha256", "task_contract_sha256"):
            value = item.get(name)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{dataset} {name} must be frozen SHA256")
    return identity


def _schedule(selected: dict[str, list[dict]], hosts: list[dict], protocol_id: str) -> tuple[list[dict], dict, list[dict]]:
    if len(hosts) != 2 or len({h.get("host") for h in hosts}) != 2 or any(not h.get("gpu_uuid") for h in hosts):
        raise ValueError("two distinct host/GPU identities required")
    blocks = [(dataset, row["id"], seed) for dataset, rows in selected.items() for row in rows for seed in SEEDS]
    first_ids = {d: {r["id"] for r in sorted(rows, key=lambda r: key(PANEL_SEED, "first", d, r["id"]))[:2]}
                 for d, rows in selected.items()}
    first = [b for b in blocks if b[2] == 101 and b[1] in first_ids[b[0]]]
    rest = [b for b in blocks if b not in first]
    first.sort(key=lambda b: (list(selected).index(b[0]), key(SCHEDULE_SEED, *b)))
    rest.sort(key=lambda b: key(SCHEDULE_SEED, *b))
    ordered = first + rest
    schedule, assignments, optional = [], {}, []
    for number, (dataset, question, seed) in enumerate(ordered):
        # Stable whole-block host allocation; each host receives a balanced block count.
        host = hosts[number % 2]
        assignments[str(number)] = dict(dataset=dataset, id=question, seed=seed, **host)
        offset = number % len(ARMS)
        order = list(ARMS[offset:] + ARMS[:offset])
        for role, arm_order in (("attempt0", order), ("warm", order[::-1])):
            for arm in arm_order:
                schedule.append(dict(index=len(schedule), block=number, dataset=dataset, id=question, seed=seed,
                                     role=role, repeat=0 if role == "attempt0" else 1, arm=arm,
                                     host=host["host"], gpu_uuid=host["gpu_uuid"],
                                     cell_id=key(protocol_id, dataset, question, seed, arm)))
        for role in ("attempt0", "warm"):
            optional.append(dict(index=len(optional), block=number, dataset=dataset, id=question, seed=seed,
                                 role=role, repeat=0 if role == "attempt0" else 1, arm=HISTORICAL,
                                 host=host["host"], gpu_uuid=host["gpu_uuid"],
                                 cell_id=key(protocol_id, dataset, question, seed, HISTORICAL)))
    if len(schedule) != 700 or len(optional) != 100 or len(first) != 6 or any(e["block"] >= 6 for e in schedule[:84]):
        raise AssertionError("v20 schedule arithmetic changed")
    return schedule, assignments, optional


def freeze(ruler: Path, aime: Path, longbench: Path, identity_file: Path, hosts_file: Path,
           private: Path, out: Path) -> dict:
    sources = {"ruler4k": Path(ruler), "aime26": Path(aime), "longbench_v2": Path(longbench)}
    identity = _identity(json.loads(Path(identity_file).read_text()), sources)
    hosts = json.loads(Path(hosts_file).read_text())
    selected = {"ruler4k": _select_ruler(_rows(ruler, "ruler")),
                "aime26": _select_aime(_rows(aime, "aime")),
                "longbench_v2": _select_lb(_rows(longbench, "longbench"))}
    protocol_id = "v20_fan_" + key(PANEL_SEED, json.dumps(identity, sort_keys=True), json.dumps(hosts, sort_keys=True),
                                  json.dumps({d: [r["id"] for r in rows] for d, rows in selected.items()}, sort_keys=True))[:16]
    schedule, assignments, optional = _schedule(selected, hosts, protocol_id)
    for assignment in assignments.values():
        assignment.update(scope=identity["scope"], policy_sha256=identity["policy_sha256"])
    manifests = {d: json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n" for d, rows in selected.items()}
    protocol = dict(schema="v20_fan_panel_v1", protocol_id=protocol_id, model_revision=REVISION,
                    status="selection_frozen_policy_pending" if identity["scope"] == "pending" else "inputs_frozen_methods_pending",
                    execution_ready=False, source_identity=identity,
                    selection_rule="one per RULER task; six exact AIME IDs; LB domain/length round-robin with SHA ties",
                    selection_seed=PANEL_SEED, schedule_seed=SCHEDULE_SEED, seeds=list(SEEDS), arms=list(ARMS),
                    ids={d: [r["id"] for r in rows] for d, rows in selected.items()},
                    strata={d: [{k: r[k] for k in ("id", "task", "domain", "bin", "prompt_token_count") if k in r}
                                for r in rows] for d, rows in selected.items()},
                    generation_manifest_sha256={d: sha(payload) for d, payload in manifests.items()},
                    block_assignments=assignments, schedule=schedule, first_checkpoint_executions=84,
                    planned_executions=700, historical_extension=dict(arm=HISTORICAL, status="qualification_pending",
                                                                       excluded_from_first84=True, planned_executions=100,
                                                                       schedule=optional),
                    scorer_note="Gold remains scorer-only; RULER official task macro, AIME final-channel, LB pinned NeMo final-channel MCQ")
    files = {private / f"{d}_generation_manifest.json": payload for d, payload in manifests.items()}
    files[out / "frozen_protocol.json"] = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    for path, payload in files.items():
        if path.exists() and path.read_text() != payload:
            raise ValueError(f"refusing to overwrite different frozen file: {path}")
    for path, payload in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
    return protocol


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("ruler", "aime", "longbench", "identity-file", "hosts-file", "private", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    result = freeze(args.ruler, args.aime, args.longbench, args.identity_file, args.hosts_file, args.private, args.out)
    print(json.dumps({k: result[k] for k in ("protocol_id", "first_checkpoint_executions", "planned_executions")}, sort_keys=True))


if __name__ == "__main__":
    main()
