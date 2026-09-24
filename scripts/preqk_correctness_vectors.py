"""Regenerate the four-question comparison from records, with machine-checked
agreement claims.

The v6 prose claimed the new M1 successor matched "native dense and fresh
Junyu T exactly (same missed question)". That is false for T: T answers
aime26/14 correctly (7276 tokens, not capped) and misses aime26/20, while
dense and the new M1 both miss aime26/14. All three score 3/4, but only
dense and M1 share a correctness VECTOR.

This module therefore refuses to describe two arms as matching unless their
per-ID vectors are equal element-wise. ``agreement_label`` is the only
sanctioned way to phrase it, and ``verify_claim`` raises on a wrong claim,
so a summary cannot restate the v6 error.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

IDS = ("aime26/2", "aime26/8", "aime26/14", "aime26/20")


def old_arm_vectors(request_table: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Per-ID rows for the frozen v5 arms, straight from request_table.csv."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    with request_table.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            arm = row["arm"]
            if row["question"] not in IDS:
                continue
            out.setdefault(arm, {})[row["question"]] = dict(
                correct=row["correct"] == "True",
                capped=row["capped"] == "True",
                unparsed=row["unparsed"] == "True",
                output_tokens=int(row["output_tokens"]),
                total_calls=int(row["total_calls"]),
                canvases=int(row["canvas_count"]),
                wall_seconds=float(row["whole_wall_seconds"]),
            )
    return out


def new_arm_vector(quality_file: Path) -> dict[str, dict[str, Any]]:
    """Per-ID rows for a new arm, from its offline quality receipt."""
    payload = json.loads(quality_file.read_text(encoding="utf-8"))
    rows = {record["id"]: record for record in payload["records"]}
    return {
        id_: dict(
            correct=bool(rows[id_]["score"]["correct"]),
            capped=bool(rows[id_]["capped"]),
            unparsed=bool(rows[id_]["unparsed"]),
            output_tokens=int(rows[id_]["output_tokens"]),
            total_calls=int(rows[id_]["total_decoder_calls"]),
            canvases=len(rows[id_]["per_canvas_calls"]),
        )
        for id_ in IDS
    }


def vector(arm_rows: dict[str, dict[str, Any]]) -> tuple[bool, ...]:
    return tuple(bool(arm_rows[id_]["correct"]) for id_ in IDS)


def agreement_label(left: tuple[bool, ...], right: tuple[bool, ...]) -> str:
    """The ONLY sanctioned phrasing. Equal counts never imply equal vectors."""
    if left == right:
        return "identical per-ID correctness vector"
    if sum(left) == sum(right):
        return (f"same 3/4-style count ({sum(left)}/{len(left)}) but DIFFERENT "
                f"per-ID vector -- different questions missed")
    return f"different count ({sum(left)}/{len(left)} vs {sum(right)}/{len(right)})"


def verify_claim(left: tuple[bool, ...], right: tuple[bool, ...],
                 claim_same_vector: bool) -> None:
    """Raise if prose asserts vector identity that the records do not support."""
    if claim_same_vector and left != right:
        raise AssertionError(
            f"Claimed identical correctness vectors but records differ: "
            f"{left} vs {right}. Equal correct-counts are not vector identity.")
    if not claim_same_vector and left == right:
        raise AssertionError(
            f"Claimed differing correctness vectors but records are identical: {left}")


def build(request_table: Path, new_quality: dict[str, Path]) -> dict[str, Any]:
    arms: dict[str, dict[str, dict[str, Any]]] = dict(old_arm_vectors(request_table))
    for name, path in new_quality.items():
        arms[name] = new_arm_vector(path)
    vectors = {name: vector(rows) for name, rows in arms.items()}
    pairs = {}
    names = sorted(vectors)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            pairs[f"{left} vs {right}"] = agreement_label(vectors[left], vectors[right])
    return dict(
        schema="preqk_correctness_vectors_v1",
        ids=list(IDS),
        note=("Correctness vectors are element-wise over the four development "
              "IDs. Equal correct-counts are NOT vector identity and are NOT "
              "population noninferiority evidence (four questions, one seed)."),
        vectors={name: list(value) for name, value in vectors.items()},
        correct_counts={name: sum(value) for name, value in vectors.items()},
        pairwise=pairs,
        per_id={name: rows for name, rows in arms.items()},
    )


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-table", type=Path, required=True)
    parser.add_argument("--new-quality", type=Path, nargs="*", default=[],
                        help="NAME=path/to/quality.json entries for new arms")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse(argv)
    new = {}
    for item in args.new_quality:
        name, _, path = str(item).partition("=")
        new[name] = Path(path)
    payload = build(args.request_table, new)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(dict(vectors=payload["vectors"], pairwise=payload["pairwise"]), indent=2))


if __name__ == "__main__":
    main()
