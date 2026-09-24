"""Offline AIME scoring of immutable native numerical-QK attempt-0 receipts.

The generation worker never loads gold answers. This script reads the authorized
manifest only after generation, uses the repository's frozen AIME numeric scorer,
and writes a separate quality file. Missing/unparsed/capped runs stay in the
denominator; timing retries cannot replace an attempt-0 answer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.numerical_qk_reuse.runner import _atomic, _receipt_path, _rows
from experiments.diffusion_gemma_aime26_modes.protocol import final_response, numeric_score


def score(manifest: Path, output: Path, phase: str, condition: str, ids: list[str], seeds: list[int]) -> dict:
    rows = {str(row["id"]): row for row in _rows(manifest)}
    if set(ids) - rows.keys():
        raise ValueError("Scoring IDs absent from manifest")
    scored = []
    for id_ in ids:
        row = rows[id_]
        if "expected" not in row:
            raise ValueError(f"No expected answer in offline manifest: {id_}")
        for seed in seeds:
            path = _receipt_path(output, phase, condition, seed, id_)
            if not path.is_file():
                raise FileNotFoundError(f"Missing attempt-0 receipt: {path}")
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if receipt.get("attempt") != 0 or receipt.get("id") != id_ or receipt.get("seed") != seed:
                raise ValueError(f"Invalid attempt-0 receipt: {path}")
            answer = final_response(receipt["raw_completion"], True)
            result = numeric_score(answer, str(row["expected"]))
            scored.append(dict(id=id_, seed=seed, condition=condition, phase=phase,
                               receipt=str(path), fingerprint=receipt["fingerprint"],
                               final_response=answer, score=result,
                               termination_reason=receipt["termination_reason"],
                               capped=receipt["termination_reason"] == "length" and receipt["output_tokens"] >= 8192,
                               unparsed=result["extracted"] is None,
                               output_tokens=receipt["output_tokens"],
                               total_decoder_calls=receipt["total_decoder_calls"],
                               per_canvas_calls=[c["decoder_calls"] for c in receipt["per_canvas"]]))
    destination = output / "quality" / f"{phase}.{condition}.json"
    payload = dict(schema="numerical_qk_quality_v1", manifest_sha256=__import__("hashlib").sha256(manifest.read_bytes()).hexdigest(),
                   n=len(scored), correct=sum(bool(r["score"]["correct"]) for r in scored),
                   capped=sum(r["capped"] for r in scored), unparsed=sum(r["unparsed"] for r in scored),
                   records=scored)
    if destination.exists() and json.loads(destination.read_text(encoding="utf-8")) != payload:
        raise RuntimeError(f"Existing quality file differs: {destination}")
    if not destination.exists():
        _atomic(destination, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = parser.parse_args()
    result = score(args.manifest, args.output, args.phase, args.condition, args.ids, args.seeds)
    print(json.dumps(dict(n=result["n"], correct=result["correct"])))


if __name__ == "__main__":
    main()
