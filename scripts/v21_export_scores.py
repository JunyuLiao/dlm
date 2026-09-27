"""Export allowlisted per-cell quality from the frozen v20 offline scorer.

Run only in the qualified CP5 scoring environment with the same frozen sources,
gold files, private receipts, and ledgers used for the published v20 summary.
The output contains no generated text, prompts, gold, or private paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.v20_score import load_records, score_firsts, verify_sources
from scripts.v18_protocol import sha


QUALITY_KEYS = ("score", "correct", "task_correct", "strict_correct", "parsed", "eos", "capped")


def export(protocol_path: Path, binding_path: Path, ledgers: list[Path], historical_ledgers: list[Path],
           gold_paths: dict[str, Path], ruler_root: Path, private_roots: dict[str, Path],
           historical_private_roots: dict[str, Path]) -> dict:
    protocol = json.loads(protocol_path.read_text())
    binding = json.loads(binding_path.read_text())
    if binding.get("status") != "frozen" or binding.get("panel_protocol_sha256") != sha(protocol_path.read_bytes()):
        raise ValueError("frozen protocol/binding mismatch")
    gold = verify_sources(protocol, gold_paths)
    core = load_records(protocol, binding_path, ledgers)
    historical = load_records(protocol, binding_path, historical_ledgers, historical=True)
    scored = score_firsts(protocol, core, gold, ruler_root=ruler_root, private_roots=private_roots)
    scored.update(score_firsts(protocol, historical, gold, ruler_root=ruler_root,
                               private_roots=historical_private_roots,
                               schedule=protocol["historical_extension"]["schedule"]))
    firsts = [e for e in protocol["schedule"] + protocol["historical_extension"]["schedule"]
              if e["role"] == "attempt0"]
    if len(firsts) != 400 or len(scored) != 400:
        raise ValueError(f"expected 400 qualified first scores, found {len(scored)}")
    cells = []
    for spec in firsts:
        row = scored[spec["cell_id"]]
        if not all(k in row for k in QUALITY_KEYS):
            raise ValueError("qualified score missing required field")
        cells.append({"cell_id": spec["cell_id"], "dataset": spec["dataset"], "id": spec["id"],
                      "seed": spec["seed"], "arm": spec["arm"],
                      **{k: row[k] for k in QUALITY_KEYS}})
    return {"schema": "v21_v20_qualified_first_scores_v1", "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha(protocol_path.read_bytes()),
            "binding_sha256": sha(binding_path.read_bytes()), "scorer": "scripts.v20_score.score_firsts",
            "source_identity": protocol["source_identity"], "first_cells": len(cells),
            "cells": sorted(cells, key=lambda r: r["cell_id"])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "binding", "ruler-gold", "aime-gold", "longbench-gold", "ruler-root", "out",
                 "private-roots", "historical-private-roots"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--ledger", type=Path, action="append", required=True)
    p.add_argument("--historical-ledger", type=Path, action="append", required=True)
    a = p.parse_args()
    result = export(a.protocol, a.binding, a.ledger, a.historical_ledger,
                    {"ruler4k": a.ruler_gold, "aime26": a.aime_gold,
                     "longbench_v2": a.longbench_gold}, a.ruler_root,
                    {k: Path(v) for k, v in json.loads(a.private_roots.read_text()).items()},
                    {k: Path(v) for k, v in json.loads(a.historical_private_roots.read_text()).items()})
    payload = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if a.out.exists() and a.out.read_text() != payload:
        raise ValueError("refusing to overwrite different qualified score export")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(payload)
    print(json.dumps({"first_cells": result["first_cells"], "protocol_id": result["protocol_id"]}))


if __name__ == "__main__":
    main()
