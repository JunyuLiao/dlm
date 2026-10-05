"""Freeze four authorized AIME26 smoke IDs and build a private input manifest.

Only two offline source forms are accepted: the exact archived v8 setup.json,
or the 30-row math-ai/aime26 aime2026.jsonl dataset. The latter uses the
existing zero-shot AIME prompt text; no AIME25 or protected data is opened.
The private manifest contains prompt and answer text for offline scoring and
must never be committed. The public identity file contains IDs and hashes only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results/diffusion_gemma_jl_aime_gaussian_dimensions_v8/dataset_audit.json"
CONTRACT = ROOT / "results/diffusion_gemma_jl_aime_gaussian_dimensions_v8/execution_contract.json"
IDENTITY = ROOT / "results/numerical_qk_reuse_20260924/smoke_manifest_identity.json"
PRIVATE = ROOT / "results/numerical_qk_reuse_20260924/private/smoke_manifest.json"
AIME26_DATASET_REVISION = "79037aebdb6580008fb960d17cb21fd3099083e3"
AIME26_DATASET_SHA256 = "52822957957a3f577d1e9706c36a66a8108a3f99b6aff424cfb72dff0094a9ee"
NATIVE_GENERATION_CONFIG_SHA256 = "99334f763c3dbe8b161aeaca1c150a05344299fda2d2e4a0e1d342c744461200"
INSTRUCTION = "Solve the problem. Show your reasoning and put the final integer answer in \\boxed{}."


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def _atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != data:
            raise RuntimeError(f"Frozen file differs; refusing overwrite: {path}")
        return
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temp.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def freeze_identity() -> dict:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    six = audit["calibration_ids"]
    if six != [f"aime26/{i}" for i in (2, 8, 14, 20, 23, 30)]:
        raise ValueError("Archived six AIME26 calibration IDs changed")
    if not audit["passed"] or audit["generation_budget"] != 2048:
        raise ValueError("Archived v8 calibration audit is not the expected completed source")
    if contract["setup_sha256"] != contract["sources"]["results/diffusion_gemma_jl_aime_gaussian_dimensions_v8/setup.json"]:
        raise ValueError("V8 setup hash conflict")
    chosen = six[:4]
    identity = dict(schema="numerical_qk_smoke_identity_v1", selected_ids=chosen,
                    selection_rule="First four IDs in archived v8 six-calibration-ID order, frozen before new outputs",
                    authorized_six_ids=six, source_audit=AUDIT.relative_to(ROOT).as_posix(),
                    source_audit_sha256=sha_file(AUDIT), source_contract=CONTRACT.relative_to(ROOT).as_posix(),
                    source_contract_sha256=sha_file(CONTRACT),
                    expected_v8_setup_sha256=contract["setup_sha256"],
                    aime26_dataset_revision=AIME26_DATASET_REVISION,
                    aime26_dataset_sha256=AIME26_DATASET_SHA256,
                    native_generation_config_sha256=NATIVE_GENERATION_CONFIG_SHA256,
                    generation=dict(max_new_tokens=8192, thinking=True, seed=42,
                                    native_adaptive=True, eos_enabled=True,
                                    canvas_length=256, max_denoising_steps=48,
                                    t_min=.4, t_max=.8, confidence_threshold=.005,
                                    stability_threshold=1, entropy_bound=.1,
                                    eos_token_ids=[1, 106, 50]))
    _atomic(IDENTITY, identity)
    return identity


def _from_setup(source: Path, ids: list[str], expected_sha: str) -> list[dict]:
    if sha_file(source) != expected_sha:
        raise ValueError("Source setup differs from exact archived v8 hash")
    setup = json.loads(source.read_text(encoding="utf-8"))
    if setup["calibration_ids"]["aime26"] != [f"aime26/{i}" for i in (2, 8, 14, 20, 23, 30)]:
        raise ValueError("Source setup calibration IDs changed")
    source_rows = {r["id"]: r for r in setup["final"] if r["benchmark"] == "aime26"}
    if len(source_rows) != 30 or set(ids) - source_rows.keys():
        raise ValueError("Source setup lacks the authorized full AIME26 final set")
    selected = []
    for id_ in ids:
        old = source_rows[id_]
        if sha_bytes(old["prompt"].encode()) != old["prompt_hash"]:
            raise ValueError(f"Historical prompt hash mismatch: {id_}")
        selected.append(dict(id=id_, source_id=str(old["source_id"]), benchmark="aime26",
                             prompt=old["prompt"], prompt_hash=old["prompt_hash"],
                             expected=str(old["expected"]), thinking=True,
                             generation_budget=8192, seed=42))
    return selected


def _from_dataset(source: Path, ids: list[str]) -> list[dict]:
    if source.name != "aime2026.jsonl":
        raise ValueError("Dataset source must be the explicit aime2026.jsonl file")
    if sha_file(source) != AIME26_DATASET_SHA256:
        raise ValueError("Offline AIME26 JSONL hash differs from the frozen authorized snapshot")
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {str(r["id"]): r for r in records}
    if len(records) != 30 or set(by_id) != {str(i) for i in range(1, 31)}:
        raise ValueError("AIME26 JSONL must contain exactly distinct IDs 1..30")
    selected = []
    for id_ in ids:
        source_id = id_.split("/", 1)[1]
        item = by_id[source_id]
        if not isinstance(item.get("problem"), str) or not item["problem"] or "answer" not in item:
            raise ValueError(f"Missing AIME26 problem/answer: {id_}")
        prompt = INSTRUCTION + "\n\n### Problem to solve\n" + item["problem"] + "\n\nSolution:"
        selected.append(dict(id=id_, source_id=source_id, benchmark="aime26",
                             prompt=prompt, prompt_hash=sha_bytes(prompt.encode()),
                             expected=str(item["answer"]), thinking=True,
                             generation_budget=8192, seed=42))
    return selected


def build(source: Path, kind: str, output: Path = PRIVATE) -> dict:
    identity = freeze_identity()
    if not source.is_file():
        raise FileNotFoundError(f"Offline AIME26 source absent: {source}")
    ids = identity["selected_ids"]
    rows = (_from_setup(source, ids, identity["expected_v8_setup_sha256"])
            if kind == "v8_setup" else _from_dataset(source, ids))
    if [r["id"] for r in rows] != ids:
        raise AssertionError("Selection drift")
    _atomic(output, rows)
    provenance = dict(source_kind=kind, source_path=str(source.resolve()), source_sha256=sha_file(source),
                      manifest_path=str(output.resolve()), manifest_sha256=sha_file(output),
                      ids=ids, prompt_hashes={r["id"]: r["prompt_hash"] for r in rows},
                      budget=8192, thinking=True, seed=42)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Offline v8 setup.json or AIME26 aime2026.jsonl")
    parser.add_argument("--kind", choices=("v8_setup", "aime26_jsonl"))
    parser.add_argument("--output", type=Path, default=PRIVATE)
    args = parser.parse_args()
    if (args.source is None) != (args.kind is None):
        parser.error("Pass both --source and --kind to materialize a private manifest")
    result = build(args.source, args.kind, args.output) if args.source else freeze_identity()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
