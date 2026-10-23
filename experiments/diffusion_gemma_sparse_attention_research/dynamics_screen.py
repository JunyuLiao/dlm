"""Freeze a score-blind 30-item native trajectory screen.

The public identity contains only source IDs, prompt hashes, and provenance.  A
private manifest is materialized beside it for local generation; it contains
the prompts and reference answers and must not be committed or copied into a
report.  Selection is deterministic and independent of dense outputs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/denoising_dynamics_census_20261022/attempt001"
MODEL = "/home/exouser/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b"
MODEL_REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"
AIME_SOURCE = ROOT / "results/v31_value_matrix_20261011/private/aime26/aime26_generation_manifest.json"
HUMANEVAL_SOURCE = ROOT / "results/v31_value_matrix_20261011/private/quick100/humaneval_generation_manifest.json"
LB_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
LB_SOURCE = Path("/home/exouser/.cache/huggingface/hub/datasets--THUDM--LongBench-v2/snapshots") / LB_REVISION / "data.json"
LB_CHECKOUT = ROOT / "reference/LongBench"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def stable_key(prefix: str, value: str) -> str:
    return sha_bytes(f"denoising-dynamics-screen-v1|{prefix}|{value}".encode())


def atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"frozen file differs; refusing overwrite: {path}")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def official_longbench_prompt(item: dict[str, Any]) -> str:
    template = (LB_CHECKOUT / "prompts/0shot.txt").read_text(encoding="utf-8")
    for marker, field in (
        ("DOC", "context"),
        ("Q", "question"),
        ("C_A", "choice_A"),
        ("C_B", "choice_B"),
        ("C_C", "choice_C"),
        ("C_D", "choice_D"),
    ):
        template = template.replace(f"${marker}$", str(item[field]).strip())
    return template


def _select(rows: Iterable[dict[str, Any]], count: int, prefix: str, id_field: str) -> list[dict[str, Any]]:
    rows = list(rows)
    ids = [str(row[id_field]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate source IDs in {prefix}")
    if len(rows) < count:
        raise ValueError(f"{prefix}: need {count}, found {len(rows)}")
    return sorted(rows, key=lambda row: stable_key(prefix, str(row[id_field])))[:count]


def build_rows() -> dict[str, list[dict[str, Any]]]:
    if not AIME_SOURCE.is_file() or not HUMANEVAL_SOURCE.is_file() or not LB_SOURCE.is_file():
        raise FileNotFoundError("one or more pinned local sources are unavailable")

    aime_all = json.loads(AIME_SOURCE.read_text(encoding="utf-8"))
    human_all = json.loads(HUMANEVAL_SOURCE.read_text(encoding="utf-8"))
    longbench_all = json.loads(LB_SOURCE.read_text(encoding="utf-8"))

    aime = []
    for row in _select(aime_all, 10, "aime26", "source_id"):
        aime.append({
            "id": f"aime26/{row['source_id']}",
            "source_id": str(row["source_id"]),
            "benchmark": "aime26",
            "prompt": row["prompt"],
            "prompt_hash": sha_bytes(row["prompt"].encode()),
            "expected": str(row["expected"]),
            "generation_budget": int(row["generation_budget"]),
            "seed": int(row.get("seed", 42)),
            "thinking": bool(row.get("thinking", True)),
        })

    human = []
    for row in _select(human_all, 10, "humaneval", "source_id"):
        human.append({
            "id": f"humaneval/{row['source_id']}",
            "source_id": str(row["source_id"]),
            "benchmark": "humaneval",
            "prompt": row["prompt"],
            "prompt_hash": sha_bytes(row["prompt"].encode()),
            "generation_budget": int(row["generation_budget"]),
            "seed": int(row.get("seed", 42)),
            "thinking": bool(row.get("thinking", True)),
        })

    # The long-context setting is the point of this study.  Select two long
    # examples from each long-capable domain.  Long-dialogue has no long-band
    # examples in the pinned release and is therefore not silently substituted.
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for row in longbench_all:
        if row.get("length") == "long":
            by_domain.setdefault(str(row["domain"]), []).append(row)
    domains = sorted(by_domain)
    if len(domains) != 5 or any(len(by_domain[d]) < 2 for d in domains):
        raise ValueError(f"unexpected LongBench long-domain inventory: { {d: len(v) for d, v in by_domain.items()} }")
    longbench = []
    for domain in domains:
        for row in _select(by_domain[domain], 2, f"longbench-v2/{domain}", "_id"):
            prompt = official_longbench_prompt(row)
            longbench.append({
                "id": f"longbench_v2/{row['_id']}",
                "source_id": str(row["_id"]),
                "benchmark": "longbench_v2",
                "domain": domain,
                "sub_domain": str(row["sub_domain"]),
                "difficulty": str(row["difficulty"]),
                "length_band": str(row["length"]),
                "prompt": prompt,
                "prompt_hash": sha_bytes(prompt.encode()),
                "expected": str(row["answer"]),
                "generation_budget": 16384,
                "seed": 42,
                "thinking": True,
            })
    longbench.sort(key=lambda row: row["id"])
    return {"aime26": aime, "longbench_v2": longbench, "humaneval": human}


def source_fingerprints() -> dict[str, Any]:
    official = {}
    for name in ("pred.py", "prompts/0shot.txt", "result.py"):
        path = LB_CHECKOUT / name
        official[name] = sha_file(path)
    return {
        "model_revision": MODEL_REVISION,
        "model_path": MODEL,
        "aime_manifest": {"path": str(AIME_SOURCE), "sha256": sha_file(AIME_SOURCE)},
        "humaneval_manifest": {"path": str(HUMANEVAL_SOURCE), "sha256": sha_file(HUMANEVAL_SOURCE)},
        "longbench_data": {"path": str(LB_SOURCE), "revision": LB_REVISION, "sha256": sha_file(LB_SOURCE)},
        "longbench_official_code": official,
    }


def public_rows(rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    result = []
    for benchmark in ("longbench_v2", "aime26", "humaneval"):
        for row in rows[benchmark]:
            result.append({key: row[key] for key in row if key not in ("prompt", "expected")})
    return result


def build(output: Path = OUT) -> dict[str, Any]:
    rows = build_rows()
    identity = public_rows(rows)
    sources = source_fingerprints()
    protocol = {
        "schema": "denoising_dynamics_census_v1",
        "purpose": "D0 native trajectory census before any sparse-selector rollout",
        "selection": "SHA256 ordering within each pinned source; ten AIME26, ten LongBench-v2 long-band, ten HumanEval; no score or dense-call filtering",
        "counts": {key: len(value) for key, value in rows.items()},
        "native_sampler": {
            "canvas_length": 256,
            "max_denoising_steps": 48,
            "temperature_schedule": "t_min=0.4, t_max=0.8, linear native schedule",
            "entropy_bound": 0.1,
            "confidence_threshold": 0.005,
            "stability_threshold": 1,
            "re_noise": "uniform native sampler for rejected positions",
            "self_conditioning": "native processed logits",
            "stopping": "whole-canvas stability AND mean processed-logit entropy; native EOS/finalization unchanged",
            "thinking": True,
            "seed": 42,
        },
        "arms": ["dense_native_hf"],
        "instrumentation": "opt-in read-only denoising-step observer; no extra forwards, no RNG draws, no sampler replacement",
        "sources": sources,
        "public_identity_sha256": sha_bytes(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()),
    }
    atomic_write(output / "protocol.json", protocol)
    atomic_write(output / "manifest_identity.json", identity)
    atomic_write(output / "source_fingerprints.json", sources)
    private = output / "private" / "manifest.json"
    atomic_write(private, [row for benchmark in ("longbench_v2", "aime26", "humaneval") for row in rows[benchmark]])
    receipt = {
        "schema": "denoising_dynamics_census_receipt_v1",
        "protocol_sha256": sha_file(output / "protocol.json"),
        "manifest_identity_sha256": sha_file(output / "manifest_identity.json"),
        "private_manifest_sha256": sha_file(private),
        "counts": protocol["counts"],
        "gpu_launched": False,
        "native_observer_installed": False,
        "status": "frozen_preflight",
    }
    atomic_write(output / "receipt.json", receipt)
    return protocol


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
