"""Versioned, compact schema for physical attention-tile traces."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "proxy-blasst-physical-v3"
READABLE_SCHEMA_VERSIONS = {
    "proxy-blasst-physical-v1", "proxy-blasst-physical-v2", SCHEMA_VERSION
}

# Strings are intentionally fixed width so shards remain mmap/concatenate
# friendly. Binary columns use uint8 and can additionally be packed by the
# collector when the storage experiment flag is enabled.
TRACE_DTYPE = np.dtype(
    [
        ("sample_id", "U64"),
        ("request_id", "U64"),
        ("seed", "<i8"),
        ("sequence_length", "<i4"),
        ("diffusion_block", "<i2"),
        ("denoising_iteration", "<i2"),
        ("remaining_mask_ratio", "<f4"),
        ("noise_bucket", "U8"),
        ("layer", "<i2"),
        ("head", "<i2"),
        ("kv_group", "<i2"),
        ("query_tile", "<i4"),
        ("kv_tile", "<i4"),
        ("traversal_index", "<i4"),
        ("valid_query_rows", "<i2"),
        ("score", "<f4"),
        ("log_score", "<f4"),
        ("exact_skip", "u1"),
        ("introduced_new_max", "u1"),
        ("row_max", "<f4"),
        ("row_q50", "<f4"),
        ("row_q90", "<f4"),
        ("row_q99", "<f4"),
        ("output_contribution_norm", "<f4"),
        ("is_local", "u1"),
        ("is_diagonal", "u1"),
        ("is_sink", "u1"),
        ("is_distant", "u1"),
    ]
)


def write_manifest(directory: Path, **metadata: object) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dtype": TRACE_DTYPE.descr,
        **metadata,
    }
    (directory / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def validate_or_write_manifest(directory: Path, **metadata: object) -> None:
    """Allow intentional multi-step appends but reject incompatible shards."""
    path = directory / "manifest.json"
    if not path.exists():
        write_manifest(directory, **metadata)
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"trace directory uses {existing.get('schema_version')!r}; choose a fresh directory for {SCHEMA_VERSION!r}"
        )
    for key, value in metadata.items():
        if existing.get(key) != value:
            raise ValueError(f"trace manifest mismatch for {key}: {existing.get(key)!r} != {value!r}")


def load_trace_directory(path: str | Path) -> np.ndarray:
    """Load all complete shards after validating their schema version."""
    directory = Path(path)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest_version = manifest.get("schema_version")
    if manifest_version not in READABLE_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported trace schema {manifest_version!r}; expected one of {sorted(READABLE_SCHEMA_VERSIONS)!r}"
        )
    arrays = []
    for shard in sorted(directory.glob("trace-*.npz")):
        with np.load(shard, allow_pickle=False) as payload:
            if str(payload["schema_version"].item()) != manifest_version:
                raise ValueError(f"schema mismatch in {shard}")
            arrays.append(payload["records"].astype(TRACE_DTYPE, copy=False))
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=TRACE_DTYPE)
