"""Versioned schema for sampled per-row physical-tile veto traces."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "blasst-veto-row-v1"

TOKEN_STATE_NAMES = {
    0: "masked",
    1: "newly_revealed",
    2: "previously_revealed",
    3: "stable_visible",
    4: "prefix",
}

METADATA_DTYPE = np.dtype(
    [
        ("sample_id", "U64"),
        ("request_id", "U64"),
        ("seed", "<i8"),
        ("sequence_length", "<i4"),
        ("diffusion_block", "<i2"),
        ("denoising_step", "<i2"),
        ("remaining_mask_ratio", "<f4"),
        ("noise_bucket", "U8"),
        ("layer", "<i2"),
        ("head", "<i2"),
        ("query_tile", "<i4"),
        ("kv_tile", "<i4"),
        ("traversal_index", "<i4"),
        ("valid_query_rows", "<i2"),
        ("threshold", "<f4"),
        ("exact_blasst_skip", "u1"),
        ("introduced_new_max", "u1"),
        ("v_mean_row_norm", "<f4"),
        ("v_max_row_norm", "<f4"),
        ("v_frobenius_per_sqrt_rows", "<f4"),
        ("v_centroid_norm", "<f4"),
        ("v_residual_max_norm", "<f4"),
        ("v_per_dimension_variance", "<f4"),
    ]
)

ROW_FLOAT_FIELDS = (
    "row_score",
    "row_log_score",
    "row_local_max",
    "row_running_max_before",
    "row_running_sum_before",
    "row_local_logsumexp",
    "row_final_logsumexp",
    "row_final_output_norm",
    "row_local_value_norm",
    "row_centroid_error_norm",
    "row_query_delta",
)

ROW_UINT8_FIELDS = ("row_keep_vote", "row_token_state")


def write_manifest(directory: Path, **metadata: object) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metadata_dtype": METADATA_DTYPE.descr,
        "row_float_fields": list(ROW_FLOAT_FIELDS),
        "row_uint8_fields": list(ROW_UINT8_FIELDS),
        "token_states": TOKEN_STATE_NAMES,
        **metadata,
    }
    (directory / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def validate_or_write_manifest(directory: Path, **metadata: object) -> None:
    path = directory / "manifest.json"
    if not path.exists():
        write_manifest(directory, **metadata)
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"trace directory uses {existing.get('schema_version')!r}; "
            f"choose a fresh directory for {SCHEMA_VERSION!r}"
        )
    for key, value in metadata.items():
        if existing.get(key) != value:
            raise ValueError(
                f"trace manifest mismatch for {key}: {existing.get(key)!r} != {value!r}"
            )


def iter_veto_trace_shards(path: str | Path):
    directory = Path(path)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported veto trace schema {manifest.get('schema_version')!r}")
    for shard in sorted(directory.glob("veto-trace-*.npz")):
        with np.load(shard, allow_pickle=False) as payload:
            if str(payload["schema_version"].item()) != SCHEMA_VERSION:
                raise ValueError(f"schema mismatch in {shard}")
            result = {"metadata": payload["metadata"].astype(METADATA_DTYPE, copy=False)}
            for name in (*ROW_FLOAT_FIELDS, *ROW_UINT8_FIELDS):
                result[name] = payload[name]
            yield shard, result
