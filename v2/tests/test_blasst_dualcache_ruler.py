from __future__ import annotations

import csv
import sys
from pathlib import Path


V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

from scripts.eval_blasst_dualcache_ruler import (  # noqa: E402
    _all_configurations,
    _write_csv,
)


def test_dualcache_configuration_matrix_is_fixed() -> None:
    configurations = _all_configurations()
    assert [row["configuration"] for row in configurations] == [
        "dense",
        "sparse_no_dualcache_b4",
        "sparse_no_dualcache_b8",
        "sparse_no_dualcache_b16",
        "sparse_no_dualcache_b32",
        "sparse_no_dualcache_b64",
        "dualcache_s4",
        "dualcache_s8",
        "dualcache_s16",
        "dualcache_s32",
    ]
    assert [
        (
            row["outer_block_size"],
            row["sub_block_size"],
            row["dual_cache_enabled"],
        )
        for row in configurations
    ] == [
        (64, 64, False),
        (4, 4, False),
        (8, 8, False),
        (16, 16, False),
        (32, 32, False),
        (64, 64, False),
        (64, 4, True),
        (64, 8, True),
        (64, 16, True),
        (64, 32, True),
    ]


def test_csv_writer_preserves_fields_present_only_in_later_rows(
    tmp_path: Path,
) -> None:
    output = tmp_path / "heterogeneous.csv"
    _write_csv(
        output,
        [
            {"configuration": "dense"},
            {
                "configuration": "sparse",
                "physical_tile_sparsity": 0.5,
            },
        ],
    )

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"configuration": "dense", "physical_tile_sparsity": ""},
        {"configuration": "sparse", "physical_tile_sparsity": "0.5"},
    ]
