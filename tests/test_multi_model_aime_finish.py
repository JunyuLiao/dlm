import hashlib
import json

import pytest

from scripts.multi_model_aime_finish import ARM, benchmark_counts


def fixture_shards(tmp_path):
    base, rerun = tmp_path / "base", tmp_path / "rerun"
    leaves = []
    for shard in range(1, 5):
        relative = f"runs/llada22/aime26/{ARM}/shard{shard:03}/attempt001"
        leaf, old = rerun / relative, base / relative
        leaf.mkdir(parents=True)
        old.mkdir(parents=True)
        rows = [{"id": f"aime26/{i // 3 + 1:02}", "seed": 42 + i % 3,
                 "counts": {"eligible_tiles": 100, "kept_tiles": 40}}
                for i in range(shard - 1, 90, 4)]
        text = "".join(json.dumps(r) + "\n" for r in rows)
        for p in (leaf, old):
            (p / "records.jsonl").write_text(text)
        (leaf / "final_complete.json").write_text('{}\n')
        (leaf / "receipt.json").write_text(json.dumps({
            "records_sha256": hashlib.sha256(text.encode()).hexdigest()}))
        leaves.append(leaf)
    return base, rerun, leaves


def test_complete_target_miss_is_disclosed(tmp_path):
    base, rerun, _ = fixture_shards(tmp_path)
    result = benchmark_counts(base, rerun, "llada22")
    assert result["requests"] == 90
    assert result["sparsity"] == .6
    assert result["target_met"] is False
    assert result["post_benchmark_recalibration"] is False


def test_tampered_shard_cannot_be_accepted(tmp_path):
    base, rerun, leaves = fixture_shards(tmp_path)
    with (leaves[0] / "records.jsonl").open("a") as f:
        f.write('{}\n')
    with pytest.raises(RuntimeError, match="hash mismatch"):
        benchmark_counts(base, rerun, "llada22")


def test_incomplete_shard_cannot_be_accepted(tmp_path):
    base, rerun, leaves = fixture_shards(tmp_path)
    (leaves[3] / "final_complete.json").unlink()
    with pytest.raises(FileNotFoundError):
        benchmark_counts(base, rerun, "llada22")
