"""Collect bounded rich supervisors on native-dense DiffusionGemma trajectories.

This observer never changes attention output.  It augments the preserved
same-state protocol with the quantities missing from the original snapshots:
token-score quantiles, exact entropy, V norms, tile contribution norms, and
the exact drop-one-tile output perturbation.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dllm.attention.blasst.core import _attention_type, _expand_valid_mask, _prepare_attention_scores
from dllm.evaluation.ruler.io import read_jsonl, write_json
from dllm.evaluation.ruler.runner import RulerRunConfig, run_evaluation
from experiments.diffusion_attention_threshold_modeling.proxy import _region_proxies
from experiments.diffusion_gemma_blasst_diagnosis.diagnostics import select_rows
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.runner import _runner_manifest
from experiments.diffusion_gemma_blasst_diagnosis.runner import BASE, MODEL, REVISION, RULER


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "results/diffusion_gemma_sparse_attention_research/rich_same_state"
SAME_STATE_DENSE = REPO / "results/diffusion_gemma_blasst_diagnosis/same_state/predictions.jsonl"


@torch.no_grad()
def summarize_tile_supervisors(
    scores: torch.Tensor,
    valid: torch.Tensor,
    probabilities: torch.Tensor,
    values: torch.Tensor,
    tile_size: int = 64,
) -> dict[str, torch.Tensor]:
    """Summarize one head/query block; inputs are [Q,K], [Q,K], [Q,K], [K,D]."""
    query_count, kv_length = scores.shape
    padding = (-kv_length) % tile_size
    tile_count = (kv_length + padding) // tile_size
    score_blocks = F.pad(scores, (0, padding), value=-torch.inf).reshape(query_count, tile_count, tile_size)
    valid_blocks = F.pad(valid, (0, padding), value=False).reshape(query_count, tile_count, tile_size)
    probability_blocks = F.pad(probabilities, (0, padding)).reshape(query_count, tile_count, tile_size)
    padded_values = F.pad(values, (0, 0, 0, padding)).reshape(tile_count, tile_size, -1)

    row_tile_mass = probability_blocks.sum(-1)
    row_tile_max = score_blocks.max(-1).values
    eligible = valid_blocks.any((0, 2))

    flattened_scores = score_blocks.permute(1, 0, 2).reshape(tile_count, -1)
    flattened_valid = valid_blocks.permute(1, 0, 2).reshape(tile_count, -1)
    tile_q95 = torch.nanquantile(
        torch.where(flattened_valid, flattened_scores, torch.nan), 0.95, dim=-1
    )
    tile_max = flattened_scores.masked_fill(~flattened_valid, -torch.inf).max(-1).values

    key_valid = valid.any(0)
    key_valid_blocks = F.pad(key_valid, (0, padding), value=False).reshape(tile_count, tile_size)
    value_norm = padded_values.norm(dim=-1)
    value_count = key_valid_blocks.sum(-1).clamp_min(1)
    tile_value_rms = (
        (value_norm.square() * key_valid_blocks).sum(-1) / value_count
    ).sqrt()
    tile_value_max = value_norm.masked_fill(~key_valid_blocks, -torch.inf).max(-1).values

    contributions = torch.einsum("qtk,tkd->qtd", probability_blocks, padded_values)
    dense_output = contributions.sum(1)
    tile_contribution_l2 = contributions.square().sum((0, 2)).sqrt()
    retained_mass = 1.0 - row_tile_mass
    drop_output = (dense_output[:, None, :] - contributions) / retained_mass[:, :, None].clamp_min(1e-30)
    drop_delta = drop_output - dense_output[:, None, :]
    tile_drop_output_l2 = drop_delta.square().sum((0, 2)).sqrt()
    dense_output_l2 = dense_output.square().sum().sqrt()

    valid_counts = valid.sum(-1)
    token_entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
    normalized_token_entropy = token_entropy / valid_counts.clamp_min(2).log()
    tile_entropy = -(row_tile_mass * row_tile_mass.clamp_min(1e-30).log()).sum(-1)
    tile_counts = valid_blocks.any(-1).sum(-1)
    normalized_tile_entropy = tile_entropy / tile_counts.clamp_min(2).log()

    sorted_mass, sorted_index = row_tile_mass.sort(dim=-1, descending=True, stable=True)
    cumulative = sorted_mass.cumsum(-1)
    support_sorted = (cumulative - sorted_mass) < 0.90
    support = torch.zeros_like(support_sorted, dtype=torch.bool)
    support.scatter_(-1, sorted_index, support_sorted)
    support &= valid_blocks.any(-1)

    return {
        "eligible": eligible,
        "row_tile_mass": row_tile_mass,
        "row_tile_max": row_tile_max,
        "tile_q95": tile_q95,
        "tile_max": tile_max,
        "tile_mass": row_tile_mass.sum(0),
        "tile_value_rms": tile_value_rms,
        "tile_value_max": tile_value_max,
        "tile_contribution_l2": tile_contribution_l2,
        "tile_drop_output_l2": tile_drop_output_l2,
        "dense_output_l2": dense_output_l2,
        "normalized_token_entropy": normalized_token_entropy,
        "normalized_tile_entropy": normalized_tile_entropy,
        "query_mass_support90": support,
    }


class RichObserver:
    def __init__(self, directory: Path, rows: list[dict]):
        self.directory = directory
        self.indices = {row["sample_id"]: index for index, row in enumerate(rows)}
        self.split = {row["sample_id"]: row["diagnostic_split"] for row in rows}
        self.example: str | None = None
        self.records: list[dict] = []
        self.arrays: dict[str, np.ndarray] = {}
        self.calls: defaultdict[int, int] = defaultdict(int)
        self.dirty = False

    def flush(self) -> None:
        if self.example is None or not self.dirty:
            return
        destination = self.directory / "shards" / self.example
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / "rich_snapshots.npz", **self.arrays)
        write_json(destination / "records.json", self.records)
        self.dirty = False

    @torch.no_grad()
    def __call__(self, module, query, key, value, attention_mask=None, **kwargs):
        runtime = module._blasst_2d_runtime
        runtime.attention_output_guard = True
        metadata = dict(runtime.metadata)
        example = metadata["example_id"]
        if example != self.example:
            self.flush()
            self.example = example
            self.records, self.arrays = [], {}
            self.calls = defaultdict(int)

        layer = int(module.layer_idx)
        call = self.calls[layer]
        self.calls[layer] += 1
        sample_index = self.indices[example]
        head = sample_index % query.shape[1]
        query_start = (sample_index % max(1, math.ceil(query.shape[-2] / 64))) * 64
        query_end = min(query_start + 64, query.shape[-2])

        expanded_key, expanded_value, all_scores, all_valid = _prepare_attention_scores(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=kwargs.get("scaling"),
            is_causal=kwargs.get("is_causal"),
            sliding_window=kwargs.get("sliding_window"),
        )
        all_valid = _expand_valid_mask(all_valid, all_scores)
        scores = all_scores[0, head, query_start:query_end].float()
        valid = all_valid[0, head, query_start:query_end]
        probabilities = scores.softmax(-1)
        probabilities = torch.where(valid.any(-1, keepdim=True), probabilities, 0.0)
        values = expanded_value[0, head].float()
        summaries = summarize_tile_supervisors(scores, valid, probabilities, values)

        scale = float(kwargs.get("scaling") or query.shape[-1] ** -0.5)
        proxy, proxy_eligible, _ = _region_proxies(
            query[:, head : head + 1],
            expanded_key[:, head : head + 1],
            valid[None, None],
            q_start=query_start,
            q_end=query_end,
            kv_start=0,
            kv_end=all_scores.shape[-1],
            kv_block_size=64,
            scaling=scale,
        )
        summaries["proxy"] = proxy[0, 0]
        if not torch.equal(summaries["eligible"], proxy_eligible[0, 0]):
            raise RuntimeError("proxy and exact supervisor eligibility disagree")

        snapshot_key = f"l{layer}_c{call}"
        for name, tensor in summaries.items():
            array = tensor.detach().cpu().numpy()
            if name == "query_mass_support90":
                array = np.packbits(array, axis=-1)
            self.arrays[f"{snapshot_key}_{name}"] = array
        self.records.append(
            {
                "example_id": example,
                "split": self.split[example],
                "layer": layer,
                "call": call,
                "head": int(head),
                "query_start": int(query_start),
                "query_length": int(query.shape[-2]),
                "kv_length": int(all_scores.shape[-1]),
                "attention_type": _attention_type(module, kwargs.get("sliding_window")),
                "metadata": metadata,
                "snapshot_key": snapshot_key,
                "packed_query_rows": int(query_end - query_start),
                "packed_tile_count": int(summaries["eligible"].numel()),
            }
        )
        self.dirty = True


def collect(root: Path = ROOT, num_samples: int = 16) -> dict:
    root = root.resolve()
    torch.backends.cuda.matmul.allow_tf32 = False
    study, rows = select_rows()
    rows = rows[:num_samples]
    root.mkdir(parents=True, exist_ok=True)
    manifest = _runner_manifest(study, rows, root / "runner_manifest.json", adapter="diffusion_gemma")
    write_json(
        root / "protocol.json",
        {
            "purpose": "rich supervisor collection; observer does not modify attention output",
            "source_selection": "same deterministic 16-prompt protocol as diffusion_gemma_blasst_diagnosis",
            "tile_shape": [64, 64],
            "head": "sample_index modulo 16",
            "query_tile": "sample_index modulo 4",
            "samples": [row["sample_id"] for row in rows],
            "model_revision": REVISION,
            "dense_reference": str(SAME_STATE_DENSE),
        },
    )
    observer = RichObserver(root, rows)
    config = RulerRunConfig(
        model_adapter="diffusion_gemma",
        model_path=MODEL,
        revision=REVISION,
        manifest_path=str(manifest),
        ruler_root=RULER,
        output_dir=str(root),
        num_samples=len(rows),
        context_length=16384,
        attention_backend="eager-dense",
        temperature=0.0,
        precision="bfloat16",
        progress_every=1,
    )
    try:
        summary = run_evaluation(config, attention_observer=observer, before_prediction_commit=observer.flush)
    finally:
        observer.flush()

    dense = {row["sample_id"]: row for row in read_jsonl(SAME_STATE_DENSE)}
    predictions = read_jsonl(root / "predictions.jsonl")
    parity = {
        row["sample_id"]: row["completion_tokens"] == dense[row["sample_id"]]["completion_tokens"]
        for row in predictions
        if row["sample_id"] in dense
    }
    write_json(root / "native_dense_parity.json", parity)
    if len(parity) != len(rows) or not all(parity.values()):
        raise RuntimeError("rich observer changed native dense output")
    audit_collection(root, len(rows))
    return summary


def audit_collection(root: Path, expected_samples: int) -> dict:
    parity = json.loads((root / "native_dense_parity.json").read_text())
    records: list[dict] = []
    failures: list[str] = []
    array_count = 0
    for shard in sorted((root / "shards").iterdir()):
        if not shard.is_dir():
            continue
        shard_records = json.loads((shard / "records.json").read_text())
        records.extend(shard_records)
        with np.load(shard / "rich_snapshots.npz") as arrays:
            array_count += len(arrays.files)
            for record in shard_records:
                key = record["snapshot_key"]
                eligible = arrays[key + "_eligible"].astype(bool)
                row_mass = arrays[key + "_row_tile_mass"]
                for name in ("tile_q95", "tile_max", "tile_value_max"):
                    if not np.isfinite(arrays[key + "_" + name][eligible]).all():
                        failures.append(f"{shard.name}:{key}_{name}:eligible_nonfinite")
                row_max = arrays[key + "_row_tile_max"]
                if not np.isfinite(row_max[row_mass > 0]).all():
                    failures.append(f"{shard.name}:{key}_row_tile_max:valid_nonfinite")
                for name in (
                    "tile_mass",
                    "tile_value_rms",
                    "tile_contribution_l2",
                    "tile_drop_output_l2",
                    "normalized_token_entropy",
                    "normalized_tile_entropy",
                    "proxy",
                ):
                    values = arrays[key + "_" + name]
                    if name.startswith("tile_") or name == "proxy":
                        values = values[eligible]
                    if not np.isfinite(values).all():
                        failures.append(f"{shard.name}:{key}_{name}:nonfinite")
    audit = {
        "complete": (
            len({record["example_id"] for record in records}) == expected_samples
            and len(parity) == expected_samples
            and all(parity.values())
            and not failures
        ),
        "samples": len({record["example_id"] for record in records}),
        "records": len(records),
        "arrays": array_count,
        "layers": sorted({record["layer"] for record in records}),
        "heads": sorted({record["head"] for record in records}),
        "attention_types": sorted({record["attention_type"] for record in records}),
        "native_dense_parity": parity,
        "eligibility_aware_finiteness_failures": failures,
        "sentinel_policy": "-inf/NaN outside structural eligibility is excluded; every eligible diagnostic is finite",
    }
    write_json(root / "audit.json", audit)
    if not audit["complete"]:
        raise RuntimeError(f"rich collection audit failed: {failures[:5]}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--num-samples", type=int, default=16)
    args = parser.parse_args()
    result = collect(args.output, args.num_samples)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
