"""Hardware-friendly selective proxy policies with a leakage-resistant API.

Policies can return only PRE_SKIP or EXACT.  ``ProxyContext`` intentionally
contains coordinates but no target score, skip label, or running maximum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
from typing import Mapping, Protocol, Sequence


class Decision(str, Enum):
    PRE_SKIP = "pre_skip"
    EXACT = "exact"


@dataclass(frozen=True)
class ProxyContext:
    sample_id: str
    layer: int
    head: int
    kv_group: int
    denoising_iteration: int
    noise_bucket: str
    query_tile: int
    kv_tile: int
    traversal_index: int
    num_kv_tiles: int
    q_to_kv_tile_ratio: int = 2

    @property
    def is_diagonal(self) -> bool:
        start = self.query_tile * self.q_to_kv_tile_ratio
        return start <= self.kv_tile < start + self.q_to_kv_tile_ratio

    @property
    def is_local(self) -> bool:
        start = self.query_tile * self.q_to_kv_tile_ratio
        return start - 1 <= self.kv_tile <= start + self.q_to_kv_tile_ratio

    @property
    def is_sink(self) -> bool:
        return self.kv_tile == 0


@dataclass(frozen=True)
class SourceMetadata:
    score: float | None
    skipped: bool
    introduced_new_max: bool
    available: bool = True
    was_pre_skipped: bool = False
    noise_bucket: str | None = None


class SourceResolver(Protocol):
    def resolve(self, context: ProxyContext, relation: str) -> SourceMetadata | None: ...


class ProxyPolicy(Protocol):
    name: str

    def decide(self, context: ProxyContext, sources: SourceResolver) -> Decision: ...


TileIdentity = tuple[str, int, int, int, int, int]


@dataclass
class MetadataStore:
    """Tile metadata keyed by sample, step, layer, head, Q tile, KV tile."""

    records: dict[TileIdentity, SourceMetadata] = field(default_factory=dict)
    leaders: Mapping[tuple[int, int, str], int] = field(default_factory=dict)
    cluster_leaders: Mapping[tuple[int, int, str], int] = field(default_factory=dict)

    @staticmethod
    def key(context: ProxyContext) -> TileIdentity:
        return (
            context.sample_id,
            context.denoising_iteration,
            context.layer,
            context.head,
            context.query_tile,
            context.kv_tile,
        )

    def write(self, context: ProxyContext, metadata: SourceMetadata) -> None:
        self.records[self.key(context)] = metadata

    def resolve(self, context: ProxyContext, relation: str) -> SourceMetadata | None:
        step, layer, head = context.denoising_iteration, context.layer, context.head
        if relation == "previous_step":
            step -= 1
        elif relation == "previous_layer":
            layer -= 1
        elif relation.startswith("head:"):
            head = int(relation.split(":", 1)[1])
            if head >= context.head:
                return None
        elif relation == "leader_head":
            head = self.leaders.get((layer, head, context.noise_bucket), -1)
            if head < 0 or head >= context.head:
                return None
        elif relation == "cluster_leader":
            head = self.cluster_leaders.get((layer, head, context.noise_bucket), -1)
            if head < 0 or head >= context.head:
                return None
        else:
            raise ValueError(f"unknown source relation: {relation}")
        if step < 0 or layer < 0:
            return None
        return self.records.get(
            (context.sample_id, step, layer, head, context.query_tile, context.kv_tile)
        )


@dataclass(frozen=True)
class ThresholdTable:
    """Transition-aware thresholds with conservative hierarchical fallbacks."""

    values: Mapping[tuple[object, ...], float]
    default: float = -math.inf

    def get(self, context: ProxyContext, source_noise_bucket: str | None = None) -> float:
        source_bucket = source_noise_bucket or "unknown"
        target_bucket = context.noise_bucket
        candidates = (
            (context.layer, context.head, source_bucket, target_bucket, context.num_kv_tiles),
            (context.layer, -1, source_bucket, target_bucket, context.num_kv_tiles),
            (-1, -1, source_bucket, target_bucket, context.num_kv_tiles),
            (context.layer, context.head, source_bucket, target_bucket),
            (context.layer, -1, source_bucket, target_bucket),
            (-1, -1, source_bucket, target_bucket),
            (context.layer, context.head, target_bucket),
            (context.layer, -1, target_bucket),
        )
        for key in candidates:
            if key in self.values:
                return float(self.values[key])
        return self.default


def load_calibrated_thresholds(
    path: str | Path,
    *,
    relation: str,
    false_skip_budget: float,
) -> ThresholdTable:
    """Load only thresholds that passed calibration confidence checks."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not payload.get("deployment_schema_eligible", False):
        return ThresholdTable({})
    values: dict[tuple[object, ...], float] = {}
    for row in payload.get("deployable_thresholds", []):
        if row["relation"] != relation or not math.isclose(
            float(row["false_skip_budget"]), false_skip_budget, rel_tol=0.0, abs_tol=1e-15
        ):
            continue
        threshold = row.get("threshold")
        if threshold is None or not row.get("certified", False):
            continue
        key: tuple[object, ...] = (
            int(row["layer"]), int(row["head"]),
            str(row["source_noise_bucket"]), str(row["target_noise_bucket"]),
        )
        if row.get("num_kv_tiles") is not None:
            key += (int(row["num_kv_tiles"]),)
        values[key] = float(threshold)
    return ThresholdTable(values)


@dataclass(frozen=True)
class DisabledPolicy:
    name: str = "disabled"

    def decide(self, context: ProxyContext, sources: SourceResolver) -> Decision:
        del context, sources
        return Decision.EXACT


@dataclass(frozen=True)
class BinaryReusePolicy:
    relation: str
    name: str = "binary_reuse"

    def decide(self, context: ProxyContext, sources: SourceResolver) -> Decision:
        source = sources.resolve(context, self.relation)
        return Decision.PRE_SKIP if source is not None and source.available and source.skipped else Decision.EXACT


@dataclass(frozen=True)
class ScoreThresholdPolicy:
    relation: str
    thresholds: ThresholdTable
    name: str = "score_threshold"

    def decide(self, context: ProxyContext, sources: SourceResolver) -> Decision:
        source = sources.resolve(context, self.relation)
        threshold = self.thresholds.get(context, source.noise_bucket if source is not None else None)
        if source is None or not source.available or source.score is None or not math.isfinite(threshold):
            return Decision.EXACT
        return Decision.PRE_SKIP if source.score < threshold else Decision.EXACT


@dataclass(frozen=True)
class MaxScorePolicy:
    """Monotonic all-source rule: max(source scores) must be below tau."""

    relations: Sequence[str]
    thresholds: ThresholdTable
    name: str = "max_source_score"

    def decide(self, context: ProxyContext, sources: SourceResolver) -> Decision:
        values = []
        source_buckets: dict[str, str | None] = {}
        for relation in self.relations:
            source = sources.resolve(context, relation)
            if source is None or not source.available or source.score is None:
                return Decision.EXACT
            values.append(source.score)
            source_buckets[relation] = source.noise_bucket
        transition_bucket = source_buckets.get("previous_step")
        if transition_bucket is None and source_buckets:
            transition_bucket = next(iter(source_buckets.values()))
        threshold = self.thresholds.get(context, transition_bucket)
        return Decision.PRE_SKIP if values and max(values) < threshold else Decision.EXACT


@dataclass(frozen=True)
class AllBinaryPolicy:
    relations: Sequence[str]
    name: str = "all_binary"

    def decide(self, context: ProxyContext, sources: SourceResolver) -> Decision:
        values = [sources.resolve(context, relation) for relation in self.relations]
        return (
            Decision.PRE_SKIP
            if values and all(item is not None and item.available and item.skipped for item in values)
            else Decision.EXACT
        )


@dataclass(frozen=True)
class LogisticPolicy:
    """Tiny calibrated linear model over source log-scores."""

    relations: Sequence[str]
    weights: Sequence[float]
    bias: float
    probability_threshold: float
    name: str = "logistic"

    def __post_init__(self) -> None:
        if len(self.relations) != len(self.weights):
            raise ValueError("one weight is required per relation")

    def decide(self, context: ProxyContext, sources: SourceResolver) -> Decision:
        value = self.bias
        for relation, weight in zip(self.relations, self.weights):
            source = sources.resolve(context, relation)
            if source is None or not source.available or source.score is None or source.score < 0:
                return Decision.EXACT
            value += weight * math.log(max(source.score, 1e-30))
        probability = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))
        return Decision.PRE_SKIP if probability >= self.probability_threshold else Decision.EXACT


@dataclass(frozen=True)
class SafetyPolicy:
    """Force exact anchors, warmup tiles, and periodic refreshes."""

    inner: ProxyPolicy
    warmup_tiles: int = 0
    periodic_refresh: int = 0
    anchor_local: bool = True
    anchor_diagonal: bool = True
    anchor_sink: bool = True
    name: str = "safe"

    def forces_exact(self, context: ProxyContext) -> bool:
        force_exact = context.traversal_index < self.warmup_tiles
        force_exact |= self.periodic_refresh > 0 and context.traversal_index % self.periodic_refresh == 0
        force_exact |= self.anchor_local and context.is_local
        force_exact |= self.anchor_diagonal and context.is_diagonal
        force_exact |= self.anchor_sink and context.is_sink
        return force_exact

    def decide(self, context: ProxyContext, sources: SourceResolver) -> Decision:
        return Decision.EXACT if self.forces_exact(context) else self.inner.decide(context, sources)
