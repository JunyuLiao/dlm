#!/usr/bin/env python3
"""H100 benchmark for compact BLASST tile-replacement predictors."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterator

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from blasst import DiffusionLambdaSchedule  # noqa: E402
from blasst.oracle_mass_pruning import (  # noqa: E402
    candidate_tiles,
    select_cumulative_budget,
    simulate_ordinary_blasst,
)
from blasst.tile_replacement_predictors import (  # noqa: E402
    analytic_log_z_predictions,
    kmeans_pool_summaries,
    normalized_mass_from_log_z,
    online_proxy_score,
    oracle_row_centroids,
    per_instance_entropy_identity,
    proxy_query_summaries,
    simple_value_predictions,
    summary_predictions,
    tiled_logits,
    uniform_pool_summaries,
)
from blasst.tile_replacement_reference import (  # noqa: E402
    ReplacementComponents,
    collect_tile_replacement_statistics,
    compose_replacement_state,
    exact_conditional_value,
    exact_log_z,
    replacement_rows,
)
from collect_proxy_traces import corpus, cyclic_contexts  # noqa: E402
from llada_eval_utils import (  # noqa: E402
    choose_mask_token_id,
    disable_use_cache,
    ensure_all_tied_weights_keys,
    patch_llada_transformers_compat,
)


class PairMoments:
    def __init__(self) -> None:
        self.n = 0
        self.x = self.y = self.xx = self.yy = self.xy = 0.0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        x = prediction.detach().double().flatten()
        y = target.detach().double().flatten()
        finite = torch.isfinite(x) & torch.isfinite(y)
        x, y = x[finite], y[finite]
        self.n += x.numel()
        self.x += float(x.sum())
        self.y += float(y.sum())
        self.xx += float(x.square().sum())
        self.yy += float(y.square().sum())
        self.xy += float((x * y).sum())

    def correlation(self) -> float:
        if self.n < 2:
            return 0.0
        covariance = self.xy - self.x * self.y / self.n
        xvar = self.xx - self.x * self.x / self.n
        yvar = self.yy - self.y * self.y / self.n
        return covariance / math.sqrt(max(xvar * yvar, 1e-30))


class ScalarMoments:
    def __init__(self) -> None:
        self.n = 0
        self.total = 0.0
        self.maximum = -math.inf
        self.minimum = math.inf

    def update(self, value: torch.Tensor) -> None:
        x = value.detach().double().flatten()
        x = x[torch.isfinite(x)]
        if not x.numel():
            return
        self.n += x.numel()
        self.total += float(x.sum())
        self.maximum = max(self.maximum, float(x.max()))
        self.minimum = min(self.minimum, float(x.min()))

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.n,
            "mean": self.total / self.n if self.n else 0.0,
            "maximum": self.maximum if self.n else 0.0,
            "minimum": self.minimum if self.n else 0.0,
        }


class MassMetrics:
    def __init__(self) -> None:
        self.log_z = PairMoments()
        self.mass = PairMoments()
        self.log_absolute = ScalarMoments()
        self.relative = ScalarMoments()
        self.worst_veto = ScalarMoments()
        self.ranking = ScalarMoments()
        self.top1 = ScalarMoments()

    def update(
        self,
        predicted_log_z: torch.Tensor,
        predicted_mass: torch.Tensor,
        exact_log_mass: torch.Tensor,
        exact_mass: torch.Tensor,
        candidates: torch.Tensor,
        veto_rows: torch.Tensor,
    ) -> None:
        selected = candidates[..., None] & veto_rows
        self.log_z.update(predicted_log_z[selected], exact_log_mass[selected])
        self.log_absolute.update((predicted_log_z - exact_log_mass).abs()[selected])
        self.mass.update(predicted_mass[selected], exact_mass[selected])
        relative = (predicted_mass - exact_mass).abs() / exact_mass.clamp_min(1e-12)
        self.relative.update(relative[selected])
        tile_worst = torch.where(veto_rows, relative, 0.0).amax(-1)
        self.worst_veto.update(tile_worst[candidates])
        exact_score = torch.where(veto_rows, exact_mass, -torch.inf).amax(-1)
        predicted_score = torch.where(veto_rows, predicted_mass, -torch.inf).amax(-1)
        for batch in range(candidates.shape[0]):
            for head in range(candidates.shape[1]):
                mask = candidates[batch, head]
                if int(mask.sum()) < 1:
                    continue
                left, right = predicted_score[batch, head, mask], exact_score[batch, head, mask]
                self.top1.update((left.argmax() == right.argmax()).float()[None])
                if left.numel() > 1:
                    lp = left[:, None] - left[None, :]
                    rp = right[:, None] - right[None, :]
                    upper = torch.triu(torch.ones_like(lp, dtype=torch.bool), diagonal=1)
                    self.ranking.update(((lp * rp) >= 0)[upper].float())

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": self.log_z.n,
            "log_z_correlation": self.log_z.correlation(),
            "absolute_log_z_error": self.log_absolute.to_dict(),
            "normalized_mass_correlation": self.mass.correlation(),
            "relative_mass_error": self.relative.to_dict(),
            "worst_veto_row_relative_mass_error": self.worst_veto.to_dict(),
            "candidate_pair_ranking_accuracy": self.ranking.to_dict()["mean"],
            "candidate_top1_ranking_accuracy": self.top1.to_dict()["mean"],
        }


class ValueMetrics:
    def __init__(self) -> None:
        self.cosine = ScalarMoments()
        self.relative = ScalarMoments()
        self.contribution = ScalarMoments()
        self.worst_veto = ScalarMoments()

    def update(
        self,
        predicted: torch.Tensor,
        exact: torch.Tensor,
        exact_mass: torch.Tensor,
        output: torch.Tensor,
        selected_rows: torch.Tensor,
        candidates: torch.Tensor,
        veto_rows: torch.Tensor,
    ) -> None:
        cosine = F.cosine_similarity(predicted.float(), exact.float(), dim=-1)
        relative = torch.linalg.vector_norm(predicted.float() - exact.float(), dim=-1) / (
            torch.linalg.vector_norm(exact.float(), dim=-1) + 1e-8
        )
        contribution = (
            exact_mass
            * torch.linalg.vector_norm(predicted.float() - exact.float(), dim=-1)
            / (torch.linalg.vector_norm(output.float(), dim=-1)[:, :, None] + 1e-8)
        )
        self.cosine.update(cosine[selected_rows])
        self.relative.update(relative[selected_rows])
        self.contribution.update(contribution[selected_rows])
        veto_selected = candidates[..., None] & veto_rows
        worst = torch.where(veto_selected, relative, 0.0).amax(-1)
        self.worst_veto.update(worst[candidates])

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": self.cosine.n,
            "u_cosine": self.cosine.to_dict(),
            "relative_u_error": self.relative.to_dict(),
            "relative_contribution_error": self.contribution.to_dict(),
            "worst_veto_row_relative_u_error": self.worst_veto.to_dict(),
        }


class AttentionMetrics:
    def __init__(self) -> None:
        self.relative = ScalarMoments()
        self.cosine = ScalarMoments()
        self.maximum_absolute = 0.0
        self.denominator = ScalarMoments()
        self.numerator = ScalarMoments()

    def update(self, candidate, reference) -> None:
        difference = candidate.output.float() - reference.output.float()
        relative = torch.linalg.vector_norm(difference, dim=-1) / (
            torch.linalg.vector_norm(reference.output.float(), dim=-1) + 1e-8
        )
        self.relative.update(relative)
        self.cosine.update(
            F.cosine_similarity(candidate.output.float(), reference.output.float(), dim=-1)
        )
        self.maximum_absolute = max(self.maximum_absolute, float(difference.abs().max()))
        denominator_relative = (
            (candidate.log_denominator - reference.log_denominator).exp() - 1.0
        ).abs()
        self.denominator.update(denominator_relative)
        common = torch.maximum(candidate.reference_maximum, reference.reference_maximum)
        left = candidate.scaled_numerator * torch.exp(
            candidate.reference_maximum - common
        )[..., None]
        right = reference.scaled_numerator * torch.exp(
            reference.reference_maximum - common
        )[..., None]
        numerator_relative = torch.linalg.vector_norm(left - right, dim=-1) / (
            torch.linalg.vector_norm(right, dim=-1) + 1e-8
        )
        self.numerator.update(numerator_relative)

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": self.relative.n,
            "relative_output_error": self.relative.to_dict(),
            "row_cosine": self.cosine.to_dict(),
            "maximum_absolute_output_error": self.maximum_absolute,
            "relative_denominator_error": self.denominator.to_dict(),
            "relative_numerator_error": self.numerator.to_dict(),
        }


class PolicyMetrics:
    def __init__(self) -> None:
        self.attention = AttentionMetrics()
        self.physical = 0
        self.kept = 0
        self.candidates = 0
        self.replaced = 0

    def update(self, candidate, reference, decisions, candidates, replaced) -> None:
        self.attention.update(candidate, reference)
        self.physical += candidates.numel()
        self.kept += int(decisions.keep.sum())
        self.candidates += int(candidates.sum())
        self.replaced += int(replaced.sum())

    def to_dict(self) -> dict[str, object]:
        return {
            **self.attention.to_dict(),
            "physical_tiles": self.physical,
            "baseline_kept_tiles": self.kept,
            "candidate_tiles": self.candidates,
            "replaced_tiles": self.replaced,
            "physical_replacement_fraction": self.replaced / max(self.physical, 1),
            "kept_replacement_fraction": self.replaced / max(self.kept, 1),
        }


class Benchmark:
    def __init__(self, k_values: tuple[int, ...]) -> None:
        self.k_values = k_values
        self.mass: dict[tuple[str, str], MassMetrics] = defaultdict(MassMetrics)
        self.value: dict[tuple[str, str, str], ValueMetrics] = defaultdict(ValueMetrics)
        self.attention: dict[tuple[str, str, str], AttentionMetrics] = defaultdict(
            AttentionMetrics
        )
        self.policy: dict[tuple[str, str, str, str], PolicyMetrics] = defaultdict(
            PolicyMetrics
        )
        self.coverage: dict[tuple[str, int], dict[str, int]] = defaultdict(
            lambda: {"physical": 0, "kept": 0, "candidates": 0}
        )
        self.breakdown: list[dict[str, object]] = []
        self.failure_breakdown: list[dict[str, object]] = []
        self.calibration_log_bias: dict[str, ScalarMoments] = defaultdict(ScalarMoments)

    def analyze(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        threshold: float,
        split: str,
        context: int,
        ratio: float,
        layer: int,
        heads: tuple[int, ...],
        query_tile: int,
        row_masked: torch.Tensor,
        softmax_scale: float | None,
    ) -> None:
        statistics = collect_tile_replacement_statistics(
            q, k, v, softmax_scale=softmax_scale
        )
        decisions = simulate_ordinary_blasst(statistics, threshold)
        logits = tiled_logits(q, k, v, softmax_scale=softmax_scale)
        exact_lz = exact_log_z(statistics)
        exact_mass = statistics.final_mass
        exact_u = exact_conditional_value(statistics)
        baseline = compose_replacement_state(statistics, decisions.keep)
        candidates_by_k = {value: candidate_tiles(decisions, value) for value in self.k_values}
        for value, candidates in candidates_by_k.items():
            coverage = self.coverage[(split, value)]
            coverage["physical"] += candidates.numel()
            coverage["kept"] += int(decisions.keep.sum())
            coverage["candidates"] += int(candidates.sum())
        candidates = candidates_by_k[8]

        log_z_predictions = analytic_log_z_predictions(logits)
        value_predictions = simple_value_predictions(logits, statistics)
        exact_capacity_log_z, exact_capacity_u = per_instance_entropy_identity(logits)
        log_z_predictions["per_instance_entropy_oracle"] = exact_capacity_log_z
        value_predictions["per_instance_entropy_oracle"] = exact_capacity_u

        summary_values: dict[str, torch.Tensor] = {}
        for slots in (1, 2, 4, 8):
            builders = {
                "uniform": uniform_pool_summaries(
                    k, v, query_heads=q.shape[2], slots=slots
                ),
                "kmeans_k": kmeans_pool_summaries(
                    k, v, query_heads=q.shape[2], slots=slots
                ),
                "kmeans_joint": kmeans_pool_summaries(
                    k, v, query_heads=q.shape[2], slots=slots, joint=True
                ),
                "proxy_mean_q": proxy_query_summaries(q, k, v, slots=slots),
                "proxy_mean_veto_q": proxy_query_summaries(
                    q, k, v, slots=slots, proxy_rows=decisions.veto_rows
                ),
            }
            for family, summary in builders.items():
                predicted_lz, predicted_u = summary_predictions(
                    q, summary, softmax_scale=softmax_scale
                )
                name = f"{family}_r{slots}"
                log_z_predictions[name] = predicted_lz
                value_predictions[name] = predicted_u
                if family in ("uniform", "kmeans_k", "kmeans_joint"):
                    summary_values[name] = predicted_u
            value_predictions[f"oracle_row_centroid_r{slots}"] = oracle_row_centroids(
                exact_u, slots
            )

        calibration_selected = candidates[..., None] & decisions.veto_rows
        for method in ("maximum_logit", "maximum_logit_count", "mean_variance"):
            if split == "calibration":
                self.calibration_log_bias[method].update(
                    (exact_lz - log_z_predictions[method])[calibration_selected]
                )
            elif self.calibration_log_bias[method].n:
                bias = self.calibration_log_bias[method].total / self.calibration_log_bias[method].n
                log_z_predictions[f"calibrated_{method}"] = log_z_predictions[method] + bias

        online_mass = online_proxy_score(statistics, decisions, decisions.traversal)
        online_log = online_mass.clamp(1e-30, 1e30).log()
        for name, predicted_lz in log_z_predictions.items():
            predicted_mass = normalized_mass_from_log_z(predicted_lz)
            self.mass[(split, name)].update(
                predicted_lz, predicted_mass, exact_lz, exact_mass, candidates, decisions.veto_rows
            )
        self.mass[(split, "existing_online_proxy")].update(
            online_log,
            online_mass,
            exact_mass.clamp_min(1e-30).log(),
            exact_mass,
            candidates,
            decisions.veto_rows,
        )

        masked = row_masked[:, None, None].expand_as(statistics.m)
        for scope, selected_rows in (
            ("veto_only", candidates[..., None] & decisions.veto_rows),
            ("all_rows", candidates[..., None].expand_as(statistics.m)),
        ):
            for name, predicted_u in value_predictions.items():
                self.value[(split, scope, name)].update(
                    predicted_u,
                    exact_u,
                    exact_mass,
                    baseline.output,
                    selected_rows,
                    candidates,
                    decisions.veto_rows,
                )

        # Required joint matrix. Summary families supply both mass and value;
        # the exact/per-instance entries are explicit diagnostic upper bounds.
        combinations: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
            "exact_mass__zero_value": (exact_lz, value_predictions["zero"]),
            "exact_mass__mean_v": (exact_lz, value_predictions["mean_v"]),
            "maximum_logit_mass__exact_u": (
                log_z_predictions["maximum_logit"],
                exact_u,
            ),
            "maximum_count_mass__exact_u": (
                log_z_predictions["maximum_logit_count"],
                exact_u,
            ),
            "mean_variance_mass__exact_u": (
                log_z_predictions["mean_variance"],
                exact_u,
            ),
            "online_mass__uniform_r8": (
                statistics.m + math.log(64),
                value_predictions["uniform_r8"],
            ),
            "exact_mass__exact_u": (exact_lz, exact_u),
        }
        for method in (
            "calibrated_maximum_logit",
            "calibrated_maximum_logit_count",
            "calibrated_mean_variance",
        ):
            if method in log_z_predictions:
                combinations[f"{method}_mass__exact_u"] = (
                    log_z_predictions[method],
                    exact_u,
                )
                combinations[f"{method}_mass__proxy_mean_q_r8"] = (
                    log_z_predictions[method],
                    value_predictions["proxy_mean_q_r8"],
                )
        for slots in (1, 2, 4, 8):
            combinations[f"exact_mass__uniform_r{slots}"] = (
                exact_lz,
                value_predictions[f"uniform_r{slots}"],
            )
            combinations[f"exact_mass__kmeans_k_r{slots}"] = (
                exact_lz,
                value_predictions[f"kmeans_k_r{slots}"],
            )
            combinations[f"exact_mass__proxy_mean_q_r{slots}"] = (
                exact_lz,
                value_predictions[f"proxy_mean_q_r{slots}"],
            )
            combinations[f"exact_mass__proxy_mean_veto_q_r{slots}"] = (
                exact_lz,
                value_predictions[f"proxy_mean_veto_q_r{slots}"],
            )
            combinations[f"exact_mass__oracle_centroid_r{slots}"] = (
                exact_lz,
                value_predictions[f"oracle_row_centroid_r{slots}"],
            )
            combinations[f"kmeans_mass_r{slots}__exact_u"] = (
                log_z_predictions[f"kmeans_k_r{slots}"],
                exact_u,
            )
            combinations[f"kmeans_joint_r{slots}"] = (
                log_z_predictions[f"kmeans_joint_r{slots}"],
                value_predictions[f"kmeans_joint_r{slots}"],
            )
            combinations[f"uniform_joint_r{slots}"] = (
                log_z_predictions[f"uniform_r{slots}"],
                value_predictions[f"uniform_r{slots}"],
            )

        for scope in ("veto_only", "all_rows"):
            rows = replacement_rows(decisions, candidates, scope)
            for name, (predicted_lz, predicted_u) in combinations.items():
                replacement = ReplacementComponents(predicted_lz, predicted_u, rows)
                candidate = compose_replacement_state(
                    statistics, decisions.keep & ~candidates, replacement
                )
                self.attention[(split, scope, name)].update(candidate, baseline)

        predicted_log_z = log_z_predictions.get(
            "calibrated_maximum_logit", log_z_predictions["maximum_logit"]
        )
        predicted_mass = normalized_mass_from_log_z(predicted_log_z)
        predicted_u = value_predictions["proxy_mean_q_r8"]
        predicted_row_error = predicted_mass * torch.linalg.vector_norm(
            predicted_u, dim=-1
        ) / (torch.linalg.vector_norm(baseline.output, dim=-1)[:, :, None] + 1e-8)
        predicted_tile_error = predicted_row_error.amax(-1)
        exact_all_max = exact_mass.amax(-1)
        exact_veto_max = torch.where(
            decisions.veto_rows, exact_mass, -torch.inf
        ).amax(-1)
        policy_candidates: dict[str, torch.Tensor] = {"all_candidates": candidates}
        for threshold_value in (0.001, 0.003, 0.01, 0.03):
            tag = f"{threshold_value:g}"
            policy_candidates[f"exact_allmax_lt_{tag}"] = candidates & (
                exact_all_max < threshold_value
            )
            policy_candidates[f"exact_vetomax_lt_{tag}"] = candidates & (
                exact_veto_max < threshold_value
            )
            policy_candidates[f"predicted_maxmass_lt_{tag}"] = candidates & (
                predicted_mass.amax(-1) < threshold_value
            )
            policy_candidates[f"predicted_error_lt_{tag}"] = candidates & (
                predicted_tile_error < threshold_value
            )
        for budget in (0.003, 0.01, 0.03):
            selected, _ = select_cumulative_budget(
                candidates,
                exact_mass,
                budget=budget,
                order="score",
                score=predicted_tile_error,
            )
            policy_candidates[f"cumulative_exact_mass_{budget:g}"] = selected
            selected, _ = select_cumulative_budget(
                candidates,
                predicted_row_error,
                budget=budget,
                order="score",
                score=predicted_tile_error,
            )
            policy_candidates[f"cumulative_predicted_error_{budget:g}"] = selected

        protection_masks = {
            "none": torch.zeros_like(candidates),
            "veto_rows": (
                decisions.introduced_new_max_rows & decisions.veto_rows
            ).any(-1),
            "all_rows": decisions.introduced_new_max_rows.any(-1),
        }
        for protection, protected in protection_masks.items():
            for policy_name, eligible in policy_candidates.items():
                selected = eligible & ~protected
                for scope in ("veto_only", "all_rows"):
                    rows = replacement_rows(decisions, selected, scope)
                    replacement = ReplacementComponents(
                        predicted_log_z, predicted_u, rows
                    )
                    candidate = compose_replacement_state(
                        statistics, decisions.keep & ~selected, replacement
                    )
                    self.policy[(split, scope, policy_name, protection)].update(
                        candidate,
                        baseline,
                        decisions,
                        candidates,
                        selected,
                    )

        compact_rows = replacement_rows(decisions, candidates, "all_rows")
        compact_state = compose_replacement_state(
            statistics,
            decisions.keep & ~candidates,
            ReplacementComponents(predicted_log_z, predicted_u, compact_rows),
        )
        veto_oracle_rows = replacement_rows(decisions, candidates, "veto_only")
        veto_oracle_state = compose_replacement_state(
            statistics,
            decisions.keep & ~candidates,
            ReplacementComponents(exact_lz, exact_u, veto_oracle_rows),
        )
        compact_relative = torch.linalg.vector_norm(
            compact_state.output - baseline.output, dim=-1
        ) / (torch.linalg.vector_norm(baseline.output, dim=-1) + 1e-8)
        compact_cosine = F.cosine_similarity(
            compact_state.output, baseline.output, dim=-1
        )
        veto_oracle_relative = torch.linalg.vector_norm(
            veto_oracle_state.output - baseline.output, dim=-1
        ) / (torch.linalg.vector_norm(baseline.output, dim=-1) + 1e-8)
        mass_relative = (predicted_mass - exact_mass).abs() / exact_mass.clamp_min(1e-12)
        value_relative = torch.linalg.vector_norm(predicted_u - exact_u, dim=-1) / (
            torch.linalg.vector_norm(exact_u, dim=-1) + 1e-8
        )
        if row_masked.shape != compact_relative[:, 0].shape:
            raise ValueError(
                "row_masked must have shape [batch, query rows] for failure analysis"
            )
        for head_index, head in enumerate(heads):
            head_candidates = candidates[:, head_index]
            veto_selected = head_candidates[..., None] & decisions.veto_rows[:, head_index]
            masked_rows = row_masked

            def selected_mean(values, selection) -> float:
                return float(values[selection].mean()) if bool(selection.any()) else 0.0

            self.failure_breakdown.append(
                {
                    "split": split,
                    "context": context,
                    "remaining_mask_ratio": ratio,
                    "noise_phase": "high" if ratio >= 0.7 else "mid" if ratio >= 0.3 else "low",
                    "layer": layer,
                    "head": head,
                    "query_tile": query_tile,
                    "candidate_tiles": int(head_candidates.sum()),
                    "new_maximum_candidates": int(
                        (
                            head_candidates
                            & decisions.introduced_new_max_rows[:, head_index].any(-1)
                        ).sum()
                    ),
                    "mean_veto_row_relative_mass_error": selected_mean(
                        mass_relative[:, head_index], veto_selected
                    ),
                    "mean_veto_row_relative_value_error": selected_mean(
                        value_relative[:, head_index], veto_selected
                    ),
                    "compact_mean_relative_output_error": float(
                        compact_relative[:, head_index].mean()
                    ),
                    "compact_masked_mean_relative_output_error": selected_mean(
                        compact_relative[:, head_index], masked_rows
                    ),
                    "compact_visible_mean_relative_output_error": selected_mean(
                        compact_relative[:, head_index], ~masked_rows
                    ),
                    "compact_minimum_row_cosine": float(
                        compact_cosine[:, head_index].amin()
                    ),
                    "veto_oracle_mean_relative_output_error": float(
                        veto_oracle_relative[:, head_index].mean()
                    ),
                }
            )

        self.breakdown.append(
            {
                "split": split,
                "context": context,
                "remaining_mask_ratio": ratio,
                "noise_phase": "high" if ratio >= 0.7 else "mid" if ratio >= 0.3 else "low",
                "layer": layer,
                "heads": list(heads),
                "query_tile": query_tile,
                "masked_rows": int(masked[:, :, 0].sum()),
                "baseline_kept_tiles": int(decisions.keep.sum()),
                "k8_candidate_tiles": int(candidates.sum()),
                "k8_new_maximum_candidates": int(
                    (candidates & decisions.introduced_new_max_rows.any(-1)).sum()
                ),
                "veto_count_histogram": {
                    str(value): int((decisions.veto_count[candidates] == value).sum())
                    for value in range(1, 9)
                },
            }
        )

    def payload(self, metadata: dict[str, object]) -> dict[str, object]:
        return {
            "schema": "blasst-tile-replacement-predictor-v1",
            **metadata,
            "coverage": [
                {
                    "split": split,
                    "k": k,
                    **values,
                    "physical_replacement_fraction": values["candidates"]
                    / max(values["physical"], 1),
                    "kept_replacement_fraction": values["candidates"]
                    / max(values["kept"], 1),
                }
                for (split, k), values in sorted(self.coverage.items())
            ],
            "mass_predictors": [
                {"split": split, "method": method, **metrics.to_dict()}
                for (split, method), metrics in sorted(self.mass.items())
            ],
            "value_predictors": [
                {
                    "split": split,
                    "scope": scope,
                    "method": method,
                    **metrics.to_dict(),
                }
                for (split, scope, method), metrics in sorted(self.value.items())
            ],
            "replacement_matrix": [
                {
                    "split": split,
                    "scope": scope,
                    "method": method,
                    **metrics.to_dict(),
                }
                for (split, scope, method), metrics in sorted(self.attention.items())
            ],
            "policy_sweep": [
                {
                    "split": split,
                    "scope": scope,
                    "policy": policy,
                    "new_maximum_protection": protection,
                    **metrics.to_dict(),
                }
                for (split, scope, policy, protection), metrics in sorted(
                    self.policy.items()
                )
            ],
            "breakdown": self.breakdown,
            "failure_breakdown": self.failure_breakdown,
            "calibration_log_z_bias": {
                method: values.to_dict()
                for method, values in sorted(self.calibration_log_bias.items())
            },
        }


@contextmanager
def install_collector(
    model: torch.nn.Module,
    benchmark: Benchmark,
    context: dict[str, object],
    selected_layers: set[int],
    selected_heads: tuple[int, ...],
) -> Iterator[None]:
    modules = [module for module in model.modules() if hasattr(module, "flash_attn_func")]
    originals = [(module, module.flash_attn_func) for module in modules]
    for layer, (module, original) in enumerate(originals):
        def call(q, k, v, *, _layer=layer, _original=original, **kwargs):
            if _original is not None:
                output = _original(q, k, v, **kwargs)
            else:
                repeats = q.shape[2] // k.shape[2]
                kh = k.repeat_interleave(repeats, dim=2).transpose(1, 2)
                vh = v.repeat_interleave(repeats, dim=2).transpose(1, 2)
                output = F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    kh,
                    vh,
                    dropout_p=float(kwargs.get("dropout_p", 0.0)),
                    is_causal=bool(kwargs.get("causal", False)),
                    scale=kwargs.get("softmax_scale"),
                ).transpose(1, 2)
            if _layer in selected_layers:
                heads = tuple(head for head in selected_heads if head < q.shape[2])
                repeats = q.shape[2] // k.shape[2]
                kv_heads = tuple(head // repeats for head in heads)
                q_start = int(context["query_tile"]) * 128
                q_end = q_start + 128
                benchmark.analyze(
                    q[:, q_start:q_end, heads],
                    k[:, :, kv_heads],
                    v[:, :, kv_heads],
                    threshold=float(context["threshold"]),
                    split=str(context["split"]),
                    context=int(context["context"]),
                    ratio=float(context["ratio"]),
                    layer=_layer,
                    heads=heads,
                    query_tile=int(context["query_tile"]),
                    row_masked=context["row_masked"][:, q_start:q_end],
                    softmax_scale=kwargs.get("softmax_scale"),
                )
            return output

        module.flash_attn_func = call
    try:
        yield
    finally:
        for module, original in originals:
            module.flash_attn_func = original


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--num-contexts", type=int, default=12)
    parser.add_argument("--calibration-contexts", type=int, default=4)
    parser.add_argument("--ratios", default="0.9,0.5,0.15")
    parser.add_argument("--layers", type=parse_ints, default=(0, 15, 31))
    parser.add_argument("--heads", type=parse_ints, default=(0, 7, 15, 31))
    parser.add_argument("--k-values", type=parse_ints, default=(1, 2, 4, 8, 16))
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("predictor benchmark requires CUDA")
    if not 0 < args.calibration_contexts < args.num_contexts:
        raise ValueError("calibration contexts must leave a non-empty test split")
    ratios = tuple(float(item) for item in args.ratios.split(",") if item.strip())

    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.flash_attention = True
    with patch_llada_transformers_compat():
        model = AutoModelForCausalLM.from_pretrained(
            args.model, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
    ensure_all_tied_weights_keys(model)
    disable_use_cache(model)
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    clean = cyclic_contexts(
        tokenizer(corpus(), add_special_tokens=False)["input_ids"],
        args.context_length,
        args.num_contexts,
    )
    mask_id = choose_mask_token_id(tokenizer, None)
    priorities = torch.stack(
        [
            torch.rand(
                args.context_length,
                generator=torch.Generator().manual_seed(args.seed + index),
            )
            for index in range(args.num_contexts)
        ]
    )
    schedule = DiffusionLambdaSchedule()
    benchmark = Benchmark(args.k_values)
    for index in range(args.num_contexts):
        digest = hashlib.blake2b(f"context-{index}".encode(), digest_size=4).digest()
        query_tile = int.from_bytes(digest, "little") % (args.context_length // 128)
        split = "calibration" if index < args.calibration_contexts else "test"
        for ratio in ratios:
            row_masked = priorities[index : index + 1] < ratio
            state = clean[index : index + 1].clone()
            state[row_masked] = mask_id
            state = state.cuda()
            row_masked = row_masked.cuda()
            context: dict[str, object] = {
                "context": index,
                "split": split,
                "ratio": ratio,
                "threshold": float(schedule.threshold(ratio)),
                "query_tile": query_tile,
                "row_masked": row_masked,
            }
            with install_collector(
                model, benchmark, context, set(args.layers), args.heads
            ), torch.inference_mode():
                model(input_ids=state)
            print(
                f"context={index} split={split} ratio={ratio:g} query_tile={query_tile}",
                flush=True,
            )

    payload = benchmark.payload(
        {
            "model": args.model,
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "context_length": args.context_length,
            "num_contexts": args.num_contexts,
            "calibration_contexts": args.calibration_contexts,
            "test_contexts": args.num_contexts - args.calibration_contexts,
            "ratios": ratios,
            "layers": args.layers,
            "heads": args.heads,
            "k_values": args.k_values,
            "seed": args.seed,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "groups": len(payload["breakdown"])}, indent=2))


if __name__ == "__main__":
    main()
