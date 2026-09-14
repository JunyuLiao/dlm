"""Fixed protocol for the 50-problem prefix-region comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass

DEFAULT_MODEL = "google/diffusiongemma-26B-A4B-it"
DEFAULT_REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"
TARGET_SPARSITIES = (0.25, 0.50, 0.75, 0.90)
ANALYTIC_BETAS = {
    0.25: -0.674490,
    0.50: 0.0,
    0.75: 0.674490,
    0.90: 1.281552,
}


@dataclass(frozen=True)
class Condition:
    name: str
    region: str | None
    target_sparsity: float
    retained_density: float
    beta: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def conditions() -> list[Condition]:
    result = [Condition("dense", None, 0.0, 1.0, None)]
    for region, prefix in (("prefix_only", "sol_prefix"), ("all", "sol_prefix_canvas")):
        for sparsity in TARGET_SPARSITIES:
            result.append(Condition(
                f"{prefix}_s{int(sparsity * 100)}",
                region,
                sparsity,
                1.0 - sparsity,
                ANALYTIC_BETAS[sparsity],
            ))
    return result


def condition_map() -> dict[str, Condition]:
    return {item.name: item for item in conditions()}
