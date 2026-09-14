"""Stable protocol constants for the compact four-benchmark sweep."""

from __future__ import annotations

from dataclasses import asdict, dataclass

MODEL = "/home/exouser/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b"
REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"
TARGETS = (0.25, 0.50, 0.75, 0.90)
BETAS = {0.25: -0.674490, 0.50: 0.0, 0.75: 0.674490, 0.90: 1.281552}


@dataclass(frozen=True)
class Condition:
    name: str
    method: str
    target_sparsity: float = 0.0
    beta: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def conditions() -> list[Condition]:
    result = [Condition("dense", "dense")]
    result += [Condition(f"sol_gaussian_s{int(s * 100)}", "sol", s, BETAS[s]) for s in TARGETS]
    result += [Condition(f"blasst_length_aware_s{int(s * 100)}", "blasst", s) for s in TARGETS]
    return result


def condition_map() -> dict[str, Condition]:
    return {item.name: item for item in conditions()}
