from __future__ import annotations

import sys
from pathlib import Path


V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

from scripts.eval_blasst_task import (  # noqa: E402
    _predicted_answer,
    _reference_answer,
    _wilson,
)


def test_gsm8k_answer_extraction_prefers_last_boxed_answer() -> None:
    text = r"intermediate \boxed{12}; corrected result \boxed{1,234.50}"
    assert _predicted_answer(text) == "1234.5"


def test_gsm8k_answer_extraction_has_numeric_fallback() -> None:
    assert _predicted_answer("First 12, then the answer is -3.50.") == "-3.5"
    assert _predicted_answer("No numeric result") is None
    assert _reference_answer("reasoning\n#### 1,234") == "1234"


def test_wilson_interval_contains_observed_rate() -> None:
    low, high = _wilson(3, 8)
    assert 0.0 <= low <= 3 / 8 <= high <= 1.0
