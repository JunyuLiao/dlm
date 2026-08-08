#!/usr/bin/env python3
"""Compatibility entry point for the universal scalar RULER evaluator.

The former Fast-dLLM-v2-specific sweep program is retained as
``eval_blasst_ruler_legacy.py`` only for the historical DualCache experiment.
New evaluations and sweeps must use :mod:`dllm.cli.eval_ruler` and the shell
matrices under ``scripts/ruler``.
"""

from __future__ import annotations

import sys
from pathlib import Path


SHARED_SOURCE = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SHARED_SOURCE))

from dllm.cli.eval_ruler import main  # noqa: E402


if __name__ == "__main__":
    main()

