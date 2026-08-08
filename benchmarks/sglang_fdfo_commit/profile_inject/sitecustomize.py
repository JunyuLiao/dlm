"""Load the FDFO profiler in child SGLang processes when explicitly enabled."""

from __future__ import annotations

import os
import sys
import traceback


if os.environ.get("SGLANG_FDFO_PROFILE_DIR"):
    try:
        from profiler_patch import install

        install()
    except Exception:  # pragma: no cover - runs in an external SGLang environment
        traceback.print_exc(file=sys.stderr)

