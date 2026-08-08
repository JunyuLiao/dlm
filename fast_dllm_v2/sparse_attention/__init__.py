"""Compatibility imports for the former Fast-dLLM-v2-local BLASST package."""

import sys
from pathlib import Path

_shared_source = Path(__file__).resolve().parents[2] / "src"
if str(_shared_source) not in sys.path:
    sys.path.insert(0, str(_shared_source))

from dllm.attention.blasst import *  # noqa: F401,F403
from dllm.attention.blasst.core import install_blasst_2d
