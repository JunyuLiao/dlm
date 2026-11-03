"""Overlay the current-canvas selector on the pinned native Gemma worker."""
import importlib.util
import os
from pathlib import Path
import runpy
import sys

root=Path(os.environ['CURRENT_TILE_SOURCE_ROOT'])
import experiments.numerical_qk_reuse.optimized_refresh_runtime as old
for name in ('v2_current_canvas_init','current_canvas_gemma_runtime'):
    spec=importlib.util.spec_from_file_location('experiments.numerical_qk_reuse.'+name,
        root/'experiments/numerical_qk_reuse'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
old.install=sys.modules['experiments.numerical_qk_reuse.current_canvas_gemma_runtime'].install
runpy.run_path(os.environ['CURRENT_TILE_ORIGINAL_WORKER'],run_name='__main__')
