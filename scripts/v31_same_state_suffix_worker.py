"""Opt-in diagnostic wrapper; delegates unchanged generation to the native bench."""
import os
from pathlib import Path
import runpy

from experiments.numerical_qk_reuse.v31_same_state_suffix import install


if __name__ == '__main__':
    directory = os.environ['SAME_STATE_SUFFIX_DIR']
    # The bench assigns its trajectory recorder per real request, after warmup.
    os.environ['STABLE_TRAJECTORY_DIR'] = directory
    controller = install(directory)
    runpy.run_path(str(Path(__file__).with_name('v31_vllm_paired_bench.py')), run_name='__main__')
