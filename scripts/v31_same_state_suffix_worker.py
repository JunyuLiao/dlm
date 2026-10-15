"""Opt-in diagnostic wrapper; delegates unchanged generation to the native bench."""
import os
from pathlib import Path
import runpy
import functools

from experiments.numerical_qk_reuse.v31_same_state_suffix import install


if __name__ == '__main__':
    directory = os.environ['SAME_STATE_SUFFIX_DIR']
    if os.environ.get('SUFFIX_EAGER')=='1':
        from vllm import LLM
        original_init=LLM.__init__
        @functools.wraps(original_init)
        def init(self,*args,**kwargs):
            kwargs['enforce_eager']=True
            return original_init(self,*args,**kwargs)
        LLM.__init__=init
    # The bench assigns its trajectory recorder per real request, after warmup.
    if os.environ.get('SUFFIX_INSTRUMENT','1')=='1':
        os.environ['STABLE_TRAJECTORY_DIR'] = directory
        controller = install(directory)
    runpy.run_path(str(Path(__file__).with_name('v31_vllm_paired_bench.py')), run_name='__main__')
