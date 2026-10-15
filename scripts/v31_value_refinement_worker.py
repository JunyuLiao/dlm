"""Explicit native worker for promoted final-layer refinements only."""
import json
import os
from pathlib import Path
import runpy

from experiments.numerical_qk_reuse.v31_value_refinement_runtime import install


if __name__=='__main__':
    # A coordinator must supply a concrete completed decision-gate receipt.
    gate=json.loads(Path(os.environ['VALUE_REFINEMENT_GATE']).read_text())
    mode=os.environ['VALUE_REFINEMENT']
    if gate.get('status')!='passed' or mode not in gate.get('promoted_modes',[]):
        raise RuntimeError('native accuracy requires a passed decision promotion gate')
    install(mode)
    runpy.run_path(str(Path(__file__).with_name('v31_vllm_paired_bench.py')),run_name='__main__')
