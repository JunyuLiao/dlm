"""Native generation under an explicit deterministic Inductor numerical policy.

No attention/input/sampler observation or CUDA-graph dispatch override. Kernel
configuration receipts are emitted on exit, outside measured requests.
"""
import os
from pathlib import Path
import runpy


if __name__ == '__main__':
    if os.environ.get('TORCHINDUCTOR_DETERMINISTIC') != '1':
        raise RuntimeError('this named worker requires deterministic Inductor policy')
    from scripts.v31_humaneval_repro_segment_worker import install_kernel_inventory
    install_kernel_inventory(Path(os.environ['HUMANEVAL_KERNEL_INVENTORY']))
    runpy.run_path(os.environ['HUMANEVAL_REPRO_BASE'], run_name='__main__')
