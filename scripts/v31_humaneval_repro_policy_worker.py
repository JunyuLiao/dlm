"""Native generation under an explicit deterministic Inductor numerical policy.

No attention/input/sampler observation or CUDA-graph dispatch override. Kernel
configuration receipts are emitted on exit, outside measured requests.
"""
import os
import atexit
import functools
import hashlib
import json
from pathlib import Path
import runpy


def install_kernel_inventory(path):
    """Self-contained source so a frozen worker has no mutable helper imports."""
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    original = CachingAutotuner.precompile
    inventory = {}

    @functools.wraps(original)
    def precompile(self, *args, **kwargs):
        inventory[id(self)] = self
        return original(self, *args, **kwargs)

    def save():
        import torch
        from torch._inductor import config
        rows = []
        for kernel in inventory.values():
            filename = Path(kernel.filename) if kernel.filename else None
            rows.append(dict(name=getattr(kernel.fn, '__name__', None),
                file=filename.name if filename else None,
                source_sha256=hashlib.sha256(filename.read_bytes()).hexdigest()
                    if filename and filename.is_file() else None,
                heuristic=str(kernel.heuristic_type), deterministic=kernel.deterministic_mode,
                selected_configs=[dict(kwargs=x.config.kwargs, num_warps=x.config.num_warps,
                    num_stages=x.config.num_stages, num_ctas=x.config.num_ctas)
                    for x in kernel.launchers]))
        if path.exists():
            raise RuntimeError('immutable kernel inventory required')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(torch_inductor_deterministic=config.deterministic,
            torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            kernels=sorted(rows, key=lambda x:(x['file'] or '', x['name'] or ''))), indent=2)+'\n')
    CachingAutotuner.precompile = precompile
    atexit.register(save)


if __name__ == '__main__':
    if os.environ.get('TORCHINDUCTOR_DETERMINISTIC') != '1':
        raise RuntimeError('this named worker requires deterministic Inductor policy')
    install_kernel_inventory(Path(os.environ['HUMANEVAL_KERNEL_INVENTORY']))
    runpy.run_path(os.environ['HUMANEVAL_REPRO_BASE'], run_name='__main__')
