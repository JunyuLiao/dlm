"""Native generation under an explicit deterministic Inductor numerical policy.

No attention/input/sampler observation or CUDA-graph dispatch override. Kernel
configuration receipts are emitted on exit, outside measured requests.
"""
import os
import atexit
import functools
import hashlib
import json
import inspect
from pathlib import Path
import runpy


def install_kernel_inventory(path, policy):
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
            policy=policy,
            kernels=sorted(rows, key=lambda x:(x['file'] or '', x['name'] or ''))), indent=2)+'\n')
    CachingAutotuner.precompile = precompile
    atexit.register(save)


def compilation_options(current):
    if current is not None and not isinstance(current, dict):
        raise RuntimeError('this frozen native worker expects dictionary compilation settings')
    options = dict(current or {})
    options['inductor_compile_config'] = dict(options.get('inductor_compile_config') or {},
        deterministic=True, benchmark_combo_kernel=False)
    return options


def recompile_native_helpers():
    import torch
    import vllm.model_executor.models.diffusion_gemma as dg
    from vllm.platforms import current_platform
    if current_platform.simple_compile_backend != 'inductor':
        raise RuntimeError('this numerical policy is pinned to the native Inductor backend')
    receipts = {}
    options = dict(deterministic=True, benchmark_combo_kernel=False)
    for name in ('_softcap_logits', '_compute_num_rejected', '_compiled_sample_step'):
        previous = getattr(dg, name)
        native = getattr(previous, '_torchdynamo_orig_callable', None)
        if native is None or native.__module__ != dg.__name__:
            raise RuntimeError('expected the untouched native compiled helper')
        replacement = torch.compile(native, dynamic=True,
            backend=current_platform.simple_compile_backend, options=options)
        if replacement._torchdynamo_orig_callable is not native:
            raise RuntimeError('native helper Python identity changed')
        setattr(dg, name, replacement)
        receipts[name] = dict(source_sha256=hashlib.sha256(inspect.getsource(native).encode()).hexdigest(),
            backend=current_platform.simple_compile_backend, dynamic=True, options=options,
            original_python_callable_preserved=True)
    return receipts


def install_compilation_policy():
    from torch._inductor import config
    from vllm import LLM
    config.deterministic = True
    original = LLM.__init__
    policy = dict(name='explicit_inductor_cfgdet', native_sampler_changed=False,
                  extra_forwards=0, intermediate_observer=False)
    policy['native_helpers'] = recompile_native_helpers()

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        kwargs['compilation_config'] = compilation_options(kwargs.get('compilation_config'))
        config.deterministic = True
        result = original(self, *args, **kwargs)
        # A model's scoped compilation options do not cover the separately
        # compiled native sampler. Keep its process-level compiler policy too.
        policy['global_after_engine_init_before_restore'] = config.deterministic
        config.deterministic = True
        effective = self.llm_engine.vllm_config.compilation_config
        policy['effective_cudagraph_mode'] = effective.cudagraph_mode.name
        policy['effective_inductor_compile_config'] = effective.inductor_compile_config
        return result
    LLM.__init__ = initialize
    return policy


if __name__ == '__main__':
    if os.environ.get('TORCHINDUCTOR_DETERMINISTIC') != '1':
        raise RuntimeError('this named worker requires deterministic Inductor policy')
    policy = install_compilation_policy()
    install_kernel_inventory(Path(os.environ['HUMANEVAL_KERNEL_INVENTORY']), policy)
    runpy.run_path(os.environ['HUMANEVAL_REPRO_BASE'], run_name='__main__')
