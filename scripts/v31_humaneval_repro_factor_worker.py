"""One-factor dense native diagnostics; no model or sampler formulas change."""
import os
import runpy


if __name__=='__main__':
    from experiments.numerical_qk_reuse.v31_value_native_audit import NativeGeometryAudit
    factor=os.environ['HUMANEVAL_REPRO_FACTOR']
    if factor=='without_geometry':
        NativeGeometryAudit.install=lambda self:None
    elif factor=='without_logit_check':
        NativeGeometryAudit.logits=lambda self,scaled:None
    elif factor in ('deterministic_fill','deterministic_no_fill'):
        import torch
        import torch.utils.deterministic
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory=(factor=='deterministic_fill')
    else:
        raise ValueError(factor)
    runpy.run_path(os.environ['HUMANEVAL_REPRO_BASE'],run_name='__main__')
