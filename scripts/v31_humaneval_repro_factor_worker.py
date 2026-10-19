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
    else:
        raise ValueError(factor)
    runpy.run_path(os.environ['HUMANEVAL_REPRO_BASE'],run_name='__main__')
