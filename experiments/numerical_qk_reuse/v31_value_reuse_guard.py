"""Causal value-residual trust and a selected-output-only reconstruction.

No query extrapolation, future output, or current omitted-key scan is used.
These are reference building blocks; native promotion needs the decision gate.
"""
from dataclasses import dataclass
import torch


def residual_trust(current_selected, anchor_selected, anchor_dense, *, kind='direction'):
    """A causal surrogate, not a bound on the unknown current dense output.

    With r = O_dense(anchor)-O_selected(anchor) and d =
    O_selected(now)-O_dense(anchor), minimize ||d + alpha*r||^2 over [0,1].
    The direction guard shrinks correction when current selected output has
    already moved toward the old dense anchor. Norm is a conservative ablation.
    Inputs should be *whole-layer post-O* vectors so head cancellation counts.
    """
    r = anchor_dense - anchor_selected
    move = current_selected - anchor_selected
    r2 = r.square().sum(-1)
    if kind == 'direction':
        alpha = 1 - (move * r).sum(-1) / r2.clamp_min(1e-20)
    elif kind == 'norm':
        alpha = 1 - move.norm(dim=-1) / r2.sqrt().clamp_min(1e-10)
    else:
        raise ValueError('unknown trust surrogate')
    return torch.where(r2 > 1e-20, alpha.clamp(0, 1), torch.zeros_like(alpha))


@dataclass
class ResidualDirectionCache:
    """Exact adjoint form of the post-O direction guard, without a held-step O GEMM.

    Store u = W.T W r at the observation. Then <W(St-S0), W r> =
    <St-S0,u>. Flatten heads together so signed cross-head effects survive.
    """
    anchor_selected: torch.Tensor
    residual: torch.Tensor
    dual: torch.Tensor
    residual_squared: torch.Tensor
    identity: object

    @classmethod
    def build(cls, dense, selected, projection, identity):
        dtype=torch.float64 if dense.dtype==torch.float64 else torch.float32
        selected=selected.to(dtype).contiguous();r=dense.to(dtype)-selected
        w=projection.to(dtype)
        if dense.shape!=selected.shape or w.shape[1]!=selected[0].numel():
            raise ValueError('incompatible whole-layer projection geometry')
        post=r.reshape(len(r),-1)@w.T
        dual=(post@w).reshape_as(r).contiguous()
        return cls(selected,r.contiguous(),dual,post.square().sum(-1),identity)

    def apply(self, current, identity):
        if identity!=self.identity or current.shape!=self.residual.shape:
            raise ValueError('canvas/support or output geometry changed; rebuild residual cache')
        movement=((current.to(self.dual.dtype)-self.anchor_selected)*self.dual).flatten(1).sum(-1)
        alpha=torch.where(self.residual_squared>1e-20,
            (1-movement/self.residual_squared.clamp_min(1e-20)).clamp(0,1),0)
        shape=(len(alpha),)+(1,)*(current.ndim-1)
        return (current.to(self.dual.dtype)+alpha.reshape(shape)*self.residual).to(current.dtype),alpha

    def apply_cuda(self,current,identity):
        if identity!=self.identity or current.shape!=self.residual.shape:
            raise ValueError('canvas/support or output geometry changed; rebuild residual cache')
        if not current.is_cuda:
            raise ValueError('CUDA tensor required')
        import triton
        from .v31_value_reuse_merge_kernel import direction_guard_kernel
        current=current.contiguous();out=torch.empty_like(current)
        alpha=torch.empty(len(current),device=current.device,dtype=torch.float32)
        d=current[0].numel()
        direction_guard_kernel[(len(current),)](current,self.anchor_selected,self.residual,self.dual,
            self.residual_squared,out,alpha,D=d,BLOCK=triton.next_power_of_2(d),num_warps=8)
        return out,alpha


@dataclass
class SelectedOnlyCorrection:
    """Cache omitted moments at an observation; consume only sparse O and LSE.

    Shapes are [rows,heads,D] for outputs and [rows,heads] for log Z. The cache
    carries a canvas/support identity so a changed complement cannot be reused.
    """
    omitted_mean: torch.Tensor
    omitted_lse: torch.Tensor
    identity: object

    @classmethod
    def from_anchor(cls, dense, dense_lse, selected, selected_lse, identity):
        share = (selected_lse.float() - dense_lse.float()).exp().clamp(0, 1)
        missing = 1 - share
        numerator = dense.float() - share[..., None] * selected.float()
        live = missing > 1e-6
        mean = torch.where(live[..., None], numerator / missing.clamp_min(1e-30)[..., None], 0)
        lse = torch.where(live, dense_lse.float() + missing.clamp_min(1e-30).log(), -torch.inf)
        return cls(mean, lse, identity)

    def reconstruct(self, selected, selected_lse, identity):
        if identity != self.identity:
            raise ValueError('canvas or support changed; rebuild omitted cache')
        total_lse = torch.logaddexp(selected_lse.float(), self.omitted_lse)
        a = (selected_lse.float() - total_lse).exp()
        b = (self.omitted_lse - total_lse).exp()
        return a[..., None] * selected.float() + b[..., None] * self.omitted_mean


def reconstruct_cuda(selected, selected_lse, cache, identity):
    """One fused merge kernel; no QK, PV, or whole-prefix tensor is accessed."""
    if identity != cache.identity:
        raise ValueError('canvas or support changed; rebuild omitted cache')
    if not selected.is_cuda:
        raise ValueError('CUDA tensor required')
    import triton
    from .v31_value_reuse_merge_kernel import merge_kernel
    shape = selected.shape
    if selected_lse.shape != shape[:-1] or cache.omitted_mean.shape != shape:
        raise ValueError('incompatible correction layout')
    selected = selected.contiguous(); selected_lse = selected_lse.contiguous()
    mean, old_lse = cache.omitted_mean.contiguous(), cache.omitted_lse.contiguous()
    out = torch.empty_like(selected)
    merge_kernel[(selected.numel() // shape[-1],)](selected, selected_lse, mean, old_lse, out,
        D=shape[-1], BLOCK=triton.next_power_of_2(shape[-1]))
    return out
