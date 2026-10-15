"""Fused cached-omission merge. Import only on a qualified CUDA runtime."""
import triton
import triton.language as tl


@triton.jit
def merge_kernel(X, L, M, A, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    current = tl.load(L + row).to(tl.float32)
    old = tl.load(A + row).to(tl.float32)
    maximum = tl.maximum(current, old)
    a, b = tl.exp(current - maximum), tl.exp(old - maximum)
    x = tl.load(X + row * D + d, d < D, 0).to(tl.float32)
    m = tl.load(M + row * D + d, d < D, 0).to(tl.float32)
    result = (a * x + b * m) / (a + b)
    tl.store(Y + row * D + d, result, d < D)


@triton.jit
def direction_guard_kernel(X,S,R,U,R2,Y,A,D:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);d=tl.arange(0,BLOCK);offset=row*D+d
    x=tl.load(X+offset,d<D,0).to(tl.float32)
    s=tl.load(S+offset,d<D,0).to(tl.float32)
    r=tl.load(R+offset,d<D,0).to(tl.float32)
    u=tl.load(U+offset,d<D,0).to(tl.float32)
    denominator=tl.load(R2+row).to(tl.float32)
    move=tl.sum((x-s)*u,0)
    alpha=tl.where(denominator>1e-20,tl.minimum(1.,tl.maximum(0.,1.-move/tl.maximum(denominator,1e-20))),0.)
    tl.store(Y+offset,x+alpha*r,d<D)
    tl.store(A+row,alpha)
