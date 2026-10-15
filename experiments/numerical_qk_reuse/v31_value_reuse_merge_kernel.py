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
