"""Device-only bounded ranking for the unchanged disjoint-swap rule."""
import triton
import triton.language as tl


@triton.jit
def disjoint_kernel(P, L, R, I, V, N:tl.constexpr, LIMIT:tl.constexpr, BLOCK:tl.constexpr):
    j = tl.arange(0, BLOCK)
    loss = tl.load(L+j, j<N, float('inf'))
    adds = tl.load(P+2*j, j<N, -1)
    rems = tl.load(P+2*j+1, j<N, -1)
    eligible = (j<N) & (loss < tl.load(R))
    for step in range(LIMIT):
        minimum = tl.min(tl.where(eligible, loss, float('inf')), 0)
        index = tl.min(tl.where(eligible & (loss==minimum), j, BLOCK), 0)
        take = index < N
        index = tl.minimum(index, N-1)
        add = tl.load(P+2*index)
        rem = tl.load(P+2*index+1)
        tl.store(I+step, index)
        tl.store(V+step, take)
        eligible = eligible & (adds!=add) & (rems!=rem) & take
